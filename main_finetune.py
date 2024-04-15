import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_mmdiffuser import LLaMA_mmdiffuser
from llama.utils import format_prompt
# from data.dataset import FinetuneDataset, transform_train
from data.ccxm_dataset import ccxm_Finetune_Dataset, transform_train

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from engine_finetune import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')
    parser.add_argument('--diffuser_path', default='/path/to/diffuser', type=str,
                        help='path to Diffuser pretrained checkpoint')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='./configs/t2i_pretrain.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--split_epoch', type=int, default=50)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_dir="./share_ckpts/llama1_weight/"
    name = "BIAS-7B"
    phase="finetune"

    model_path = "./share_ckpts/llama-adapter/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth"
    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    # model, preprocess = llama.load("BIAS-7B", llama_dir, device)

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    # llama_ckpt_dir = "./output_qformer/pretrain_20240224_instruction/ckpt_weight"
    # llama_ckpt_dir = llama_dir
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    query_path = './output_qformer/pretrain_20240224_instruction/checkpoints/epoch0_queryblock.pkl'
    # query_path = './checkpoints/20231106/epoch2_queryblock.pkl'

    diffuser_ckpt_dir = "./share_ckpts/stabilityai/stable-diffusion-xl-base-1.0/"
    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-mmdiffuser from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_mmdiffuser(
        llama_ckpt_dir, llama_tokenzier_path, diffuser_ckpt_dir,
        max_seq_len=512, max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    query_ckpt = torch.load(query_path, map_location='cpu')
    load_result = model.load_state_dict(ckpt['model'], strict=False)
    query_load_result1 = model.query_block.load_state_dict(query_ckpt['query_block'], strict=False)
    query_load_result2 = model.sd_query.load_state_dict(query_ckpt['sd_query'], strict=False)

    # print(query_load_result1)
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # misc.load_model(model_without_ddp, args.pretrained_path)

    dataset_train = ccxm_Finetune_Dataset(args.data_config, transform=transform_train,
                                max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            with torch.cuda.amp.autocast():
                prompt = "A door with a sticker of a cat door on it"
                formprompts = format_prompt('generate an image', prompt) + prompt + '<|img|>'*128
                # print(formprompts)
                img = model_without_ddp.t2i_generate(prompts=[formprompts])
                sdxl_img = model_without_ddp.t2i_generate(prompts=[prompt], use_origin=True, seed=epoch)
            img_resize = [img[0].resize((256, 256))]
            np_image = np.asarray(img_resize)

            sdxl_img_resize = [sdxl_img[0].resize((256, 256))]
            sdxl_np_image = np.asarray(sdxl_img_resize)
            if misc.is_main_process():
                log_writer.add_images("validation", np_image, epoch, dataformats="NHWC")
                log_writer.add_images("sdxl", sdxl_np_image, epoch, dataformats="NHWC")
                aligner_to_save = {     #'sd_query', 'sd_query_block', 'text_unpooled_proj', 'text_pooled_proj'
                    "sd_query":model_without_ddp.sd_query.state_dict(),
                    "query_block":model_without_ddp.query_block.state_dict(),
                    "epoch":epoch
                }
                ckpt_dir = os.path.join(args.log_dir,"checkpoints/")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                save_dir = os.path.join(ckpt_dir,f"epoch{epoch}_queryblock.pkl")
                torch.save(aligner_to_save,save_dir)
            if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):    
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
