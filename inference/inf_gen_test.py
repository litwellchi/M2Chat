import cv2
import os
import numpy as np
# from llama.utils import format_prompt
# from llama.llama_mmdiffuser_clip import LLaMA_mmdiffuser
from llama.llama_mmdiffuser import LLaMA_mmdiffuser
import torch
import util.misc as misc
import torch.backends.cudnn as cudnn
from PIL import Image
from data.MMDialog_dataset import format_mmDialog_sprompt
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = "cuda" if torch.cuda.is_available() else "cpu"


# fix the seed for reproducibility
seed = 42 + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True


llama_dir = "llama1_weight/"
name = "BIAS-7B"
phase="finetune"

model_path = "llama-adapter/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth"
# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
# model, preprocess = llama.load("BIAS-7B", llama_dir, device)

# BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
llama_type = name.split('.')[0].split('-')[-1]
llama_ckpt_dir = os.path.join(llama_dir, llama_type)
llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
query_path = '/aifs4su/mmcode/videogen/chatillusion/mmllama/output_qformer/pretrain_20240224_instruction/checkpoints/epoch0_queryblock.pkl'
diffuser_ckpt_dir = "stabilityai/stable-diffusion-xl-base-1.0/"
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

print(query_load_result1)
assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
# assert len(query_load_result.unexpected_keys) == 0, f"Unexpected keys: {query_load_result.unexpected_keys}"
model.to(device)
preprocess = model.clip_transform


# eval
model.eval()

context1 = "Introducing today's FlashbackFridayz theme desserts &amp; drinks! As many know I :red_heart: :red_heart: DESSERT :grinning_squinting_face: :shortcake:Use (with Z!) :cocktail_glass:Tag hosts &amp; FAB guest hosts &amp; your friendz Retweet &amp; HAVE FUN! :backhand_index_pointing_down:Sorghum pecan pie:backhand_index_pointing_down:"
caption1 = "a piece of cake is sitting on a plate . "
context2 = "Oh Yeh! Desserts &amp; drinks! A fab theme to celebrate 2020! Tweet your pics with hashtag FlashbackFridayz :backhand_index_pointing_right:Tag &amp; Retweet your friends &amp; hosts &amp; guest hosts :camera_with_flash:We :red_heart:our Sunset Mojitos, Koh Tao Thailand"
caption2 = "a person holding a drink on a beach with palm trees in the background . "
answer = context2+"\n###Image:"+caption2
formprompts = format_mmDialog_sprompt('answer based on the input dialog', context1, caption1)
formprompts = formprompts + answer
img = Image.fromarray(cv2.imread('videogen/MMDialogDataset/MMDialogDataset/test/-5402952495417911867.jpg'))
img = model.clip_transform(img).unsqueeze(0).to(device)
# img = model.t2i_generate(prompts=[formprompts])
img = model.ti2ti_generate(img, [formprompts])
sdxl_img = model.t2i_generate(prompts=[caption2], use_origin=True, seed=23)
img_resize = [img[0].resize((256, 256))]
np_image = np.asarray(img_resize)