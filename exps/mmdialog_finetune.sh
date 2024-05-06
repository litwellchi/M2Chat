#!/usr/bin/bash

current_time=$(date +%Y%m%d%H%M%S)
PRETRAINED_PATH=output_qformer/pretrain_20240224_instruction/checkpoint-0.pth # path to pre-trained checkpoint
LLAMA_PATH="share_ckpts/llama1_weight/"
DIFFUSER_PATH="share_ckpts/stabilityai/stable-diffusion-xl-base-1.0/"
CONFIG="configs/t2i_pretrain.yaml"
OUTPUT_DIR="output_qformer/1_MMDialog_30k_0pretrain"
mkdir -p "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u -m torch.distributed.launch --master_port=11345 --nproc_per_node=8 --use_env \
 main_dialog.py --data_config "$CONFIG" --batch_size 16 --num_workers 1 \
 --epochs 20 --split_epoch 1 --warmup_epochs 2 --lr 1.0e-5 --weight_decay 2 \
 --llama_path "$LLAMA_PATH" \
 --diffuser_path "$DIFFUSER_PATH" \
 --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR" \
  --pretrained_path "$PRETRAINED_PATH" \
&>> "$OUTPUT_DIR"/output.log &


