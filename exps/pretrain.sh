#!/usr/bin/bash

LLAMA_PATH="share_ckpts/llama1_weight/"
DIFFUSER_PATH="share_ckpts/stabilityai/stable-diffusion-xl-base-1.0/"
CONFIG="configs/t2i_pretrain.yaml"
OUTPUT_DIR="output_qformer/20240131"
CHECKPOINT_DIR="output_qformer/20240131"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u -m torch.distributed.launch --master_port=11345 --nproc_per_node=8 --use_env \
 main_pretrain.py --data_config "$CONFIG" --batch_size 96 --num_workers 32 \
 --epochs 5 --split_epoch 1 --warmup_epochs 2 --lr 1.0e-4 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --diffuser_path "$DIFFUSER_PATH" \
 --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR" \
&>> "$OUTPUT_DIR"/output.log &

