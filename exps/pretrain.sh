#!/usr/bin/bash

CONFIG="configs/t2i_pretrain.yaml"
OUTPUT_DIR="output/20240705"
CHECKPOINT_DIR="output/20240705"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -u -m torch.distributed.launch --master_port=11345 --nproc_per_node=4 --use_env \
 main_pretrain.py --data_config "$CONFIG" --batch_size 16 --num_workers 32 \
 --epochs 15 --split_epoch 1 --warmup_epochs 2 --lr 1.0e-4 --weight_decay 0.05 \
 --llama_path "checkpoints/share_ckpts/llama1_weight" \
 --llama_bias_path "checkpoints/share_ckpts/llama-adapter/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth" \
 --querry_path 'checkpoints/20231106/epoch2_queryblock.pkl' \
 --diffuser_path "checkpoints/share_ckpts/stabilityai/stable-diffusion-xl-base-1.0" \
 --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR" 
# &>> "$OUTPUT_DIR"/output.log &

