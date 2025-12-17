#!/bin/bash
# Stereo video generation inference script

CUDA_VISIBLE_DEVICES=1 python sample.py \
  --config toml/infer.toml \
  --input ./sample/sample1.mp4 \
  --output_folder ./output \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python sample.py \
  --config toml/infer.toml \
  --input ./sample/sample2.mp4 \
  --output_folder ./output \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python sample.py \
  --config toml/infer.toml \
  --input ./sample/sample3.mp4 \
  --output_folder ./output \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python sample.py \
  --config toml/infer.toml \
  --input ./sample/sample4.mp4 \
  --output_folder ./output \
  --device cuda:0

