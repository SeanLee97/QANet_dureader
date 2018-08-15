# !/usr/bin/bash

python cli.py --evaluate --batch_size 16 --learning_rate 1e-2 --weight_decay 0.9999 --clip_weight --max_norm_grad 5.0 --dropout 0.9 --head_size 1 --hidden_size 128 --epochs 50 --gpu 3
