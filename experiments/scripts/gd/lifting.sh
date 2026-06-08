#!/bin/bash

python trajopt/gd.py \
    --n_epochs 201 \
    --n_steps 100 \
    --n_inner_steps 20 \
    --pos_bound 0.01 \
    --min_z 0.03 \
    --use_adam \
    --lr_scheduler cosine \
    --lr 0.0001 \
    --task lifting \
    --exp_name gd-01