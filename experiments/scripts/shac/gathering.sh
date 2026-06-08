#!/bin/bash

python rl/shac.py \
    --task gathering \
    --n_envs 100 \
    --n_steps 100 \
    --horizon 20 \
    --n_epochs 80 \
    --critic_method td-lambda \
    --lr_schedule linear \
    --exp_name shac-01 \
    --seed 123