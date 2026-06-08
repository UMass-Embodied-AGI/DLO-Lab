#!/bin/bash

python rl/shac.py \
    --task wrapping \
    --n_envs 100 \
    --n_steps 100 \
    --horizon 20 \
    --n_epochs 200 \
    --critic_method td-lambda \
    --lr_schedule linear \
    --exp_name shac-01 \
    --seed 123