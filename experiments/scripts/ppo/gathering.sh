#!/bin/bash

python rl/continuous_run.py \
    --task gathering \
    --n_envs 100 \
    --n_steps 100 \
    --n_traj 4 \
    --n_epochs 20 \
    --exp_name rudin-01 \
    --seed 123