#!/bin/bash

python rl/continuous_run.py \
    --task slingshot \
    --n_envs 100 \
    --n_steps 30 \
    --n_traj 4 \
    --n_epochs 50 \
    --exp_name rudin-01 \
    --seed 123