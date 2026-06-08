#!/bin/bash

python rl/continuous_run.py \
    --task unknotting \
    --n_envs 100 \
    --n_steps 100 \
    --n_traj 4 \
    --n_epochs 25 \
    --exp_name sac-01 \
    --seed 123