#!/bin/bash

python trajopt/continuous_run.py \
    --task lifting \
    --n_envs 100 \
    --max_iter 26 \
    --n_steps 10 \
    --n_steps_sub 10 \
    --exp_name cmaes-01 \
    --angle_bound 1.0 \
    --sigma 0.005 \
    --seed 123