#!/bin/bash

python rl/shac.py \
    --task lifting \
    --n_envs 100 \
    --n_steps 100 \
    --horizon 20 \
    --n_epochs 100 \
    --critic_method td-lambda \
    --lr_schedule linear \
    --activation silu \
    --use_squashed_normal \
    --no_target_critic \
    --with_entropy \
    --with_logprobs \
    --use_distr_ent \
    --scale_by_target_entropy \
    --offset_by_target_entropy \
    --unscale_entropy_alpha \
    --entropy_in_return \
    --entropy_in_targets \
    --exp_name sapo-01 \
    --seed 123