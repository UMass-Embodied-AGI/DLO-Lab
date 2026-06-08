"""
Training script for SHAC on manipulation tasks.
Follows the structure of experiments/rl/rudinppo.py for consistency.
"""
import os
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import trange
from pathlib import Path
from natsort import natsorted
from omegaconf import DictConfig

import sys
sys.path.append('.')
from envs.base import Train_Env
from envs.env_coiling import Train_Env_Coiling
from envs.env_gathering import Train_Env_Gathering
from envs.env_lifting import Train_Env_Lifting
from envs.env_separation import Train_Env_Separation
from envs.env_slingshot import Train_Env_Slingshot
from envs.env_unknotting import Train_Env_Unknotting
from envs.env_wiring_post import Train_Env_Wiring_post
from envs.env_wrapping import Train_Env_Wrapping

from diff_rl.shac import SHACAgent
from utils.logging import color_print
from utils.domain_randomization import randomized_config

def experiment(
    n_envs, n_epochs, n_outer_steps, n_inner_steps, steps_interval_split,
    pos_bound=0.1, angle_bound=5.0,
    task="wrapping", exp_name="SHAC",
    args=None
):
    """
    Main experiment function for SHAC training.

    Args:
        n_envs: Number of parallel environments
        n_epochs: Number of training epochs
        n_outer_steps: Horizon length (steps per episode)
        n_inner_steps: Substeps per step (physics simulation)
        steps_interval_split: Split interval for action application
        pos_bound: Position action bound
        angle_bound: Angle action bound
        task: Task name
        exp_name: Experiment name
        args: Command line arguments
    """

    # Environment dictionary
    env_dict = {
        "coiling": Train_Env_Coiling,
        "gathering": Train_Env_Gathering,
        "lifting": Train_Env_Lifting,
        "separation": Train_Env_Separation,
        "slingshot": Train_Env_Slingshot,
        "unknotting": Train_Env_Unknotting,
        "wiring_post": Train_Env_Wiring_post,
        "wrapping": Train_Env_Wrapping,
    }
    cfg = DictConfig({
        "task": task,
        "log_dir": os.path.join("logs", task, exp_name),
        "n_envs": n_envs,
        "n_substeps_per_step": n_inner_steps,
        "GUI": args.gui,
        "camera": False if args.test is None else True,
        "raytracer": args.raytracer,
        "requires_grad": True if args.test is None else False,
        "grad_clip": args.grad_clip,
        "disable_constraint_grad": args.disable_constraint_grad,
        "bptt_window": args.bptt_window,
    })
    # Create environment with differentiable physics
    mdp: Train_Env = env_dict[task](config=cfg)

    # Initialize environment for differentiable RL
    n_additional_obj_dict = {
        "coiling": 1,
        "gathering": 3,
        "lifting": 2,
        "separation": 0,
        "slingshot": 2,
        "unknotting": 0,
        "wiring_post": 2,
        "wrapping": 3,
    }
    n_additional_obj = n_additional_obj_dict.get(task, 0)
    if task == "separation":
        n_additional_obj = mdp.rope2.n_vertices

    mdp.init_diff_rl_env(
        n_steps=n_outer_steps,
        pos_bound=pos_bound,
        angle_bound=angle_bound,
        n_additional_obj=n_additional_obj,
        steps_interval_split=steps_interval_split,
        debug=args.gui
    )

    print(f'Max moving distance {mdp._l2_limit}x{n_outer_steps}={mdp._l2_limit * n_outer_steps} m for each control point')
    print(f'Total substeps: {n_outer_steps}x{n_inner_steps}={n_outer_steps * n_inner_steps}')
    print(f'Bound: {mdp._act_magnitude}')
    print(f'Observation dimension: {mdp._obs_dim}, Action dimension: {mdp._act_dim}')
    print(f'steps_interval_split: {steps_interval_split}')

    # Setup logging directories
    curve_dir = Path("logs") / task / exp_name
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_path = curve_dir / f"summary.csv"
    full_log_path = curve_dir / f"rewards_all.csv"
    args_path = curve_dir / f"run_config.json"

    resume = os.path.exists(args_path)

    if resume:
        curve_file = open(curve_path, "a")
        full_log_file = open(full_log_path, "a")
    else:
        curve_file = open(curve_path, "w")
        curve_file.write(f"epoch,R_mean,R_std,R_best,F_mean,F_std,F_best,best_so_far,actor_lr,critic_lr,epoch_duration\n")

        full_log_file = open(full_log_path, "w")
        full_log_file.write(f"epoch,idx,R,F,last_idx\n")

    # Create SHAC agent
    shac_config = {
        'obs_dim': mdp._obs_dim,
        'action_dim': mdp._act_dim,
        'device': 'cuda:0',

        # Training
        'max_agent_steps': args.max_steps,
        'horizon_len': args.horizon,
        'gamma': 0.99,

        # Networks
        'actor_hidden_dims': (256, 128, 64),
        'critic_hidden_dims': (256, 256),
        'num_critics': 2,
        'activation': args.activation,
        'norm_type': args.norm_type,
        'dist_kwargs': {
            'dist_type': 'squashed_normal',
            'minlogstd': -10.0,
            'maxlogstd': 0.005,
        } if args.use_squashed_normal else {
            'dist_type': 'normal',
        },

        # Optimization
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
        'alpha_lr': args.alpha_lr,
        'max_grad_norm': args.max_grad_norm,

        # Learning rate scheduler
        'lr_schedule': args.lr_schedule,
        'critic_lrschedule': True,
        'min_lr': args.min_lr,
        'max_lr': args.max_lr if args.max_lr is not None else max(args.actor_lr, args.critic_lr),
        'max_epochs': args.max_epochs if args.max_epochs is not None else n_epochs,

        # Critic training
        'critic_iterations': 16,
        'num_critic_batches': 4,
        'critic_method': args.critic_method,
        'no_target_critic': args.no_target_critic,
        'target_critic_alpha': 0.4,

        # Entropy (SAC-style auto-tuning)
        'with_entropy': args.with_entropy,
        'with_logprobs': args.with_logprobs,
        'entropy_coef': args.entropy_coef,
        'use_distr_ent': args.use_distr_ent,
        'init_alpha': args.init_alpha,
        'target_entropy_scalar': args.target_entropy_scalar,

        # Entropy scaling/offsetting
        'scale_by_target_entropy': args.scale_by_target_entropy,
        'offset_by_target_entropy': args.offset_by_target_entropy,
        'unscale_entropy_alpha': args.unscale_entropy_alpha,

        # Entropy in returns and targets
        'entropy_in_return': args.entropy_in_return,
        'entropy_in_targets': args.entropy_in_targets,
        'no_actor_entropy': args.no_actor_entropy,

        # Normalization
        'normalize_obs': args.normalize_obs,
        'reward_scale': 1.0,
        'reward_shift': 0.0,

        # Debug
        'debug': args.debug,
    }

    # Save config
    if args.test is None:
        with open(args_path, "w") as f:
            info = {'args': vars(args)}
            info.update(shac_config=shac_config)
            info.update(randomized_args=randomized_config.get(task, {}))
            json.dump(info, f, indent=4)

    ckpt_dir = curve_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent = SHACAgent(mdp, shac_config)

    # Load checkpoint if resuming
    best_so_far = -np.inf
    start_epoch = 0
    if resume:
        if args.test is None:
            latest_ckpt = ckpt_dir / "latest_shac.pkl"
            if os.path.exists(latest_ckpt):
                print(f"Resuming from checkpoint: {latest_ckpt}")
                agent.load(latest_ckpt)
            else:
                available_ckpts = natsorted(ckpt_dir.glob("*.pkl"))
                if len(available_ckpts) > 0:
                    print(f"Resuming from checkpoint: {available_ckpts[-1]}")
                    agent.load(available_ckpts[-1])
        else:
            ckpt_path = curve_dir / args.test
            print(f"Loading checkpoint: {ckpt_path}")
            agent.load(ckpt_path)

        record = ckpt_dir / "record.json"
        if os.path.exists(record):
            with open(record, "r") as f:
                record_data = json.load(f)
                best_so_far = record_data.get("best_so_far", -np.inf)
                start_epoch = record_data.get("epoch", -1) + 1

    print(f"Best so far loaded: {best_so_far}")

    # Testing mode
    if args.test is not None:
        color_print("Testing mode", "magenta")
        test_R = list()
        test_F = list()

        test_log_path = curve_dir / f"test_results.csv"
        test_log_file = open(test_log_path, "w")
        test_log_file.write(f"epoch,idx,R,F,last_idx\n")

        # Run test episodes (each test episode runs all n_envs in parallel)
        for test_ep in range(args.n_test_episodes):
            mdp.reset()
            episode_rewards = torch.zeros(n_envs)
            episode_lengths = torch.zeros(n_envs, dtype=torch.int32)
            env_mask = torch.ones(n_envs, dtype=torch.bool)

            for _ in range(n_outer_steps):
                obs = mdp.compute_observation()
                action = agent.get_action(obs, deterministic=True)

                # Apply action
                _, env_mask, r, _ = mdp.step_diff_rl(env_mask, action)
                episode_rewards += r
                episode_lengths[env_mask] += 1

            # Log results for each environment in this test episode
            for j in range(n_envs):
                test_R.append(episode_rewards[j].item())
                test_F.append(r[j].item())
                test_log_file.write(f"{args.test},{test_ep * n_envs + j},{episode_rewards[j].item()},{r[j].item()},{episode_lengths[j].item()}\n")
                test_log_file.flush()
                os.fsync(test_log_file.fileno())

        test_R = np.array(test_R)
        test_F = np.array(test_F)
        print(f"Test | Return: {test_R.mean()} ± {test_R.std()}, Final Reward: {test_F.mean()} ± {test_F.std()}")
        print(f"Test | Best Return: {test_R.max()}, Best Final Reward: {test_F.max()}")

        test_log_file.close()

        mdp.save_animation(save_dir=curve_dir.as_posix())

        return

    if start_epoch == 0:
        # Evaluate initial policy before training
        mdp.reset()
        epoch_start = time.time()
        episode_rewards = torch.zeros(n_envs)
        episode_lengths = torch.zeros(n_envs, dtype=torch.int32)
        env_mask = torch.ones(n_envs, dtype=torch.bool)

        # Run episode for all environments in parallel
        for _ in range(n_outer_steps):
            obs = mdp.compute_observation()
            action = agent.get_action(obs, deterministic=False)

            _, env_mask, r, _ = mdp.step_diff_rl(env_mask, action)
            episode_rewards += r
            episode_lengths[env_mask] += 1

        # Collect results for each environment
        batch_R = list()
        batch_F = list()
        for j in range(n_envs):
            batch_R.append(episode_rewards[j].item())
            batch_F.append(r[j].item())  # Final step reward

        batch_R = np.array(batch_R)
        batch_F = np.array(batch_F)

        Return_opt = np.max(batch_R)
        Return = np.mean(batch_R)
        Return_std = np.std(batch_R)
        FinalReward_opt = np.max(batch_F)
        FinalReward = np.mean(batch_F)
        FinalReward_std = np.std(batch_F)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        # Initial evaluation
        curve_file.write(f"-1,{Return},{Return_std},{Return_opt},{FinalReward},{FinalReward_std},{FinalReward_opt},{best_so_far},0.0,0.0,{epoch_duration}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())

    # Initialize domain randomization
    randomized_args = randomized_config.get(task, {})
    mdp.init_domain_randomization(**randomized_args)

    # Training loop
    print(f"Starting training from {start_epoch}. Total: {n_epochs} epochs.")
    for it in trange(start_epoch, n_epochs, leave=False):
        epoch_start = time.time()

        # Enable randomization during training
        mdp.randomization_initialized = True

        # Train one epoch
        agent.train_one_epoch()

        # Disable randomization for evaluation
        mdp.randomization_initialized = False

        # Evaluate all n_envs in parallel
        mdp.reset()
        episode_rewards = torch.zeros(n_envs)
        episode_lengths = torch.zeros(n_envs, dtype=torch.int32)
        env_mask = torch.ones(n_envs, dtype=torch.bool)

        # Run episode for all environments in parallel
        for _ in range(n_outer_steps):
            obs = mdp.compute_observation()
            action = agent.get_action(obs, deterministic=False)

            _, env_mask, r, _ = mdp.step_diff_rl(env_mask, action)
            episode_rewards += r
            episode_lengths[env_mask] += 1

        # Collect results for each environment
        batch_R = list()
        batch_F = list()
        for j in range(n_envs):
            batch_R.append(episode_rewards[j].item())
            batch_F.append(r[j].item())  # Final step reward

            full_log_file.write(f"{it},{j},{episode_rewards[j].item()},{r[j].item()},{episode_lengths[j].item()}\n")
            full_log_file.flush()
            os.fsync(full_log_file.fileno())

        batch_R = np.array(batch_R)
        batch_F = np.array(batch_F)

        Return_opt = np.max(batch_R)
        Return = np.mean(batch_R)
        Return_std = np.std(batch_R)
        FinalReward_opt = np.max(batch_F)
        FinalReward = np.mean(batch_F)
        FinalReward_std = np.std(batch_F)

        # Save checkpoints
        if it % 10 == 0 or it == n_epochs - 1:
            agent.save(ckpt_dir / f"{it}_shac.pkl")
        # only save 3 most recent ckpts + latest to save space
        ckpt_files = list((ckpt_dir / '').glob("*_shac.pkl"))
        # exclude latest
        ckpt_files = [f for f in ckpt_files if 'latest' not in f.name]
        if len(ckpt_files) > 3:
            ckpt_files = natsorted(ckpt_files)
            for f in ckpt_files[:-3]:
                os.remove(f.as_posix())
        agent.save(ckpt_dir / "latest_shac.pkl")
        if Return > best_so_far:
            agent.save(curve_dir / "best_shac.pkl")
            best_so_far = Return

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        actor_lr = agent.actor_optim.param_groups[0]['lr']
        critic_lr = agent.critic_optim.param_groups[0]['lr']

        # Log
        curve_file.write(f"{it},{Return},{Return_std},{Return_opt},{FinalReward},{FinalReward_std},{FinalReward_opt},{best_so_far},{actor_lr:.4e},{critic_lr:.4e},{epoch_duration}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())

        # Update record
        with open(ckpt_dir / "record.json", "w") as f:
            json.dump({"best_so_far": float(best_so_far), "epoch": int(it)}, f, indent=4)

        print(f"Epoch {it} | R={Return:.2f}±{Return_std:.2f} | F={FinalReward:.2f} | Best={best_so_far:.2f} | A LR={actor_lr:.4e} | C LR={critic_lr:.4e}")

    # Close files
    curve_file.close()
    full_log_file.close()

    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wrapping', help='Task name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=100, help='Horizon (steps per episode)')
    parser.add_argument('--n_substeps_per_step', type=int, default=20)
    parser.add_argument('--steps_interval_split', type=int, default=1)
    parser.add_argument('--disable_constraint_grad', action='store_true', help='Disable constraint gradients in physics simulation')
    parser.add_argument('--bound', type=float, default=0.01, help='Position bound')
    parser.add_argument('--angle_bound', type=float, default=0.1, help='Angle bound')
    parser.add_argument('--horizon', type=int, default=16, help='SHAC horizon length')
    parser.add_argument('--activation', type=str, default='elu', choices=['relu', 'elu', 'tanh', 'silu', 'gelu'], help='Network activation function')
    parser.add_argument('--norm_type', type=str, default=None, choices=[None, 'batchnorm', 'layernorm'], help='Normalization type for networks')
    parser.add_argument('--use_squashed_normal', action='store_true', help='Use squashed normal distribution for actor')
    parser.add_argument('--actor_lr', type=float, default=2e-3)
    parser.add_argument('--critic_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_lr', type=float, default=5e-3)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--bptt_window', type=int, default=0, help='Truncated-BPTT window in scene.step() units (0=full BPTT; must be < n_substeps_per_step to truncate within a macro-step)')
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument('--critic_method', type=str, default='one-step', choices=['one-step', 'td-lambda'], help='Critic training method')
    parser.add_argument('--no_target_critic', action='store_true', help='Disable target critic networks (use online critic for stability)')

    # Learning rate scheduler arguments
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'linear', 'kl'], help='Learning rate schedule type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for schedulers')
    parser.add_argument('--max_lr', type=float, default=None, help='Maximum learning rate for KL scheduler (defaults to actor_lr)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Max epochs for linear scheduler (0 = disabled, use max_steps instead)')
    parser.add_argument('--normalize_obs', action='store_true')

    # Entropy regularization arguments
    parser.add_argument('--with_entropy', action='store_true', help='Enable learnable entropy coefficient (SAC-style)')
    parser.add_argument('--with_logprobs', action='store_true', help='Store log probabilities and distribution entropy')
    parser.add_argument('--entropy_coef', type=float, default=None, help='Fixed entropy coefficient (alternative to auto-tuning)')
    parser.add_argument('--use_distr_ent', action='store_true', help='Use distribution entropy instead of -logprob')
    parser.add_argument('--init_alpha', type=float, default=1.0, help='Initial entropy coefficient')
    parser.add_argument('--target_entropy_scalar', type=float, default=0.5, help='Target entropy scalar (target = -action_dim * scalar)')

    # Entropy scaling/offsetting
    parser.add_argument('--scale_by_target_entropy', action='store_true', help='Scale entropy by 1/|target_entropy|')
    parser.add_argument('--offset_by_target_entropy', action='store_true', help='Offset entropy by |target_entropy|/2')
    parser.add_argument('--unscale_entropy_alpha', action='store_true', help='Unscale alpha when computing alpha loss')

    # Entropy in returns and targets
    parser.add_argument('--entropy_in_return', action='store_true', help='Include entropy term in returns computation')
    parser.add_argument('--entropy_in_targets', action='store_true', help='Include entropy in critic target values')
    parser.add_argument('--no_actor_entropy', action='store_true', help='Disable entropy term in actor loss')

    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--raytracer', '-r', action='store_true', help='Enable raytracer')
    parser.add_argument('--n_test_episodes', type=int, default=1)
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda:0')

    experiment(
        n_envs=args.n_envs,
        n_epochs=args.n_epochs,
        n_outer_steps=args.n_steps,
        n_inner_steps=args.n_substeps_per_step,
        steps_interval_split=args.steps_interval_split,
        pos_bound=args.bound,
        angle_bound=args.angle_bound,
        task=args.task,
        exp_name=args.exp_name,
        args=args
    )
