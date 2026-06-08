import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import time
from pathlib import Path
from natsort import natsorted
from omegaconf import DictConfig
from typing_extensions import Literal

from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.utils import TorchUtils

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

from utils.logging import color_print
from utils.domain_randomization import randomized_config


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.uniform_(self._h3.weight, -0.001, 0.001)
        nn.init.constant_(self._h3.bias, 0.0)

        # Ensure parameters are float32 to avoid Float/Double mismatches
        self.float()

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, type: Literal['mu', 'sigma'] = 'mu', **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], n_features[2])
        self._h4 = nn.Linear(n_features[2], n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))

        # Ensure parameters are float32 to avoid Float/Double mismatches
        self.float()

    def forward(self, state, **kwargs):
        x = torch.squeeze(state, 1)
        # Align dtype/device with layer weights to avoid mismatches
        x = x.to(dtype=self._h1.weight.dtype, device=self._h1.weight.device)
        features1 = F.relu(self._h1(x))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = self._h4(features3)

        return a

def experiment(
    alg, n_envs, n_epochs, n_outer_steps, n_inner_steps, steps_interval_split, n_steps, n_steps_per_fit, n_episodes_test,
    alg_params, critic_params, task, exp_name="SAC", pos_bound=0.1, angle_bound=5.0, args=None
):

    # n_outer_steps is the HORIZON (steps per episode)

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
    })
    mdp: Train_Env = env_dict[task](config=cfg)

    # NOTE: SACPolicy scales the action based on mdp.info.action_space
    act_mag = [pos_bound] * 3 * len(mdp.control_idx) + [angle_bound] * 3 * len(mdp.control_idx)

    if task == "coiling":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 1, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "gathering":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 3, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "lifting":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 2, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "separation":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, mdp.rope2.n_vertices, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "slingshot":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 2, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "unknotting":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 0, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "wiring_post":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 2, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    elif task == "wrapping":
        mdp.init_rl_env(n_outer_steps, 1.0, 1.0, 3, steps_interval_split, l2_limit=pos_bound, action_magnitude=act_mag, debug=args.gui)
    else:
        raise ValueError(f"Unknown env_name: {task}")

    # Initialize domain randomization
    randomized_args = randomized_config.get(task, {})

    print(f'Max moving distance {mdp._l2_limit}x{n_outer_steps}={mdp._l2_limit * n_outer_steps} m for each control point')
    print(f'Total substeps: {n_outer_steps}x{n_inner_steps}={n_outer_steps * n_inner_steps}')
    print(f'Total RL steps: {n_steps}x{n_envs}={n_steps * n_envs}')
    print(f'Act scale: {mdp._act_magnitude}, Act space: {mdp.info.action_space._low}~{mdp.info.action_space._high}')
    print(f'Observation dimension: {mdp._obs_dim}, Action dimension: {mdp._act_dim}, Target entropy: {-mdp.info.action_space.shape[0] * 0.5}')
    print(f'n_steps_per_fit: {n_steps_per_fit}, steps_interval_split: {steps_interval_split}')

    curve_dir = Path("logs") / task / exp_name
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_path = curve_dir / f"summary.csv"
    full_log_path = curve_dir / f"rewards_all.csv"
    args_path = curve_dir / f"run_config.json"
    diagnostic_log_path = curve_dir / f"diagnose.txt"

    if os.path.exists(args_path):
        resume = True
    else:
        resume = False

    if resume:
        curve_file = open(curve_path, "a")
        full_log_file = open(full_log_path, "a")
        diagnostic_log_file = open(diagnostic_log_path, "a")
    else:
        curve_file = open(curve_path, "w")
        curve_file.write(f"epoch,R_mean,R_std,R_best,F_mean,F_std,F_best,best_so_far,epoch_duration\n")

        full_log_file = open(full_log_path, "w")
        full_log_file.write(f"epoch,idx,R,F,last_idx\n")

        diagnostic_log_file = open(diagnostic_log_path, "w")

    # Approximators
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=alg_params['actor_n_features'],
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           type='mu')
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=alg_params['actor_n_features'],
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              type='sigma')
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params.update({'input_shape': critic_input_shape})
    critic_params.update({'n_features': alg_params['critic_n_features']})

    if args.test is None:   # Only save args if in training mode
        import copy

        with open(args_path, "w") as f:
            info = vars(args)
            alg_info = copy.deepcopy(alg_params)
            alg_info['actor_optimizer']['class'] = alg_info['actor_optimizer']['class'].__name__
            if alg_info['actor_optimizer'].get('clipping', None) is not None:
                alg_info['actor_optimizer']['clipping']['method'] = alg_info['actor_optimizer']['clipping']['method'].__name__
            critic_info = copy.deepcopy(critic_params)
            critic_info['network'] = critic_info['network'].__name__
            critic_info['optimizer']['class'] = critic_info['optimizer']['class'].__name__  # type: ignore
            critic_info['loss'] = critic_info['loss'].__name__

            info.update(critic_params=critic_info)
            info.update(alg_params=alg_info)
            info.update(randomized_args=randomized_args)
            json.dump(info, f, indent=4)

    ckpt_dir = curve_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                alg_params['actor_optimizer'], critic_params,
                alg_params['batch_size'], alg_params['initial_replay_size'],
                alg_params['max_replay_size'], alg_params['warmup_transitions'],
                alg_params['tau'], alg_params['lr_alpha'],
                use_log_alpha_loss=True,
                log_std_min=alg_params['log_std_min'], log_std_max=alg_params['log_std_max'],
                target_entropy=-mdp.info.action_space.shape[0] * 0.5,
                critic_fit_params=None)
    best_so_far = -np.inf
    start_epoch = 0
    if resume:
        if args.test is None:
            latest_sac = ckpt_dir / "latest_sac.pkl"
            if os.path.exists(latest_sac):
                print(f"Resuming from checkpoint: {latest_sac}")
                agent = alg.load(path=latest_sac)
            else:
                available_ckpts = natsorted(ckpt_dir.glob("*.pkl"))
                if len(available_ckpts) > 0:
                    print(f"Resuming from checkpoint: {available_ckpts[-1]}")
                    agent = alg.load(path=available_ckpts[-1])
        else:
            if args.test == "best":
                ckpt_path = curve_dir / "best_sac.pkl"
            else:
                ckpt_path = ckpt_dir / f"{args.test}_sac.pkl"
            print(f"Resuming from checkpoint: {ckpt_path}")
            agent = alg.load(path=ckpt_path)

        record = ckpt_dir / "record.json"
        if os.path.exists(record):
            with open(record, "r") as f:
                record_data = json.load(f)
                best_so_far = record_data.get("best_so_far", -np.inf)
                start_epoch = record_data.get("epoch", -1) + 1
    print(f"Best so far loaded: {best_so_far}")

    print('replay init size: ', agent._replay_memory._initial_size, 'replay max size: ', agent._replay_memory._max_size, 'replay idx: ', agent._replay_memory._idx)
    if agent._replay_memory._dataset is None:
        agent._replay_memory.reset()
    print('reset agent dataset ', agent._replay_memory._dataset, 'full ', agent._replay_memory._full)
    core = VectorCore(agent, mdp)

    if args.test is not None:
        color_print("Testing mode", "magenta")
        test_R = list()
        test_F = list()

        test_log_path = curve_dir / f"test_results.csv"
        test_log_file = open(test_log_path, "a")
        if not os.path.exists(test_log_path):
            test_log_file.write(f"epoch,idx,R,F,last_idx\n")

        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

        ur = dataset.undiscounted_return
        fsr = dataset.reward_sequence
        episode_length = dataset.episodes_length
        last_idx = episode_length - 1

        n_data = len(ur)
        for j in range(n_data):
            actual_last_idx = episode_length[j] - 1
            last_reward = fsr[j, actual_last_idx]
            test_R.append(ur[j])
            test_F.append(last_reward)

            test_log_file.write(f"{args.test},{j},{ur[j]},{last_reward},{actual_last_idx}\n")
            test_log_file.flush()
            os.fsync(test_log_file.fileno())

        del dataset

        test_R = torch.as_tensor(test_R)
        test_R = test_R.cpu().numpy()
        test_F = torch.as_tensor(test_F)
        test_F = test_F.cpu().numpy()
        Return_opt = np.max(test_R)
        Return = np.mean(test_R)
        Return_std = np.std(test_R)
        FinalReward_opt = np.max(test_F)
        FinalReward = np.mean(test_F)
        FinalReward_std = np.std(test_F)
        print(f"Test | Return: {Return} ± {Return_std}, Final Reward: {FinalReward} ± {FinalReward_std}")
        print(f"Test | Best Return: {Return_opt}, Best Final Reward: {FinalReward_opt}")

        test_log_file.close()

        mdp.save_animation(save_dir=curve_dir.as_posix())

        return

    if start_epoch == 0:
        epoch_start = time.time()
        batch_R = list()
        batch_F = list()

        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)
        ur = dataset.undiscounted_return
        fsr = dataset.reward_sequence
        episode_length = dataset.episodes_length

        n_data = len(ur)
        for j in range(n_data):
            actual_last_idx = episode_length[j] - 1
            last_reward = fsr[j, actual_last_idx]
            batch_R.append(ur[j])
            batch_F.append(last_reward)

        # (n_envs * batch_size, )
        batch_R = torch.as_tensor(batch_R)
        batch_R = batch_R.cpu().numpy()
        batch_F = torch.as_tensor(batch_F)
        batch_F = batch_F.cpu().numpy()

        Return_opt = np.max(batch_R)
        Return = np.mean(batch_R)
        Return_std = np.std(batch_R)
        FinalReward_opt = np.max(batch_F)
        FinalReward = np.mean(batch_F)
        FinalReward_std = np.std(batch_F)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        # Initial evaluation
        curve_file.write(f"-1,{Return},{Return_std},{Return_opt},{FinalReward},{FinalReward_std},{FinalReward_opt},{best_so_far},{epoch_duration}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())

    logger = Logger(log_name=exp_name, results_dir=Path("logs") / task, log_console=True, append=True, log_file_name="train")
    agent.set_logger(logger)

    mdp.init_domain_randomization(**randomized_args)

    # Warmup
    if start_epoch == 0 or agent._replay_memory._dataset is None:
        print("Starting warmup...")
        core.learn(n_steps=alg_params['initial_replay_size'], n_steps_per_fit=alg_params['initial_replay_size'])

    print(f"Starting training from {start_epoch}. Total: {n_epochs} epochs.")
    for it in trange(start_epoch, n_epochs, leave=False):
        epoch_start = time.time()
        batch_R = list()
        batch_F = list()

        mdp.randomization_initialized = True    # enable randomization
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        mdp.randomization_initialized = False   # disable randomization for evaluation
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

        ur = dataset.undiscounted_return
        fsr = dataset.reward_sequence
        episode_length = dataset.episodes_length
        last_idx = episode_length - 1
        action = dataset.action

        # Diagnostic
        diagnostic_log_file.write(f"\n\n{'='*60}\n")
        diagnostic_log_file.write(f"DIAGNOSTICS (epoch {it})\n")
        diagnostic_log_file.write(f"{'='*60}\n")
        diagnostic_log_file.write(f"Episode lengths: min={episode_length.min().item()}, max={episode_length.max().item()}, mean={episode_length.float().mean().item():.1f}\n")
        diagnostic_log_file.write(f"Action stats - Pos: mean={action[:, :3].mean():.4f}, std={action[:, :3].std():.4f}\n")
        diagnostic_log_file.write(f"Action stats - Rot: mean={action[:, 3:].mean():.4f}, std={action[:, 3:].std():.4f}\n")
        success_rate = (episode_length == n_outer_steps).float().mean().item() * 100
        diagnostic_log_file.write(f"Success rate: {success_rate:.1f}% ({(episode_length == n_outer_steps).sum().item()}/{len(episode_length)} episodes)\n")
        diagnostic_log_file.write(f"Best final reward: {fsr[:, last_idx].max():.2f}\n")
        diagnostic_log_file.write(f"{'='*60}\n\n")
        diagnostic_log_file.flush()
        os.fsync(diagnostic_log_file.fileno())

        n_data = len(ur)
        for j in range(n_data):
            actual_last_idx = episode_length[j] - 1
            last_reward = fsr[j, actual_last_idx]
            batch_R.append(ur[j])
            batch_F.append(last_reward)

            full_log_file.write(f"{it},{j},{ur[j]},{last_reward},{actual_last_idx}\n")
            full_log_file.flush()
            os.fsync(full_log_file.fileno())

        # (n_envs * batch_size, )
        batch_R = torch.as_tensor(batch_R)
        batch_R = batch_R.cpu().numpy()
        batch_F = torch.as_tensor(batch_F)
        batch_F = batch_F.cpu().numpy()

        # print(f"batch_R: {batch_R.shape}, batch_F: {batch_F.shape}")

        Return_opt = np.max(batch_R)
        Return = np.mean(batch_R)
        Return_std = np.std(batch_R)
        FinalReward_opt = np.max(batch_F)
        FinalReward = np.mean(batch_F)
        FinalReward_std = np.std(batch_F)
        if it % 10 == 0 or it == n_epochs - 1:
            agent.save(path=ckpt_dir / f"{it}_sac.pkl", full_save=False)
        agent.save(path=ckpt_dir / "latest_sac.pkl", full_save=False)
        if Return > best_so_far:
            agent.save(path=curve_dir / "best_sac.pkl", full_save=False)
            best_so_far = Return

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        # Log reward for this iteration to curve file
        curve_file.write(f"{it},{Return},{Return_std},{Return_opt},{FinalReward},{FinalReward_std},{FinalReward_opt},{best_so_far},{epoch_duration}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())
        logger.epoch_info(it, R=Return, F=FinalReward, best_so_far=best_so_far, E=agent.policy.entropy(dataset.state).item())

        # Update record file
        record = ckpt_dir / "record.json"
        with open(record, "w") as f:
            json.dump({"best_so_far": float(best_so_far), "epoch": int(it)}, f, indent=4)

    # Close curve file after training
    curve_file.close()
    full_log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wiring_ring', help='Task name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--n_traj', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_substeps_per_step', type=int, default=20)
    parser.add_argument('--steps_interval_split', type=int, default=1)
    parser.add_argument('--bound', type=float, default=0.01)
    parser.add_argument('--angle_bound', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--raytracer', '-r', action='store_true', help='Enable raytracer for rendering')
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Enforce float32 globally for new tensors/modules
    torch.set_default_dtype(torch.float32)
    TorchUtils.set_default_device('cuda:0')
    n_envs = args.n_envs
    critic_params = dict(
        network=CriticNetwork,
        optimizer={'class': optim.Adam,
                'params': {'lr': 5e-4}},
        loss=F.smooth_l1_loss,
        output_shape=(1,)
    )
    sac_params = {
        'actor_n_features': [256, 128, 64],
        'critic_n_features': 256,
        'actor_optimizer': {
            'class': optim.Adam,
            'params': {'lr': 5e-4},
            'clipping': {
                'method': torch.nn.utils.clip_grad_norm_,
                'params': {'max_norm': 0.5}
            }
        },
        'batch_size': 2048,
        'initial_replay_size': 20000,
        'max_replay_size': 500000,
        'warmup_transitions': 20000,
        'tau': 0.01,
        'lr_alpha': 1e-3,
        'log_std_min': -10.0,
        'log_std_max': 0.005
    }

    # Setup for: 10 envs, 10 steps/trajectory, 20 trajectories before policy update
    n_trajectories = args.n_traj  # Number of trajectories to collect per env

    experiment(
        alg=SAC,
        n_envs=n_envs,
        n_epochs=args.n_epochs,
        n_outer_steps=args.n_steps,
        n_inner_steps=args.n_substeps_per_step,
        steps_interval_split=args.steps_interval_split,
        n_steps=n_envs * args.n_steps * n_trajectories,
        n_steps_per_fit=n_envs,
        n_episodes_test=n_envs,
        alg_params=sac_params,
        critic_params=critic_params,
        task=args.task,
        exp_name=args.exp_name,
        pos_bound=args.bound,
        angle_bound=args.angle_bound,
        args=args
    )
