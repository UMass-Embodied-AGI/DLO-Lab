import os
import json
import torch
import random
import argparse
import numpy as np
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

from utils.logging import color_print


def experiment(args):
    ########################## init ##########################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env_dict = {
        'coiling': Train_Env_Coiling,
        'gathering': Train_Env_Gathering,
        'lifting': Train_Env_Lifting,
        'separation': Train_Env_Separation,
        'slingshot': Train_Env_Slingshot,
        'unknotting': Train_Env_Unknotting,
        'wiring_post': Train_Env_Wiring_post,
        'wrapping': Train_Env_Wrapping,
    }
    cfg = DictConfig({
        "task": args.task,
        "log_dir": os.path.join("logs", args.task, args.exp_name),
        "n_envs": 1,
        "n_substeps_per_step": args.n_inner_steps,
        "GUI": args.gui,
        "camera": False if args.test is None else True,
        "raytracer": args.raytracer,
        "requires_grad": True,
        "bptt_window": args.bptt_window,
        "grad_clip": args.grad_clip
    })
    gd_env: Train_Env = env_dict[args.task](config=cfg)

    task = args.task
    exp_name = args.exp_name

    gd_env.init_gd_env(
        args.n_steps, args.pos_bound, args.angle_bound, args.min_z, 
        scale_method=args.scale_method, exp_base=args.exp_base, lr=args.lr, lr_min=args.lr_min, debug=args.debug
    )

    print(f'Max moving distance {gd_env._max_ddist}x{gd_env._n_steps}={gd_env._max_ddist * gd_env._n_steps} m for each control point')
    print(f'Total substeps: {gd_env._n_steps}x{gd_env.steps_interval}={gd_env._n_steps * gd_env.steps_interval}')
    print(f'Feasible region: {gd_env._feasible_region}')

    gd_env.construct_traj_optim(
        max_ddist=gd_env._max_ddist,
        max_grad_norm=args.max_grad_norm,
        controller="TrajOptimController",
        debug=args.debug,
        use_adam=args.use_adam,
        lr_scheduler=args.lr_scheduler,
    )

    log_dir = Path("logs") / task / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.csv"
    args_path = log_dir / "run_config.json"

    if os.path.exists(args_path):
        resume = True
    else:
        resume = False

    ckpt_dir = log_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        if args.test is None:
            summary_file = open(summary_path, 'a')
        else:
            # in test mode, load specified test ckpt
            traj_path = log_dir / args.test
            traj = torch.load(traj_path)
            if task == 'separation':    # two ropes
                traj_ca = traj[:, :, :1, :].clone()
                traj_cb = traj[:, :, 1:, :].clone()
                gd_env.ca.traj = traj_ca.to(gd_env.ca.traj.device)
                gd_env.cb.traj = traj_cb.to(gd_env.cb.traj.device)
            else:
                gd_env.c.traj = traj.to(gd_env.c.traj.device)
            color_print(f'Loaded test traj from {traj_path.as_posix()}', "magenta")

            out = gd_env.train_one_iter_gd(it=0, max_it=args.n_epochs, skip_backward=True)
            print(f'Single traj reward: {out["reward"].item():.4f}')
            forward_time = out["forward_time"]
            mean_step_time = forward_time / (gd_env._n_steps * gd_env.steps_interval)
            forward_FPS = (gd_env._n_steps * gd_env.steps_interval) / forward_time
            print(f'Forward time: {mean_step_time * 1000:.2f}ms, forward FPS: {forward_FPS:.2f}, eq steps: {gd_env._n_steps * gd_env.steps_interval}')
            gd_env.save_animation(save_dir=log_dir.as_posix())
            return
    else:
        summary_file = open(summary_path, 'w')
        summary_file.write("epoch,R_mean,best_so_far,forward_elapsed,backward_elapsed,lr\n")

    max_reward = -float('inf')

    if resume:
        latest_iter = -1
        for i in os.listdir(ckpt_dir.as_posix()):
            # find the latest traj
            if i.endswith('_traj.pt') and i != 'best_traj.pt':
                iter_id = int(i.split('_')[0])
                if iter_id > latest_iter:
                    latest_iter = iter_id
        if latest_iter >= 0:
            iter_start = latest_iter + 1
            traj_path = ckpt_dir / f'{latest_iter:03d}_traj.pt'
            traj = torch.load(traj_path)
            if task == 'separation':    # two ropes
                traj_ca = traj[:, :, :1, :].clone()
                traj_cb = traj[:, :, 1:, :].clone()
                gd_env.ca.traj = traj_ca.to(gd_env.ca.traj.device)
                gd_env.cb.traj = traj_cb.to(gd_env.cb.traj.device)
            else:
                gd_env.c.traj = traj.to(gd_env.c.traj.device)
            print(f'Loaded existing traj from {traj_path.as_posix()}')
            if args.use_adam:
                adam_path = ckpt_dir / 'adam_state.pt'
                if os.path.exists(adam_path.as_posix()):
                    adam_state = torch.load(adam_path.as_posix())
                    if task == 'separation':
                        gd_env.ca.c.m_buffer = adam_state['m_buffer_a'].to(gd_env.ca.traj.device)
                        gd_env.ca.c.v_buffer = adam_state['v_buffer_a'].to(gd_env.ca.traj.device)
                        gd_env.cb.c.m_buffer = adam_state['m_buffer_b'].to(gd_env.cb.traj.device)
                        gd_env.cb.c.v_buffer = adam_state['v_buffer_b'].to(gd_env.cb.traj.device)
                    else:
                        gd_env.c.m_buffer = adam_state['m_buffer'].to(gd_env.c.traj.device)
                        gd_env.c.v_buffer = adam_state['v_buffer'].to(gd_env.c.traj.device)
                    print(f'Loaded existing Adam state from {adam_path.as_posix()}. Previous ends at iter {adam_state["cur_iter"]}.')
                else:
                    print(f'No existing Adam state found at {adam_path.as_posix()}. Starting Adam fresh.')
            # read summary to find max reward so far
            max_ = list()
            with open(summary_path, 'r') as f:
                lines = f.readlines()[1:]  # skip header
                for line in lines:
                    items = line.strip().split(',')
                    max_.append(float(items[2]))
            if len(max_) > 0:
                max_reward = max(max_)
            print(f'Resumed from existing ckpt dir: {ckpt_dir.as_posix()}, best_so_far: {max_reward:.4f}, will start from iter {iter_start}')
        else:
            iter_start = 0
            print(f'No existing traj found in {ckpt_dir.as_posix()}. Starting from scratch.')
    else:
        iter_start = 0
        print(f'No existing ckpt dir found. Created new: {ckpt_dir.as_posix()}')

    if args.test is None:

        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

    print(f'Iter from {iter_start} to {args.n_epochs-1}')

    ########################## train ##########################

    for it in range(iter_start, args.n_epochs):
        out = gd_env.train_one_iter_gd(it=it, max_it=args.n_epochs)
        iter_rewards = out["reward"]
        max_iter_reward_idx = np.argmax(iter_rewards)
        iter_reward = iter_rewards[max_iter_reward_idx].item()

        print(f'Epoch {it}: loss={out["loss"]:.4f}, reward_accum={iter_reward:.4f}, forw={out["forward_time"]:.2f}s, back={out["backward_time"]:.2f}s, lr={out["lr"]:.3e}')

        if iter_reward > max_reward:
            max_reward = iter_reward
            # save best traj
            if task == 'separation':
                best_traj = torch.cat([gd_env.ca.traj, gd_env.cb.traj], dim=2)
            else:
                best_traj = gd_env.c.traj
            torch.save(best_traj.cpu(), (log_dir / 'best_traj.pt').as_posix())
            print(f'Saved best traj at epoch {it} with reward {max_reward:.4f}')
            best_qpos = out["qpos_seq"][:, max_iter_reward_idx]
            np.save((log_dir / 'best_qpos.npy').as_posix(), best_qpos)

        if task == 'separation':
            traj = torch.cat([gd_env.ca.traj, gd_env.cb.traj], dim=2)
        else:
            traj = gd_env.c.traj
        torch.save(traj.cpu(), (ckpt_dir / f'{it:03d}_traj.pt').as_posix())
        # only save 3 most recent ckpts to save space
        ckpt_files = list((ckpt_dir / '').glob('*_traj.pt'))
        if len(ckpt_files) > 3:
            ckpt_files = natsorted(ckpt_files)
            for f in ckpt_files[:-3]:
                os.remove(f.as_posix())

        if args.use_adam:
            adam_state = dict()
            if task == 'separation':
                adam_state['m_buffer_a'] = gd_env.ca.m_buffer.cpu()
                adam_state['v_buffer_a'] = gd_env.ca.v_buffer.cpu()
                adam_state['m_buffer_b'] = gd_env.cb.m_buffer.cpu()
                adam_state['v_buffer_b'] = gd_env.cb.v_buffer.cpu()
            else:
                adam_state['m_buffer'] = gd_env.c.m_buffer.cpu()
                adam_state['v_buffer'] = gd_env.c.v_buffer.cpu()
            adam_state['cur_iter'] = it
            torch.save(adam_state, (ckpt_dir / 'adam_state.pt').as_posix())

        # log
        # write csv
        summary_file.write(f'{it},{iter_reward},{max_reward},{out["forward_time"]:.2f},{out["backward_time"]:.2f},{out["lr"]:.4e}\n')
        summary_file.flush()
        os.fsync(summary_file.fileno())

    summary_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wiring_ring', choices=
                        ['coiling', 'gathering', 'lifting', 'separation', 'slingshot', 'unknotting', 'wiring_post', 'wrapping'])
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_inner_steps', type=int, default=20)
    parser.add_argument('--pos_bound', type=float, default=0.01)
    parser.add_argument('--angle_bound', type=float, default=0.1)
    parser.add_argument('--min_z', type=float, default=0.013)
    parser.add_argument('--scale_method', type=str, default=None, choices=[None, 'linear', 'exp', 'custom'])
    parser.add_argument('--exp_base', type=float, default=1.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_min', type=float, default=0.000001)
    parser.add_argument('--bptt_window', type=int, default=0,
                        help='Truncated-BPTT window in scene.step() units (0=full BPTT; must be < n_inner_steps to truncate within a macro-step)')
    parser.add_argument('--max_grad_norm', type=float, default=1000.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'cosine'])
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--raytracer', '-r', action='store_true')
    args = parser.parse_args()

    experiment(args)
