import os
import json
import time
import pickle
import random
from omegaconf import DictConfig
from argparse import ArgumentParser
from typing import Tuple, List, Dict, Any, Optional, Sequence

import numpy as np
import cma

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


# ----------------------------
# Helpers: shape & constraints
# ----------------------------
def reshape_to_traj(x: np.ndarray, n_steps: int, act_dim: int) -> np.ndarray:
    return x.reshape(n_steps, act_dim)

def _as_per_comp_array(per_comp_bound: Optional[Sequence[float]], act_dim: int) -> np.ndarray:
    if per_comp_bound is None:
        return np.full((act_dim,), np.inf, dtype=np.float32)
    if np.isscalar(per_comp_bound):
        return np.full((act_dim,), float(per_comp_bound), dtype=np.float32)
    arr = np.asarray(per_comp_bound, dtype=np.float32).reshape(-1)
    if arr.size != act_dim:
        raise ValueError(f"per_comp_bound length {arr.size} != act_dim {act_dim}")
    return arr

def project_deltas(
    traj: np.ndarray,
    pcb: np.ndarray,
    angle_scale: np.ndarray,
    max_l2_per_step: Optional[float],
) -> np.ndarray:
    n_steps, act_dim = traj.shape
    assert pcb.shape == (act_dim,)
    n_repeat = (act_dim // 2) // angle_scale.shape[0]
    angle_scale = np.tile(angle_scale, n_repeat)
    assert angle_scale.shape == (act_dim // 2,)
    traj[:, act_dim // 2:] = traj[:, act_dim // 2:] * angle_scale
    if np.isfinite(pcb).any():
        traj = np.clip(traj, -pcb, pcb)
    if max_l2_per_step is not None and np.isfinite(max_l2_per_step):
        # only limit the first half of the action dimensions (x,y,z)
        # the second half (roll,pitch,yaw) are not limited using l2 norm
        norms = np.linalg.norm(traj[:, :act_dim // 2], axis=1, keepdims=True)
        scale = np.ones_like(norms, dtype=traj.dtype)
        over = norms > max_l2_per_step
        scale[over] = max_l2_per_step / (norms[over] + 1e-12)
        traj[:, :act_dim // 2] = traj[:, :act_dim // 2] * scale
    return traj

def _save_best_traj(log_dir: str, best_traj: np.ndarray):
    np.save(os.path.join(log_dir, "best_traj.npy"), best_traj)

def _save_traj(log_dir: str, traj: np.ndarray, iter: int):
    ckpt_dir = os.path.join(log_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, f"traj_iter{iter:03d}.npy"), traj)

def _save_best_qpos(log_dir: str, best_qpos: np.ndarray):
    np.save(os.path.join(log_dir, "best_qpos.npy"), best_qpos)

def _save_qpos(log_dir: str, qpos: np.ndarray, iter: int):
    ckpt_dir = os.path.join(log_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, f"qpos_iter{iter:03d}.npy"), qpos)

def _save_cmaes_ckpt(es, log_dir: str, it: int, best_reward: float):
    with open(os.path.join(log_dir, "cmaes_ckpt.pkl"), 'wb') as f:
        f.write(es.pickle_dumps())
    with open(os.path.join(log_dir, "resume_meta.json"), 'w') as f:
        json.dump({
            "iter": it,
            "best_reward": best_reward,
        }, f, indent=4)

# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env: Train_Env, traj_list: List[np.ndarray]) -> Dict[str, Any]:
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr

    return env.eval_traj(trajs)

def evaluate_single(env: Train_Env, traj: np.ndarray, log_dir: str, n_steps: int) -> float:
    print(f'Traj shape: {traj.shape}')
    # TODO: hack here
    # if getattr(env, "c1", None) is not None:
    #     env.c1.debug = True
    # if getattr(env, "c2", None) is not None:
    #     env.c2.debug = True

    if os.path.exists(os.path.join(log_dir, "best_traj.npy")):
        placeholder = np.load(os.path.join(log_dir, "best_traj.npy"))
        placeholder = placeholder[None, ...]
    else:
        # Resolve shapes
        act_dim = env._act_dim
        placeholder = np.zeros((1, n_steps, act_dim), dtype=np.float32)
    out = env.eval_traj(placeholder, debug=True, qpos=traj)
    cum_rewards = out['cum_reward']
    final_rewards = out['final_reward']
    forward_time = out['forward_time']
    mean_step_time = forward_time / (n_steps * 200)
    forward_FPS = (n_steps * 200) / forward_time
    print(f'Single traj cum reward: {cum_rewards[0]:.4f}, final reward: {final_rewards[0]:.4f}')
    print(f'Forward time: {mean_step_time * 1000:.2f}ms, forward FPS: {forward_FPS:.2f}, eq steps: {n_steps * 200}')
    rewards = cum_rewards

    env.save_animation(save_dir=log_dir)

    return rewards[0]

def optimize_trajectory(
    env: Train_Env,
    task: str,
    exp_name: str,
    n_steps: int,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    angle_bound: Optional[float] = None,
    angle_scale: float = 1.0,
    max_iters: int = 200,
    seed: int = 42,
    # NEW: checkpointing
    resume: bool = False,
    save_every: int = 1,
    use_last_state_reward: bool = False,
    randomized_args: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Adds CMA-ES checkpointing via (work_dir/trial_name)/cmaes_ckpt.pkl.
    - If resume=True and the file exists, loads the CMA state and continues.
    - Saves checkpoint every `save_every` iters and at the end.

    Other behavior: general shapes, logging, and optional bound inference unchanged.
    """
    # Resolve shapes
    if act_dim is None:
        act_dim = env._act_dim
    if act_dim is None:
        raise ValueError("act_dim could not be inferred; please pass act_dim explicitly.")
    if n_steps is None:
        raise ValueError("n_steps could not be inferred; please pass n_steps explicitly.")

    # Resolve l2 bound (optional)
    if l2_bound is None and hasattr(env, "l2_bound"):
        l2_bound = float(getattr(env, "l2_bound"))

    # Resolve log_dir
    log_dir = os.path.join("logs", task, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    args_path = os.path.join(log_dir, "run_config.json")

    with open(args_path, 'w') as f:
        info = {
            "n_envs": getattr(env, "n_envs", None),
            "n_steps": n_steps,
            "n_steps_sub": getattr(env, "_cmaes_n_steps_sub", None),
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                                else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "angle_bound": angle_bound,
            "angle_scale": angle_scale,
            "max_iters": max_iters,
            "seed": seed,
            "use_last_state_reward": use_last_state_reward,
            "randomized_args": randomized_args,
        }
        json.dump(info, f, indent=4)

    dim = n_steps * act_dim
    pcb = _as_per_comp_array(per_comp_bound, act_dim // 2)
    pcb_angle = _as_per_comp_array(angle_bound, act_dim // 2)
    pcb = np.concatenate([pcb, pcb_angle])
    angle_scale = np.asarray(angle_scale, dtype=np.float32).reshape(-1)

    print(f'Bound: {pcb}')
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    print(f'Max moving distance {l2_bound}x{n_steps}={l2_bound * n_steps} m for each control point')

    best_traj = None
    best_reward = -np.inf

    # Try to resume CMA-ES
    es = None
    start_iter = 0

    rewards_all_path = os.path.join(log_dir, "rewards_all.csv")
    summary_path = os.path.join(log_dir, "summary.csv")
    if resume:
        pkl_path = os.path.join(log_dir, "cmaes_ckpt.pkl")
        meta_path = os.path.join(log_dir, "resume_meta.json")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                es = pickle.load(f)
            start_iter = 0
            best_reward = -np.inf
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    content = json.load(f)
                start_iter = int(content.get("iter", 0)) + 1
                best_reward = float(content.get("best_reward", -np.inf))
            rewards_all_file = open(rewards_all_path, 'a')
            summary_file = open(summary_path, 'a')
        else:
            es = None
            start_iter = 0
            best_reward = -np.inf

            rewards_all_file = open(rewards_all_path, 'w')
            rewards_all_file.write('iter,idx,final,cum\n')

            summary_file = open(summary_path, 'w')
            if use_last_state_reward:
                summary_file.write(
                    'iter,pop,chunks,mean,std,min,max,mean_last,std_last,min_last,max_last,best_so_far,sigma,forward_time,t_iter_s,t_total_s\n'
                )
            else:
                summary_file.write(
                    'iter,pop,chunks,mean,std,min,max,best_so_far,sigma,forward_time,t_iter_s,t_total_s\n'
                )

        if es is not None:
            # quick sanity: check dimension matches
            if getattr(es, "N", dim) != dim:
                raise ValueError(f"Loaded CMA-ES dimension {getattr(es, 'N', None)} "
                                 f"does not match expected dim {dim}.")
            # Note: popsize is internal in es.opts; we trust the checkpoint.
            print(f"[resume] Loaded CMA-ES from iteration {start_iter} "
                  f"with dim={dim}, expected max_iters={max_iters}, "
                  f"loaded best reward {best_reward:.4f}.")
        else:
            print("[resume] No checkpoint found; starting fresh.")

    # Fresh CMA-ES if not resumed
    if es is None:
        es = cma.CMAEvolutionStrategy(
            x0=[0.0] * dim,
            sigma0=sigma0,
            inopts={
                'bounds': [lower, upper],
                'popsize': popsize,
                'seed': seed,
                'CMA_elitist': True,
                'verb_disp': 0,
            }
        )

    assert es.popsize == popsize, f"CMA-ES popsize {es.popsize} != expected {popsize}"

    batch_size = env.n_envs
    it = start_iter
    t0_all = time.time()

    # If resuming, keep the previous total time in resume_meta (optional)
    print(f"{'iter':>5} | {'pop':>4} | {'chunks':>6} | {'mean':>8} | {'std':>8} | "
          f"{'min':>8} | {'max':>8} | {'best':>8} | {'sigma':>7} | {'t_iter(s)':>8} | {'t_total(s)':>9}")

    while it < max_iters:
        t_iter = time.time()
        X = es.ask()    # (popsize, n_steps * act_dim)
        pop = len(X)
        n_chunks = (pop + batch_size - 1) // batch_size
        qpos_list = list()

        all_final_rewards = []
        all_cum_rewards = []
        forward_time = 0.0
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, pcb, angle_scale, l2_bound)
                trajs.append(tr)
            out = evaluate_batch(env, trajs)
            qpos_list.extend(env.qpos_seq.tolist())
            if out.get('final_reward') is not None:
                all_final_rewards.extend(out['final_reward'].tolist())
            if out.get('cum_reward') is not None:
                all_cum_rewards.extend(out['cum_reward'].tolist())
            if out.get('forward_time') is not None:
                forward_time += out['forward_time']
            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_cum_rewards, dtype=np.float32)
        assert all_rewards.shape[0] == len(X), f"all_rewards {all_rewards.shape[0]} vs X {len(X)} length mismatch"

        # Log raw rewards for this generation
        for idx in range(len(all_final_rewards)):
            rewards_all_file.write(f"{it},{idx},{all_final_rewards[idx]},{all_cum_rewards[idx]}\n")
        rewards_all_file.flush()
        os.fsync(rewards_all_file.fileno())

        # CMA-ES minimizes; negate to maximize reward
        if use_last_state_reward:
            all_final_rewards = np.asarray(all_final_rewards, dtype=np.float32)
            es.tell(X, (-all_final_rewards).tolist())
            # Track best of gen
            gen_best_idx = int(np.argmax(all_final_rewards))
            gen_best_reward = float(all_final_rewards[gen_best_idx])
        else:
            es.tell(X, (-all_rewards).tolist())
            # Track best of gen
            gen_best_idx = int(np.argmax(all_rewards))
            gen_best_reward = float(all_rewards[gen_best_idx])

        qpos_array = np.asarray(qpos_list, dtype=np.float32)
        assert qpos_array.shape[0] == len(X), f"qpos_array {qpos_array.shape[0]} and X lengths {len(X)} do not match"
        gen_best_qpos = np.asarray(qpos_array[gen_best_idx], dtype=np.float32)

        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps, act_dim),
            pcb, angle_scale, l2_bound
        )

        _save_traj(log_dir, gen_best_traj, it)
        _save_qpos(log_dir, gen_best_qpos, it)

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)
                _save_best_qpos(log_dir, gen_best_qpos)

        # Iteration summary
        m = float(all_rewards.mean()) if all_rewards.size else float('nan')
        s = float(all_rewards.std()) if all_rewards.size else float('nan')
        mn = float(all_rewards.min()) if all_rewards.size else float('nan')
        mx = float(all_rewards.max()) if all_rewards.size else float('nan')
        if use_last_state_reward:
            ml = float(all_final_rewards.mean()) if all_final_rewards.size else float('nan')
            sl = float(all_final_rewards.std()) if all_final_rewards.size else float('nan')
            mnl = float(all_final_rewards.min()) if all_final_rewards.size else float('nan')
            mxl = float(all_final_rewards.max()) if all_final_rewards.size else float('nan')
        try:
            sigma_now = float(es.sigma)
        except Exception:
            sigma_now = float(es.sigma0) if hasattr(es, 'sigma0') else float('nan')

        t_iter_sec = time.time() - t_iter
        t_total_sec = time.time() - t0_all

        print(f"{it:5d} | {pop:4d} | {n_chunks:6d} | {m:8.4f} | {s:8.4f} | "
              f"{mn:8.4f} | {mx:8.4f} | {best_reward:8.4f} | {sigma_now:7.4f} | "
              f"{t_iter_sec:8.3f} | {t_total_sec:9.3f}")

        if use_last_state_reward:
            summary_file.write(
                f"{it},{pop},{n_chunks},{m},{s},{mn},{mx},{ml},{sl},{mnl},{mxl},{best_reward},{sigma_now},"
                f"{forward_time},{t_iter_sec},{t_total_sec}\n"
            )
        else:
            summary_file.write(
                f"{it},{pop},{n_chunks},{m},{s},{mn},{mx},{best_reward},{sigma_now},"
                f"{forward_time},{t_iter_sec},{t_total_sec}\n"
            )
        summary_file.flush()
        os.fsync(summary_file.fileno())

        # Save checkpoint periodically
        if save_every > 0 and (it % save_every == 0):
            _save_cmaes_ckpt(es, log_dir, it, best_reward)

        it += 1

    # Final checkpoint
    _save_cmaes_ckpt(es, log_dir, it - 1, best_reward)

    return best_traj, best_reward

def _build_env(
        task: str, log_dir: str, n_envs: int,
        vis_traj: Optional[str] = None, gui: bool = False,
        raytracer: bool = False
    ) -> Train_Env:
    task = task.lower()
    task_to_env = {
        "coiling":   Train_Env_Coiling,
        "gathering": Train_Env_Gathering,
        "lifting":   Train_Env_Lifting,
        "separation": Train_Env_Separation,
        "slingshot": Train_Env_Slingshot,
        "unknotting": Train_Env_Unknotting,
        "wiring_post": Train_Env_Wiring_post,
        "wrapping":  Train_Env_Wrapping,
    }
    if task not in task_to_env:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(task_to_env.keys())}")
    EnvCls = task_to_env[task]
    if vis_traj is None:
        camera = False
    else:
        n_envs = 1
        camera = True

    cfg = DictConfig({
        "task": task,
        "log_dir": log_dir,
        "n_envs": n_envs,
        "GUI": gui,
        "camera": camera,
        "raytracer": raytracer,
        "requires_grad": False,
    })
    return EnvCls(config=cfg)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--task', type=str, default="coiling",
        help="Task / environment to optimize."
    )
    parser.add_argument(
        '--popsize', type=int, default=400,
        help="CMA-ES population size."
    )
    parser.add_argument(
        '--seed', type=int, default=123,
    )
    parser.add_argument(
        '--n_envs', type=int, default=10,
    )
    parser.add_argument(
        '--max_iter', type=int, default=20,
    )
    parser.add_argument(
        '--n_steps', type=int, default=10,
    )
    parser.add_argument(
        '--n_steps_sub', type=int, default=10,
    )
    parser.add_argument(
        '--use_last_state_reward', action='store_true',
        help="Whether to use the last state reward instead of accumulated reward."
    )
    parser.add_argument(
        '--vis_traj', type=str, default=None, 
        help="Path to saved trajectory .npy for visualization. If None, runs optimization."
    )
    parser.add_argument(
        '--bound', type=float, default=0.1,
        help="Per-step L2 bound for each control point."
    )
    parser.add_argument(
        '--angle_bound', type=float, default=10.0,
        help="Per-step angle bound for each control point."
    )
    parser.add_argument(
        '--angle_scale', nargs='+', type=float, default=[1.0, 1.0, 1.0],
        help="Scaling factor for angle dimensions in optimization."
    )
    parser.add_argument(
        '--sigma', type=float, default=0.05
    )
    parser.add_argument(
        '--exp_name', type=str, default=None,
    )
    parser.add_argument('--gui', action='store_true', help="Whether to show GUI.")
    parser.add_argument('--raytracer', '-r', action='store_true', help='Enable raytracer for rendering')
    args = parser.parse_args()

    exp_name = f"{args.exp_name}" if args.exp_name is not None else "cmaes"
    log_dir = f"logs/{args.task}/{exp_name}"
    env = _build_env(args.task, log_dir, args.n_envs, args.vis_traj, args.gui, args.raytracer)
    env.init_cmaes_env(
        n_steps_sub=args.n_steps_sub,
    )
    print(f'CMA-ES n_steps_sub: {env._cmaes_n_steps_sub}')
    # Initialize domain randomization
    # randomized_args = randomized_config.get(args.task, {})
    # env.init_domain_randomization(**randomized_args)
    n_steps = args.n_steps

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.vis_traj is None:

        assert not env.requires_grad, "CMA-ES optimization does not need env with gradients."
        
        best_traj, best_reward = optimize_trajectory(
            env,
            task=args.task,
            exp_name=exp_name,
            n_steps=n_steps,
            act_dim=None,           # infer if available
            popsize=args.popsize,
            sigma0=args.sigma,
            per_comp_bound=args.bound,
            l2_bound=args.bound,          # use env.l2_bound if present
            angle_bound=args.angle_bound,
            angle_scale=args.angle_scale,
            max_iters=args.max_iter,
            seed=args.seed,
            resume=True,            # set True to load if checkpoint exists
            save_every=1,           # save each generation
            use_last_state_reward=args.use_last_state_reward,
            # randomized_args=randomized_args,
        )

    else:

        color_print(f'Visualizing CMA-ES trajectory from {args.vis_traj}', "magenta")
        evaluate_single(env, np.load(args.vis_traj), log_dir, n_steps)
