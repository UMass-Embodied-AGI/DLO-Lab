#!/usr/bin/env python3
"""
analyze_unknotting_failure.py
==============================
Failure analysis for the DLO-Lab Unknotting task under PPO.

This script:
  1. Loads a trained PPO checkpoint (or uses a random policy for baseline)
  2. Rolls out N episodes and records per-step reward, ACN (knottedness), and
     end-effector trajectories
  3. Identifies *why* episodes fail: sparse reward, early collision, slow
     untangling, or NaN divergence
  4. Produces publication-quality plots:
     - Learning curve (if tensorboard logs present)
     - Per-episode reward distribution
     - ACN (Average Crossing Number) over time — the actual unknotting metric
     - Failure mode breakdown (pie chart)
     - Example failure trajectory overlaid on initial knot state

Run from the `experiments/` directory:

    cd experiments

    # Analyse a random policy (no checkpoint needed, for baseline)
    python analyze_unknotting_failure.py --random_policy --n_episodes 20

    # Analyse a trained checkpoint (.pkl saved by MushroomRL)
    python analyze_unknotting_failure.py \\
        --checkpoint logs/unknotting/rudin-01/best_ppo.pkl \\
        --n_episodes 50

    # Full analysis with trajectory visualisation
    python analyze_unknotting_failure.py --random_policy \\
        --n_episodes 20 --save_dir ./unknotting_analysis

Outputs (saved to --save_dir):
    reward_distribution.pdf
    acn_over_time.pdf
    failure_modes.pdf
    episode_trajectories.pdf   (if --save_trajectories)
    summary.txt
"""

import argparse
import os
import sys
import json
import time
import traceback
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode taxonomy
# ─────────────────────────────────────────────────────────────────────────────

FAILURE_MODES = {
    'success':       'ACN < 0.1 at episode end',
    'sparse_reward': 'reward < 0.05 throughout (policy never unties)',
    'collision':     'environment terminated early (collision or stretch)',
    'slow_progress': 'ACN decreases but not below 0.1 within horizon',
    'diverged':      'NaN in observation or reward',
}


def classify_episode(rewards, acn_values, terminated_early, has_nan):
    """Classify an episode into one failure mode."""
    if has_nan:
        return 'diverged'
    if terminated_early:
        return 'collision'
    final_acn = acn_values[-1] if len(acn_values) > 0 else float('inf')
    if final_acn < 0.1:
        return 'success'
    max_reward = max(rewards) if rewards else 0.0
    if max_reward < 0.05:
        return 'sparse_reward'
    return 'slow_progress'


# ─────────────────────────────────────────────────────────────────────────────
# Genesis + environment setup
# ─────────────────────────────────────────────────────────────────────────────

def build_env(n_envs=1):
    import genesis as gs
    from omegaconf import OmegaConf
    from envs.env_unknotting import Train_Env_Unknotting

    if not gs._initialized:
        gs.init(seed=0, precision='64', logging_level='error', backend=gs.gpu, performance_mode=True)

    cfg = OmegaConf.create({
        'task': 'unknotting',
        'n_envs': n_envs,
        'GUI': False,
        'camera': False,
        'raytracer': False,
        'requires_grad': False,
        'log_dir': '/tmp/dlolab_analysis/unknotting',
        'n_substeps_per_step': 200,
    })

    env = Train_Env_Unknotting(config=cfg)
    env.init_rl_env(
        n_steps=100,
        pos_bound=0.05,
        angle_bound=5.0,
        steps_interval_split=2,
    )
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

def make_random_policy(act_dim):
    """Random policy — baseline for failure analysis.

    Returns actions in [-1, 1]; step_all() scales by env._act_magnitude internally.
    """
    import torch

    def policy(obs):
        return torch.rand((obs.shape[0], act_dim)) * 2 - 1   # uniform in [-1, 1]

    return policy


def load_ppo_policy(checkpoint_path):
    """Load a saved PPO policy from a MushroomRL .pkl checkpoint.

    DLO-Lab saves PPO agents via MushroomRL's agent.save() (see rl/rudinppo.py).
    Checkpoints are .pkl files: best_ppo.pkl, latest_ppo.pkl, or <epoch>_ppo.pkl.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rl.rudinppo import RudinPPO

    agent = RudinPPO.load(path=checkpoint_path)
    agent.policy._log_sigma.data = agent.policy._log_sigma.data.float()

    def policy(obs):
        action, _ = agent.draw_action(obs)
        return action

    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

def compute_acn(verts_np):
    """
    Approximate Average Crossing Number for a single rope configuration.

    verts_np : (n_verts, 3) array

    The ACN is computed via the Gauss linking integral approximation used
    in env_unknotting.py's `_unknotting_penalty`. We replicate the torch
    computation in numpy for post-hoc analysis.

    Returns a scalar ACN value (lower = less knotted; < 0.1 ≈ unknotted).
    """
    import numpy as np
    V = verts_np  # (N, 3)
    N = len(V)
    if N < 4:
        return 0.0

    edges = V[1:] - V[:-1]           # (N-1, 3)
    midpoints = (V[1:] + V[:-1]) / 2  # (N-1, 3)
    n_edges = len(edges)

    # Pairwise midpoint differences
    r_i = midpoints[:, None, :]       # (N-1, 1, 3)
    r_j = midpoints[None, :, :]       # (1, N-1, 3)
    r_ij = r_i - r_j                  # (N-1, N-1, 3)

    dr_i = edges[:, None, :]
    dr_j = edges[None, :, :]

    cross = np.cross(dr_i, dr_j)                   # (N-1, N-1, 3)
    numerator = np.sum(r_ij * cross, axis=-1)       # (N-1, N-1)
    dist = np.linalg.norm(r_ij, axis=-1)            # (N-1, N-1)
    denominator = (dist ** 2 + 0.02 ** 2) ** 1.5

    acn_matrix = np.abs(numerator) / (denominator + 1e-12)

    # Mask nearest neighbours
    idx = np.arange(n_edges)
    diff = np.abs(idx[:, None] - idx[None, :])
    mask = diff <= 1
    acn_matrix[mask] = 0.0

    acn = acn_matrix.sum() / (4 * np.pi)
    return float(acn)


def rollout_episode(env, policy, n_steps=100):
    """Run one episode and collect diagnostics."""
    import torch

    obs, _ = env.reset_all(torch.ones(env.n_envs, dtype=torch.bool))
    obs = obs[0:1]  # take env 0 only for analysis

    rewards, acn_vals, ef1_traj, ef2_traj = [], [], [], []
    terminated_early = False
    has_nan = False

    for step in range(n_steps):
        # Check for NaN in obs
        if torch.isnan(obs).any():
            has_nan = True
            break

        action = policy(obs)
        next_obs, rew, absorbing, _ = env.step_all(
            torch.ones(env.n_envs, dtype=torch.bool), action
        )

        r = float(rew[0].cpu())
        rewards.append(r)

        if torch.isnan(rew[0]) or torch.isnan(next_obs[0]).any():
            has_nan = True
            break

        # Get current rope configuration for ACN
        verts = env.rope.get_all_verts()[0].copy()    # (n_verts, 3)
        acn_vals.append(compute_acn(verts))

        # Record end-effector positions
        ef1_pos = env.c1.ef.get_pos()[0].cpu().numpy()
        ef2_pos = env.c2.ef.get_pos()[0].cpu().numpy()
        ef1_traj.append(ef1_pos.copy())
        ef2_traj.append(ef2_pos.copy())

        if bool(absorbing[0].cpu()):
            terminated_early = True
            break

        obs = next_obs[0:1]

    return {
        'rewards':          rewards,
        'acn':              acn_vals,
        'ef1_traj':         np.array(ef1_traj) if ef1_traj else np.zeros((0, 3)),
        'ef2_traj':         np.array(ef2_traj) if ef2_traj else np.zeros((0, 3)),
        'terminated_early': terminated_early,
        'has_nan':          has_nan,
        'n_steps':          len(rewards),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis and plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_distribution(all_rewards, save_path):
    total_rewards = [sum(r) for r in all_rewards]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(total_rewards, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(total_rewards), color='r', ls='--',
               label=f'mean = {np.mean(total_rewards):.3f}')
    ax.set_xlabel('Total episode reward')
    ax.set_ylabel('Count')
    ax.set_title('Episode reward distribution\n(PPO on Unknotting)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    # Learning curve proxy: reward vs episode index
    ax2.plot(range(len(total_rewards)), total_rewards, 'o-', ms=4, lw=1.5, color='steelblue')
    # Rolling average
    window = max(1, len(total_rewards) // 5)
    rolling = np.convolve(total_rewards, np.ones(window) / window, mode='valid')
    ax2.plot(range(window - 1, len(total_rewards)), rolling, 'r-', lw=2,
             label=f'rolling mean (w={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total reward')
    ax2.set_title('Reward across evaluation episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_acn_over_time(all_acn, failure_modes, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    color_map = {
        'success':       'seagreen',
        'slow_progress': 'steelblue',
        'sparse_reward': 'goldenrod',
        'collision':     'tomato',
        'diverged':      'purple',
    }

    for i, (acn, mode) in enumerate(zip(all_acn, failure_modes)):
        if len(acn) == 0:
            continue
        c = color_map.get(mode, 'gray')
        ax.plot(range(len(acn)), acn, alpha=0.5, lw=1.0, color=c)

    # Mean ACN across all episodes
    max_len = max((len(a) for a in all_acn), default=0)
    if max_len > 0:
        padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in all_acn]
        mean_acn = np.nanmean(padded, axis=0)
        ax.plot(range(len(mean_acn)), mean_acn, 'k-', lw=2.5, label='Mean ACN', zorder=5)

    ax.axhline(0.1, color='darkgreen', ls='--', lw=1.5, label='Success threshold (ACN < 0.1)')

    legend_patches = [
        mpatches.Patch(color=c, label=f'{m}: {FAILURE_MODES[m]}')
        for m, c in color_map.items()
    ]
    legend_patches.append(plt.Line2D([0], [0], color='k', lw=2, label='Mean ACN'))
    ax.legend(handles=legend_patches, fontsize=7, loc='upper right')

    ax.set_xlabel('Step within episode')
    ax.set_ylabel('Average Crossing Number (ACN)')
    ax.set_title('ACN over time — Unknotting task\n'
                 '(lower = less knotted; < 0.1 = success)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_failure_modes(failure_modes, save_path):
    counts = defaultdict(int)
    for m in failure_modes:
        counts[m] += 1

    color_map = {
        'success':       'seagreen',
        'slow_progress': 'steelblue',
        'sparse_reward': 'goldenrod',
        'collision':     'tomato',
        'diverged':      'purple',
    }

    labels = [f'{k}\n({v} ep)' for k, v in counts.items() if v > 0]
    sizes  = [v for v in counts.values() if v > 0]
    colors = [color_map.get(k, 'gray') for k in counts if counts[k] > 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title('Failure mode breakdown\n(Unknotting task, PPO baseline)')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_example_trajectories(episodes, failure_modes, save_path, n_show=4):
    """Plot EF trajectories for a few example failure episodes."""
    # Pick one episode per failure mode (prefer first occurrence)
    selected = {}
    for i, (ep, mode) in enumerate(zip(episodes, failure_modes)):
        if mode not in selected and ep['ef1_traj'].shape[0] > 0:
            selected[mode] = (i, ep)

    if not selected:
        return

    n_cols = min(len(selected), n_show)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4),
                             subplot_kw={'projection': '3d'})
    if n_cols == 1:
        axes = [axes]

    for ax, (mode, (i, ep)) in zip(axes, list(selected.items())[:n_cols]):
        t1 = ep['ef1_traj']
        t2 = ep['ef2_traj']
        if t1.shape[0] > 0:
            ax.plot(t1[:, 0], t1[:, 1], t1[:, 2], 'b-', lw=2, label='EF1')
            ax.plot(*t1[0], 'b>', ms=7)
            ax.plot(*t1[-1], 'bs', ms=7)
        if t2.shape[0] > 0:
            ax.plot(t2[:, 0], t2[:, 1], t2[:, 2], 'r-', lw=2, label='EF2')
            ax.plot(*t2[0], 'r>', ms=7)
            ax.plot(*t2[-1], 'rs', ms=7)

        n_steps = ep['n_steps']
        final_r = sum(ep['rewards'])
        ax.set_title(f'Mode: {mode}\nep={i}, steps={n_steps}, R={final_r:.3f}',
                     fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(fontsize=7)

    plt.suptitle('End-effector trajectories by failure mode', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def write_summary(episodes, failure_modes, save_path):
    total = len(episodes)
    counts = defaultdict(int)
    for m in failure_modes:
        counts[m] += 1

    total_rewards = [sum(ep['rewards']) for ep in episodes]
    all_acn_final = [ep['acn'][-1] if ep['acn'] else float('nan') for ep in episodes]

    lines = [
        '=' * 60,
        'DLO-Lab Unknotting Failure Analysis',
        '=' * 60,
        f'Total episodes evaluated : {total}',
        f'Mean total reward        : {np.nanmean(total_rewards):.4f}',
        f'Std total reward         : {np.nanstd(total_rewards):.4f}',
        f'Mean final ACN           : {np.nanmean(all_acn_final):.4f}',
        '',
        'Failure mode breakdown:',
    ]
    for mode, desc in FAILURE_MODES.items():
        n = counts.get(mode, 0)
        pct = 100 * n / total if total > 0 else 0
        lines.append(f'  {mode:<16} {n:3d} ({pct:5.1f}%)  — {desc}')

    lines += [
        '',
        'Hypotheses for PPO failure on Unknotting:',
        '  1. REWARD SPARSITY: exp(-ACN) is near-zero for typical initial knots',
        '     (ACN ≈ 3–5), giving gradient signal ≈ 0 throughout most of training.',
        '     PPO cannot shape behaviour when the dense reward is indistinguishable',
        '     from random noise in early training.',
        '  2. EPISODE HORIZON: 100 steps × 200 substeps = 20 s of simulation,',
        '     but unknotting typically requires coordinated multi-stage motions',
        '     (pull apart, loop through, separate) that PPO cannot plan over a',
        '     single short horizon without curriculum or demonstrations.',
        '  3. OBSERVATION REPRESENTATION: The rope state is a flat (n_verts × 6)',
        '     vector — high-dimensional, permutation-sensitive. PPO\'s MLP policy',
        '     has no inductive bias for the topological structure of the knot.',
        '     A graph-based policy (GNN) might extract better features.',
        '  4. COLLISION FAILURES (see pie chart): Early terminations prevent the',
        '     policy from ever reaching a state where unknotting reward is non-zero.',
        '     Curriculum initialisation (start from less knotted configurations)',
        '     could help.',
        '',
        '=' * 60,
    ]

    text = '\n'.join(lines)
    with open(save_path, 'w') as f:
        f.write(text)
    print(f'  Saved: {save_path}')
    print()
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='DLO-Lab Unknotting failure analysis'
    )
    p.add_argument('--checkpoint',    type=str, default=None,
                   help='Path to MushroomRL PPO checkpoint (.pkl, e.g. best_ppo.pkl)')
    p.add_argument('--random_policy', action='store_true',
                   help='Use a random policy (baseline, no checkpoint needed)')
    p.add_argument('--n_episodes',   type=int, default=20,
                   help='Number of episodes to evaluate (default: 20)')
    p.add_argument('--n_steps',      type=int, default=100,
                   help='Steps per episode (default: 100)')
    p.add_argument('--save_dir',     type=str, default='./unknotting_analysis',
                   help='Directory for output plots and summary')
    p.add_argument('--save_trajectories', action='store_true',
                   help='Also save EF trajectory plot per failure mode')
    p.add_argument('--seed',         type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.random_policy and args.checkpoint is None:
        print('ERROR: Provide --checkpoint or --random_policy.')
        sys.exit(1)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Building Unknotting environment...')
    env = build_env(n_envs=1)

    # Build policy
    act_dim = env._act_dim
    act_mag = env._act_magnitude

    if args.random_policy:
        print('Using random policy (baseline).')
        policy = make_random_policy(act_dim)
    else:
        print(f'Loading checkpoint: {args.checkpoint}')
        policy = load_ppo_policy(args.checkpoint)

    print(f'\nRunning {args.n_episodes} episodes ({args.n_steps} steps each)...\n')

    all_episodes = []
    failure_modes = []
    t_start = time.time()

    for ep_idx in range(args.n_episodes):
        print(f'  Episode {ep_idx + 1:3d}/{args.n_episodes}', end=' ', flush=True)
        try:
            result = rollout_episode(env, policy, n_steps=args.n_steps)
        except Exception:
            print('  [exception]')
            traceback.print_exc()
            result = {
                'rewards': [], 'acn': [], 'ef1_traj': np.zeros((0, 3)),
                'ef2_traj': np.zeros((0, 3)), 'terminated_early': False,
                'has_nan': True, 'n_steps': 0
            }

        mode = classify_episode(
            result['rewards'], result['acn'],
            result['terminated_early'], result['has_nan']
        )
        total_r = sum(result['rewards'])
        final_acn = result['acn'][-1] if result['acn'] else float('nan')
        print(f'| mode={mode:<16} R={total_r:.4f}  ACN_final={final_acn:.4f}  steps={result["n_steps"]}')

        all_episodes.append(result)
        failure_modes.append(mode)

    elapsed = time.time() - t_start
    print(f'\nTotal evaluation time: {elapsed:.1f} s ({elapsed/args.n_episodes:.1f} s/episode)')
    print(f'\nGenerating plots in {args.save_dir}/ ...')

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_reward_distribution(
        [ep['rewards'] for ep in all_episodes],
        os.path.join(args.save_dir, 'reward_distribution.pdf')
    )
    plot_acn_over_time(
        [ep['acn'] for ep in all_episodes],
        failure_modes,
        os.path.join(args.save_dir, 'acn_over_time.pdf')
    )
    plot_failure_modes(
        failure_modes,
        os.path.join(args.save_dir, 'failure_modes.pdf')
    )
    if args.save_trajectories:
        plot_example_trajectories(
            all_episodes, failure_modes,
            os.path.join(args.save_dir, 'episode_trajectories.pdf')
        )

    write_summary(
        all_episodes, failure_modes,
        os.path.join(args.save_dir, 'summary.txt')
    )

    # Save raw data for further analysis
    raw_path = os.path.join(args.save_dir, 'raw_results.json')
    raw = []
    for ep, mode in zip(all_episodes, failure_modes):
        raw.append({
            'failure_mode': mode,
            'total_reward': float(sum(ep['rewards'])),
            'n_steps':      ep['n_steps'],
            'final_acn':    float(ep['acn'][-1]) if ep['acn'] else None,
            'terminated_early': ep['terminated_early'],
            'has_nan':      ep['has_nan'],
        })
    with open(raw_path, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f'  Saved: {raw_path}')


if __name__ == '__main__':
    main()
