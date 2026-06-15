"""Uniform policy interface for PPO, SAC, SHAC, and CMA-ES."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# Clip action magnitude
def clip_action(action: np.ndarray, act_magnitude: float = 1.0, max_mag: float = 1.0) -> np.ndarray:
    action *= act_magnitude
    action = np.clip(action, -max_mag, max_mag)
    mag = np.linalg.norm(action)

    if mag > max_mag:
        return action / mag * max_mag
    else:
        return action


class PolicyWrapper:
    """Base class for policy wrappers."""

    def get_action(self, obs_or_step):
        """Return action as numpy array (act_dim,)."""
        raise NotImplementedError

    @property
    def is_trajectory_based(self) -> bool:
        return False


class PPOPolicyWrapper(PolicyWrapper):
    """Wrapper for MushroomRL RudinPPO agents."""

    def __init__(self, checkpoint_path: str, mdp_info=None):
        import sys
        import __main__
        from mushroom_rl.algorithms.actor_critic import RudinPPO
        from rl.rudinppo import Network

        # Torch pickle expects Network in __main__ (where checkpoint was saved)
        if not hasattr(__main__, "Network"):
            __main__.Network = Network

        self.agent = RudinPPO.load(path=checkpoint_path)
        self.agent.policy._log_sigma.data = self.agent.policy._log_sigma.data.float()
        print(f"Loaded PPO from {checkpoint_path}")

    def get_action(self, obs):
        # draw_action returns (action, next_policy_state)
        action, _ = self.agent.draw_action(obs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return np.asarray(action, dtype=np.float32).flatten()


class SACPolicyWrapper(PolicyWrapper):
    """Wrapper for MushroomRL SAC agents."""

    def __init__(self, checkpoint_path: str, mdp_info=None):
        import __main__
        from mushroom_rl.algorithms.actor_critic import SAC
        from rl.sac import ActorNetwork, CriticNetwork

        # Torch pickle expects these classes in __main__
        if not hasattr(__main__, "ActorNetwork"):
            __main__.ActorNetwork = ActorNetwork
        if not hasattr(__main__, "CriticNetwork"):
            __main__.CriticNetwork = CriticNetwork

        self.agent = SAC.load(path=checkpoint_path)
        print(f"Loaded SAC from {checkpoint_path}")

    def get_action(self, obs):
        # draw_action returns (action, next_policy_state)
        action, _ = self.agent.draw_action(obs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return np.asarray(action, dtype=np.float32).flatten()


class SHACPolicyWrapper(PolicyWrapper):
    """Wrapper for SHAC agents."""

    def __init__(self, checkpoint_path: str, env, config_path: str | None = None):
        from diff_rl.shac import SHACAgent

        # Load config from run_config.json next to checkpoint
        ckpt_path = Path(checkpoint_path)
        if config_path is None:
            # Try to find run_config.json in parent dirs
            for parent in [ckpt_path.parent, ckpt_path.parent.parent]:
                candidate = parent / "run_config.json"
                if candidate.exists():
                    config_path = str(candidate)
                    break

        if config_path is not None:
            with open(config_path) as f:
                saved = json.load(f)
            shac_config = saved.get("shac_config", saved)
        else:
            # Fallback: minimal config from env
            print(f"Warning: run_config.json not found near {checkpoint_path}, using minimal SHAC config with env dimensions")
            shac_config = {
                "obs_dim": env._obs_dim,
                "action_dim": env._act_dim,
                "device": "cuda:0",
                "max_agent_steps": 2000000,
                "horizon_len": 16,
                "gamma": 0.99,
                "actor_hidden_dims": (256, 128, 64),
                "critic_hidden_dims": (256, 256),
                "num_critics": 2,
                "activation": "elu",
                "norm_type": None,
                "dist_kwargs": {"dist_type": "normal"},
                "actor_lr": 2e-3,
                "critic_lr": 5e-4,
                "alpha_lr": 5e-3,
                "max_grad_norm": 0.5,
                "lr_schedule": "constant",
                "critic_lrschedule": True,
                "min_lr": 1e-6,
                "max_lr": 2e-3,
                "max_epochs": 100,
                "critic_iterations": 16,
                "num_critic_batches": 4,
                "critic_method": "one-step",
                "no_target_critic": False,
                "target_critic_alpha": 0.4,
                "with_entropy": False,
                "with_logprobs": False,
                "entropy_coef": None,
                "use_distr_ent": False,
                "init_alpha": 1.0,
                "target_entropy_scalar": 0.5,
                "scale_by_target_entropy": False,
                "offset_by_target_entropy": False,
                "unscale_entropy_alpha": False,
                "entropy_in_return": False,
                "entropy_in_targets": False,
                "no_actor_entropy": False,
                "normalize_obs": False,
                "reward_scale": 1.0,
                "reward_shift": 0.0,
                "debug": False,
            }

        # Infer action_dim from the checkpoint — saved configs/env may differ.
        # SHACAgent.__init__ ignores config["action_dim"] and reads env._act_dim
        # directly, so temporarily patch the env to match the checkpoint.
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_action_dim = ckpt["actor"]["mu_layer.weight"].shape[0]

        orig_act_dim = env._act_dim
        env._act_dim = ckpt_action_dim
        try:
            self.agent = SHACAgent(env, shac_config)
            self.agent.load(checkpoint_path)
        finally:
            env._act_dim = orig_act_dim  # restore so env works normally

        # Detect whether the checkpoint uses xyz-only actions (3D per arm).
        # If so, get_action() will pad zeros for roll/pitch/yaw and return
        # a 12D action in step_all format: [all_xyz, all_rot_zeros].
        self._shac_act_dim = shac_config["action_dim"]
        self._env_act_dim  = env._act_dim
        self._n_arms = len(env.control_idx)                        # 1 for single-arm, 2 for bimanual
        self._xyz_only = (self._shac_act_dim == self._n_arms * 3)  # True iff 3D-per-arm xyz-only
        print(f"Loaded SHAC from {checkpoint_path} "
              f"(shac_act_dim={self._shac_act_dim}, env_act_dim={self._env_act_dim}, "
              f"xyz_only={self._xyz_only})")

    def get_action(self, obs):
        action = self.agent.get_action(obs, deterministic=True)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float32).flatten()

        if self._xyz_only:
            # Pad to step_all format: [r1_xyz, r2_xyz, ..., r1_rot(0), r2_rot(0), ...]
            # Consistent with PPO/SAC step_all convention.
            padded = np.zeros(self._n_arms * 6, dtype=np.float32)
            for i in range(self._n_arms):
                padded[i * 3 : i * 3 + 3] = clip_action(action[i * 3 : i * 3 + 3])  # clip xyz per arm
            # rotation slots [n_arms*3 : n_arms*6] remain zeros
            action = padded

        return action


class CMAESPolicyWrapper(PolicyWrapper):
    """Wrapper for CMA-ES policy that samples trajectories from the fitted distribution.

    Loads ``cmaes_ckpt.pkl`` (the serialised ``cma.CMAEvolutionStrategy`` object) and
    samples a new trajectory on every ``reset()`` call.  Falls back to ``best_traj.npy``
    when the pickle is absent.

    The sampled trajectory (``n_steps`` waypoints) is linearly interpolated to
    ``target_steps`` so that the dataset has the same number of frames as RL episodes.
    Bounds projection replicates ``trajopt/cmaes.py::project_deltas`` inline to avoid
    importing the heavy environment classes in that module.
    """

    def __init__(self, checkpoint_dir: str, target_steps: int | None = None):
        import pickle

        ckpt_dir = Path(checkpoint_dir)

        # --- Load run_config for CMA-ES distribution parameters ---
        config_path = ckpt_dir / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self._n_steps = int(config.get("n_steps", 10))
            self._act_dim = int(config.get("act_dim", 12))
            pcb = config.get("per_comp_bound", 0.1)
            self._per_comp_bound = float(pcb) if pcb is not None else 0.1
            ab = config.get("angle_bound", 0.1)
            self._angle_bound = float(ab) if ab is not None else 0.1
            lb = config.get("l2_bound")
            self._l2_bound = float(lb) if lb is not None else None
            self._angle_scale = np.asarray(
                config.get("angle_scale", [1.0, 1.0, 1.0]), dtype=np.float32
            ).reshape(-1)
        else:
            self._n_steps = None
            self._act_dim = None
            self._per_comp_bound = 0.1
            self._angle_bound = 0.1
            self._l2_bound = None
            self._angle_scale = np.ones(3, dtype=np.float32)

        # --- Load CMA-ES entity for stochastic sampling ---
        ckpt_pkl = ckpt_dir / "cmaes_ckpt.pkl"
        if ckpt_pkl.exists():
            with open(ckpt_pkl, "rb") as f:
                self._es = pickle.load(f)
            if self._n_steps is None:
                # Infer n_steps from CMA-ES dimension and act_dim
                dim = self._es.N
                if self._act_dim is not None:
                    self._n_steps = dim // self._act_dim
                else:
                    raise ValueError(
                        "run_config.json not found; cannot infer n_steps from cmaes_ckpt.pkl"
                    )
            print(f"Loaded CMA-ES checkpoint from {ckpt_pkl} "
                  f"(n_steps={self._n_steps}, act_dim={self._act_dim})")
        else:
            self._es = None
            print(f"No cmaes_ckpt.pkl in {ckpt_dir} — falling back to best_traj.npy")

        # --- Fallback: best_traj.npy ---
        traj_path = ckpt_dir / "best_traj.npy"
        if traj_path.exists():
            self._best_traj = np.load(traj_path).astype(np.float32)
            if self._n_steps is None:
                self._n_steps = self._best_traj.shape[0]
            if self._act_dim is None:
                self._act_dim = self._best_traj.shape[1]
        else:
            self._best_traj = None

        if self._es is None and self._best_traj is None:
            raise FileNotFoundError(
                f"Neither cmaes_ckpt.pkl nor best_traj.npy found in {ckpt_dir}"
            )

        self._target_steps = target_steps
        self.traj = None
        self._step = 0
        self.reset()  # initialise first trajectory

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _project_bounds(self, traj: np.ndarray) -> np.ndarray:
        """Clip trajectory to CMA-ES training bounds (replicates project_deltas)."""
        act_dim = traj.shape[1]
        half = act_dim // 2
        pcb = np.concatenate([
            np.full(half, self._per_comp_bound, dtype=np.float32),
            np.full(half, self._angle_bound, dtype=np.float32),
        ])
        n_repeat = half // len(self._angle_scale)
        angle_tiled = np.tile(self._angle_scale, n_repeat)
        traj[:, half:] = traj[:, half:] * angle_tiled
        traj = np.clip(traj, -pcb, pcb)
        if self._l2_bound is not None:
            norms = np.linalg.norm(traj[:, :half], axis=1, keepdims=True)
            scale = np.where(norms > self._l2_bound,
                             self._l2_bound / (norms + 1e-12), 1.0)
            traj[:, :half] = traj[:, :half] * scale
        return traj

    def _interpolate(self, traj: np.ndarray, target_steps: int) -> np.ndarray:
        """Linearly interpolate trajectory from n_steps to target_steps."""
        x_old = np.linspace(0, 1, len(traj))
        x_new = np.linspace(0, 1, target_steps)
        result = np.zeros((target_steps, traj.shape[1]), dtype=np.float32)
        for d in range(traj.shape[1]):
            result[:, d] = np.interp(x_new, x_old, traj[:, d])
        return result

    # ------------------------------------------------------------------
    # PolicyWrapper interface
    # ------------------------------------------------------------------

    @property
    def is_trajectory_based(self) -> bool:
        return True

    @property
    def n_steps(self) -> int:
        return len(self.traj) if self.traj is not None else 0

    def reset(self):
        """Sample a new trajectory from the CMA-ES distribution."""
        if self._es is not None:
            x = np.asarray(self._es.ask(1)[0], dtype=np.float32)
            traj = x.reshape(self._n_steps, self._act_dim)
            traj = self._project_bounds(traj.copy())
        else:
            traj = self._best_traj.copy()

        if self._target_steps is not None and len(traj) != self._target_steps:
            traj = self._interpolate(traj, self._target_steps)

        self.traj = traj
        self._step = 0

    def get_action(self, step_idx=None):
        idx = step_idx if step_idx is not None else self._step
        if idx >= len(self.traj):
            return np.zeros(self.traj.shape[1], dtype=np.float32)
        action = self.traj[idx].copy()
        self._step = idx + 1
        return action


def create_policy(
    algo: str,
    checkpoint: str,
    env=None,
    config_path: str | None = None,
    horizon: int | None = None,
) -> PolicyWrapper:
    """Factory function to create the appropriate policy wrapper."""
    algo = algo.lower()
    if algo == "ppo":
        return PPOPolicyWrapper(checkpoint)
    elif algo == "sac":
        return SACPolicyWrapper(checkpoint)
    elif algo == "shac":
        if env is None:
            raise ValueError("SHAC requires env for agent initialization")
        return SHACPolicyWrapper(checkpoint, env, config_path)
    elif algo == "cmaes":
        return CMAESPolicyWrapper(checkpoint, target_steps=horizon)
    else:
        raise ValueError(f"Unknown algo: {algo}. Valid: ppo, sac, shac, cmaes")
