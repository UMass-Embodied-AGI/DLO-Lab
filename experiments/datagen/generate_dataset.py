"""Generate LeRobot v3.0 datasets from trained RL/TO policy rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(".")

from datagen.config import (
    TASK_REGISTRY,
    FRONT_CAMERA_PARAMS,
    ARM_STATE_DIM,
    ARM_ACTION_DIM,
    ARM_JOINT_DIM,
    STATE_DIM_UNIFIED,
    ACTION_DIM_UNIFIED,
    JOINT_DIM_UNIFIED,
    get_env_class,
    get_n_additional_obj,
    get_n_controllers,
    get_camera_names,
    build_state_names,
    build_action_names,
    build_joint_action_names,
)
from datagen.cameras import (
    make_construct_extra_cameras,
    attach_wrist_cameras,
    collect_datagen_cameras,
    capture_images,
)
from datagen.export import (
    save_episode_parquet,
    save_episode_videos,
    finalize_dataset,
    TASK_DESCRIPTIONS,
)
from datagen.policy_wrappers import create_policy
from utils.domain_randomization import randomized_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training-config auto-derivation
# ---------------------------------------------------------------------------

DEFAULT_POS_BOUND = 0.01
DEFAULT_ANGLE_BOUND = 0.1
DEFAULT_N_SUBSTEPS_PER_STEP = 20
DEFAULT_STEPS_INTERVAL_SPLIT = 1


def resolve_run_config_path(checkpoint: str) -> Path:
    """Locate ``run_config.json`` for a checkpoint.

    The checkpoint is either a file (PPO/SAC/SHAC ``.pkl``) — config sits in the
    same directory — or a directory (CMA-ES) that contains the config directly.
    """
    ckpt = Path(checkpoint)
    base = ckpt if ckpt.is_dir() else ckpt.parent
    return base / "run_config.json"


def derive_training_config(checkpoint: str) -> dict:
    """Read action bounds / substeps / config path from the run's run_config.json.

    Handles the per-algo layouts: PPO/SAC store ``vars(args)`` flat; SHAC nests
    them under an ``"args"`` key; CMA-ES stores a flat dict that uses
    ``per_comp_bound`` instead of ``bound`` and omits the substep keys. Missing
    keys fall back to the module defaults with a warning.

    Returns a dict with: ``config_path`` (run_config.json path, consumed by the
    SHAC wrapper; None if absent), ``pos_bound``, ``angle_bound``,
    ``n_substeps_per_step``, ``steps_interval_split``.
    """
    cfg_path = resolve_run_config_path(checkpoint)
    if not cfg_path.exists():
        logger.warning(
            "run_config.json not found at %s — falling back to defaults "
            "(pos_bound=%s, angle_bound=%s, n_substeps_per_step=%s, steps_interval_split=%s)",
            cfg_path, DEFAULT_POS_BOUND, DEFAULT_ANGLE_BOUND,
            DEFAULT_N_SUBSTEPS_PER_STEP, DEFAULT_STEPS_INTERVAL_SPLIT,
        )
        return {
            "config_path": None,
            "pos_bound": DEFAULT_POS_BOUND,
            "angle_bound": DEFAULT_ANGLE_BOUND,
            "n_substeps_per_step": DEFAULT_N_SUBSTEPS_PER_STEP,
            "steps_interval_split": DEFAULT_STEPS_INTERVAL_SPLIT,
        }

    with open(cfg_path) as f:
        raw = json.load(f)
    # SHAC nests the CLI args under "args"; PPO/SAC/CMA-ES keep them flat.
    targs = raw.get("args", raw)

    # pos_bound: PPO/SAC/SHAC store it as "bound"; CMA-ES as "per_comp_bound".
    pos_bound = targs.get("bound", targs.get("per_comp_bound", targs.get("pos_bound")))
    if pos_bound is None:
        logger.warning("No action bound in %s; using default %s", cfg_path, DEFAULT_POS_BOUND)
        pos_bound = DEFAULT_POS_BOUND

    angle_bound = targs.get("angle_bound", DEFAULT_ANGLE_BOUND)
    # CMA-ES run_config omits n_substeps_per_step / steps_interval_split.
    n_substeps_per_step = targs.get("n_substeps_per_step", DEFAULT_N_SUBSTEPS_PER_STEP)
    steps_interval_split = targs.get("steps_interval_split", DEFAULT_STEPS_INTERVAL_SPLIT)

    return {
        "config_path": str(cfg_path),
        "pos_bound": float(pos_bound),
        "angle_bound": float(angle_bound),
        "n_substeps_per_step": int(n_substeps_per_step),
        "steps_interval_split": int(steps_interval_split),
    }


# ---------------------------------------------------------------------------
# Unified-dimension padding helpers
# ---------------------------------------------------------------------------


def _pad_state_to_unified(state: np.ndarray) -> np.ndarray:
    """Pad single-arm 8D state to unified 16D (zeros for missing arm)."""
    if len(state) >= STATE_DIM_UNIFIED:
        return state
    return np.concatenate([state, np.zeros(STATE_DIM_UNIFIED - len(state), dtype=np.float32)])


def _pad_action_to_unified(action: np.ndarray, n_controllers: int) -> np.ndarray:
    """Pad single-arm 6D action to unified 12D step_all format.

    Single-arm action [dx, dy, dz, droll, dpitch, dyaw] →
    unified [right_dx, right_dy, right_dz, left_dx(0), left_dy(0), left_dz(0),
             right_droll, right_dpitch, right_dyaw, left_droll(0), left_dpitch(0), left_dyaw(0)]
    """
    if n_controllers >= 2 or len(action) >= ACTION_DIM_UNIFIED:
        return action
    padded = np.zeros(ACTION_DIM_UNIFIED, dtype=np.float32)
    padded[0:3] = action[0:3]  # xyz  → right_xyz (slot 0-2)
    padded[6:9] = action[3:6]  # rot  → right_rot (slot 6-8)
    return padded


def _pad_joint_to_unified(joint: np.ndarray) -> np.ndarray:
    """Pad single-arm 9D joint to unified 18D (zeros for missing arm)."""
    if len(joint) >= JOINT_DIM_UNIFIED:
        return joint
    return np.concatenate([joint, np.zeros(JOINT_DIM_UNIFIED - len(joint), dtype=np.float32)])


# ---------------------------------------------------------------------------
# Data capture
# ---------------------------------------------------------------------------


def get_controllers(env, n_controllers: int):
    """Get list of RobotControllerPink instances from env."""
    controllers = []
    for i in range(n_controllers):
        c = getattr(env, f"c{i + 1}", None)
        if c is None:
            raise AttributeError(f"env has no controller 'c{i + 1}'")
        controllers.append(c)
    return controllers


def capture_frame(
    env,
    controllers: list,
    action: np.ndarray,
    step_idx: int,
    episode_idx: int,
    global_idx: int,
    dt: float,
    env_idx: int = 0,
) -> dict:
    """Capture one frame of state + action data from env_idx."""
    state_parts = []
    joint_parts = []

    for c in controllers:
        joints = c.robot.get_dofs_position(c.motors_dof)[env_idx].cpu().numpy()
        gripper = c.robot.get_dofs_position(c.fingers_dof)[env_idx].cpu().numpy()
        state_parts.append(joints)
        state_parts.append(np.array([gripper.sum()], dtype=np.float32))
        joint_parts.append(joints)
        joint_parts.append(gripper)

    return {
        "observation.state": _pad_state_to_unified(np.concatenate(state_parts).astype(np.float32)),
        "action": np.asarray(action, dtype=np.float32).flatten(),  # already unified by caller
        "action.joint": _pad_joint_to_unified(np.concatenate(joint_parts).astype(np.float32)),
        "timestamp": float(step_idx * dt),
        "frame_index": step_idx,
        "episode_index": episode_idx,
        "index": global_idx,
        "task_index": 0,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def _diagnose_failure(env, env_idx: int) -> str:
    """Return a human-readable string describing which constraint killed env_idx.

    Mirrors the checks inside step_diff_rl / step_all so the caller can log
    the specific reason without modifying the env step functions.
    """
    reasons = []

    # 1. NaN in rope vertices
    verts = env.rope.get_all_verts()           # (n_envs, n_verts, 3)
    if np.isnan(verts[env_idx]).any():
        reasons.append("NaN-in-rope-vertices")

    # 2. Rope–world collision (non-gripper geometry)
    try:
        collided     = env.rope._solver.vertices_collision.collided.to_numpy().T   # (n_envs, n_verts)
        geom_idx_arr = env.rope._solver.vertices_collision.geom_idx.to_numpy().T   # (n_envs, n_verts)
        verts_to_check = np.arange(env.rope.n_vertices) + env.rope._v_start
        col_env  = collided[env_idx, verts_to_check]           # (n_verts,)
        geom_env = geom_idx_arr[env_idx, verts_to_check]       # (n_verts,)
        registered = np.zeros_like(col_env, dtype=bool)
        for gidx in getattr(env, "gripper_geom_indices", []):
            registered |= (geom_env == gidx)
        if (col_env & ~registered).any():
            reasons.append("rope-world-collision")
    except Exception:
        pass

    # 3. Rope stretch between control points
    try:
        if getattr(env, "control_dist_init", None) is not None:
            dist_now = env.rope.get_geodesic_distance(
                env.control_idx[0], env.control_idx[1]
            )
            ratio = float(dist_now[env_idx]) / float(env.control_dist_init)
            if ratio > 1.05:
                reasons.append(f"rope-stretch(ratio={ratio:.3f}>1.05)")
    except Exception:
        pass

    return ", ".join(reasons) if reasons else "unknown"


def _env_step(algo: str, env, env_mask, action_tensor):
    """Call the algo-appropriate env step function.

    Returns:
        obs_batch: next observation tensor (n_envs, obs_dim), or None for cmaes
        absorbing: bool array/tensor (n_envs,) — True means episode ended
    """
    if algo in ("ppo", "sac", "cmaes"):
        obs_batch, _reward, absorbing, _info = env.step_all(env_mask, action_tensor)
        return obs_batch, absorbing

    elif algo == "shac":
        # step_diff_rl expects xyz-only action (n_envs, _act_dim=n_ctrl*3).
        # The policy wrapper returns step_all-padded 12D; strip the rotation part.
        xyz_action = action_tensor[:, :env._act_dim]
        _loss, alive, _rewards, _s_global = env.step_diff_rl(env_mask, xyz_action)
        absorbing = ~alive
        obs_batch = env.compute_observation()
        return obs_batch, absorbing

    else:
        raise ValueError(f"Unknown algo for _env_step: {algo}")


def generate_episodes(
    env,
    algo: str,
    policy,
    datagen_cameras: dict,
    camera_names: list[str],
    controllers: list,
    n_controllers: int,
    n_episodes: int,
    horizon: int,
    dt: float,
    fps: float,
    save_images: bool,
    output_dir: str,
    front_params: dict,
    no_filter_failed: bool = False,
    debug_failures: bool = False,
):
    """Run policy rollouts and save per-episode data.

    Supports n_envs > 1: each outer iteration runs n_envs episodes in parallel
    (physics). Images are captured serially — for each env i, cameras are moved
    to that env's world-space offset before rendering, so ropes are visible via
    the standard rasterizer (no BatchRenderer needed).

    Step functions dispatched by algo:
      ppo/sac/cmaes → env.step_all     (requires init_rl_env)
      shac          → env.step_diff_rl (requires init_diff_rl_env; xyz-only actions)

    CMA-ES uses is_trajectory_based=True; policy.reset() samples a new trajectory
    from the CMA-ES distribution on each batch, interpolated to `horizon` steps.
    """
    global_idx = 0
    act_dim = env._act_dim
    n_envs = env.n_envs
    saved_ep_idx = 0   # monotonically increasing index for saved (successful) episodes
    envs_offset = env.scene.envs_offset  # (n_envs, 3) world offsets from env_spacing

    batch = 0
    while saved_ep_idx < n_episodes:
        n_active = min(n_envs, n_episodes - saved_ep_idx)
        t0 = time.time()

        if hasattr(policy, "reset"):
            policy.reset()

        # -------------------------------------------------------------- #
        # Per-step loop: PPO / SAC / SHAC / CMA-ES                        #
        # -------------------------------------------------------------- #
        env.reset()

        # Per-env accumulators
        episode_data   = [[] for _ in range(n_active)]
        episode_images = [{cam: [] for cam in camera_names} for _ in range(n_active)]
        failed         = [False] * n_active

        obs_batch = env.compute_observation()  # (n_envs, obs_dim)

        # Initial frame (step 0, no action) — store in unified dims
        action_zero = np.zeros(ACTION_DIM_UNIFIED, dtype=np.float32)
        for ei in range(n_active):
            frame = capture_frame(env, controllers, action_zero, 0,
                                  saved_ep_idx + ei, global_idx + ei, dt, env_idx=ei)
            episode_data[ei].append(frame)
            if save_images:
                imgs = capture_images(datagen_cameras, camera_names, ei, envs_offset[ei], front_params)
                for cam in camera_names:
                    episode_images[ei][cam].append(imgs.get(cam))
        global_idx += n_active

        for step in tqdm(range(horizon), desc=f"Batch {batch}", leave=False):
            obs0 = obs_batch[0] if n_envs > 1 else obs_batch
            if policy.is_trajectory_based:
                raw_action = policy.get_action(step)
            else:
                raw_action = policy.get_action(obs0)

            action_tensor = torch.tensor(
                np.tile(raw_action, (n_envs, 1)), dtype=torch.float32
            )
            env_mask = torch.ones(n_envs, dtype=torch.bool)
            obs_batch, absorbing = _env_step(algo, env, env_mask, action_tensor)

            # Pad action to unified dims for dataset storage (env step uses raw_action)
            store_action = _pad_action_to_unified(raw_action, n_controllers)

            for ei in range(n_active):
                if failed[ei]:
                    continue
                if absorbing[ei]:
                    if debug_failures:
                        reason = _diagnose_failure(env, ei)
                        logger.info(
                            f"  [debug] Batch {batch} env {ei} failed at step {step+1}/{horizon}"
                            f" — {reason}"
                        )
                    if not no_filter_failed:
                        failed[ei] = True
                        continue
                frame = capture_frame(env, controllers, store_action, step + 1,
                                     saved_ep_idx + ei, global_idx + ei, dt, env_idx=ei)
                episode_data[ei].append(frame)
                if save_images:
                    imgs = capture_images(datagen_cameras, camera_names, ei, envs_offset[ei], front_params)
                    for cam in camera_names:
                        episode_images[ei][cam].append(imgs.get(cam))
            global_idx += n_active

        # Save episodes; skip failed ones unless --no_filter_failed is set
        n_saved = 0
        for ei in range(n_active):
            if failed[ei]:
                if no_filter_failed:
                    logger.info(f"Batch {batch} env {ei}: early stop — saving anyway (no_filter_failed)")
                else:
                    logger.info(f"Batch {batch} env {ei}: early stop — discarded")
                    continue
            ep_idx = saved_ep_idx + n_saved
            data = episode_data[ei]
            # Last frame action = copy from t-1
            if len(data) >= 2:
                data[-1]["action"] = data[-2]["action"].copy()
                data[-1]["action.joint"] = data[-2]["action.joint"].copy()
            # Fix episode_index in frames to match final ep_idx
            for f in data:
                f["episode_index"] = ep_idx
            save_episode_parquet(
                data, output_dir, ep_idx,
                save_images=save_images, camera_names=camera_names,
            )
            if save_images:
                save_episode_videos(episode_images[ei], output_dir, ep_idx, fps=fps)
            n_saved += 1

        saved_ep_idx += n_saved
        elapsed = time.time() - t0
        n_failed = n_active - n_saved
        logger.info(
            f"Batch {batch}: {n_saved}/{n_active} saved, {n_failed} failed, "
            f"total saved {saved_ep_idx}/{n_episodes}, time={elapsed:.1f}s"
        )
        batch += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate LeRobot v3.0 dataset")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--algo", type=str, required=True,
        choices=["ppo", "sac", "shac", "cmaes"],
        help="Algorithm type",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint path (a .pkl file for ppo/sac/shac, or the run dir for cmaes)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset dir")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments (physics runs in parallel; images captured serially)")
    parser.add_argument("--horizon", type=int, default=100, help="Steps per episode")
    parser.add_argument(
        "--img_resolution", type=int, nargs=2, default=[384, 240],
        help="Front camera resolution (width height)",
    )
    parser.add_argument(
        "--wrist_resolution", type=int, nargs=2, default=[384, 240],
        help="Wrist camera resolution (width height)",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--no_filter_failed", action="store_true",
                        help="Save all episodes including ones that ended early (absorbing state)")
    parser.add_argument("--debug_failures", action="store_true",
                        help="Log which constraint (NaN/collision/stretch) caused each env to fail")
    parser.add_argument("--domain_randomize", action="store_true",
                        help="Enable full domain randomization (rope positions will be randomized regardless of this argument)")
    parser.add_argument("--raytracer", "-r", action="store_true")
    parser.add_argument("--exr_path", type=str, default="dlo-lab/exrs/brown_photostudio_02_4k.exr",
                        help="HDRI environment map for the raytracer env surface "
                             "(no effect unless --raytracer is set)")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)

    task = args.task

    # Auto-derive the training-matched args (action bounds, substeps, SHAC config)
    # from the checkpoint's run_config.json so the rollout reproduces training.
    train_cfg = derive_training_config(args.checkpoint)
    logger.info(
        "Derived from run_config.json: pos_bound=%s, angle_bound=%s, "
        "n_substeps_per_step=%s, steps_interval_split=%s, config_path=%s",
        train_cfg["pos_bound"], train_cfg["angle_bound"],
        train_cfg["n_substeps_per_step"], train_cfg["steps_interval_split"],
        train_cfg["config_path"],
    )

    n_controllers = get_n_controllers(task)
    camera_names = get_camera_names(n_controllers)
    img_resolution = tuple(args.img_resolution)
    wrist_resolution = tuple(args.wrist_resolution)

    # Build per-camera resolution mapping
    camera_resolutions = {}
    for cam in camera_names:
        if "wrist" in cam:
            camera_resolutions[cam] = wrist_resolution
        else:
            camera_resolutions[cam] = img_resolution

    # Monkey-patch construct_extra_cameras to add front + wrist cameras before scene.build()
    EnvCls = get_env_class(task)
    EnvCls.construct_extra_cameras = make_construct_extra_cameras(
        task, n_controllers, img_resolution, wrist_resolution
    )

    # Create environment with camera=False to skip default high-res cameras.
    # n_envs > 1 runs parallel physics; images are captured serially per env using
    # the standard rasterizer (which renders ropes correctly via mesh_from_centerline).
    env_config = DictConfig({
        "task": task,
        "log_dir": args.output_dir,  # env only mkdirs it; keep everything under the dataset dir
        "n_envs": args.n_envs,
        "n_substeps_per_step": train_cfg["n_substeps_per_step"],
        "GUI": args.gui,
        "camera": False,
        "raytracer": args.raytracer,
        "exr_path": args.exr_path,  # raytracer env map; ignored when raytracer is off
    })
    env = EnvCls(config=env_config)

    # Initialize env with algo-appropriate init function.
    # - PPO/SAC: init_rl_env  (sets _mdp_info, required by step_all)
    # - SHAC:    init_diff_rl_env (sets _act_low/_act_high, _act_dim=3*n_ctrl xyz-only)
    # - CMA-ES:  init_cmaes_env  (sets cmaes_initialized, required by eval_traj)
    n_additional_obj = get_n_additional_obj(task, env)
    algo = args.algo
    if algo in ("ppo", "sac"):
        env.init_rl_env(
            n_steps=args.horizon,
            pos_bound=train_cfg["pos_bound"],
            angle_bound=train_cfg["angle_bound"],
            n_additional_obj=n_additional_obj,
            steps_interval_split=train_cfg["steps_interval_split"],
            debug=args.gui,
        )
    elif algo == "shac":
        env.init_diff_rl_env(
            n_steps=args.horizon,
            pos_bound=train_cfg["pos_bound"],
            angle_bound=train_cfg["angle_bound"],
            n_additional_obj=n_additional_obj,
            steps_interval_split=train_cfg["steps_interval_split"],
            debug=args.gui,
        )
    elif algo == "cmaes":
        # CMA-ES datagen replays sampled trajectories step-by-step via step_all,
        # same as PPO/SAC.  The CMA-ES training bounds (per_comp_bound/angle_bound,
        # derived from run_config.json) are passed so step_all doesn't clip them.
        env.init_rl_env(
            n_steps=args.horizon,
            pos_bound=train_cfg["pos_bound"],
            angle_bound=train_cfg["angle_bound"],
            n_additional_obj=n_additional_obj,
            steps_interval_split=train_cfg["steps_interval_split"],
            debug=args.gui,
        )

    # Domain randomization — always ensure rope position variation for datagen
    randomized_args = randomized_config.get(task, {})
    default_pos_bound = (-0.05, -0.05, 0.05, 0.05)  # ±5cm in xy
    if "pos_bound" not in randomized_args:
        randomized_args["pos_bound"] = default_pos_bound
    if args.domain_randomize:
        env.init_domain_randomization(**randomized_args)
    else:
        # Even without full DR, always randomize initial rope position
        env.init_domain_randomization(pos_bound=randomized_args["pos_bound"])

    # Attach wrist cameras to robot links (must be after scene.build())
    attach_wrist_cameras(env, n_controllers)

    # Collect datagen cameras (front from env + wrist from construct_extra_cameras hook)
    datagen_cameras = collect_datagen_cameras(env)
    logger.info(f"Cameras: {list(datagen_cameras.keys())}")

    # Controllers
    controllers = get_controllers(env, n_controllers)

    # Always use unified (bimanual) dimensions so all tasks share the same schema.
    # Single-arm data is zero-padded to these sizes during capture.
    state_dim        = STATE_DIM_UNIFIED   # 16
    action_dim       = ACTION_DIM_UNIFIED  # 12
    joint_action_dim = JOINT_DIM_UNIFIED   # 18

    state_names = build_state_names(n_controllers)
    action_names = build_action_names(n_controllers)
    joint_action_names = build_joint_action_names(n_controllers)

    # dt per policy step = physics substep dt * n_substeps_per_step
    n_substeps_per_step = train_cfg["n_substeps_per_step"]
    dt = env.scene.dt * n_substeps_per_step
    fps = 1.0 / dt

    logger.info(f"Task: {task}, Algo: {args.algo}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}, Joint action dim: {joint_action_dim}")
    logger.info(f"Scene dt: {env.scene.dt:.4f}s, Substeps per step: {n_substeps_per_step}, Policy dt: {dt:.4f}s, FPS: {fps:.1f}")
    logger.info(f"Controllers: {n_controllers}, Horizon: {args.horizon}")

    # Load policy. config_path (the run's run_config.json) is consumed by the
    # SHAC wrapper to rebuild the actor; the other wrappers ignore it.
    policy = create_policy(
        args.algo, args.checkpoint,
        env=env, config_path=train_cfg["config_path"],
        horizon=args.horizon,
    )

    # Generate
    os.makedirs(args.output_dir, exist_ok=True)
    generate_episodes(
        env=env,
        algo=algo,
        policy=policy,
        datagen_cameras=datagen_cameras,
        camera_names=camera_names,
        controllers=controllers,
        n_controllers=n_controllers,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        dt=dt,
        fps=fps,
        save_images=args.save_images,
        output_dir=args.output_dir,
        front_params=FRONT_CAMERA_PARAMS[task],
        no_filter_failed=args.no_filter_failed,
        debug_failures=args.debug_failures,
    )

    # Finalize dataset
    logger.info("Finalizing dataset...")
    task_description = TASK_DESCRIPTIONS.get(task, f"{task}_manipulation")
    finalize_dataset(
        output_dir=args.output_dir,
        save_images=args.save_images,
        fps=fps,
        camera_resolutions=camera_resolutions,
        state_dim=state_dim,
        action_dim=action_dim,
        joint_action_dim=joint_action_dim,
        state_names=state_names,
        action_names=action_names,
        joint_action_names=joint_action_names,
        camera_names=camera_names,
        task_description=task_description,
    )

    logger.info(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
