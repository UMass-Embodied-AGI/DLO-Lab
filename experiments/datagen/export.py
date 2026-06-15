"""LeRobot v3.0 dataset export utilities.

Two-phase pipeline:
  Phase 1 (per-episode): save_episode_parquet() + save_episode_videos()
  Phase 2 (finalize):    finalize_dataset() consolidates into v3.0 dataset.

Produces:
  - data/chunk-000/file-000.parquet (all frames, video path struct columns)
  - videos/chunk-000/observation.images.{cam}/episode_NNNNNN.mp4
  - meta/info.json (v3.0), meta/stats.json, meta/tasks.parquet
  - meta/episodes/chunk-000/file-000.parquet (per-episode metadata + stats)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical task descriptions used in meta/tasks.parquet and SmolVLA language conditioning.
TASK_DESCRIPTIONS: dict[str, str] = {
    "coiling":     "The task scene consists of a fixed cone and a rope. The agent needs to wind the rope around the cone's surface.",
    "gathering":   "This task starts with several rigid and deformable bodies randomly placed on the ground. The objective is to use a rope to gather these objects together.",
    "lifting":     "Given two \"C\"-shaped rings, the agent is required to manipulate a rope to lift the rings.",
    "separation":  "In this task, the agent needs to use two robot arms to separate two ropes that are initially tangled together.",
    "slingshot":   "In this scenario, a rigid ball and a rigid cube are placed on a table. In front of the ball, there is a slingshot made from a rope with high stretching stiffness. The agent should operate a robot arm and use the slingshot to launch the ball and hit the cube.",
    "unknotting":  "This task requires the agent to untangle a rope that has an overhand knot.",
    "wiring_post": "The task involves manipulating a rope from a straight position into a specific \"S\"-shaped path that winds through two posts fixed to the table.",
    "wrapping":    "The agent is asked to wrap a rubber band around three cylinders fixed to the table.",
}

logger = logging.getLogger(__name__)

# Arrow struct type matching HF Image() feature layout
_IMAGE_STRUCT_TYPE = pa.struct([("path", pa.string()), ("bytes", pa.binary())])

# Numeric features for which stats are computed
_NUMERIC_FEATURES = ["observation.state", "action", "action.joint"]

# Quantile values matching LeRobot convention
_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


# ---------------------------------------------------------------------------
# Video path helpers
# ---------------------------------------------------------------------------


def video_rel_path(
    cam_name: str, episode_idx: int, chunk_idx: int = 0
) -> str:
    """Build the relative video path (from dataset root) for one episode."""
    return (
        f"videos/chunk-{chunk_idx:03d}/observation.images.{cam_name}"
        f"/episode_{episode_idx:06d}.mp4"
    )


# ---------------------------------------------------------------------------
# Per-episode saving
# ---------------------------------------------------------------------------


def save_episode_parquet(
    episode_data: list[dict],
    output_dir: Path | str,
    episode_idx: int,
    save_images: bool = False,
    chunk_idx: int = 0,
    camera_names: list[str] | None = None,
) -> None:
    """Save a single episode as a parquet file.

    Args:
        episode_data: List of per-frame dicts with keys:
            observation.state, action, action.joint, timestamp,
            frame_index, episode_index, index, task_index
        output_dir: Root output directory.
        episode_idx: Episode number.
        save_images: Whether to add image path struct columns.
        chunk_idx: Chunk number (default 0).
        camera_names: List of camera names.
    """
    output_dir = Path(output_dir)
    chunk_name = f"chunk-{chunk_idx:03d}"
    data_dir = output_dir / "data" / chunk_name
    data_dir.mkdir(parents=True, exist_ok=True)

    columns = {
        "observation.state": pa.array(
            [d["observation.state"].tolist() for d in episode_data],
            type=pa.list_(pa.float32()),
        ),
        "action": pa.array(
            [d["action"].tolist() for d in episode_data],
            type=pa.list_(pa.float32()),
        ),
        "action.joint": pa.array(
            [d["action.joint"].tolist() for d in episode_data],
            type=pa.list_(pa.float32()),
        ),
        "timestamp": pa.array(
            [d["timestamp"] for d in episode_data], type=pa.float32()
        ),
        "frame_index": pa.array(
            [d["frame_index"] for d in episode_data], type=pa.int64()
        ),
        "episode_index": pa.array(
            [d["episode_index"] for d in episode_data], type=pa.int64()
        ),
        "index": pa.array(
            [d["index"] for d in episode_data], type=pa.int64()
        ),
        "task_index": pa.array(
            [d["task_index"] for d in episode_data], type=pa.int64()
        ),
    }

    # Note: image/video columns are intentionally NOT written to the parquet.
    # Lerobot v3.0 reconstructs video paths from the template in info.json
    # (video_path = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4")
    # using episode_index from each row.  Including image path struct columns
    # in the parquet causes a schema mismatch when lerobot loads the dataset.

    table = pa.table(columns)
    filename = f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, data_dir / filename)


def save_episode_videos(
    frames: dict[str, list[np.ndarray]],
    output_dir: Path | str,
    episode_idx: int,
    fps: float = 5.0,
    chunk_idx: int = 0,
) -> None:
    """Save per-episode camera observations as mp4 videos.

    Args:
        frames: {camera_name: [frame_0, frame_1, ...]} where each frame
                is (H, W, 3) uint8.
        output_dir: Root output directory.
        episode_idx: Episode number.
        fps: Video frame rate.
        chunk_idx: Chunk number.
    """
    import imageio

    output_dir = Path(output_dir)
    chunk_name = f"chunk-{chunk_idx:03d}"

    for cam_name, frame_list in frames.items():
        vid_dir = (
            output_dir / "videos" / chunk_name
            / f"observation.images.{cam_name}"
        )
        vid_dir.mkdir(parents=True, exist_ok=True)

        vid_path = vid_dir / f"episode_{episode_idx:06d}.mp4"
        imageio.mimwrite(
            str(vid_path), frame_list, fps=fps, codec="libx264",
            pixelformat="yuv420p",
        )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def compute_numeric_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute stats for a 2D numeric array (N, D) -> per-feature stats."""
    stats = {
        "min": arr.min(axis=0).astype(np.float32),
        "max": arr.max(axis=0).astype(np.float32),
        "mean": arr.mean(axis=0).astype(np.float32),
        "std": arr.std(axis=0).astype(np.float32),
        "count": np.array([arr.shape[0]]),
    }
    for q in _QUANTILES:
        stats[f"q{int(q * 100):02d}"] = np.quantile(arr, q, axis=0).astype(
            np.float32
        )
    return stats


def default_image_stats(n_frames: int) -> dict[str, np.ndarray]:
    """Return placeholder per-channel image stats (3, 1, 1) for [0, 1] range."""
    return {
        "min": np.zeros((3, 1, 1), dtype=np.float32),
        "max": np.ones((3, 1, 1), dtype=np.float32),
        "mean": np.full((3, 1, 1), 0.5, dtype=np.float32),
        "std": np.full((3, 1, 1), 0.25, dtype=np.float32),
        "count": np.array([n_frames]),
        **{
            f"q{int(q * 100):02d}": np.full((3, 1, 1), q, dtype=np.float32)
            for q in _QUANTILES
        },
    }


def _pad_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Zero-pad arr to target_shape (used for mixed-dim multi-task stats)."""
    if arr.shape == target_shape:
        return arr
    pad_width = [(0, target_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def aggregate_feature_stats(
    stats_list: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Aggregate per-episode stats for one feature.

    Handles mixed-dimension features (e.g. state_dim=8 for single-arm tasks and
    state_dim=16 for dual-arm tasks in a combined dataset) by zero-padding all
    stat arrays to the maximum shape before aggregation.
    """
    # Resolve max shape across all episodes (handles mixed-dim multi-task datasets)
    feat_shape = stats_list[0]["mean"].shape
    if any(s["mean"].shape != feat_shape for s in stats_list):
        max_shape = tuple(
            max(s["mean"].shape[i] for s in stats_list)
            for i in range(len(feat_shape))
        )
        stats_list = [
            {
                k: _pad_to_shape(v, max_shape) if isinstance(v, np.ndarray) and v.shape == s["mean"].shape else v
                for k, v in s.items()
            }
            for s in stats_list
        ]

    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    aggregated: dict[str, np.ndarray] = {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean.astype(np.float32),
        "std": np.sqrt(np.maximum(total_variance, 0)).astype(np.float32),
        "count": total_count,
    }

    quantile_keys = [
        k for k in stats_list[0] if k.startswith("q") and k[1:].isdigit()
    ]
    for q_key in quantile_keys:
        if all(q_key in s for s in stats_list):
            q_vals = np.stack([s[q_key] for s in stats_list])
            aggregated[q_key] = (q_vals * counts).sum(axis=0) / total_count

    return aggregated


def aggregate_stats(
    all_ep_stats: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate per-episode stats into global stats across all features."""
    all_keys: set[str] = set()
    for stats in all_ep_stats:
        all_keys.update(stats.keys())

    global_stats = {}
    for key in sorted(all_keys):
        feature_stats = [s[key] for s in all_ep_stats if key in s]
        if feature_stats:
            global_stats[key] = aggregate_feature_stats(feature_stats)

    return global_stats


def compute_active_only_stats(
    merged_df: pd.DataFrame,
    feature_keys: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-dimension stats using only episodes where that dimension is active.

    A dimension is considered "active" in an episode if it has non-zero variance.
    This avoids zero-padded dimensions (e.g. left arm in single-arm tasks) from
    distorting the global mean/std used for normalization.

    Falls back to standard stats for dimensions that are always active (non-zero
    variance in every episode).
    """
    stats = {}
    for feat in feature_keys:
        if feat not in merged_df.columns:
            continue
        all_data = np.array(merged_df[feat].tolist(), dtype=np.float32)  # (N, D)
        n_total, n_dims = all_data.shape

        ep_indices = merged_df["episode_index"].values
        unique_eps = np.unique(ep_indices)

        per_dim_mean = np.zeros(n_dims, dtype=np.float32)
        per_dim_std = np.zeros(n_dims, dtype=np.float32)
        per_dim_min = np.full(n_dims, np.inf, dtype=np.float32)
        per_dim_max = np.full(n_dims, -np.inf, dtype=np.float32)
        per_dim_count = np.zeros(n_dims, dtype=np.int64)

        for ep in unique_eps:
            mask = ep_indices == ep
            ep_data = all_data[mask]  # (T, D)
            ep_std = ep_data.std(axis=0)
            active = ep_std > 1e-8  # dims with non-zero variance in this episode

            for d in range(n_dims):
                if active[d]:
                    col = ep_data[:, d]
                    n = len(col)
                    per_dim_count[d] += n
                    per_dim_min[d] = min(per_dim_min[d], col.min())
                    per_dim_max[d] = max(per_dim_max[d], col.max())
                    # Welford-style online accumulation (batch per episode)
                    per_dim_mean[d] += col.sum()

        # Finalize mean
        for d in range(n_dims):
            if per_dim_count[d] > 0:
                per_dim_mean[d] /= per_dim_count[d]
            else:
                per_dim_mean[d] = 0.0
                per_dim_min[d] = 0.0
                per_dim_max[d] = 0.0

        # Second pass for std (needs final mean)
        for ep in unique_eps:
            mask = ep_indices == ep
            ep_data = all_data[mask]
            ep_std_check = ep_data.std(axis=0)
            active = ep_std_check > 1e-8

            for d in range(n_dims):
                if active[d]:
                    col = ep_data[:, d]
                    per_dim_std[d] += ((col - per_dim_mean[d]) ** 2).sum()

        for d in range(n_dims):
            if per_dim_count[d] > 1:
                per_dim_std[d] = np.sqrt(per_dim_std[d] / per_dim_count[d])
            else:
                per_dim_std[d] = 1.0  # avoid division by zero

        stats[feat] = {
            "min": per_dim_min,
            "max": per_dim_max,
            "mean": per_dim_mean,
            "std": per_dim_std,
            "count": np.array([int(per_dim_count.max())]),
        }

    return stats


def serialize_stats(stats: dict) -> dict:
    """Convert numpy arrays in stats dict to JSON-serializable lists."""
    out = {}
    for feat, feat_stats in stats.items():
        out[feat] = {}
        for k, v in feat_stats.items():
            out[feat][k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


# ---------------------------------------------------------------------------
# v3.0 features definition
# ---------------------------------------------------------------------------


def build_v30_features(
    state_dim: int,
    action_dim: int,
    joint_action_dim: int,
    state_names: list[str] | None,
    action_names: list[str] | None,
    joint_action_names: list[str] | None,
    camera_names: list[str],
    save_images: bool,
    fps: int,
    camera_resolutions: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Build the features dict for info.json (v3.0 format).

    camera_resolutions: mapping from camera name to (width, height).
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
            "fps": fps,
        },
        "action.joint": {
            "dtype": "float32",
            "shape": [joint_action_dim],
            "names": joint_action_names,
            "fps": fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
    }

    if save_images and camera_resolutions:
        for cam in camera_names:
            w, h = camera_resolutions[cam]
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "fps": fps,
            }

    return features


# ---------------------------------------------------------------------------
# Phase 2: Finalize dataset to v3.0
# ---------------------------------------------------------------------------


def finalize_dataset(
    output_dir: Path | str,
    save_images: bool,
    fps: float,
    camera_resolutions: dict[str, tuple[int, int]],
    state_dim: int,
    action_dim: int,
    joint_action_dim: int,
    state_names: list[str] | None,
    action_names: list[str] | None,
    joint_action_names: list[str] | None,
    camera_names: list[str],
    task_description: str = "dlo_manipulation",
) -> None:
    """Consolidate per-episode parquets into a v3.0 LeRobot dataset.

    Reads all episode_*.parquet files, computes stats, and writes:
      - data/chunk-000/file-000.parquet (concatenated frames)
      - meta/episodes/chunk-000/file-000.parquet (per-episode metadata + stats)
      - meta/tasks.parquet
      - meta/stats.json
      - meta/info.json (v3.0)

    Then removes the per-episode parquets.
    """
    output_dir = Path(output_dir).resolve()
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    fps_int = int(fps)

    # 1. Discover per-episode parquets
    data_dir = output_dir / "data"
    ep_paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not ep_paths:
        logger.warning("No per-episode parquets found, nothing to finalize.")
        return

    logger.info(f"Finalizing {len(ep_paths)} episodes into v3.0 format")

    image_keys = (
        [f"observation.images.{cam}" for cam in camera_names]
        if save_images
        else []
    )

    # 2. Read per-episode parquets, compute stats, build metadata
    all_ep_stats: list[dict[str, dict[str, np.ndarray]]] = []
    ep_metadata_list: list[dict] = []
    dfs: list[pd.DataFrame] = []
    global_idx = 0

    for i, ep_path in enumerate(ep_paths):
        df = pd.read_parquet(ep_path)
        n_frames = len(df)

        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for feat in _NUMERIC_FEATURES:
            if feat in df.columns:
                arr = np.array(df[feat].tolist(), dtype=np.float32)
                ep_stats[feat] = compute_numeric_stats(arr)

        if save_images:
            for key in image_keys:
                ep_stats[key] = default_image_stats(n_frames)

        all_ep_stats.append(ep_stats)

        # Episode metadata (v3.0)
        ep_md: dict = {
            "episode_index": i,
            "tasks": [task_description],
            "length": n_frames,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_idx,
            "dataset_to_index": global_idx + n_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        # Video index + timestamp columns — required by lerobot to locate video
        # files and query frames by timestamp.
        ts = df["timestamp"].astype(float)
        for cam in camera_names:
            vid_key = f"observation.images.{cam}"
            ep_md[f"videos/{vid_key}/chunk_index"]    = 0
            ep_md[f"videos/{vid_key}/file_index"]     = i
            ep_md[f"videos/{vid_key}/from_timestamp"] = float(ts.min())
            ep_md[f"videos/{vid_key}/to_timestamp"]   = float(ts.max())
        for feat, stats in ep_stats.items():
            for stat_name, stat_val in stats.items():
                ep_md[f"stats/{feat}/{stat_name}"] = stat_val.tolist()

        ep_metadata_list.append(ep_md)

        df["episode_index"] = i
        df["index"] = np.arange(global_idx, global_idx + n_frames, dtype=np.int64)

        if save_images:
            for key in image_keys:
                if key in df.columns:
                    df[key] = df[key].apply(
                        lambda x: {
                            "path": str(output_dir / x["path"]),
                            "bytes": x.get("bytes"),
                        }
                    )

        dfs.append(df)
        global_idx += n_frames

    total_episodes = len(ep_metadata_list)
    total_frames = global_idx

    # 3. Concatenate into single data parquet
    logger.info(f"Concatenating {total_episodes} episodes ({total_frames} frames)")
    concat_df = pd.concat(dfs, ignore_index=True)

    final_path = data_dir / "chunk-000" / "file-000.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    present_image_keys = [k for k in image_keys if k in concat_df.columns]
    if save_images and present_image_keys:
        from datasets import Features, Image

        schema = pa.Schema.from_pandas(concat_df)
        hf_features = Features.from_arrow_schema(schema)
        for key in present_image_keys:
            hf_features[key] = Image()
        schema = hf_features.arrow_schema
        concat_df.to_parquet(final_path, index=False, schema=schema)
    else:
        concat_df.to_parquet(final_path, index=False)

    logger.info(f"Wrote {final_path}")

    # 4. Compute global stats
    global_stats = aggregate_stats(all_ep_stats)

    # 5. Write meta/stats.json
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(serialize_stats(global_stats), f, indent=2)
    logger.info("Wrote meta/stats.json")

    # 6. Write meta/tasks.parquet
    tasks_df = pd.DataFrame(
        {"task_index": [0]},
        index=[task_description],
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")
    logger.info("Wrote meta/tasks.parquet")

    # 7. Write meta/episodes/chunk-000/file-000.parquet
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes_df = pd.DataFrame(ep_metadata_list)
    episodes_df.to_parquet(episodes_dir / "file-000.parquet", index=False)
    logger.info(
        f"Wrote meta/episodes/chunk-000/file-000.parquet ({total_episodes} episodes)"
    )

    # 8. Write meta/info.json (v3.0)
    features = build_v30_features(
        state_dim, action_dim, joint_action_dim,
        state_names, action_names, joint_action_names,
        camera_names, save_images, fps_int, camera_resolutions,
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps_int,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(
        f"Wrote meta/info.json: v3.0, {total_episodes} episodes, {total_frames} frames"
    )

    # 9. Clean up per-episode parquets
    for ep_path in ep_paths:
        ep_path.unlink()
    logger.info(f"Removed {len(ep_paths)} per-episode parquet files")
