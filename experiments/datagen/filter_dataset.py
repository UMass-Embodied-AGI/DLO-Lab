"""Filter episodes from a finalized LeRobot v3.0 dataset by episode ID.

Removes specified episodes, re-indexes episode_index and frame index
contiguously, renames video files to match new indices, and rewrites all
metadata (info.json, stats.json, tasks.parquet, episodes parquet).

Usage:
    cd experiments
    python datagen/filter_dataset.py \
        --dataset_dir visual_dataset/wiring_post/ppo \
        --remove_episodes 3 7 12

    # Or randomly sample a target number of episodes:
    python datagen/filter_dataset.py \
        --dataset_dir visual_dataset/wiring_post/ppo \
        --target_num 10
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def load_dataset(dataset_dir: Path):
    """Load the consolidated parquet and info.json."""
    data_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    info_path = dataset_dir / "meta" / "info.json"
    if not data_path.exists():
        raise FileNotFoundError(f"No dataset found at {data_path}")
    df = pd.read_parquet(data_path)
    with open(info_path) as f:
        info = json.load(f)
    return df, info


def filter_and_reindex(df: pd.DataFrame, remove_ids: set[int]):
    """Remove episodes and re-index episode_index and frame index."""
    keep = ~df["episode_index"].isin(remove_ids)
    df = df[keep].copy()

    # Build old→new episode index mapping (contiguous from 0)
    old_ep_ids = sorted(df["episode_index"].unique())
    ep_remap = {old: new for new, old in enumerate(old_ep_ids)}
    df["episode_index"] = df["episode_index"].map(ep_remap)

    # Re-assign global frame index contiguously
    df["index"] = np.arange(len(df), dtype=np.int64)

    # Re-assign frame_index within each episode
    df["frame_index"] = df.groupby("episode_index").cumcount()

    # Re-assign timestamps within each episode
    if "timestamp" in df.columns:
        first_ts = df.groupby("episode_index")["timestamp"].transform("first")
        df["timestamp"] = df["timestamp"] - first_ts

    return df, ep_remap


def rename_videos(dataset_dir: Path, ep_remap: dict[int, int], camera_names: list[str]):
    """Rename episode video files to match new episode indices."""
    vid_base = dataset_dir / "videos" / "chunk-000"
    if not vid_base.exists():
        return

    for cam in camera_names:
        cam_dir = vid_base / f"observation.images.{cam}"
        if not cam_dir.exists():
            continue

        # Two-pass rename: first to temp names to avoid conflicts
        for old_idx, new_idx in ep_remap.items():
            src = cam_dir / f"episode_{old_idx:06d}.mp4"
            if src.exists():
                src.rename(cam_dir / f"episode_{old_idx:06d}.mp4.tmp")

        for old_idx, new_idx in ep_remap.items():
            tmp = cam_dir / f"episode_{old_idx:06d}.mp4.tmp"
            if tmp.exists():
                tmp.rename(cam_dir / f"episode_{new_idx:06d}.mp4")

        # Remove videos for deleted episodes (no .tmp → they were removed)
        for f in cam_dir.glob("episode_*.mp4.tmp"):
            f.unlink()
        # Remove any stale videos with indices beyond the new total
        for f in cam_dir.glob("episode_*.mp4"):
            idx = int(f.stem.split("_")[1])
            if idx >= len(ep_remap):
                f.unlink()
                logger.info(f"Removed stale video {f.name}")


def recompute_stats(df: pd.DataFrame, numeric_keys: list[str], image_keys: list[str]):
    """Recompute global stats from the filtered dataframe."""
    from datagen.export import compute_numeric_stats, default_image_stats, aggregate_stats

    all_ep_stats = []
    for ep_idx, ep_df in df.groupby("episode_index"):
        ep_stats = {}
        for feat in numeric_keys:
            if feat in ep_df.columns:
                arr = np.array(ep_df[feat].tolist(), dtype=np.float32)
                ep_stats[feat] = compute_numeric_stats(arr)
        for key in image_keys:
            ep_stats[key] = default_image_stats(len(ep_df))
        all_ep_stats.append(ep_stats)

    return aggregate_stats(all_ep_stats) if all_ep_stats else {}


def build_episodes_metadata(df: pd.DataFrame, task_description: str, numeric_keys, image_keys):
    """Rebuild per-episode metadata rows."""
    from datagen.export import compute_numeric_stats, default_image_stats

    rows = []
    global_start = 0
    for ep_idx, ep_df in df.groupby("episode_index"):
        n = len(ep_df)
        ep_stats = {}
        for feat in numeric_keys:
            if feat in ep_df.columns:
                arr = np.array(ep_df[feat].tolist(), dtype=np.float32)
                ep_stats[feat] = compute_numeric_stats(arr)
        for key in image_keys:
            ep_stats[key] = default_image_stats(n)

        row = {
            "episode_index": int(ep_idx),
            "tasks": [task_description],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_start,
            "dataset_to_index": global_start + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feat, stats in ep_stats.items():
            for stat_name, stat_val in stats.items():
                v = stat_val.tolist() if isinstance(stat_val, np.ndarray) else stat_val
                row[f"stats/{feat}/{stat_name}"] = v
        rows.append(row)
        global_start += n
    return rows


def update_video_paths(df: pd.DataFrame, dataset_dir: Path, camera_names: list[str]):
    """Update the video path struct column to reflect new episode indices."""
    for cam in camera_names:
        key = f"observation.images.{cam}"
        if key not in df.columns:
            continue
        def _new_path(row):
            ep = int(row["episode_index"])
            return {"path": f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4", "bytes": None}
        df[key] = df.apply(_new_path, axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Filter episodes from a LeRobot v3.0 dataset")
    parser.add_argument("--dataset_dir", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--remove_episodes", type=int, nargs="+",
        help="Episode indices to remove (0-based)",
    )
    group.add_argument(
        "--target_num", type=int,
        help="Randomly sample this many episodes and remove the rest",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --target_num sampling")
    parser.add_argument("--force_recompute_stat", action="store_true",
                        help="Recompute stats even when no episodes are removed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_dir = Path(args.dataset_dir).resolve()

    df, info = load_dataset(dataset_dir)
    all_ep_ids = set(df["episode_index"].unique())
    total_before = int(len(all_ep_ids))

    if args.target_num is not None:
        if args.target_num >= total_before and not args.force_recompute_stat:
            logger.info(f"target_num ({args.target_num}) >= total episodes ({total_before}), nothing to do.")
            return
        if args.target_num >= total_before and args.force_recompute_stat:
            remove_ids = set()
        else:
            rng = np.random.default_rng(args.seed)
            keep_ids = set(rng.choice(sorted(all_ep_ids), size=args.target_num, replace=False).tolist())
            remove_ids = all_ep_ids - keep_ids
    else:
        remove_ids = set(args.remove_episodes)

    logger.info(f"Dataset: {total_before} episodes, {len(df)} frames")
    logger.info(f"Removing episodes: {sorted(remove_ids)}")

    # Validate
    invalid = remove_ids - all_ep_ids
    if invalid:
        raise ValueError(f"Episode IDs not found in dataset: {sorted(invalid)}")

    # Detect camera names from info
    camera_names = [
        k.replace("observation.images.", "")
        for k in info.get("features", {})
        if k.startswith("observation.images.")
    ]
    has_videos = bool(camera_names) and (dataset_dir / "videos").exists()

    # Filter and re-index
    df, ep_remap = filter_and_reindex(df, remove_ids)
    total_after = int(df["episode_index"].nunique())
    logger.info(f"After filtering: {total_after} episodes, {len(df)} frames")

    # Update video path columns before saving
    if has_videos:
        df = update_video_paths(df, dataset_dir, camera_names)

    # Rewrite main parquet
    data_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    df.to_parquet(data_path, index=False)
    logger.info(f"Rewrote {data_path}")

    # Rename video files
    if has_videos:
        rename_videos(dataset_dir, ep_remap, camera_names)
        logger.info("Renamed video files")

    # Recompute stats
    numeric_keys = ["observation.state", "action", "action.joint"]
    image_keys = [f"observation.images.{cam}" for cam in camera_names] if has_videos else []
    from datagen.export import serialize_stats
    global_stats = recompute_stats(df, numeric_keys, image_keys)
    with open(dataset_dir / "meta" / "stats.json", "w") as f:
        json.dump(serialize_stats(global_stats), f, indent=2)
    logger.info("Rewrote meta/stats.json")

    # Rewrite episodes metadata
    names = info.get("features", {}).get("task_index", {}).get("names") or ["unknown"]
    task_description = names[0]
    # Fall back: read from tasks.parquet
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        task_description = tasks_df.index[0] if len(tasks_df) > 0 else task_description

    ep_rows = build_episodes_metadata(df, task_description, numeric_keys, image_keys)
    ep_df = pd.DataFrame(ep_rows)
    ep_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_df.to_parquet(ep_path, index=False)
    logger.info(f"Rewrote {ep_path}")

    # Rewrite info.json
    info["total_episodes"] = total_after
    info["total_frames"] = len(df)
    info["splits"] = {"train": f"0:{total_after}"}
    with open(dataset_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Rewrote meta/info.json: {total_after} episodes, {len(df)} frames")

    logger.info("Done.")


if __name__ == "__main__":
    main()
