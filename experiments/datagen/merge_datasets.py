"""Merge multiple LeRobot v3.0 datasets into one.

Takes a base directory and a list of sub-directory names, reads each
dataset, re-indexes episodes and frames contiguously, copies videos,
and writes a merged dataset.

Usage (subdir list):
    cd experiments
    python datagen/merge_datasets.py \
        --base_dir visual_dataset/wiring_post \
        --subdirs ppo1 ppo2 ppo3 \
        --output_dir visual_dataset/wiring_post/merged

Usage (config file with per-subdir episode counts):
    cd experiments
    python datagen/merge_datasets.py \
        --base_dir visual_dataset/coiling \
        --config datagen/merge_configs/coiling.json \
        --output_dir visual_dataset/coiling/merged

Config file format (JSON):
    [
        {"subdir": "sapo1_02", "n_episodes": 10},
        {"subdir": "shac2_05", "n_episodes":  5},
        {"subdir": "sapo2_04"}
    ]
    Omitting "n_episodes" (or setting it to null) uses all episodes in that subdir.
    Episodes are sampled randomly when n_episodes < total available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_source(src_dir: Path, n_episodes: int | None = None, seed: int = 0):
    """Load parquet + info from one source dataset.

    Args:
        src_dir:    Path to the source dataset directory.
        n_episodes: If set, randomly sample this many episodes (without
                    replacement).  ``None`` uses all episodes.
        seed:       Random seed for reproducible sampling.
    """
    data_path = src_dir / "data" / "chunk-000" / "file-000.parquet"
    info_path  = src_dir / "meta" / "info.json"
    if not data_path.exists():
        raise FileNotFoundError(f"No parquet found at {data_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json found at {info_path}")
    df = pd.read_parquet(data_path)
    with open(info_path) as f:
        info = json.load(f)

    if n_episodes is not None:
        all_ep_ids = sorted(df["episode_index"].unique())
        if n_episodes > len(all_ep_ids):
            raise ValueError(
                f"{src_dir.name}: requested n_episodes={n_episodes} but only "
                f"{len(all_ep_ids)} episodes are available"
            )
        rng = random.Random(seed)
        chosen = sorted(rng.sample(all_ep_ids, n_episodes))
        df = df[df["episode_index"].isin(chosen)].copy()

    return df, info


def load_config(config_path: Path) -> list[dict]:
    """Parse a merge config JSON file.

    Returns a list of dicts with keys ``subdir`` (str) and optionally
    ``n_episodes`` (int | None).
    """
    with open(config_path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Config file must contain a JSON array, got {type(entries)}")
    for e in entries:
        if "subdir" not in e:
            raise ValueError(f"Each config entry must have a 'subdir' key, got: {e}")
        e.setdefault("n_episodes", None)
    return entries


def camera_names_from_info(info: dict) -> list[str]:
    return sorted(
        k.replace("observation.images.", "")
        for k in info.get("features", {})
        if k.startswith("observation.images.")
    )


def copy_videos(
    src_dir: Path,
    dst_dir: Path,
    camera_names: list[str],
    old_ep_ids: list[int],
    ep_remap: dict[int, int],
):
    """Copy video files from src to dst with re-indexed episode names."""
    for cam in camera_names:
        src_cam = src_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
        dst_cam = dst_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
        dst_cam.mkdir(parents=True, exist_ok=True)
        for old_ep in old_ep_ids:
            src_vid = src_cam / f"episode_{old_ep:06d}.mp4"
            if src_vid.exists():
                new_ep = ep_remap[old_ep]
                shutil.copy2(src_vid, dst_cam / f"episode_{new_ep:06d}.mp4")


def update_video_paths(df: pd.DataFrame, camera_names: list[str]) -> pd.DataFrame:
    """Rewrite video path structs to match the new episode indices."""
    for cam in camera_names:
        key = f"observation.images.{cam}"
        if key not in df.columns:
            continue
        df[key] = df["episode_index"].apply(
            lambda ep, c=cam: {
                "path": f"videos/chunk-000/observation.images.{c}/episode_{ep:06d}.mp4",
                "bytes": None,
            }
        )
    return df


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot v3.0 datasets")
    parser.add_argument("--base_dir",   type=str, required=True,
                        help="Parent directory containing the sub-datasets")
    parser.add_argument("--subdirs",    type=str, nargs="+", default=None,
                        help="Sub-directory names to merge (e.g. ppo1 ppo2 ppo3). "
                             "Mutually exclusive with --config.")
    parser.add_argument("--config",     type=str, default=None,
                        help="JSON config file specifying subdirs and per-subdir "
                             "episode counts.  Mutually exclusive with --subdirs.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the merged dataset")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for episode sampling (default: 0)")
    args = parser.parse_args()

    if args.subdirs is None and args.config is None:
        parser.error("Provide either --subdirs or --config")
    if args.subdirs is not None and args.config is not None:
        parser.error("--subdirs and --config are mutually exclusive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base_dir   = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build (src_dir, n_episodes) list from either --subdirs or --config
    if args.subdirs is not None:
        entries = [{"subdir": s, "n_episodes": None} for s in args.subdirs]
    else:
        entries = load_config(Path(args.config))

    src_dirs = [base_dir / e["subdir"] for e in entries]
    for d in src_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Source directory not found: {d}")

    # ------------------------------------------------------------------ #
    # Load all sources
    # ------------------------------------------------------------------ #
    sources = []
    for e, d in zip(entries, src_dirs):
        n_ep = e["n_episodes"]
        df, info = load_source(d, n_episodes=n_ep, seed=args.seed)
        sources.append((d, df, info))
        sample_note = f" (sampled {n_ep})" if n_ep is not None else ""
        logger.info(f"Loaded {d.name}{sample_note}: {df['episode_index'].nunique()} episodes, {len(df)} frames")

    # Use first source's info as template for features / fps / etc.
    _, _, ref_info = sources[0]
    camera_names = camera_names_from_info(ref_info)
    has_videos   = bool(camera_names) and (src_dirs[0] / "videos").exists()

    # ------------------------------------------------------------------ #
    # Merge and re-index
    # ------------------------------------------------------------------ #
    merged_dfs   = []
    global_ep    = 0   # next episode_index in merged dataset
    global_frame = 0   # next frame index

    for src_dir, df, info in sources:
        old_ep_ids = sorted(df["episode_index"].unique())
        ep_remap   = {old: global_ep + i for i, old in enumerate(old_ep_ids)}

        # Re-map episode indices
        df = df.copy()
        df["episode_index"] = df["episode_index"].map(ep_remap)

        # Re-map global frame index
        n = len(df)
        df["index"] = np.arange(global_frame, global_frame + n, dtype=np.int64)

        # Re-map frame_index within each episode
        df["frame_index"] = df.groupby("episode_index").cumcount()

        # Update video path structs
        if has_videos:
            df = update_video_paths(df, camera_names)
            copy_videos(src_dir, output_dir, camera_names, old_ep_ids, ep_remap)

        merged_dfs.append(df)
        logger.info(f"  {src_dir.name}: episodes {global_ep}–{global_ep + len(old_ep_ids) - 1}")

        global_ep    += len(old_ep_ids)
        global_frame += n

    merged_df = pd.concat(merged_dfs, ignore_index=True)
    total_episodes = global_ep
    total_frames   = global_frame
    logger.info(f"Merged: {total_episodes} episodes, {total_frames} frames")

    # ------------------------------------------------------------------ #
    # Write data parquet
    # ------------------------------------------------------------------ #
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(data_dir / "file-000.parquet", index=False)
    logger.info("Wrote data/chunk-000/file-000.parquet")

    # ------------------------------------------------------------------ #
    # Recompute stats
    # ------------------------------------------------------------------ #
    from datagen.export import (
        compute_numeric_stats, default_image_stats,
        aggregate_stats, serialize_stats,
    )

    numeric_keys = ["observation.state", "action", "action.joint"]
    image_keys   = [f"observation.images.{c}" for c in camera_names] if has_videos else []

    all_ep_stats = []
    for ep_idx, ep_df in merged_df.groupby("episode_index"):
        ep_stats = {}
        for feat in numeric_keys:
            if feat in ep_df.columns:
                arr = np.array(ep_df[feat].tolist(), dtype=np.float32)
                ep_stats[feat] = compute_numeric_stats(arr)
        for key in image_keys:
            ep_stats[key] = default_image_stats(len(ep_df))
        all_ep_stats.append(ep_stats)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    global_stats = aggregate_stats(all_ep_stats)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(serialize_stats(global_stats), f, indent=2)
    logger.info("Wrote meta/stats.json")

    # ------------------------------------------------------------------ #
    # Episodes metadata
    # ------------------------------------------------------------------ #
    tasks_path = src_dirs[0] / "meta" / "tasks.parquet"
    task_description = "dlo_manipulation"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if len(tasks_df) > 0:
            task_description = tasks_df.index[0]

    ep_rows  = []
    g_start  = 0
    for ep_idx, ep_df in merged_df.groupby("episode_index"):
        n = len(ep_df)
        ep_stats = all_ep_stats[ep_idx]
        row = {
            "episode_index": int(ep_idx),
            "tasks": [task_description],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": g_start,
            "dataset_to_index": g_start + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feat, stats in ep_stats.items():
            for stat_name, stat_val in stats.items():
                row[f"stats/{feat}/{stat_name}"] = (
                    stat_val.tolist() if isinstance(stat_val, np.ndarray) else stat_val
                )
        ep_rows.append(row)
        g_start += n

    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_rows).to_parquet(ep_meta_dir / "file-000.parquet", index=False)
    logger.info("Wrote meta/episodes/chunk-000/file-000.parquet")

    # tasks.parquet
    tasks_out = pd.DataFrame({"task_index": [0]}, index=[task_description])
    tasks_out.to_parquet(meta_dir / "tasks.parquet")
    logger.info("Wrote meta/tasks.parquet")

    # ------------------------------------------------------------------ #
    # info.json
    # ------------------------------------------------------------------ #
    info_out = dict(ref_info)
    info_out["total_episodes"] = total_episodes
    info_out["total_frames"]   = total_frames
    info_out["splits"]         = {"train": f"0:{total_episodes}"}
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_out, f, indent=2)
    logger.info(f"Wrote meta/info.json: {total_episodes} episodes, {total_frames} frames")

    logger.info(f"Done. Merged dataset at {output_dir}")


if __name__ == "__main__":
    main()
