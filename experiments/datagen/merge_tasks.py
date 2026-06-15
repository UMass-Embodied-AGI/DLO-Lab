"""merge_tasks.py

Merge LeRobot v3.0 datasets from multiple tasks into a single multi-task dataset.

Each source directory may contain a different task.  Episodes are re-indexed
contiguously; each task gets a unique task_index in meta/tasks.parquet, and
the task_index column in the data parquet is updated accordingly.

Usage:
    cd experiments
    python datagen/merge_tasks.py \
        --dirs visual_dataset/separation/final \
               visual_dataset/unknotting/final \
        --output_dir visual_dataset/multi_task/final

    # Optionally override the task description for any source dir:
    python datagen/merge_tasks.py \
        --dirs visual_dataset/separation/final visual_dataset/unknotting/final \
        --output_dir visual_dataset/multi_task/final \
        --descriptions "separate the ropes" "unknot the rope"
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source(src_dir: Path):
    data_path = src_dir / "data" / "chunk-000" / "file-000.parquet"
    info_path  = src_dir / "meta" / "info.json"
    tasks_path = src_dir / "meta" / "tasks.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"No parquet found at {data_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json found at {info_path}")

    df = pd.read_parquet(data_path)
    with open(info_path) as f:
        info = json.load(f)

    task_description = None
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if len(tasks_df) > 0:
            task_description = tasks_df.index[0]

    return df, info, task_description


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge LeRobot v3.0 datasets from multiple tasks into one multi-task dataset"
    )
    parser.add_argument(
        "--dirs", nargs="+", required=True,
        help="Source dataset directories (one per task)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for the merged dataset",
    )
    parser.add_argument(
        "--descriptions", nargs="+", default=None,
        help="Optional: override task description strings, one per --dirs entry",
    )
    parser.add_argument(
        "--norm_strategy", choices=["global", "active_only"], default="active_only",
        help="Stats normalization strategy. 'global' = naive stats over all data. "
             "'active_only' = compute mean/std only from episodes where each dim "
             "has non-zero variance (avoids zero-padded single-arm dims distorting stats).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src_dirs   = [Path(d).resolve() for d in args.dirs]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in src_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Source directory not found: {d}")

    if args.descriptions is not None and len(args.descriptions) != len(src_dirs):
        raise ValueError(
            f"--descriptions must have the same number of entries as --dirs "
            f"({len(src_dirs)}), got {len(args.descriptions)}"
        )

    # ------------------------------------------------------------------ #
    # Load all sources
    # ------------------------------------------------------------------ #
    sources = []  # (src_dir, df, info, task_description)
    for i, d in enumerate(src_dirs):
        df, info, task_desc_from_file = load_source(d)
        # CLI override takes priority, then tasks.parquet, then dir name
        if args.descriptions is not None:
            task_desc = args.descriptions[i]
        elif task_desc_from_file is not None:
            task_desc = task_desc_from_file
        else:
            task_desc = d.name
        sources.append((d, df, info, task_desc))
        logger.info(
            f"Loaded {d.name}: {df['episode_index'].nunique()} episodes, "
            f"{len(df)} frames  |  task='{task_desc}'"
        )

    # ------------------------------------------------------------------ #
    # Build task registry: description → task_index
    # Preserve insertion order; deduplicate (same description = same index).
    # ------------------------------------------------------------------ #
    task_registry: dict[str, int] = {}
    for _, _, _, task_desc in sources:
        if task_desc not in task_registry:
            task_registry[task_desc] = len(task_registry)
    logger.info(f"Tasks ({len(task_registry)}): {list(task_registry.keys())}")

    # ------------------------------------------------------------------ #
    # Merge and re-index
    # ------------------------------------------------------------------ #
    ref_info     = sources[0][2]
    camera_names = camera_names_from_info(ref_info)
    has_videos   = bool(camera_names) and (src_dirs[0] / "videos").exists()

    merged_dfs   = []
    global_ep    = 0
    global_frame = 0

    for src_dir, df, info, task_desc in sources:
        new_task_index = task_registry[task_desc]
        old_ep_ids     = sorted(df["episode_index"].unique())
        ep_remap       = {old: global_ep + i for i, old in enumerate(old_ep_ids)}

        df = df.copy()
        df["episode_index"] = df["episode_index"].map(ep_remap)
        df["task_index"]    = new_task_index

        n = len(df)
        df["index"]       = np.arange(global_frame, global_frame + n, dtype=np.int64)
        df["frame_index"] = df.groupby("episode_index").cumcount()

        if has_videos:
            df = update_video_paths(df, camera_names)
            copy_videos(src_dir, output_dir, camera_names, old_ep_ids, ep_remap)

        merged_dfs.append(df)
        logger.info(
            f"  {src_dir.name}: episodes {global_ep}–{global_ep + len(old_ep_ids) - 1}"
            f"  task_index={new_task_index}"
        )
        global_ep    += len(old_ep_ids)
        global_frame += n

    merged_df    = pd.concat(merged_dfs, ignore_index=True)
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
    # Recompute stats from merged data
    # ------------------------------------------------------------------ #
    from datagen.export import (
        compute_numeric_stats, default_image_stats,
        aggregate_stats, serialize_stats, compute_active_only_stats,
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

    if args.norm_strategy == "active_only":
        global_stats = compute_active_only_stats(merged_df, numeric_keys)
        for key in image_keys:
            global_stats[key] = default_image_stats(total_frames)
        logger.info("Using active-only normalization (zero-padded dims excluded from mean/std)")
    else:
        global_stats = aggregate_stats(all_ep_stats)

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(serialize_stats(global_stats), f, indent=2)
    logger.info("Wrote meta/stats.json")

    # ------------------------------------------------------------------ #
    # Episodes metadata parquet
    # ------------------------------------------------------------------ #
    ep_task_index = merged_df.groupby("episode_index")["task_index"].first()
    index_to_desc = {v: k for k, v in task_registry.items()}

    ep_rows = []
    g_start = 0
    for ep_idx, ep_df in merged_df.groupby("episode_index"):
        n        = len(ep_df)
        ep_stats = all_ep_stats[ep_idx]
        task_desc = index_to_desc[int(ep_task_index[ep_idx])]
        row = {
            "episode_index":             int(ep_idx),
            "tasks":                     [task_desc],
            "length":                    n,
            "data/chunk_index":          0,
            "data/file_index":           0,
            "dataset_from_index":        g_start,
            "dataset_to_index":          g_start + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index":  0,
        }
        ts = ep_df["timestamp"].astype(float)
        for cam in camera_names:
            vid_key = f"observation.images.{cam}"
            row[f"videos/{vid_key}/chunk_index"]    = 0
            row[f"videos/{vid_key}/file_index"]     = int(ep_idx)
            row[f"videos/{vid_key}/from_timestamp"] = float(ts.min())
            row[f"videos/{vid_key}/to_timestamp"]   = float(ts.max())
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

    # ------------------------------------------------------------------ #
    # tasks.parquet  — one row per unique task
    # ------------------------------------------------------------------ #
    tasks_out = pd.DataFrame(
        {"task_index": list(task_registry.values())},
        index=list(task_registry.keys()),
    )
    tasks_out.to_parquet(meta_dir / "tasks.parquet")
    logger.info(f"Wrote meta/tasks.parquet ({len(task_registry)} tasks)")

    # ------------------------------------------------------------------ #
    # info.json
    # ------------------------------------------------------------------ #
    info_out = dict(ref_info)
    info_out["total_episodes"] = total_episodes
    info_out["total_frames"]   = total_frames
    info_out["total_tasks"]    = len(task_registry)
    info_out["splits"]         = {"train": f"0:{total_episodes}"}
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_out, f, indent=2)
    logger.info(f"Wrote meta/info.json: {total_episodes} episodes, {total_frames} frames, {len(task_registry)} tasks")

    logger.info(f"Done. Multi-task dataset at {output_dir}")


if __name__ == "__main__":
    main()
