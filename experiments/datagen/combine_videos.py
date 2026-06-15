"""Pack per-episode MP4s into multi-episode video files for a LeRobot v3.0 dataset.

Concatenates individual episode videos (stream-copy, no re-encode) into fewer
packed files, rolling over when the accumulated size exceeds --target-size-mb.
Updates info.json video_path template and adds video timestamp columns to the
episodes metadata parquet.

Usage:
    cd experiments
    python datagen/combine_videos.py visual_dataset/coiling/final
    python datagen/combine_videos.py visual_dataset/multi_task/final --target-size-mb 300

    # Preview without writing:
    python datagen/combine_videos.py visual_dataset/coiling/final --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s

logger = logging.getLogger(__name__)

PACKED_VIDEO_PATH_TPL = (
    "videos/chunk-{chunk_index:03d}/{video_key}/file-{file_index:03d}.mp4"
)


def _video_keys(info: dict) -> list[str]:
    return sorted(
        k for k in info.get("features", {}) if k.startswith("observation.images.")
    )


def _src_video_path(root: Path, video_key: str, episode_idx: int) -> Path:
    return (
        root / "videos" / "chunk-000" / video_key / f"episode_{episode_idx:06d}.mp4"
    )


def _plan_packing(
    episode_indices: list[int],
    root: Path,
    keys: list[str],
    target_size_mb: int,
    fps: int,
) -> list[dict]:
    """Decide chunk/file assignment and timestamps for each episode."""
    target_bytes = target_size_mb * 1024 * 1024
    chunk_idx, file_idx = 0, 0
    accum_bytes: dict[str, int] = {k: 0 for k in keys}
    accum_dur: dict[str, float] = {k: 0.0 for k in keys}
    plan = []

    for i, ep_idx in enumerate(episode_indices):
        sizes: dict[str, int] = {}
        durations: dict[str, float] = {}
        srcs: dict[str, Path] = {}

        for k in keys:
            src = _src_video_path(root, k, ep_idx)
            if not src.exists():
                raise FileNotFoundError(f"Missing source video: {src}")
            sizes[k] = src.stat().st_size
            srcs[k] = src
            durations[k] = get_video_duration_in_s(src)

        # Roll over to next file if adding this episode exceeds target
        if i > 0 and any(accum_bytes[k] + sizes[k] >= target_bytes for k in keys):
            file_idx += 1
            accum_bytes = {k: 0 for k in keys}
            accum_dur = {k: 0.0 for k in keys}

        per_key: dict[str, dict] = {}
        for k in keys:
            from_ts = accum_dur[k]
            to_ts = from_ts + durations[k]
            per_key[k] = {"from_ts": from_ts, "to_ts": to_ts, "src": srcs[k]}
            accum_bytes[k] += sizes[k]
            accum_dur[k] = to_ts

        plan.append({
            "episode_index": ep_idx,
            "chunk_idx": chunk_idx,
            "file_idx": file_idx,
            "per_key": per_key,
        })

    return plan


def _group_sources(
    plan: list[dict], keys: list[str]
) -> dict[tuple[str, int, int], list[Path]]:
    groups: dict[tuple[str, int, int], list[Path]] = {}
    for ep in plan:
        for k in keys:
            grp = (k, ep["chunk_idx"], ep["file_idx"])
            groups.setdefault(grp, []).append(ep["per_key"][k]["src"])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_dir", type=Path, help="Dataset root directory.")
    parser.add_argument(
        "--target-size-mb", type=int, default=200,
        help="Rollover threshold per packed video file (default: 200 MB).",
    )
    parser.add_argument(
        "--keep-originals", action="store_true",
        help="Keep source per-episode mp4s after packing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan but do not write anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root: Path = args.dataset_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    # Check if already packed
    if "file-{file_index" in info.get("video_path", ""):
        logger.info("Already packed (video_path uses file-{file_index} template). Nothing to do.")
        return

    keys = _video_keys(info)
    if not keys:
        raise ValueError("No observation.images.* features in info.json")

    fps = int(info["fps"])
    total_episodes = int(info["total_episodes"])
    episode_indices = list(range(total_episodes))

    logger.info(
        "Dataset: %d episodes, %d cameras (%s), target=%d MB",
        total_episodes, len(keys), [k.split(".")[-1] for k in keys], args.target_size_mb,
    )

    plan = _plan_packing(episode_indices, root, keys, args.target_size_mb, fps)
    groups = _group_sources(plan, keys)

    n_packed_files = len(set((k, e["chunk_idx"], e["file_idx"]) for e in plan for k in keys)) // len(keys)
    logger.info("Will pack %d episodes into %d file(s) per camera", total_episodes, n_packed_files)

    if args.dry_run:
        for ep in plan[:5]:
            logger.info(
                "  ep %d -> chunk=%d file=%d", ep["episode_index"], ep["chunk_idx"], ep["file_idx"]
            )
            for k, pk in ep["per_key"].items():
                logger.info("      %s [%.3f, %.3f]s", k.split(".")[-1], pk["from_ts"], pk["to_ts"])
        if len(plan) > 5:
            logger.info("  ... (%d more)", len(plan) - 5)
        return

    # Concatenate videos
    for (key, chunk_idx, file_idx), srcs in sorted(groups.items()):
        out = root / PACKED_VIDEO_PATH_TPL.format(
            video_key=key, chunk_index=chunk_idx, file_index=file_idx
        )
        if out.exists():
            logger.info("Skip existing %s", out.relative_to(root))
            continue
        logger.info(
            "Concat %s chunk=%d file=%d (%d episodes)",
            key.split(".")[-1], chunk_idx, file_idx, len(srcs),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        concatenate_video_files(srcs, out, overwrite=True)

    # Update episodes metadata with video timestamps
    ep_meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if ep_meta_path.exists():
        ep_df = pd.read_parquet(ep_meta_path)
        for ep in plan:
            ei = ep["episode_index"]
            mask = ep_df["episode_index"] == ei
            for k in keys:
                ep_df.loc[mask, f"videos/{k}/chunk_index"] = ep["chunk_idx"]
                ep_df.loc[mask, f"videos/{k}/file_index"] = ep["file_idx"]
                ep_df.loc[mask, f"videos/{k}/from_timestamp"] = ep["per_key"][k]["from_ts"]
                ep_df.loc[mask, f"videos/{k}/to_timestamp"] = ep["per_key"][k]["to_ts"]
        ep_df.to_parquet(ep_meta_path, index=False)
        logger.info("Updated meta/episodes parquet with video timestamps")

    # Update info.json
    info["video_path"] = PACKED_VIDEO_PATH_TPL
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Updated meta/info.json video_path template")

    # Clean up originals
    if args.keep_originals:
        logger.info("--keep-originals: source per-episode mp4s left in place")
    else:
        deleted = 0
        for ep_idx in episode_indices:
            for k in keys:
                src = _src_video_path(root, k, ep_idx)
                if src.exists():
                    src.unlink()
                    deleted += 1
        logger.info("Deleted %d source per-episode mp4(s)", deleted)

    logger.info("Done.")


if __name__ == "__main__":
    main()
