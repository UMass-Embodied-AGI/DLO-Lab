"""Interactive web UI for manually filtering dataset episodes.

Displays front + wrist camera videos per episode with prev/next navigation.
Mark episodes for discard, then click Complete to apply filtering.

Usage:
    cd experiments
    python datagen/filter_ui.py --dataset_dir visual_dataset/wiring_post/ppo
    # Then open http://localhost:7860 in your browser
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def build_app(dataset_dir: Path, camera_names: list[str], episode_ids: list[int]):
    from flask import Flask, jsonify, request, send_file, Response

    app = Flask(__name__)
    app.config["DATASET_DIR"] = dataset_dir
    app.config["CAMERA_NAMES"] = camera_names
    app.config["EPISODE_IDS"] = episode_ids

    # ------------------------------------------------------------------ #
    # Video serving
    # ------------------------------------------------------------------ #

    @app.route("/video/<cam>/<int:ep_idx>")
    def serve_video(cam, ep_idx):
        vid_path = (
            dataset_dir / "videos" / "chunk-000"
            / f"observation.images.{cam}"
            / f"episode_{ep_idx:06d}.mp4"
        )
        if not vid_path.exists():
            return Response("Video not found", status=404)
        return send_file(str(vid_path), mimetype="video/mp4", conditional=True)

    # ------------------------------------------------------------------ #
    # Apply discard
    # ------------------------------------------------------------------ #

    @app.route("/apply", methods=["POST"])
    def apply_filter():
        data = request.get_json()
        discard_ids = sorted(set(data.get("discard", [])))
        if not discard_ids:
            return jsonify({"status": "ok", "message": "Nothing to discard."})
        try:
            from datagen.filter_dataset import (
                load_dataset, filter_and_reindex, rename_videos,
                recompute_stats, build_episodes_metadata, update_video_paths,
            )
            from datagen.export import serialize_stats
            df, info = load_dataset(dataset_dir)
            df, ep_remap = filter_and_reindex(df, set(discard_ids))
            has_videos = bool(camera_names) and (dataset_dir / "videos").exists()
            if has_videos:
                df = update_video_paths(df, dataset_dir, camera_names)
            data_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
            df.to_parquet(data_path, index=False)
            if has_videos:
                rename_videos(dataset_dir, ep_remap, camera_names)
            numeric_keys = ["observation.state", "action", "action.joint"]
            image_keys = [f"observation.images.{c}" for c in camera_names] if has_videos else []
            global_stats = recompute_stats(df, numeric_keys, image_keys)
            with open(dataset_dir / "meta" / "stats.json", "w") as f:
                json.dump(serialize_stats(global_stats), f, indent=2)
            import pandas as pd
            tasks_path = dataset_dir / "meta" / "tasks.parquet"
            task_description = "dlo_manipulation"
            if tasks_path.exists():
                tasks_df = pd.read_parquet(tasks_path)
                if len(tasks_df) > 0:
                    task_description = tasks_df.index[0]
            ep_rows = build_episodes_metadata(df, task_description, numeric_keys, image_keys)
            ep_df = pd.DataFrame(ep_rows)
            ep_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            ep_df.to_parquet(ep_path, index=False)
            total_after = int(df["episode_index"].nunique())
            info["total_episodes"] = total_after
            info["total_frames"] = len(df)
            info["splits"] = {"train": f"0:{total_after}"}
            with open(dataset_dir / "meta" / "info.json", "w") as f:
                json.dump(info, f, indent=2)
            msg = f"Discarded {len(discard_ids)} episode(s). {total_after} episodes remaining."
            logger.info(msg)
            return jsonify({"status": "ok", "message": msg})
        except Exception as e:
            logger.exception("Filter failed")
            return jsonify({"status": "error", "message": str(e)}), 500

    # ------------------------------------------------------------------ #
    # Main page
    # ------------------------------------------------------------------ #

    HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Filter UI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; }
  header {
    display: flex; align-items: center; gap: 16px;
    padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333;
    position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 1rem; font-weight: 600; flex: 1; }
  .counter { font-size: 0.85rem; color: #aaa; }
  .ep-badge {
    border-radius: 6px; padding: 3px 10px;
    font-size: 0.8rem; font-weight: 600;
    background: #c0392b;  /* default: discard = red */
  }
  .ep-badge.keep { background: #27ae60; }
  .main { padding: 20px; }
  .ep-header {
    display: flex; align-items: center; gap: 16px; margin-bottom: 16px;
  }
  .ep-id { font-size: 1.4rem; font-weight: 700; }
  .discard-check { display: flex; align-items: center; gap: 8px; cursor: pointer; }
  .discard-check input { width: 18px; height: 18px; cursor: pointer; accent-color: #27ae60; }
  .discard-check label { font-size: 0.95rem; cursor: pointer; color: #27ae60; font-weight: 500; }
  .videos { display: flex; gap: 16px; align-items: stretch; height: 60vh; }
  .video-card { flex: 1; min-width: 0; background: #1a1a1a; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
  .video-card h3 { padding: 8px 12px; font-size: 0.8rem; color: #aaa; background: #222; flex-shrink: 0; }
  .video-card video { flex: 1; min-height: 0; width: 100%; object-fit: contain; display: block; background: #000; }
  .nav { display: flex; align-items: center; gap: 12px; margin-top: 20px; }
  button {
    padding: 9px 20px; border: none; border-radius: 6px;
    font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: opacity 0.15s;
  }
  button:disabled { opacity: 0.35; cursor: default; }
  .btn-nav { background: #2c3e50; color: #fff; }
  .btn-nav:hover:not(:disabled) { background: #34495e; }
  .btn-complete { background: #27ae60; color: #fff; margin-left: auto; }
  .btn-complete:hover { background: #2ecc71; }
  .discard-count { font-size: 0.85rem; color: #27ae60; min-width: 140px; }
  .progress-bar { height: 3px; background: #333; margin-bottom: 0; }
  .progress-fill { height: 100%; background: #3498db; transition: width 0.2s; }
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.7); z-index: 100;
    align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: #1e1e1e; border-radius: 10px; padding: 28px;
    max-width: 480px; width: 90%; border: 1px solid #333;
  }
  .modal h2 { margin-bottom: 12px; font-size: 1.1rem; }
  .modal p { color: #aaa; font-size: 0.9rem; margin-bottom: 16px; line-height: 1.5; }
  .modal-ids { font-family: monospace; color: #e74c3c; word-break: break-all; margin-bottom: 20px; font-size: 0.9rem; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .btn-cancel { background: #333; color: #eee; }
  .btn-confirm { background: #c0392b; color: #fff; }
  .btn-confirm:hover { background: #e74c3c; }
  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #27ae60; color: #fff; padding: 12px 24px;
    border-radius: 8px; font-size: 0.9rem; font-weight: 500;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
    z-index: 200;
  }
  .toast.error { background: #c0392b; }
  .toast.show { opacity: 1; }
  kbd {
    background: #333; border-radius: 3px; padding: 1px 5px;
    font-size: 0.75rem; font-family: monospace; color: #ccc;
  }
  .hint { font-size: 0.75rem; color: #666; margin-left: auto; }
</style>
</head>
<body>

<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>

<header>
  <h1 id="dataset-title">Dataset Filter</h1>
  <span class="counter" id="counter">Episode 1 / N</span>
  <span class="ep-badge" id="ep-badge">DISCARD</span>
</header>

<div class="main">
  <div class="ep-header">
    <span class="ep-id" id="ep-id">Episode 0</span>
    <label class="discard-check">
      <input type="checkbox" id="keep-check" onchange="toggleKeep()">
      <label for="keep-check">Keep this episode</label>
    </label>
  </div>
  <div class="videos" id="videos"></div>
  <div class="nav">
    <button class="btn-nav" id="btn-prev" onclick="navigate(-1)">&#8592; Prev</button>
    <button class="btn-nav" id="btn-next" onclick="navigate(1)">Next &#8594;</button>
    <span class="discard-count" id="keep-count"></span>
    <span class="hint"><kbd>A</kbd><kbd>D</kbd> navigate &nbsp; <kbd>K</kbd> keep episode</span>
    <button class="btn-complete" onclick="openConfirm()">Complete &amp; Apply &#10003;</button>
  </div>
</div>

<!-- Confirm modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2>Confirm Discard</h2>
    <p id="modal-summary"></p>
    <div class="modal-ids" id="modal-ids"></div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeConfirm()">Cancel</button>
      <button class="btn-confirm" onclick="applyFilter()">Yes, Discard</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const EPISODES = __EPISODES__;
const CAMERAS  = __CAMERAS__;
const TITLE    = "__TITLE__";
document.getElementById("dataset-title").textContent = TITLE;

let cur = 0;
// Default: all episodes are discarded; user presses K to keep specific ones.
const kept = new Set();

function render() {
  const ep = EPISODES[cur];
  const isKept = kept.has(ep);

  document.getElementById("ep-id").textContent = `Episode ${ep}`;
  document.getElementById("counter").textContent = `${cur + 1} / ${EPISODES.length}`;
  document.getElementById("progress").style.width = `${(cur + 1) / EPISODES.length * 100}%`;

  const chk = document.getElementById("keep-check");
  chk.checked = isKept;

  const badge = document.getElementById("ep-badge");
  badge.textContent = isKept ? "KEEP" : "DISCARD";
  badge.classList.toggle("keep", isKept);

  document.getElementById("btn-prev").disabled = cur === 0;
  document.getElementById("btn-next").disabled = cur === EPISODES.length - 1;

  const nKept = kept.size;
  const nDiscard = EPISODES.length - nKept;
  document.getElementById("keep-count").textContent =
    `${nKept} kept / ${nDiscard} to discard`;

  // Videos
  const container = document.getElementById("videos");
  container.innerHTML = "";
  for (const cam of CAMERAS) {
    const card = document.createElement("div");
    card.className = "video-card";
    const title = document.createElement("h3");
    title.textContent = cam;
    const video = document.createElement("video");
    video.src = `/video/${cam}/${ep}`;
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    card.appendChild(title);
    card.appendChild(video);
    container.appendChild(card);
  }
}

function navigate(delta) {
  const next = cur + delta;
  if (next < 0 || next >= EPISODES.length) return;
  cur = next;
  render();
}

function toggleKeep() {
  const ep = EPISODES[cur];
  if (document.getElementById("keep-check").checked) {
    kept.add(ep);
  } else {
    kept.delete(ep);
  }
  const isKept = kept.has(ep);
  const badge = document.getElementById("ep-badge");
  badge.textContent = isKept ? "KEEP" : "DISCARD";
  badge.classList.toggle("keep", isKept);
  const nKept = kept.size;
  document.getElementById("keep-count").textContent =
    `${nKept} kept / ${EPISODES.length - nKept} to discard`;
}

function openConfirm() {
  const discardIds = EPISODES.filter(ep => !kept.has(ep));
  if (discardIds.length === 0) {
    showToast("All episodes are kept — nothing to discard.", false);
    return;
  }
  document.getElementById("modal-summary").textContent =
    `You are about to permanently remove ${discardIds.length} episode(s) and keep ${kept.size}. This cannot be undone.`;
  document.getElementById("modal-ids").textContent =
    "Discarding: " + discardIds.join(", ");
  document.getElementById("modal").classList.add("open");
}

function closeConfirm() {
  document.getElementById("modal").classList.remove("open");
}

async function applyFilter() {
  closeConfirm();
  const ids = EPISODES.filter(ep => !kept.has(ep));
  try {
    const res = await fetch("/apply", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({discard: ids}),
    });
    const data = await res.json();
    if (data.status === "ok") {
      showToast(data.message, false);
      kept.clear();
      // Reload page after short delay
      setTimeout(() => location.reload(), 2000);
    } else {
      showToast("Error: " + data.message, true);
    }
  } catch(e) {
    showToast("Request failed: " + e, true);
  }
}

function showToast(msg, isError) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "") + " show";
  setTimeout(() => t.classList.remove("show"), 3500);
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft"  || e.key === "a" || e.key === "A") navigate(-1);
  if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") navigate(1);
  if (e.key === "k" || e.key === "K") {
    const chk = document.getElementById("keep-check");
    chk.checked = !chk.checked;
    toggleKeep();
  }
});

render();
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        html = HTML
        html = html.replace("__EPISODES__", json.dumps(episode_ids))
        html = html.replace("__CAMERAS__", json.dumps(camera_names))
        html = html.replace("__TITLE__", dataset_dir.name)
        return html

    return app


def main():
    parser = argparse.ArgumentParser(description="Web UI for manual dataset filtering")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Load info.json
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No meta/info.json found in {dataset_dir}")
    with open(info_path) as f:
        info = json.load(f)

    # Detect cameras
    camera_names = sorted(
        k.replace("observation.images.", "")
        for k in info.get("features", {})
        if k.startswith("observation.images.")
    )
    if not camera_names:
        logger.warning("No camera features found in info.json — video display disabled")

    # Detect episodes from video files or parquet
    vid_base = dataset_dir / "videos" / "chunk-000"
    if camera_names and vid_base.exists():
        cam_dir = vid_base / f"observation.images.{camera_names[0]}"
        episode_ids = sorted(
            int(p.stem.split("_")[1])
            for p in cam_dir.glob("episode_*.mp4")
        )
    else:
        import pandas as pd
        df = pd.read_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
        episode_ids = sorted(df["episode_index"].unique().tolist())

    logger.info(f"Dataset: {dataset_dir.name}")
    logger.info(f"Episodes: {len(episode_ids)}  Cameras: {camera_names}")
    logger.info(f"Starting UI at http://localhost:{args.port}")

    app = build_app(dataset_dir, camera_names, episode_ids)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
