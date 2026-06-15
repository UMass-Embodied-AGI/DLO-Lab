## LeRobot v3.0 Dataset Generation

Generates [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets for training visuomotor (VLA) policies by rolling out trained RL / trajectory-optimization policies in the DLO-Lab Genesis environments.

### Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | **Main entry point** — policy rollouts → per-episode parquet + MP4 → finalized v3.0 dataset |
| `config.py` | Task registry, unified schema dims, env class resolution, per-task front-camera poses |
| `cameras.py` | Genesis camera setup: front + wrist cameras, multi-env rendering, wrist offset transform |
| `export.py` | LeRobot v3.0 export: parquet/video writing, stats aggregation, dataset finalization |
| `policy_wrappers.py` | Uniform `get_action()` interface for PPO/SAC (MushroomRL), SHAC (diff_rl), CMA-ES (trajopt) |
| `filter_dataset.py` | CLI: remove episodes by index / random subsample, re-index, regenerate metadata |
| `filter_ui.py` | Flask web UI for manual episode curation (keep/discard with video preview) |
| `merge_datasets.py` | Merge multiple same-task datasets (optionally sampling episodes per source) |
| `merge_tasks.py` | Merge datasets of different tasks into one multi-task dataset (`--norm_strategy active_only` recommended) |
| `combine_videos.py` | Pack per-episode MP4s into larger files (requires the `lerobot` package) |

### Usage

**Note: All commands below are run from the `experiments/` directory.**

#### Generate a dataset

```bash
python datagen/generate_dataset.py \
    --task coiling \
    --algo ppo \
    --checkpoint logs/coiling/rudin-01/best_ppo.pkl \
    --n_episodes 100 \
    --n_envs 10 \
    --horizon 100 \
    --save_images \
    --output_dir datasets/coiling_ppo
```

Algo ↔ checkpoint mapping in this repo:

| `--algo` | Trainer | `--checkpoint` points at | Typical exp name |
|----------|---------|--------------------------|------------------|
| `ppo` | `rl/rudinppo.py` | `logs/{task}/{exp}/best_ppo.pkl` | `rudin-*` |
| `sac` | `rl/sac.py` | `logs/{task}/{exp}/best_sac.pkl` | `sac-*` |
| `shac` | `rl/shac.py` (SHAC/SAPO) | `logs/{task}/{exp}/best_shac.pkl` | `sapo-*`, `shac-*` |
| `cmaes` | `trajopt/cmaes.py` | `logs/{task}/{exp}/` (the run **dir**, with `cmaes_ckpt.pkl` / `best_traj.npy`) | `cmaes-*` |


#### Raytraced rendering

Please first ensure LuisaRender has been installed following [this guide]((https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html#photo-realistic-rendering-with-luisa-deprecating)). 
Add `--raytracer` (alias `-r`) to render photo-realistic frames with the LuisaRender path tracer instead of the rasterizer. This swaps in a textured wooden table and an HDRI environment map (`genesis/assets/dlo-lab/exrs/`, `meshes/`) and produces proper lighting and shadows. 

```bash
python datagen/generate_dataset.py \
    --task coiling \
    --algo ppo \
    --checkpoint logs/coiling/rudin-01/best_ppo.pkl \
    --n_episodes 100 \
    --n_envs 1 \
    --horizon 100 \
    --save_images \
    --raytracer \
    --output_dir datasets/coiling_ppo_rt
```

**Always use `--n_envs 1` with `--raytracer`.** Multi-env rendering relies on a rasterizer trick (moving a single camera to each env's world-space offset before each render); the path tracer is not validated for it, and raytraced frames are slow (~5× the rasterizer) so batching buys little. Keep `--n_envs 1` and increase `--n_episodes` instead.

Use `--exr_path` to choose the HDRI environment map for the raytracer's env surface (default `dlo-lab/exrs/brown_photostudio_02_4k.exr`; alternatives `brown_photostudio_{01,04,05}_4k.exr`). It has no effect without `--raytracer`. The map is fixed for the whole run — to vary it across a dataset, generate separate runs with different `--exr_path` into sub-dirs and `merge_datasets.py` them.

#### Post-processing

```bash
# Remove bad episodes / subsample
python datagen/filter_dataset.py --dataset_dir datasets/coiling_ppo --remove_episodes 3 7
python datagen/filter_dataset.py --dataset_dir datasets/coiling_ppo --target_num 50

# Manual curation in the browser
python datagen/filter_ui.py --dataset_dir datasets/coiling_ppo --port 7860

# Merge same-task datasets
python datagen/merge_datasets.py --base_dir datasets --subdirs coiling_ppo coiling_sac \
    --output_dir datasets/coiling_merged

# Merge different tasks into one multi-task dataset
python datagen/merge_tasks.py --dirs datasets/coiling_ppo datasets/unknotting_sapo \
    --output_dir datasets/multitask --norm_strategy active_only

# Pack per-episode videos into larger files (needs `lerobot`, already in the dlo-lab extra)
python datagen/combine_videos.py datasets/coiling_ppo --target-size-mb 200
```

---

## Unified Dataset Schema

All tasks share a **single fixed tensor schema** regardless of how many robot arms they use. This allows training a single VLA policy across all tasks without per-task input/output adapters.

### Task Categorization

| Task | Arms | n_controllers |
|------|------|---------------|
| coiling | single | 1 |
| slingshot | single | 1 |
| wiring_post | single | 1 |
| gathering | bimanual | 2 |
| lifting | bimanual | 2 |
| separation | bimanual | 2 |
| unknotting | bimanual | 2 |
| wrapping | bimanual | 2 |

### Unified Tensor Dimensions

| Field | Dim | Layout |
|-------|-----|--------|
| `observation.state` | **16** | `[right_joint_1..7, right_gripper_width, left_joint_1..7, left_gripper_width]` |
| `action` | **12** | step_all: `[right_xyz, left_xyz, right_rot, left_rot]` |
| `action.joint` | **18** | `[right_motor_1..7, right_finger_1, right_finger_2, left_motor_1..7, left_finger_1, left_finger_2]` |

Constants defined in `config.py`:
```python
N_ARMS_UNIFIED      = 2
STATE_DIM_UNIFIED   = 16   # N_ARMS_UNIFIED * ARM_STATE_DIM
ACTION_DIM_UNIFIED  = 12   # N_ARMS_UNIFIED * ARM_ACTION_DIM
JOINT_DIM_UNIFIED   = 18   # N_ARMS_UNIFIED * ARM_JOINT_DIM
```

#### Single-arm padding rules

Single-arm tasks collect only 8D state / 6D action / 9D joint from the real arm. These are zero-padded at datagen time to fill the unified dims:

**State (8D → 16D):** zeros appended for the missing left arm.
```
[right_joint_1..7, right_gripper_width, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Action (6D → 12D):** split into xyz + rot slots, left arm slots are zero.
```
original:  [dx, dy, dz, droll, dpitch, dyaw]
unified:   [dx, dy, dz,  0,  0,  0,  droll, dpitch, dyaw,  0,  0,  0]
           ├── right_xyz ──┤├── left_xyz ──┤├── right_rot ──┤├── left_rot ──┤
```

**Joint (9D → 18D):** zeros appended for the missing left arm.
```
[right_motor_1..7, right_finger_1, right_finger_2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## Action Format: step_all Convention

All tasks — single-arm (padded) and bimanual — store actions in **step_all format**:

```
[right_dx, right_dy, right_dz, left_dx, left_dy, left_dz,
 right_droll, right_dpitch, right_dyaw, left_droll, left_dpitch, left_dyaw]
 ├──────────── all xyz ────────────────┤├──────────── all rot ───────────────┤
```

### Camera Schema

All tasks always produce **three cameras**:

| Camera | Single-arm | Bimanual |
|--------|-----------|----------|
| `front` | real | real |
| `wrist_right` | real (right arm) | real (right arm) |
| `wrist_left` | **black frame** | real (left arm) |

For single-arm tasks, `wrist_left` is stored as a static black (zeros) frame — no additional GPU camera object is created. The video file is saved as usual so the schema remains identical across all tasks.

Front camera poses are hardcoded per task in `config.py::FRONT_CAMERA_PARAMS`, and the wrist camera offset is `cameras.py::WRIST_OFFSET_T`.

### Inference

A consumer (e.g. a VLA inference script) should match the unified schema:

- **State input**: zero-pad env state to `STATE_DIM_UNIFIED = 16` before passing to the VLA.
- **Action output**: the VLA always outputs 12D unified action. For single-arm tasks extract the real DOF before calling `step_all`:
  ```python
  env_action = np.concatenate([action_np[0:3], action_np[6:9]])  # right_xyz + right_rot
  ```

### Output Layout (LeRobot v3.0)

```
dataset_dir/
├── data/chunk-000/file-000.parquet      # all frames, concatenated
├── videos/chunk-000/observation.images.{front,wrist_right,wrist_left}/episode_NNNNNN.mp4
└── meta/
    ├── info.json                        # features, fps, shapes, paths
    ├── stats.json                       # per-feature normalization stats
    ├── tasks.parquet                    # task descriptions ↔ task_index
    └── episodes/chunk-000/file-000.parquet
```
