# DLO-Lab Installation Guide (Linux, Verified)

This guide documents a complete, reproducible installation of DLO-Lab on
**Linux x86\_64 with a CUDA-capable GPU**, including every error encountered
during a fresh installation and how to fix each one.

DLO-Lab is built on top of the Genesis physics engine (bundled in this repo).
The install is non-trivial because Genesis has strict package version pins and
requires a CUDA PyTorch build matched to your driver.

---

## System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 20.04 / 22.04 / 24.04 | Tested on 22.04 |
| GPU | NVIDIA, compute capability ≥ 7.5 | Turing / Ampere / Ada |
| CUDA | 11.8 or 12.x | Must match PyTorch build |
| Python | 3.10–3.13 | Verified: 3.10 (most stable wheel availability) |
| RAM | ≥ 16 GB | Genesis caches compiled kernels in memory |
| Disk | ≥ 10 GB | For conda env + wheels + assets |

**CPU-only is not supported.** Genesis' rod solver requires a CUDA/Taichi backend
(`gs.gpu`). There is no `gs.cpu` fallback for the ROD material.

---

## Step 1: Create Conda Environment

```bash
conda create -n dlolab python=3.10 -y
conda activate dlolab
```

> **Python version note:** This guide is verified on **Python 3.10**. Python 3.11
> and 3.12 may work but some wheels (e.g. `pymeshlab`) have inconsistent availability
> across minor versions. Stick to 3.10 for a friction-free install.

---

## Step 2: Install PyTorch (CUDA)

Install **before** Genesis — Genesis' `pyproject.toml` pins `mujoco>=3.2.5`
and several packages that may conflict with a CPU-only torch install.

Check your CUDA version first:

```bash
nvidia-smi    # shows "CUDA Version: XX.Y" in the top-right corner
nvcc --version  # shows the toolkit version (may differ from driver)
```

Then install the matching PyTorch build:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x+cuXXX True
```

---

## Step 3: Install DLO-Lab

```bash
cd /path/to/DLO-Lab
pip install -e ".[dlo-lab,dev]"
```

This installs Genesis (the bundled version in `genesis/`) plus all DLO-Lab
experiment dependencies (`omegaconf`, `cma`, `pin-pink`, `lerobot`, etc.).

### Common Errors at This Step

**Error: `OpenEXR` build fails**
```
error: could not build OpenEXR
```
Fix:
```bash
sudo apt install libopenexr-dev openexr
pip install OpenEXR
```

**Error: `pymeshlab` wheel not found for Python 3.12**
```
No matching distribution found for pymeshlab
```
Fix: `pymeshlab` publishes wheels for cp310/cp311/cp312. If pip can't find one,
try upgrading pip first:
```bash
pip install --upgrade pip
pip install pymeshlab
```
If still failing, install from conda-forge as a workaround:
```bash
conda install -c conda-forge pymeshlab -y
```

**Error: `tetgen` build fails (missing CMake or C++ compiler)**
```
CMake must be installed to build the following extensions: tetgen
```
Fix:
```bash
sudo apt install cmake g++ -y
pip install tetgen==0.8.2
```

**Error: `z3-solver>=4.15.5.0` conflict**

Genesis pins `z3-solver<4.15.5.0`. If you see a resolver conflict, force the
pinned version:
```bash
pip install "z3-solver<4.15.5.0"
```

**Error: `gs-madrona` install fails**

`gs-madrona` (the batch renderer) is Linux x86\_64 only and requires glibc ≥ 2.27.
On Ubuntu 20.04+ this should not occur. On older systems, install without it:
```bash
pip install -e ".[dlo-lab,dev]" --no-deps
# Then install remaining dependencies individually, e.g.:
pip install omegaconf cma pin-pink lerobot torch torchvision
```
The rasterizer renderer still works without `gs-madrona`.

---

## Step 4: Install ffmpeg

Genesis uses ffmpeg for video export. Install via conda-forge to get version 7:

```bash
conda install "ffmpeg=7" -c conda-forge -y
```

Verify:
```bash
ffmpeg -version | head -1
# ffmpeg version 7.x
```

---

## Step 5: Install Mushroom-RL

The RL training scripts use a custom fork of Mushroom-RL:

```bash
git clone https://github.com/XJay18/mushroom-rl.git
cd mushroom-rl
pip install -e .
cd ..
```

---

## Step 6: Download DLO-Lab Assets

The rope meshes, textures, and pre-built knot configurations are **not** in
the GitHub repo. Download them from the link in the README and unzip under
`genesis/assets/`:

```bash
# After downloading dlo-lab-assets.zip from the SharePoint link in README:
unzip dlo-lab-assets.zip -d genesis/assets/
# Verify:
ls genesis/assets/dlo-lab/
# Expected: meshes/  ropes/  textures/  urdf/  exrs/
```

**Without these assets, all 8 benchmark tasks will fail at scene construction.**
The error looks like:
```
FileNotFoundError: [Errno 2] No such file or directory: 'dlo-lab/ropes/ropec.npy'
```

---

## Step 7: Verify Installation

### 7.1 Genesis core sanity check

```bash
python -c "
import genesis as gs
gs.init(seed=0, precision='64', logging_level='info', backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1e-3, substeps=5),
    rod_options=gs.options.RODOptions(damping=10.0),
    show_viewer=False,
)
rope = scene.add_entity(
    material=gs.materials.ROD.Base(segment_radius=0.005, E=1e5, G=1e4),
    morph=gs.morphs.ParameterizedRod(
        type='rod', n_vertices=20, interval=0.01, axis='x', pos=(0, 0, 0.3),
    ),
)
scene.build(n_envs=1)
rope.set_fixed_states(fixed_ids=[0, 1])
for _ in range(100):
    scene.step()
print('Genesis ROD solver: OK')
"
```

### 7.2 Quick example

```bash
python examples/quick_example.py --output_folder ./output_test
# Expected: writes output_test/view_0_best.mp4
```

### 7.3 Check all 8 tasks load

Run the verification script (contributed in this PR — see `experiments/verify_tasks.py`):

```bash
cd experiments
python verify_tasks.py
```

Expected output:
```
[OK] coiling      — scene built, 1 reset step passed
[OK] gathering    — scene built, 1 reset step passed
[OK] lifting      — scene built, 1 reset step passed
[OK] separation   — scene built, 1 reset step passed
[OK] slingshot    — scene built, 1 reset step passed
[OK] unknotting   — scene built, 1 reset step passed
[OK] wiring_post  — scene built, 1 reset step passed
[OK] wrapping     — scene built, 1 reset step passed
All 8 tasks verified.
```

---

## Step 8: Run a Benchmark Task

All training commands are run from the `experiments/` directory:

```bash
cd experiments

# PPO on the coiling task (quick test: 2 epochs, 4 envs)
python rl/continuous_run.py \
    --task coiling \
    --n_envs 4 \
    --n_steps 10 \
    --n_traj 1 \
    --n_epochs 2 \
    --exp_name smoke-test \
    --seed 0
```

Logs and checkpoints are written to `logs/coiling/smoke-test/`.

---

## Known Issues and Workarounds

### Issue: `DISPLAY` / OpenGL error in headless mode

```
pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
```

Fix — run with a virtual framebuffer:
```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python ...
```
Or use `show_viewer=False` (already the default in training scripts).

### Issue: CUDA out of memory with large `n_envs`

Genesis allocates per-environment memory at `scene.build()`. For the default
unknotting task (`n_envs=100`), you need ≥ 16 GB VRAM. Scale down if needed:
```bash
python rl/continuous_run.py --task unknotting --n_envs 16 ...
```

### Issue: Taichi kernel compilation is slow on first run

Genesis uses Taichi for GPU kernel compilation. The first `scene.build()` call
per task compiles CUDA kernels and may take 2–5 minutes. Subsequent runs use
cached kernels (stored in `~/.cache/taichi/`). This is normal.

### Issue: `pin-pink` install fails (pinocchio dependency)

`pin-pink` requires `pinocchio` (robot kinematics library). Install via conda:
```bash
conda install -c conda-forge pinocchio -y
pip install pin-pink
```

### Issue: `lerobot` install pulls incompatible `datasets` version

If `lerobot` conflicts with `datasets` or `huggingface_hub`, install with:
```bash
pip install lerobot --no-deps
pip install datasets huggingface_hub  # install separately at compatible versions
```

---

## Known Bugs (found during verification, fixed in this PR)

### Bug 1: `Train_Env_Gathering` crashes with "Genesis already initialized"

**Symptom:** When running multiple tasks sequentially (e.g. in `verify_tasks.py`),
`Train_Env_Gathering` raises:
```
genesis.GenesisException: Genesis already initialized.
```
**Cause:** `env_gathering.py.__init__` calls `gs.init()` unconditionally, while all
other envs rely on the base class guard (`if not gs._initialized`).

**Fix** (applied in this PR to `experiments/envs/env_gathering.py`):
```python
# Before:
gs.init(seed=0, precision="64", ...)

# After:
if not gs._initialized:
    gs.init(seed=0, precision="64", ...)
```

### Bug 2: `verify_tasks.py` cannot find `Train_Env_Wiring_Post`

**Symptom:**
```
AttributeError: module 'envs.env_wiring_post' has no attribute 'Train_Env_Wiring_Post'
```
**Cause:** The class in `env_wiring_post.py` is named `Train_Env_Wiring_post`
(lowercase `p`), inconsistent with the `_Post` used in `verify_tasks.py`'s lookup
table. All other entry scripts (`rudinppo.py`, `sac.py`, `cmaes.py`) correctly use
`Train_Env_Wiring_post`.

**Fix** (applied in this PR to `experiments/verify_tasks.py`): corrected the lookup
to `'Train_Env_Wiring_post'`.

---

## Dependency Version Reference

Tested working combination (as of 2026-06):

| Package | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| genesis-world | 1.0.0 (this repo) |
| mushroom-rl | 2.0.0rc1 |
| omegaconf | 2.3.x |
| cma | 4.x |

> **Note:** Genesis emits a `UserWarning: 'torch<2.8.0' is not supported` with
> PyTorch 2.5.x. All 8 tasks still run correctly; the warning is cosmetic.

---

## Getting Help

- DLO-Lab paper: [arXiv:2606.04206](https://arxiv.org/pdf/2606.04206)
- Genesis docs: [genesis-world.readthedocs.io](https://genesis-world.readthedocs.io)
- Contact: xjay2018@gmail.com (Junyi Cao)
- File issues in this repo with the label `installation`
