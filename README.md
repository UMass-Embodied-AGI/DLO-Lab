<p align="center">

  <h1 align="center">DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics</h1>
  <p align="center">
    Junyi Cao
    ·
    Yian Wang
    ·
    Ziyan Xiong
    ·
    Chunru Lin
    ·
    Zhehuan Chen
    ·
    Chuang Gan

  </p>
  <h3 align="center"> <a href="https://arxiv.org/pdf/2606.04206" target="_blank"> Paper </a> &nbsp;&nbsp; | &nbsp;&nbsp; <a href="https://dlo-lab-26.github.io/" target="_blank"> Website </a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="teaser.gif" alt="DLO-Lab teaser" width="85%">
  </a>
</p>

<p style="text-align:justify">
DLO-Lab supports various material properties for deformable linear objects and their interactions with rigid and soft bodies, allowing for versatile manipulations.
</p>

**Please consider citing our paper if you find it interesting or helpful to your research.**
```
@inproceedings{Cao_2026_DLOLab,
    author      = {Cao, Junyi and Wang, Yian and Xiong, Ziyan and Lin, Chunru and Chen, Zhehuan and Gan, Chuang},
    title       = {DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics},
    booktitle   = {International Conference on Machine Learning (ICML)},
    year        = {2026}
}
```


---


### Installation

1. Create conda environment
```
conda create -n dlolab python=3.12
conda activate dlolab
```

2. Install [PyTorch](https://pytorch.org/get-started/locally/) following the official instructions

3. Install DLO-Lab via PyPI
```
git clone https://github.com/UMass-Embodied-AGI/DLO-Lab.git
cd DLO-Lab
pip install -e ".[dlo-lab,dev]"
```

4. Install ffmpeg
```
conda install "ffmpeg=7" -c conda-forge
```

5. Install Mushroom-RL
```
git clone https://github.com/XJay18/mushroom-rl.git
cd mushroom-rl
pip install -e .
```

6. Download DLO-Lab assets from [this link](https://umass-my.sharepoint.com/:u:/g/personal/junyicao_umass_edu/IQBoxs6PdxUuQJiBAbJrfiDQAfwE6oDKd_NZ8j2H8qH-OC0?e=oFir1a), unzip it under `genesis/assets`, make sure you have this folder: `genesis/assets/dlo-lab`

---

### Quick Example

DLO-Lab is built on top of [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world). A deformable linear object is added to a scene as an entity with a `ROD` material and a `Rod/ParameterizedRod` morph, then simulated by stepping the scene. The snippet below simulates two ropes (each with one end fixed) colliding with two fixed capsules:

```python
import genesis as gs

gs.init(seed=0, precision="64", logging_level="info", backend=gs.gpu)

# Create a scene with the rod solver activated
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1e-3, substeps=5),
    rod_options=gs.options.RODOptions(damping=10.0, angular_damping=5.0),
    show_viewer=True,
)

# Ground plane
scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.1),
    morph=gs.morphs.Plane(fixed=True),
)

# Two thin ropes. E/G control bending/twisting stiffness
v1 = scene.add_entity(
    material=gs.materials.ROD.Base(segment_radius=0.005, E=1e5, G=1e4),
    morph=gs.morphs.ParameterizedRod(
        type="rod", n_vertices=100, interval=0.01, axis="x",
        pos=(0.5, 0.5, 0.3), euler=(0.0, 0.0, 15.0),
    ),
)
v2 = scene.add_entity(
    material=gs.materials.ROD.Base(segment_radius=0.005, E=1e5, G=1e4),
    morph=gs.morphs.ParameterizedRod(
        type="rod", n_vertices=80, interval=0.01, axis="x",
        pos=(0.55, 0.43, 0.4), euler=(0.0, 0.0, 0.0),
    ),
)

# Two thick, short capsules for the ropes to collide with
b1 = scene.add_entity(
    material=gs.materials.ROD.Base(segment_radius=0.02),
    morph=gs.morphs.ParameterizedRod(
        type="rod", n_vertices=3, interval=0.1, axis="x",
        pos=(0.75, 0.435, 0.25), euler=(0.0, 0.0, -75.0),
    ),
)
b2 = scene.add_entity(
    material=gs.materials.ROD.Base(segment_radius=0.02),
    morph=gs.morphs.ParameterizedRod(
        type="rod", n_vertices=3, interval=0.1, axis="x",
        pos=(1.05, 0.435, 0.25), euler=(0.0, 0.0, -75.0),
    ),
)

# Build, then pin selected vertices: one end of each rope, and the capsules
scene.build(n_envs=1)
v1.set_fixed_states(fixed_ids=[0, 1])
v2.set_fixed_states(fixed_ids=[78, 79])
b1.set_fixed_states(fixed_ids=[0, 1, 2])
b2.set_fixed_states(fixed_ids=[0, 1, 2])

# Roll out the simulation
for _ in range(2000):
    scene.step()
```

For complete, runnable scripts — grasping with a Franka arm, coupling rods with elastic/sand/liquid bodies, and differentiable system identification — see [`examples/`](examples).

```bash
# Run the quick example, saving a video
python examples/quick_example.py --output_folder ./output
```

---

### DLO-Lab Benchmark

DLO-Lab benchmarks eight manipulation tasks — `coiling`, `gathering`, `lifting`, `separation`, `slingshot`, `unknotting`, `wiring_post`, `wrapping` — across several learning and trajectory-optimization methods. You could also extend beyond these tasks with the provided APIs.

**Note: All commands below are run from the `experiments/` directory.**

#### Training

Ready-to-run launch scripts live in [`experiments/scripts/`](experiments/scripts), organized one folder per method, one script per task:

| Method | Scripts | Entry point |
| --- | --- | --- |
| PPO | [`scripts/ppo/`](experiments/scripts/ppo) | `rl/rudinppo.py` |
| SAC | [`scripts/sac/`](experiments/scripts/sac) | `rl/sac.py` |
| SHAC | [`scripts/shac/`](experiments/scripts/shac) | `rl/shac.py` |
| SAPO | [`scripts/sapo/`](experiments/scripts/sapo) | `rl/shac.py` |
| CMA-ES | [`scripts/cmaes/`](experiments/scripts/cmaes) | `trajopt/cmaes.py` |
| Gradient Descent | [`scripts/gd/`](experiments/scripts/gd) | `trajopt/gd.py` |

For example, to train PPO on the coiling task:

```bash
cd experiments
bash scripts/ppo/coiling.sh
```

Logs and checkpoints are written to `logs/<task>/<exp_name>/`, where `<exp_name>` is set by the `--exp_name` flag inside each script (e.g. `rudin-01` for PPO, `sac-01` for SAC, `cmaes-01` for CMA-ES).

#### Visualizing a trained policy (test)

To roll out and render a trained policy/trajectory, re-run the **entry point directly** with the same `--task` and `--exp_name`, plus the method-specific test flag. This evaluates the checkpoint and saves an animation to `logs/<task>/<exp_name>/`. Add `-r` for ray-traced rendering, or `--gui` for the interactive viewer.

```bash
# RL methods — load a checkpoint with --test
python rl/rudinppo.py --task coiling --exp_name rudin-01 --test best   # PPO: "best" or an epoch number
python rl/sac.py      --task coiling --exp_name sac-01   --test best   # SAC: "best" or an epoch number
python rl/shac.py     --task coiling --exp_name shac-01  --test best_shac.pkl   # SHAC/SAPO: checkpoint filename

# Trajectory optimization — replay the optimized trajectory
python trajopt/cmaes.py --task coiling --n_steps $n_steps --exp_name cmaes-01 --vis_traj logs/coiling/cmaes-01/best_qpos.npy  # $n_steps should match the training script
python trajopt/gd.py    --task coiling --n_steps $n_steps --exp_name gd-01    --test best_traj.pt # $n_steps should match the training script
```

> RL checkpoints are saved under `logs/<task>/<exp_name>/ckpts/`; `--test best` loads the best-return checkpoint, while `--test <epoch>` loads a specific one. For trajectory optimization, point `--vis_traj` / `--test` at the saved trajectory file under the run's log directory.

#### Ray-traced rendering

If you want to use the ray-traced renderer, please follow [the instruction](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html#photo-realistic-rendering-with-luisa-deprecating) to install LuisaRenderer first.

#### Adding a new task

Please follow [this guide](experiments/envs/README.md) for adding a new manipulation environment.

---

Please feel free to contact Junyi Cao (xjay2018@gmail.com) if you have any questions about this work.
