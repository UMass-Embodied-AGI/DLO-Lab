## Adding a New Manipulation Environment

This folder holds the manipulation task environments. Every task is a subclass of [`Train_Env`](base.py) that lives in its own `env_<task>.py` file (e.g. [`env_coiling.py`](env_coiling.py), [`env_wrapping.py`](env_wrapping.py)). The base class implements everything shared across tasks — scene/sim setup, the optimizer harness entry points, domain randomization, and video saving — and leaves the task-specific pieces as `raise NotImplementedError()` hooks for you to fill in.

The **same env class** is reused by four different optimizers, but each one calls a different subset of hooks and is wired up through a different entry script:

| Optimizer | What it is | Entry script | Needs gradients? |
| --- | --- | --- | --- |
| **RL** | PPO / SAC | [`../rl/rudinppo.py`](../rl/rudinppo.py), [`../rl/sac.py`](../rl/sac.py) | No |
| **DiffRL** | SHAC / SAPO | [`../rl/shac.py`](../rl/shac.py) | Yes |
| **CMA-ES** | Gradient-free trajectory optimization | [`../trajopt/cmaes.py`](../trajopt/cmaes.py) | No |
| **GD** | Gradient-descent trajectory optimization | [`../trajopt/gd.py`](../trajopt/gd.py) | Yes |

This guide first covers the **common foundation** that every env needs, then gives the extra steps **per optimizer**. Throughout we add a new task called `myenv`.

The cleanest starting point is to **copy the closest existing task** — [`env_coiling.py`](env_coiling.py) is the smallest single-rope, single-arm example and implements every hook below:

```bash
cp envs/env_coiling.py envs/env_myenv.py   # then rename the class to Train_Env_Myenv
```

### Contents

- [Common foundation (required for every optimizer)](#common-foundation-required-for-every-optimizer)
- [To build an env for RL (PPO / SAC)](#to-build-an-env-for-rl-ppo--sac)
- [To build an env for DiffRL (SHAC / SAPO)](#to-build-an-env-for-diffrl-shac--sapo)
- [To build an env for CMA-ES](#to-build-an-env-for-cma-es)
- [To build an env for GD (gradient-descent trajectory optimization)](#to-build-an-env-for-gd-gradient-descent-trajectory-optimization)
- [Running & visualizing](#running--visualizing)
- [Checklist](#checklist)

---

### Common foundation (required for every optimizer)

Regardless of which optimizer you target, your `Train_Env_Myenv(Train_Env)` must implement these shared hooks:

| Method | Purpose |
| --- | --- |
| `construct_scene(self, camera)` | Add the plane, rope, obstacles/targets, and robot(s). **Must set** `self.rope` (primary `ROD.Base` rod), `self.control_idx` (list of grasped rod-vertex indices), and one controller per control point (`self.c1`, `self.c2`, …). Call `self.construct_cameras()` when `camera` is true and finish with `self.scene.build(...)`. |
| `construct_cameras(self)` | Append `self.scene.add_camera(...)` views to `self.cameras` (used for video). |
| `reward(self)` | Per-env scalar reward, NumPy/list of shape `(n_envs,)`, **higher = better**, non-negative. Defines the task for RL & CMA-ES. |
| `reset(self, envs_idx=None)` | `self.scene.reset(...)`, re-seat the robot, (re)attach grasped vertices, and apply the base `self._randomize_*` helpers. |

Two attributes set in `construct_scene` drive the observation/action sizing for the whole class:

- `self.rope` → `self.rope.n_vertices` sizes the observation.
- `self.control_idx` → number of grasp points; action dimension is derived from it.

Everything below is **on top of** this foundation.

---

### To build an env for RL (PPO / SAC)

**1. Implement these extra hooks:**

| Method | Purpose |
| --- | --- |
| `compute_observation(self)` | Return the observation tensor `(n_envs, obs_dim)` — typically rope positions + velocities, object states, and end-effector pose concatenated. |
| `step_all(self, env_mask, action)` | Apply the action (6 DoF per control point: `xyz` + rotation), micro-step the sim, and return `(next_obs, rewards, absorbing, info)`. |

**2. Action / observation specs.** RL uses `init_rl_env`, which sets `act_dim = len(control_idx) * 6` and

```
_obs_dim = (rope.n_vertices + n_additional_obj) * 6 + len(control_idx) * 14
```

(`* 6` = position + velocity per point; `* 14` = pose + joints per gripper.) The length your `compute_observation` returns **must** equal `_obs_dim`, so `n_additional_obj` has to match the object states you concatenate.

**3. Register** in both [`../rl/rudinppo.py`](../rl/rudinppo.py) (PPO) and [`../rl/sac.py`](../rl/sac.py) (SAC):

```python
from envs.env_myenv import Train_Env_Myenv       # add import
...
env_dict = { ..., "myenv": Train_Env_Myenv }     # add to env_dict
```

Then add a per-task branch in the same file that calls `init_rl_env` with the right `n_additional_obj`:

```python
elif task == "myenv":
    mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound,
                    n_additional_obj=<N>, steps_interval_split=steps_interval_split, debug=args.gui)
```

**4. Launch script:**

```bash
cp ../scripts/ppo/coiling.sh ../scripts/ppo/myenv.sh   # edit --task myenv
# or scripts/sac/ for SAC
```

---

### To build an env for DiffRL (SHAC / SAPO)

DiffRL backpropagates through the simulator, so the scene runs with `requires_grad=True` (the entry script sets this automatically).

**1. Implement these extra hooks:**

| Method | Purpose |
| --- | --- |
| `compute_observation(self)` | Same as RL — observation tensor `(n_envs, obs_dim)`. |
| `loss_criterion(self, state)` | **Differentiable** loss `(n_envs,)` computed from `state.pos` (a Torch tensor), **lower = better**. Keep it consistent with `reward` (loss ≈ negative reward). |
| `step_diff_rl(self, env_mask, action)` | Differentiable step; return `(loss, alive, rewards, s_global)`. Action is position-only here. |

**2. Action specs.** DiffRL uses `init_diff_rl_env`, which sets `act_dim = len(control_idx) * 3` (translation only). This differs from RL's `* 6`: when this project was developed, Genesis did not yet support differentiability through the rigid solver, so the control signal is the gradient w.r.t. the grasped **rope vertices** rather than the gripper pose — hence translation-only, no rotation DoF. The observation formula is identical to RL.

**3. Register** in [`../rl/shac.py`](../rl/shac.py) (used by both SHAC and SAPO):

```python
from envs.env_myenv import Train_Env_Myenv
...
env_dict = { ..., "myenv": Train_Env_Myenv }
```

`shac.py` initializes the env via `mdp.init_diff_rl_env(...)`; make sure its `n_additional_obj` matches your `compute_observation`.

**4. Launch script:**

```bash
cp ../scripts/shac/coiling.sh ../scripts/shac/myenv.sh   # edit --task myenv
# or scripts/sapo/ for SAPO
```

---

### To build an env for CMA-ES

CMA-ES is gradient-free; it rolls out sampled open-loop trajectories and keeps the best. The scene runs with `requires_grad=False`.

**1. Implement this extra hook:**

| Method | Purpose |
| --- | --- |
| `eval_traj(self, trajs, **kwargs)` | Roll out the fixed trajectories `trajs` of shape `(n_envs, n_steps, 6 * len(control_idx))`, and return a dict with `final_reward`, `cum_reward`, and `forward_time` (each `(n_envs,)`). See `Train_Env_Coiling.eval_traj` for the failure/NaN-handling template. |

No `compute_observation` or `step_*` is needed (there is no policy network). `init_cmaes_env` only stores `n_steps_sub`.

**2. Register** in [`../trajopt/cmaes.py`](../trajopt/cmaes.py):

```python
from envs.env_myenv import Train_Env_Myenv
...
env_dict = { ..., "myenv": Train_Env_Myenv }
```

**3. Launch script:**

```bash
cp ../scripts/cmaes/coiling.sh ../scripts/cmaes/myenv.sh   # edit --task myenv
```

---

### To build an env for GD (gradient-descent trajectory optimization)

GD optimizes a single trajectory by backpropagating the loss through the simulator (`requires_grad=True`), using a `TrajOptimController` (`self.c`) built via `construct_traj_optim`.

**1. Implement these extra hooks:**

| Method | Purpose |
| --- | --- |
| `loss_criterion(self, state)` | Differentiable loss `(n_envs,)` from `state.pos` (same as DiffRL). |
| `train_one_iter_gd(self, it, max_it, ...)` | One forward rollout driving `self.c.pre_apply_grad / post_check`, accumulate `loss` (typically `loss_criterion + self.loss_above_plane`), then `loss.backward()` and `self.c.gather_grad(...)`. See `Train_Env_Coiling.train_one_iter_gd` for the template. |

`gd.py` initializes the env with `init_gd_env(...)` followed by `construct_traj_optim(...)`, which creates `self.c` from `self.rope` and `self.control_idx` — so the common foundation must set both.

**2. Register** in [`../trajopt/gd.py`](../trajopt/gd.py):

```python
from envs.env_myenv import Train_Env_Myenv
...
env_dict = { ..., "myenv": Train_Env_Myenv }
```

**3. Launch script:**

```bash
cp ../scripts/gd/coiling.sh ../scripts/gd/myenv.sh   # edit --task myenv
```

---

### Running & visualizing

Run training from the `experiments/` directory, e.g. `bash scripts/ppo/myenv.sh`. See the repository [README](../../README.md#benchmark-example) for how to train each method and how to visualize a trained policy/trajectory (the `--test` / `--vis_traj` flags).

---

### Checklist

- [ ] `envs/env_myenv.py` with `class Train_Env_Myenv(Train_Env)`.
- [ ] **Foundation:** `construct_scene` sets `self.rope`, `self.control_idx`, a controller (`self.c1`, …) and calls `self.scene.build(...)`; plus `construct_cameras`, `reward`, `reset`.
- [ ] **RL:** `compute_observation` + `step_all`; registered in `rudinppo.py` & `sac.py` with a matching `init_rl_env` branch; launch scripts under `scripts/ppo|sac/`.
- [ ] **DiffRL:** `compute_observation` + `loss_criterion` + `step_diff_rl`; registered in `shac.py`; launch scripts under `scripts/shac|sapo/`.
- [ ] **CMA-ES:** `eval_traj`; registered in `cmaes.py`; launch script under `scripts/cmaes/`.
- [ ] **GD:** `loss_criterion` + `train_one_iter_gd`; registered in `gd.py`; launch script under `scripts/gd/`.
- [ ] `compute_observation` length matches `_obs_dim` (keep `n_additional_obj` in sync).
