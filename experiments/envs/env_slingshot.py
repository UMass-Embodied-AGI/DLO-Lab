import genesis as gs
import os
import time
import torch
import numpy as np
from envs.base import Train_Env
from utils.controller import RobotControllerPink

asset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "genesis", "assets", "dlo-lab")


class Train_Env_Slingshot(Train_Env):
    def __init__(self, config):
        super().__init__(config=config)

        # Use the counter to track how many steps have been taken for each env
        self._env_step_counter = np.array([0] * self.n_envs)

        initial_gripper_qpos = np.load(os.path.join(asset_dir, "target_pos", "slingshot_pregrasp_qpos.npy"))
        self.initial_gripper_qpos = torch.tensor(initial_gripper_qpos, dtype=gs.tc_float)

    def construct_scene(self, camera):
        plane = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
                visualization=not self.raytracer
            ),
        )

        if self.raytracer:
            table = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="dlo-lab/meshes/wooden_table.glb",
                    pos=(-0., 0, -0.799418 * 4),
                    euler=(0, 0, 90),
                    scale=4,
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Default()
            )

        segment_radius = 0.01
        self.rope = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                K=8e5,
                E=1e5,
                G=0,
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=12,
                interval=0.02,
                axis="x",
                pos=(0.0, 0.0, 0.21),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="dlo-lab/textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.b1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0.22, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.sphere = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=200, coup_friction=0.02,
            ),
            morph=gs.morphs.Sphere(
                radius=0.02,
                pos=(0.12, 0.06, 0.2),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 1.0)
            )
        )

        self.cube = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=20, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 0.23, 0.22),
                size=(0.08, 0.08, 0.08),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.7, 0.7, 1.0)
            )
        )

        self.table = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 1.0, 0.09),
                size=(0.8, 1.9, 0.18),
                euler=(0, 0, 0),
                fixed=True,
            ),
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.33, -0.65, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget', 'camera_link'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        if camera:
            self.construct_cameras()

        self.construct_extra_cameras()
        self.scene.build(n_envs=self.n_envs, env_spacing=(10, 10), center_envs_at_origin=False)

        # candidate mode grasp proposal
        self.control_idx = [6]

        # Construct controller
        for f in [self.franka1]:
            f.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 30, 30]),
            )
            f.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
            )
            f.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
                np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
            )
        self._ef1 = self.franka1.get_link("panda_grasptarget")

        # move to pre-grasp pose
        x0 = 0.12
        y0 = -0.1
        z0 = 0.216
        open_gap = 0.03

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_pos=(x0, y0, z0),
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(1, -1.4, 1.5), up=(0, 0, 1),
            lookat=(0.12, 0.2, 0.18), fov=24, GUI=False
        ))
        if not self.raytracer:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(-0.15, 1.4, 1.2), up=(0, 0, 1),
                lookat=(0.12, 0.25, 0.35), fov=33, GUI=False
            ))

        self.cameras = cameras

    def reward(self):
        # [n_envs, 3]
        cube_pos = self.cube.get_pos().cpu().numpy()

        rewards = []
        for i in range(self.n_envs):
            cube_pos_y = cube_pos[i, 1]

            # NOTE: New: max y position is 5, min y position is 0
            cube_pos_y = min(max(cube_pos_y, 0), 5)

            # we want the cube to be as far as possible in +y direction
            rewards.append(cube_pos_y)

        return rewards

    def loss_criterion(self, state):
        cube_pos = self.cube.get_pos()   # [n_envs, 3]
        cube_pos_y = cube_pos[:, 1]        # [n_envs,]
        cube_pos_y = torch.clamp(cube_pos_y, min=0.0, max=5.0)

        return -cube_pos_y  # maximize y position

    def _randomize_sphere_and_cube_positions(self, envs_idx=None):
        if not self.randomization_initialized:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        # Randomize sphere and cube x positions from -0.03 to 0.03
        x_pos = np.random.uniform(-0.03, 0.03, size=(len(envs_idx), 1)) # [n_envs_idx, 1]
        y_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        z_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        rel_pos = np.hstack((x_pos, y_pos, z_pos))  # [n_envs_idx, 3]
        rel_pos = torch.tensor(rel_pos, dtype=gs.tc_float)
        self.sphere.set_pos(rel_pos, envs_idx=envs_idx, relative=True)
        self.cube.set_pos(rel_pos, envs_idx=envs_idx, relative=True)

    def reset(self, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)
        self._randomize_masses([self.rope], envs_idx=envs_idx)
        self._randomize_bending_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_stretching_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_sphere_and_cube_positions(envs_idx=envs_idx)

        envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

        for f in [self.franka1]:
            f.set_qpos(
                np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                envs_idx=envs_idx_
            )

        self.rope.set_fixed_states(fixed_ids=[0, 1, 10, 11])

        self.c1.set_initial_dofs_position(self.initial_gripper_qpos, False, envs_idx=envs_idx)

        force = -1.0
        self.c1.control_robot(force, force, g_dof_use_force=True)
        for i in range(100):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

        self._env_step_counter[np.asarray(envs_idx_)] = 0

    def eval_traj(self, trajs, **kwargs):
        """
        Evaluate trajectories using cumulative reward.
        """
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 6 == 0 and dof // 6 == n_ctrl, (
            f"dof must be 6 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        n_steps_sub = self._cmaes_n_steps_sub
        if kwargs.get("qpos", None) is None:
            self.qpos_seq = np.zeros((n_steps * n_steps_sub, self.n_envs, len(self.control_idx) * 9))
            self.use_qpos = False
        else:
            self.qpos_seq = kwargs["qpos"]
            self.use_qpos = True

        self.reset()

        steps_interval = self.steps_interval
        total_micro_steps = int(n_steps * n_steps_sub)
        if total_micro_steps <= 0:
            # Degenerate case: no steps → everyone "survives"; defer to env reward (or -100 if NaN)
            rewards = np.asarray(self.reward(), dtype=np.float32)
            rewards[np.isnan(rewards)] = -100.0
            return rewards.astype(np.float32)

        # Per-env status
        alive = np.ones((self.n_envs,), dtype=bool)              # True until first failure (collision or NaN)
        ever_nan = np.zeros((self.n_envs,), dtype=bool)          # True if verts ever became NaN

        reward_accum = np.zeros((self.n_envs,), dtype=np.float32)

        forward_elapsed = 0.0

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            newly_nan = nan_now & alive
            if newly_nan.any():
                ever_nan[newly_nan] = True
                alive[newly_nan] = False

            # Early exit if everyone is already NaN
            if ever_nan.all():
                break

            # If no env is alive anymore, we can stop
            if not alive.any():
                break

            # Prepare interpolation to targets for this macro-step
            delta = trajs[:, i].reshape(self.n_envs, 6)            # (n_envs, 6), n_ctrl == 1!
            delta = torch.tensor(delta, dtype=gs.tc_float)

            n_intervals_per_substep = steps_interval // n_steps_sub

            for j in range(n_steps_sub):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta[~alive, :] = 0.0

                alpha = 1 / n_steps_sub
                dxyz = alpha * delta[:, :3]
                drot = alpha * delta[:, 3:]

                if self.use_qpos:
                    qpos = self.qpos_seq[i * n_steps_sub + j]
                    qpos = torch.tensor(qpos, dtype=gs.tc_float)
                    self.c1.robot.control_dofs_position(qpos[..., :-2], self.c1.motors_dof)
                    gripper_arg = torch.tensor([[-3, -3]] * self.scene.n_envs)
                    self.c1.robot.control_dofs_force(gripper_arg, self.c1.fingers_dof)

                    self.c1.draw_debug_point(dxyz, min_z=0.03)
                else:
                    qpos = self.c1.control_robot(
                        -3.0, -3.0, g_dof_use_force=True,
                        dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], di=drot[:, 0], dj=drot[:, 1], dk=drot[:, 2], min_z=0.03
                    )
                    self.qpos_seq[i * n_steps_sub + j] = qpos.cpu().numpy()

                forward_start_time = time.time()
                for k in range(n_intervals_per_substep):
                    self.scene.step()

                    if (k + j * n_intervals_per_substep) % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)
                forward_elapsed += time.time() - forward_start_time

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

                # Collect reward here
                is_final_step = (i == n_steps - 1) and (j == n_steps_sub - 1)
                should_release = is_final_step & alive
                if should_release.any():
                    release_envs_idx = np.where(should_release)[0].tolist()
                    self.c1.control_robot(0.08, 0.08, envs_idx=release_envs_idx)
                    for s in range(200):
                        self.scene.step()
                        if s % 10 == 0:
                            for cid, cam in enumerate(self.cameras):
                                img = cam.render()[0]
                                self.frames[cid].append(img)

                substep_rewards_pre = np.asarray(self.reward(), dtype=np.float32)
                substep_rewards_nan = np.isnan(substep_rewards_pre)

                substep_rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
                failed = ~alive | substep_rewards_nan
                substep_rewards[failed] = 0.0
                substep_rewards[~failed] = substep_rewards_pre[~failed]
                reward_accum += substep_rewards

        if not self.use_qpos:
            self.qpos_seq = self.qpos_seq.transpose(1, 0, 2)  # (n_envs, n_steps * n_steps_sub, n_dofs)
            self.qpos_seq = self.qpos_seq.astype(np.float32)

        out = dict()
        out['forward_time'] = forward_elapsed
        out['final_reward'] = reward_accum.astype(np.float32)
        out['cum_reward'] = reward_accum.astype(np.float32)

        return out

    def compute_observation(self):
        verts_rope = self.rope.get_all_verts_tc()                   # (n_envs, n_verts, 3)
        obs_rope_pos = verts_rope.reshape(self.n_envs, -1).to(torch.float32)

        vels_rope = self.rope.get_all_vels_tc()                     # (n_envs, n_verts, 3)
        obs_rope_vel = vels_rope.reshape(self.n_envs, -1).to(torch.float32)

        obs_rope = torch.cat([obs_rope_pos, obs_rope_vel], dim=1)

        sphere_pos = self.sphere.get_pos().to(torch.float32)
        sphere_vel = self.sphere.get_vel().to(torch.float32)

        cube_pos = self.cube.get_pos().to(torch.float32)
        cube_vel = self.cube.get_vel().to(torch.float32)

        ef_pos = self.c1.ef.get_pos().to(torch.float32)
        ef_quat = self.c1.ef.get_quat().to(torch.float32)
        joint_qpos = self.c1.robot.get_dofs_position(self.c1.motors_dof).to(torch.float32)

        obs = torch.cat([obs_rope, sphere_pos, sphere_vel, cube_pos, cube_vel, ef_pos, ef_quat, joint_qpos], dim=1)
        return obs

    def step_all(self, env_mask, action):
        """ Used in MushroomRL """
        # Accept torch or numpy; operate and return torch for torch backend
        if isinstance(action, np.ndarray):
            action = torch.tensor(action)
        else:
            action = torch.as_tensor(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        if isinstance(env_mask, np.ndarray):
            env_mask_np = torch.tensor(env_mask, dtype=torch.bool)
        else:
            env_mask_np = torch.as_tensor(env_mask, dtype=torch.bool)

        assert action.shape == (self.n_envs, self._act_dim), \
            f"Expected action shape {(self.n_envs, self._act_dim)}, got {action.shape}"

        # Track failure states and absorbing flags (only track masked envs)
        absorbing = np.zeros((self.n_envs,), dtype=bool)
        tracked = env_mask_np.clone().cpu().numpy()
        alive = tracked.copy()

        action = action.to(torch.float32)
        action = action * self._act_magnitude
        action = torch.clamp(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        action_xyz = action[:, :self._act_dim // 2]
        action_rot = action[:, self._act_dim // 2:]

        action_xyz_norm = torch.linalg.norm(action_xyz, dim=1, keepdim=True)
        scale = torch.ones_like(action_xyz_norm)
        over = action_xyz_norm > self._l2_limit
        scale[over] = self._l2_limit / (action_xyz_norm[over] + gs.EPS)
        action_xyz = action_xyz * scale

        # Check NaNs BEFORE micro-stepping this macro-step
        verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
        nan_now = np.isnan(verts_rope).any(axis=(1, 2))
        newly_nan = nan_now & alive
        if newly_nan.any():
            # Failure occurs before any micro-step of this macro-step
            absorbing[newly_nan] = True
            alive[newly_nan] = False

        n_steps_sub = self._steps_interval_split
        n_intervals_per_substep = self._steps_per_action // n_steps_sub

        for j in range(n_steps_sub):
            if not (alive & tracked).any():
                break

            # NOTE: Do not move already-failed envs
            action_xyz[~alive, :] = 0.0
            action_rot[~alive, :] = 0.0

            alpha = 1 / n_steps_sub
            dxyz = alpha * action_xyz
            drot = alpha * action_rot

            qpos = self.c1.control_robot(
                -3.0, -3.0, g_dof_use_force=True,
                dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], di=drot[:, 0], dj=drot[:, 1], dk=drot[:, 2], min_z=0.03
            )

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                absorbing[newly_nan_after] = True
                alive[newly_nan_after] = False

        # release gripper
        is_final_step = self._env_step_counter >= self._horizon - 1
        should_release = is_final_step & (~absorbing)
        if should_release.any():
            release_envs_idx = np.where(should_release)[0].tolist()
            self.c1.control_robot(0.08, 0.08, envs_idx=release_envs_idx)
            for s in range(200):
                self.scene.step()
                if s % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

        self._env_step_counter[~absorbing] += 1

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
        failed = absorbing | env_rewards_nan
        rewards[failed] = 0.0
        rewards[~failed] = env_rewards[~failed]
        rewards = torch.as_tensor(rewards).reshape((self.n_envs,))
        absorbing = torch.as_tensor(absorbing).reshape((self.n_envs,))

        next_obs = self.compute_observation()

        return next_obs, rewards, absorbing, [{}] * self.n_envs

    def step_diff_rl(self, env_mask, action):
        if action.ndim == 1:
            action = action.unsqueeze(0)

        assert action.shape == (self.n_envs, self._act_dim), \
            f"Expected action shape {(self.n_envs, self._act_dim)}, got {action.shape}"

        # Track failure states and absorbing flags (only track masked envs)
        tracked = env_mask.clone()
        alive = env_mask.clone()

        action = action.to(torch.float32)
        action = action * self._act_magnitude
        action = torch.clamp(action, self._act_low, self._act_high)

        action_norm = torch.linalg.norm(action, dim=1, keepdim=True)
        scale = torch.ones_like(action_norm)
        over = action_norm > self._l2_limit
        scale[over] = self._l2_limit / (action_norm[over] + gs.EPS)
        action = action * scale

        # Check NaNs BEFORE micro-stepping this macro-step
        verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
        nan_now = np.isnan(verts_rope).any(axis=(1, 2))
        newly_nan = nan_now & alive.cpu().numpy()
        if newly_nan.any():
            # Failure occurs before any micro-step of this macro-step
            alive[newly_nan] = False

        n_steps_sub = self._steps_interval_split
        n_intervals_per_substep = self._steps_per_action // n_steps_sub

        for j in range(n_steps_sub):
            if not (alive & tracked).any():
                break

            alpha = 1 / n_steps_sub
            dxyz = alpha * action

            for i_g in range(len(self.control_idx)):
                controller = getattr(self, f"c{i_g+1}")
                qpos = controller.control_robot(
                    0, 0,
                    dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], min_z=0.03
                )

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive.cpu().numpy()
            if newly_nan_after.any():
                alive[newly_nan_after] = False

        # release gripper
        alive_np = alive.cpu().numpy()
        is_final_step = self._env_step_counter >= self._horizon - 1
        should_release = is_final_step & alive_np
        if should_release.any():
            release_envs_idx = np.where(should_release)[0].tolist()
            self.c1.control_robot(0.08, 0.08, envs_idx=release_envs_idx)
            for s in range(200):
                self.scene.step()
                if s % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

        self._env_step_counter[alive_np] += 1

        # Compute loss
        state = self.rope.get_state()
        loss = self.loss_criterion(state)   # (n_envs,)

        # Collect reward here
        substep_rewards_pre = torch.as_tensor(self.reward(), dtype=torch.float32)
        substep_rewards_nan = torch.isnan(substep_rewards_pre)

        substep_rewards = torch.full((self.n_envs,), 0.0, dtype=torch.float32)
        failed = ~alive | substep_rewards_nan
        substep_rewards[failed] = 0.0
        substep_rewards[~failed] = substep_rewards_pre[~failed]

        return loss, alive, substep_rewards, state.s_global

    def train_one_iter_gd(self, it=None, max_it=None, skip_backward=False):
        self.qpos_seq = np.zeros((self._n_steps, self.n_envs, len(self.control_idx) * 9))
        self.use_qpos = False

        self.reset()

        loss = 0.
        total_horizon = 100
        horizon_ids = list()

        alive = np.ones((self.n_envs,), dtype=bool)
        reward_accum = np.zeros((self.n_envs,), dtype=np.float32)
        
        forward_elapsed = 0.0

        for i in range(self._n_steps):
            local_loss = 0.
            n_horizons = self.steps_interval
            # Do not move already-failed envs
            delta_pos = self.c.pre_apply_grad(stage_idx=i)

            step_qpos = list()
            for i_g in range(len(self.control_idx)):
                controller = getattr(self, f"c{i_g+1}")
                qpos = controller.control_robot(
                    0, 0,
                    dx=delta_pos[:, i_g, 0],
                    dy=delta_pos[:, i_g, 1],
                    dz=delta_pos[:, i_g, 2],
                    min_z=self._min_z,
                )
                step_qpos.append(qpos.cpu().numpy())
            step_qpos = np.concatenate(step_qpos, axis=-1) # (n_envs, n_ctrl * 9)
            self.qpos_seq[i] = step_qpos

            forward_start_time = time.time()
            for j in range(n_horizons):
                self.scene.step()
                if j % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)
            forward_elapsed += time.time() - forward_start_time

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                alive[newly_nan_after] = False

            # Compute loss
            state = self.rope.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            loss_c = self.loss_criterion(state) + self.loss_above_plane(state)
            local_loss += loss_c.mean()

            scale = self.scale_array[i]
            loss += scale * local_loss

            self.c.post_check(stage_idx=i, alive=torch.as_tensor(alive))

            # Collect reward here
            is_final_step = (i == self._n_steps - 1)
            should_release = is_final_step & alive
            if should_release.any():
                release_envs_idx = np.where(should_release)[0].tolist()
                self.c1.control_robot(0.08, 0.08, envs_idx=release_envs_idx)
                for s in range(200):
                    self.scene.step()
                    if s % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)

            # Collect reward here
            substep_rewards_pre = np.asarray(self.reward(), dtype=np.float32)
            substep_rewards_nan = np.isnan(substep_rewards_pre)

            substep_rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
            failed = ~alive | substep_rewards_nan
            substep_rewards[failed] = 0.0
            substep_rewards[~failed] = substep_rewards_pre[~failed]
            reward_accum += substep_rewards

        out = dict()
        out['loss'] = loss.item()
        out['reward'] = reward_accum

        backward_elapsed = 0.0

        if not skip_backward:

            backward_start_time = time.time()
            loss.backward()
            backward_elapsed = time.time() - backward_start_time

            for stage_idx, horizon_idx in enumerate(horizon_ids):
                self.c.gather_grad(
                    stage_idx=stage_idx,
                    horizon_idx=horizon_idx,
                    cur_step=it,
                    max_step=max_it,
                    lr=self.lr,
                    lr_min=self.lr_min,
                )
        
        out['forward_time'] = forward_elapsed
        out['backward_time'] = backward_elapsed
        out['lr'] = self.c._lr

        out['qpos_seq'] = self.qpos_seq  # (n_steps, n_envs, n_ctrl * 9)
        return out
