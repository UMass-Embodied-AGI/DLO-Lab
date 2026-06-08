import genesis as gs
import os
import time
import torch
import numpy as np
from envs.base import Train_Env
from utils.controller import (
    rod_vertex_attached_to_gripper,
    rod_vertex_detached_from_gripper,
    RobotControllerPink,
)

asset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "genesis", "assets", "dlo-lab")


class Train_Env_Wiring_post(Train_Env):
    def __init__(self, config):
        super().__init__(config=config)

        self.center_pos = np.array([0.198, 0.198, 0.01], dtype=gs.np_float)
        self.center_pos_tc = torch.tensor([0.198, 0.198, 0.01], dtype=gs.tc_float)

        self.target_pos = np.load(os.path.join(asset_dir, "target_pos", "wiring_post_finalpos.npy"))
        print(f'Loaded target pos from "wiring_post_finalpos.npy", shape = {self.target_pos.shape}')

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
                    pos=(-0., 0, -0.799418 * 2),
                    euler=(0, 0, 0),
                    scale=2,
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
                E=1e4,
                G=1e3,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=30,
                interval=0.02,
                axis="x",
                pos=(0.3, 0.05, 0.012),
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

        self.stick1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7, rho=50
            ),
            morph=gs.morphs.Cylinder(
                radius=0.02,
                height=0.04,
                pos=(0.28, 0.14, 0.02),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.stick2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7, rho=50
            ),
            morph=gs.morphs.Cylinder(
                radius=0.02,
                height=0.04,
                pos=(0.1, 0.275, 0.02),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        # no visual but could still collide with the rope
        self.stick1_hidden = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.02,
                segment_mass=1.0,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=3,
                interval=0.02,
                axis="z",
                pos=(0.28, 0.14, -0.02),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4, 0)
            )
        )

        # no visual but could still collide with the rope
        self.stick2_hidden = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.02,
                segment_mass=1.0,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=3,
                interval=0.02,
                axis="z",
                pos=(0.1, 0.275, -0.02),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4, 0)
            )
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.2, -0.25, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget', 'camera_link'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        if camera:
            self.construct_cameras()

        gripper_geom_indices = list()
        for gi in self.franka1.get_link("panda_leftfinger")._geoms:
            gripper_geom_indices.append(gi.idx)
        for gi in self.franka1.get_link("panda_rightfinger")._geoms:
            gripper_geom_indices.append(gi.idx)

        self.gripper_geom_indices = gripper_geom_indices
        self.construct_extra_cameras()
        self.scene.build(n_envs=self.n_envs, env_spacing=(10, 10), center_envs_at_origin=False)

        # candidate mode grasp proposal
        self.control_idx = [3]

        # Construct controller
        for f in [self.franka1]:
            f.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
            )
            f.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
            )
            f.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
                np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
            )
        self._ef1 = self.franka1.get_link("panda_grasptarget")

        open_gap = 0.01

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.2, 1.2, 1.5), up=(0, 0, 1),
            lookat=(0.3, 0.2, 0), fov=30, GUI=False
        ))
        if not self.raytracer:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(0.3, 1.2, 0.7), up=(0, 0, 1),
                lookat=(0.3, 0.2, 0), fov=30, GUI=False
            ))

        self.cameras = cameras

    def reward(self):
        # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()
        assert verts_batch.shape[1] == self.target_pos.shape[0]
        # [n_envs, n_verts, 3]
        target_batch = np.tile(self.target_pos[None, :, :], (self.n_envs, 1, 1))

        # Pairwise distances between rope vertices and target positions
        # [n_envs, n_verts, n_verts, 3]
        diff = verts_batch[:, :, None, :] - target_batch[:, None, :, :]
        D = np.linalg.norm(diff, axis=-1)  # [n_envs, n_verts, n_verts]

        # For each target point, find the closest rope vertex
        min_dist_to_target = D.min(axis=2)
        min_dist_to_verts = D.min(axis=1)

        # z-axis penalty to encourage rope beyond target height
        z_penalty = np.maximum(verts_batch[:, :, 2] - 0.04, 0.0)  # [n_envs, n_verts]
        z_penalty = z_penalty.mean(axis=1)

        # compute distance from rope center to target center, xy only
        rope_center = verts_batch[:, 15, :2]
        target_center = self.center_pos[None, :2]
        center_diff = rope_center - target_center
        center_dist = np.linalg.norm(center_diff, axis=-1)  # [n_envs,]
        # center_dist below 0.02 is considered "good"
        center_dist = np.clip(center_dist - 0.02, min=0.0)

        rewards = - min_dist_to_target.mean(axis=1) - min_dist_to_verts.mean(axis=1)
        rewards -= z_penalty
        rewards -= center_dist * 5.0
        rewards = np.exp(rewards)

        return rewards.tolist()

    def loss_criterion(self, state):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos
        target = torch.tensor(self.target_pos, dtype=verts_batch.dtype, device=verts_batch.device)

        # Pairwise distances between rope vertices and target positions
        diff = verts_batch[:, :, None, :] - target[None, None, :, :]  # (n_envs, n_verts, n_verts, 3)
        D = torch.norm(diff, dim=-1)  # (n_envs, n_verts, n_verts)

        min_dist_to_target, _ = torch.min(D, dim=2)  # (n_envs, n_verts)
        min_dist_to_verts, _ = torch.min(D, dim=1)   # (n_envs, n_verts)

        z_penalty = torch.clamp(verts_batch[:, :, 2] - 0.04, min=0.0)  # (n_envs, n_verts)
        z_penalty = torch.mean(z_penalty, dim=1)  # (n_envs,)

        rope_center = verts_batch[:, 15, :2]  # (n_envs, 2)
        target_center = self.center_pos_tc[None, :2]  # (1, 2)
        center_diff = rope_center - target_center  # (n_envs, 2)
        center_dist = torch.norm(center_diff, dim=-1)  # (n_envs,)
        center_dist = torch.clamp(center_dist - 0.02, min=0.0)

        loss = torch.mean(min_dist_to_target, dim=1) + torch.mean(min_dist_to_verts, dim=1)
        loss += z_penalty
        loss += center_dist * 5.0

        return loss

    def reset(self, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)
        self._randomize_positions([self.rope], envs_idx=envs_idx)
        self._randomize_masses([self.rope], envs_idx=envs_idx)
        self._randomize_radii([self.rope], envs_idx=envs_idx)
        self._randomize_bending_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_twisting_stiffness([self.rope], envs_idx=envs_idx)

        envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

        for f in [self.franka1]:
            f.set_qpos(
                np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                envs_idx=envs_idx_
            )

        for i in envs_idx_:
            rod_vertex_detached_from_gripper(self.rope, self.control_idx[0], envs_idx=i)

        init_pos = self.rope.get_all_verts()
        init_pos_f1 = init_pos[:, self.control_idx[0]]

        qpos = self.c1.set_initial_position(init_pos=init_pos_f1, envs_idx=envs_idx)
        if self.cmaes_initialized or self.gd_initialized:
            if not self.use_qpos:
                qpos = qpos.cpu().numpy()
                self.qpos_seq[0] = qpos

        for i in envs_idx_:
            rod_vertex_attached_to_gripper(self.rope, self.control_idx[0], self._ef1, envs_idx=i)

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
            self.qpos_seq = np.zeros((n_steps * n_steps_sub + 1, self.n_envs, len(self.control_idx) * 9))
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
        first_fail_step = np.full((self.n_envs,), total_micro_steps, dtype=np.int32)  # micro-step index of first failure

        reward_accum = np.zeros((self.n_envs,), dtype=np.float32)

        forward_elapsed = 0.0

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            newly_nan = nan_now & alive
            if newly_nan.any():
                step_at_nan = i * n_steps_sub
                step_at_nan = max(1, step_at_nan)
                first_fail_step[newly_nan] = step_at_nan
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
                    qpos = self.qpos_seq[i * n_steps_sub + j + 1]
                    qpos = torch.tensor(qpos, dtype=gs.tc_float)
                    self.c1.robot.control_dofs_position(qpos[..., :-2], self.c1.motors_dof)
                    self.c1.robot.control_dofs_position(qpos[..., -2:], self.c1.fingers_dof)

                    self.c1.draw_debug_point(dxyz, min_z=0.03)
                else:
                    qpos = self.c1.control_robot(
                        0, 0,
                        dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], di=drot[:, 0], dj=drot[:, 1], dk=drot[:, 2], min_z=0.03
                    )
                    self.qpos_seq[i * n_steps_sub + j + 1] = qpos.cpu().numpy()

                forward_start_time = time.time()
                for k in range(n_intervals_per_substep):
                    self.scene.step()

                    if (k + j * n_intervals_per_substep) % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)
                forward_elapsed += time.time() - forward_start_time

                global_step = i * n_steps_sub + (j + 1)
                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    first_fail_step[newly_nan_after] = np.minimum(first_fail_step[newly_nan_after], global_step)
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

                # Collect reward here
                substep_rewards_pre = np.asarray(self.reward(), dtype=np.float32)
                substep_rewards_nan = np.isnan(substep_rewards_pre)

                substep_rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
                failed = ~alive | substep_rewards_nan
                substep_rewards[failed] = 0.0
                substep_rewards[~failed] = substep_rewards_pre[~failed]
                reward_accum += substep_rewards

        # Compute final state rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)
        # Compose final rewards
        final = np.empty((n_envs,), dtype=np.float32)
        failed = ~alive  # failed due to collision or NaN during rollout
        survived = alive
        # Failed: reward = survival_ratio (counts both collision and NaN cases)
        if failed.any():
            survival_ratio = first_fail_step.astype(np.float32) / float(total_micro_steps)
            final[failed] = survival_ratio[failed] - 100
        # Survived full rollout: take env reward; if it's NaN, clamp to -100
        final[survived] = env_rewards[survived]
        if env_rewards_nan.any():
            final[env_rewards_nan] = -100.0

        if not self.use_qpos:
            self.qpos_seq = self.qpos_seq.transpose(1, 0, 2)  # (n_envs, n_steps * n_steps_sub + 1, n_dofs)
            self.qpos_seq = self.qpos_seq.astype(np.float32)

        out = dict()
        out['forward_time'] = forward_elapsed
        out['final_reward'] = final.astype(np.float32)
        out['cum_reward'] = reward_accum.astype(np.float32)

        return out

    def compute_observation(self):
        verts_rope = self.rope.get_all_verts_tc()                   # (n_envs, n_verts, 3)
        obs_rope_pos = verts_rope.reshape(self.n_envs, -1).to(torch.float32)

        vels_rope = self.rope.get_all_vels_tc()                     # (n_envs, n_verts, 3)
        obs_rope_vel = vels_rope.reshape(self.n_envs, -1).to(torch.float32)

        obs_rope = torch.cat([obs_rope_pos, obs_rope_vel], dim=1)

        stick1_pos = self.stick1.get_pos().to(torch.float32)
        stick1_vel = self.stick1.get_vel().to(torch.float32)
        stick2_pos = self.stick2.get_pos().to(torch.float32)
        stick2_vel = self.stick2.get_vel().to(torch.float32)

        stick_obs = torch.cat([stick1_pos, stick1_vel, stick2_pos, stick2_vel], dim=1)

        ef1_pos = self.c1.ef.get_pos().to(torch.float32)
        ef1_quat = self.c1.ef.get_quat().to(torch.float32)
        joint1_qpos = self.c1.robot.get_dofs_position(self.c1.motors_dof).to(torch.float32)
        c1_obs = torch.cat([ef1_pos, ef1_quat, joint1_qpos], dim=1)

        obs = torch.cat([obs_rope, stick_obs, c1_obs], dim=1)

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
                0, 0,
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
        self.qpos_seq = np.zeros((self._n_steps + 1, self.n_envs, len(self.control_idx) * 9))
        self.use_qpos = False

        self.reset()

        loss = 0.
        total_horizon = 0
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
            self.qpos_seq[i + 1] = step_qpos

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
