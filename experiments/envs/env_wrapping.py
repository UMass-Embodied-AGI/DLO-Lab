import genesis as gs
import time
import torch
import numpy as np
from envs.base import Train_Env
from utils.controller import (
    rod_vertex_attached_to_gripper,
    rod_vertex_detached_from_gripper,
    RobotControllerPink,
)


class Train_Env_Wrapping(Train_Env):
    def __init__(self, config):
        super().__init__(config=config)

        # initial total length
        self.control_dist_init = self.rope.get_total_length()   # (n_envs,)

        print(f'Initial total length: {self.control_dist_init[0]:.4f}')

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
                K=1e5,
                E=1e4,
                G=0,
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=50,
                radius=0.14,
                axis="x",
                pos=(0.6, 0, 0.012),
                euler=(0.0, 0.0, 0.0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="dlo-lab/textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.post1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7, rho=50
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.04,
                pos=(0.45, -0.115, 0.02),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.post2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7, rho=50
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.04,
                pos=(0.45, 0.115, 0.02),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.post3 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7, rho=50
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.04,
                pos=(0.25, 0.0, 0.02),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.05, 0.45, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget', 'camera_link'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        self.franka2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.05, -0.45, 0),
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
        for gi in self.franka2.get_link("panda_leftfinger")._geoms:
            gripper_geom_indices.append(gi.idx)
        for gi in self.franka2.get_link("panda_rightfinger")._geoms:
            gripper_geom_indices.append(gi.idx)

        self.gripper_geom_indices = gripper_geom_indices
        self.construct_extra_cameras()
        self.scene.build(n_envs=self.n_envs, env_spacing=(10, 10), center_envs_at_origin=False)

        # candidate mode grasp proposal
        self.control_idx = [17, 33]

        # Construct controller
        for f in [self.franka1, self.franka2]:
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
        self._ef2 = self.franka2.get_link("panda_grasptarget")

        open_gap = 0.01

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_quat=(0, 0.9239, 0.3827, 0),
            initial_gripper_gap=open_gap,
        )

        self.c2 = RobotControllerPink(
            self.scene, self.franka2, self._ef2,
            initial_quat=(0, 0.3827, 0.9239, 0),
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(1.0, -0.8, 0.8), up=(0, 0, 1),
            lookat=(0.4, 0., 0), fov=30, GUI=False
        ))
        if not self.raytracer:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(2, 0, 1.2), up=(0, 0, 1),
                lookat=(0.3, 0., 0), fov=24, GUI=False
            ))

        self.cameras = cameras

    @staticmethod
    def _winding_angle_loss_np(rope_verts, posts_pos):
        """
        Computes the scaled winding angle loss for a single environment using NumPy.

        Args:
            rope_verts: (N_verts, 3) Array of rope vertices [x, y, z]
            posts_pos:  (N_posts, 3) Array of post positions [x, y, z]

        Returns:
            loss: float (0.0 = fully wrapped, 1.0 = not wrapped)
        """
        # rope_xy: (N, 2)
        rope_xy = rope_verts[:, :2]

        # posts_xy: (P, 2)
        posts_xy = posts_pos[:, :2]

        # We broaden dimensions to create a matrix of shape (N, P, 2)
        # rope becomes (N, 1, 2)
        # posts becomes (1, P, 2)
        # rel_pos[i, j] is the vector from Post j to Vertex i
        rel_pos = rope_xy[:, np.newaxis, :] - posts_xy[np.newaxis, :, :] # (N, P, 2)

        # v_curr: (N, P, 2)
        # v_next: (N, P, 2) (shifted cyclically along the vertex axis 0)
        v_curr = rel_pos
        v_next = np.roll(rel_pos, shift=-1, axis=0)

        # Cross Product (2D): x1*y2 - x2*y1
        cross = v_curr[..., 0] * v_next[..., 1] - v_curr[..., 1] * v_next[..., 0]

        # Dot Product (2D): x1*x2 + y1*y2
        dot   = v_curr[..., 0] * v_next[..., 0] + v_curr[..., 1] * v_next[..., 1]

        # arctan2(y, x) -> returns values in (-pi, pi]
        angle_steps = np.arctan2(cross, dot)  # Shape (N, P)

        # Sum along axis 0 (the vertex dimension)
        total_winding = np.sum(angle_steps, axis=0)  # Shape (P,)

        # Convert radians to "turns" (0 to 1 scale)
        turns = total_winding / (2 * np.pi)

        # Use abs() so that -1 turn (CW) and +1 turn (CCW) are both valid
        turns_mag = np.abs(turns)

        # Target is 1.0 full wrap.
        # If turns_mag is 0 (outside), loss is (0-1)^2 = 1.0
        # If turns_mag is 1 (wrapped), loss is (1-1)^2 = 0.0
        loss = np.mean((turns_mag - 1.0) ** 2)

        return float(loss)

    @staticmethod
    def _winding_angle_loss_tc(rope_verts, posts_pos):
        """
        Computes the scaled winding angle loss for a single environment using PyTorch tensors.

        Args:
            rope_verts: (B, N_verts, 3)  The rope geometry
            posts_pos:  (B, N_posts, 3)  The positions of the posts

        Returns:
            loss: (B,)  The winding angle loss per batch
        """
        assert rope_verts.ndim == 3
        assert posts_pos.ndim == 3
        assert rope_verts.shape[0] == posts_pos.shape[0]

        # rope: (B, N, 1, 2)
        rope_xy = rope_verts[:, :, :2].unsqueeze(2)

        # posts: (B, 1, P, 2)
        posts_xy = posts_pos[:, :, :2].unsqueeze(1)

        # rel_pos: (B, N, P, 2) -> The vector from Post P to Vertex N
        rel_pos = rope_xy - posts_xy

        # v_curr: (B, N, P, 2)
        # v_next: (B, N, P, 2) (shifted along vertex dim)
        v_curr = rel_pos
        v_next = torch.roll(rel_pos, shifts=-1, dims=1)

        # Cross Product (2D): x1*y2 - x2*y1
        cross = v_curr[..., 0] * v_next[..., 1] - v_curr[..., 1] * v_next[..., 0]

        # Dot Product (2D): x1*x2 + y1*y2
        dot   = v_curr[..., 0] * v_next[..., 0] + v_curr[..., 1] * v_next[..., 1]

        # atan2(y, x)
        angle_steps = torch.atan2(cross, dot)  # Shape (B, N, P)

        total_winding = torch.sum(angle_steps, dim=1)  # Shape (B, P)

        turns = total_winding / (2 * torch.pi)

        # We want exactly 1 turn (or -1 turn)
        # Using abs() allows either Clockwise or Counter-Clockwise wrapping
        turns_mag = torch.abs(turns)

        loss = torch.mean((turns_mag - 1.0) ** 2, dim=1)  # Shape (B,)

        return loss

    def _control_proximity_loss(self, verts_batch, posts_all):
        """
        Auxiliary loss to encourage control vertices to approach posts.
        This provides strong gradient signals directly to the control vertices.

        Args:
            verts_batch: (B, N_verts, 3)  The rope geometry
            posts_all:   (B, N_posts, 3)  The positions of the posts

        Returns:
            loss: (B,)  The proximity loss per batch
        """
        # Extract control vertices [n_envs, 2, 3]
        control_verts = verts_batch[:, self.control_idx, :]

        # Distance from each control vertex to each post [n_envs, 2, 3]
        dists = torch.cdist(control_verts, posts_all)

        # For each control vertex, find distance to closest post [n_envs, 2]
        min_dists = torch.min(dists, dim=2)[0]

        # Average over control vertices [n_envs]
        loss = torch.mean(min_dists, dim=1)

        return loss

    def reward(self):
        # [n_envs, 3]
        post1_pos_batch = self.post1.get_pos().cpu().numpy()
        post2_pos_batch = self.post2.get_pos().cpu().numpy()
        post3_pos_batch = self.post3.get_pos().cpu().numpy()
        # [n_envs, 3, 3]
        posts_all = np.stack([post1_pos_batch, post2_pos_batch, post3_pos_batch], axis=1)
        # # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()

        rewards = []
        for i in range(self.n_envs):
            verts = verts_batch[i]
            post1_pos = post1_pos_batch[i]
            post2_pos = post2_pos_batch[i]
            post3_pos = post3_pos_batch[i]

            dist_post1 = np.linalg.norm(verts - post1_pos, axis=1)
            min_dists_post1 = np.min(dist_post1)    # initial: 0.0509
            # we do not penalize distances below 0.015m because this is the radius of the post
            dist_penalty_post1 = np.maximum(min_dists_post1 - 0.015, 0)

            dist_post2 = np.linalg.norm(verts - post2_pos, axis=1)
            min_dists_post2 = np.min(dist_post2)    # initial: 0.0509
            # we do not penalize distances below 0.015m because this is the radius of the post
            dist_penalty_post2 = np.maximum(min_dists_post2 - 0.015, 0)

            dist_post3 = np.linalg.norm(verts - post3_pos, axis=1)
            min_dists_post3 = np.min(dist_post3)    # initial: 0.2104
            # we do not penalize distances below 0.015m because this is the radius of the post
            dist_penalty_post3 = np.maximum(min_dists_post3 - 0.015, 0)

            winding_loss = self._winding_angle_loss_np(rope_verts=verts, posts_pos=posts_all[i])
            winding_reward = 1.0 - winding_loss

            # winding_reward in [0, 1] -> 1: fully wrapped, 0: not wrapped at all
            # dist: we want to minimize the closest distance to each post
            reward = winding_reward - (dist_penalty_post1 + dist_penalty_post2 + dist_penalty_post3)

            rewards.append(reward)

        return rewards

    def loss_criterion(self, state):
        verts_batch = state.pos

        # [n_envs, 3]
        post1_pos_batch = self.post1.get_pos()
        post2_pos_batch = self.post2.get_pos()
        post3_pos_batch = self.post3.get_pos()
        # [n_envs, 3, 3]
        posts_all = torch.stack([post1_pos_batch, post2_pos_batch, post3_pos_batch], dim=1)

        dists_post1 = torch.cdist(verts_batch, post1_pos_batch.unsqueeze(1))  # [n_envs, n_verts, 1]
        min_dists_post1 = torch.min(dists_post1.squeeze(-1), dim=1)[0]    # [n_envs]
        dist_penalty_post1 = torch.maximum(min_dists_post1 - 0.015, torch.zeros_like(min_dists_post1))

        dists_post2 = torch.cdist(verts_batch, post2_pos_batch.unsqueeze(1))  # [n_envs, n_verts, 1]
        min_dists_post2 = torch.min(dists_post2.squeeze(-1), dim=1)[0]    # [n_envs]
        dist_penalty_post2 = torch.maximum(min_dists_post2 - 0.015, torch.zeros_like(min_dists_post2))

        dists_post3 = torch.cdist(verts_batch, post3_pos_batch.unsqueeze(1))  # [n_envs, n_verts, 1]
        min_dists_post3 = torch.min(dists_post3.squeeze(-1), dim=1)[0]    # [n_envs]
        dist_penalty_post3 = torch.maximum(min_dists_post3 - 0.015, torch.zeros_like(min_dists_post3))

        winding_loss = self._winding_angle_loss_tc(rope_verts=verts_batch, posts_pos=posts_all)

        # Control proximity loss - provides strong gradients to control vertices
        control_proximity = self._control_proximity_loss(verts_batch, posts_all)

        loss = (winding_loss +
                (dist_penalty_post1 + dist_penalty_post2 + dist_penalty_post3) +
                0.1 * control_proximity)  # Weight=0.1 to balance with winding loss
        return loss     # (n_envs,)

    def reset(self, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)
        self._randomize_positions([self.rope], envs_idx=envs_idx)
        self._randomize_masses([self.rope], envs_idx=envs_idx)
        self._randomize_radii([self.rope], envs_idx=envs_idx)
        self._randomize_bending_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_stretching_stiffness([self.rope], envs_idx=envs_idx)

        envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

        for f in [self.franka1, self.franka2]:
            f.set_qpos(
                np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                envs_idx=envs_idx_
            )

        for i in envs_idx_:
            rod_vertex_detached_from_gripper(self.rope, self.control_idx[0], envs_idx=i)
            rod_vertex_detached_from_gripper(self.rope, self.control_idx[1], envs_idx=i)

        init_pos = self.rope.get_all_verts()
        init_pos_f1 = init_pos[:, self.control_idx[0]]
        init_pos_f2 = init_pos[:, self.control_idx[1]]

        qpos1 = self.c1.set_initial_position(init_pos=init_pos_f1, envs_idx=envs_idx)
        qpos2 = self.c2.set_initial_position(init_pos=init_pos_f2, envs_idx=envs_idx)
        if self.cmaes_initialized or self.gd_initialized:
            if not self.use_qpos:
                qpos1 = qpos1.cpu().numpy()
                qpos2 = qpos2.cpu().numpy()
                qpos = np.concatenate([qpos1, qpos2], axis=-1)
                self.qpos_seq[0] = qpos

        for i in envs_idx_:
            rod_vertex_attached_to_gripper(self.rope, self.control_idx[0], self._ef1, envs_idx=i)
            rod_vertex_attached_to_gripper(self.rope, self.control_idx[1], self._ef2, envs_idx=i)

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
            delta = trajs[:, i].reshape(self.n_envs, 2 * 6)            # (n_envs, 2 * 6), n_ctrl == 2
            # first half: translation
            delta1_xyz = torch.tensor(delta[:, 0:3], dtype=gs.tc_float)
            delta2_xyz = torch.tensor(delta[:, 3:6], dtype=gs.tc_float)
            # second half: rotation
            delta1_rot = torch.tensor(delta[:, 6:9], dtype=gs.tc_float)
            delta2_rot = torch.tensor(delta[:, 9:12], dtype=gs.tc_float)

            n_intervals_per_substep = steps_interval // n_steps_sub

            for j in range(n_steps_sub):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta1_xyz[~alive, :] = 0.0
                delta2_xyz[~alive, :] = 0.0
                delta1_rot[~alive, :] = 0.0
                delta2_rot[~alive, :] = 0.0

                alpha = 1 / n_steps_sub
                dxyz1 = alpha * delta1_xyz
                drot1 = alpha * delta1_rot
                dxyz2 = alpha * delta2_xyz
                drot2 = alpha * delta2_rot

                if self.use_qpos:
                    qpos = self.qpos_seq[i * n_steps_sub + j + 1]
                    qpos = torch.tensor(qpos, dtype=gs.tc_float)
                    qpos1, qpos2 = torch.split(qpos, qpos.shape[0] // 2)
                    self.c1.robot.control_dofs_position(qpos1[..., :-2], self.c1.motors_dof)
                    self.c2.robot.control_dofs_position(qpos2[..., :-2], self.c2.motors_dof)
                    self.c1.robot.control_dofs_position(qpos1[..., -2:], self.c1.fingers_dof)
                    self.c2.robot.control_dofs_position(qpos2[..., -2:], self.c2.fingers_dof)

                    self.c1.draw_debug_point(dxyz1, min_z=0.04)
                    self.c2.draw_debug_point(dxyz2, min_z=0.04)
                else:
                    qpos1 = self.c1.control_robot(
                        0, 0,
                        dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.04
                    )
                    qpos2 = self.c2.control_robot(
                        0, 0,
                        dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.04
                    )
                    qpos1 = qpos1.cpu().numpy()
                    qpos2 = qpos2.cpu().numpy()
                    # (n_envs, n_dofs * 2)
                    qpos = np.concatenate([qpos1, qpos2], axis=-1)
                    self.qpos_seq[i * n_steps_sub + j + 1] = qpos

                forward_start_time = time.time()
                for k in range(n_intervals_per_substep):
                    self.scene.step()

                    if (k + j * n_intervals_per_substep) % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)
                forward_elapsed += time.time() - forward_start_time

                global_step = i * n_steps_sub + (j + 1)
                # Post-step: detect stretching failures
                if self.control_dist_init is not None:
                    # (n_envs,)
                    control_dist_now = self.rope.get_geodesic_distance(
                        self.control_idx[0], self.control_idx[1]
                    )
                    # 20% stretch allowed
                    stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.2
                    newly_stretched = stretched_between_ctrl & alive
                    if newly_stretched.any():
                        first_fail_step[newly_stretched] = np.minimum(first_fail_step[newly_stretched], global_step)
                        alive[newly_stretched] = False

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
                substep_rewards[~failed] = substep_rewards_pre[~failed] + 1.0
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

        vels_rope = self.rope.get_all_vels_tc()                   # (n_envs, n_verts, 3)
        obs_rope_vel = vels_rope.reshape(self.n_envs, -1).to(torch.float32)

        obs_rope = torch.cat([obs_rope_pos, obs_rope_vel], dim=1)

        post1_pos = self.post1.get_pos().to(torch.float32)
        post1_vel = self.post1.get_vel().to(torch.float32)
        post2_pos = self.post2.get_pos().to(torch.float32)
        post2_vel = self.post2.get_vel().to(torch.float32)
        post3_pos = self.post3.get_pos().to(torch.float32)
        post3_vel = self.post3.get_vel().to(torch.float32)

        post_obs = torch.cat([post1_pos, post1_vel, post2_pos, post2_vel, post3_pos, post3_vel], dim=1)

        ef1_pos = self.c1.ef.get_pos().to(torch.float32)
        ef1_quat = self.c1.ef.get_quat().to(torch.float32)
        joint1_qpos = self.c1.robot.get_dofs_position(self.c1.motors_dof).to(torch.float32)
        c1_obs = torch.cat([ef1_pos, ef1_quat, joint1_qpos], dim=1)

        ef2_pos = self.c2.ef.get_pos().to(torch.float32)
        ef2_quat = self.c2.ef.get_quat().to(torch.float32)
        joint2_qpos = self.c2.robot.get_dofs_position(self.c2.motors_dof).to(torch.float32)
        c2_obs = torch.cat([ef2_pos, ef2_quat, joint2_qpos], dim=1)

        obs = torch.cat([obs_rope, post_obs, c1_obs, c2_obs], dim=1)
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

        # Split action for two controllers: first half for controller 1, second half for controller 2
        action1_xyz = action[:, :self._act_dim // 4]
        action2_xyz = action[:, self._act_dim // 4:self._act_dim // 2]
        action1_rot = action[:, self._act_dim // 2:self._act_dim // 2 + self._act_dim // 4]
        action2_rot = action[:, self._act_dim // 2 + self._act_dim // 4:]

        # Apply L2 limit to translation actions
        action1_xyz_norm = torch.linalg.norm(action1_xyz, dim=1, keepdim=True)
        scale1 = torch.ones_like(action1_xyz_norm)
        over1 = action1_xyz_norm > self._l2_limit
        scale1[over1] = self._l2_limit / (action1_xyz_norm[over1] + gs.EPS)
        action1_xyz = action1_xyz * scale1

        action2_xyz_norm = torch.linalg.norm(action2_xyz, dim=1, keepdim=True)
        scale2 = torch.ones_like(action2_xyz_norm)
        over2 = action2_xyz_norm > self._l2_limit
        scale2[over2] = self._l2_limit / (action2_xyz_norm[over2] + gs.EPS)
        action2_xyz = action2_xyz * scale2

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
            action1_xyz[~alive, :] = 0.0
            action1_rot[~alive, :] = 0.0
            action2_xyz[~alive, :] = 0.0
            action2_rot[~alive, :] = 0.0

            alpha = 1 / n_steps_sub
            dxyz1 = alpha * action1_xyz
            drot1 = alpha * action1_rot
            dxyz2 = alpha * action2_xyz
            drot2 = alpha * action2_rot

            qpos1 = self.c1.control_robot(
                0, 0,
                dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.04
            )
            qpos2 = self.c2.control_robot(
                0, 0,
                dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.04
            )

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            # Post-step: detect stretching failures
            if self.control_dist_init is not None:
                # (n_envs,)
                control_dist_now = self.rope.get_geodesic_distance(
                    self.control_idx[0], self.control_idx[1]
                )
                # 20% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.2
                newly_stretched = stretched_between_ctrl & alive
                if newly_stretched.any():
                    absorbing[newly_stretched] = True
                    alive[newly_stretched] = False

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
        rewards[~failed] = env_rewards[~failed] + 1.0
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

        action1 = action[:, :self._act_dim // 2]
        action2 = action[:, self._act_dim // 2:]

        action1_norm = torch.linalg.norm(action1, dim=1, keepdim=True)
        scale = torch.ones_like(action1_norm)
        over = action1_norm > self._l2_limit
        scale[over] = self._l2_limit / (action1_norm[over] + gs.EPS)
        action1 = action1 * scale

        action2_norm = torch.linalg.norm(action2, dim=1, keepdim=True)
        scale2 = torch.ones_like(action2_norm)
        over2 = action2_norm > self._l2_limit
        scale2[over2] = self._l2_limit / (action2_norm[over2] + gs.EPS)
        action2 = action2 * scale2

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
            dxyz = [alpha * action1, alpha * action2]

            for i_g in range(len(self.control_idx)):
                controller = getattr(self, f"c{i_g+1}")
                qpos = controller.control_robot(
                    0, 0,
                    dx=dxyz[i_g][:, 0], dy=dxyz[i_g][:, 1], dz=dxyz[i_g][:, 2], min_z=0.04
                )

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            # Post-step: detect stretching failures
            if self.control_dist_init is not None:
                # (n_envs,)
                control_dist_now = self.rope.get_geodesic_distance(
                    self.control_idx[0], self.control_idx[1]
                )
                # 20% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.20
                newly_stretched = stretched_between_ctrl & alive.cpu().numpy()
                if newly_stretched.any():
                    alive[newly_stretched] = False

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
        substep_rewards[~failed] = substep_rewards_pre[~failed] + 1.0

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

            # Post-step: detect stretching failures
            if self.control_dist_init is not None:
                # (n_envs,)
                control_dist_now = self.rope.get_geodesic_distance(
                    self.control_idx[0], self.control_idx[1]
                )
                # 20% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.2
                newly_stretched = stretched_between_ctrl & alive
                if newly_stretched.any():
                    alive[newly_stretched] = False

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
            substep_rewards[~failed] = substep_rewards_pre[~failed] + 1.0
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
