import genesis as gs
import time
import torch
import numpy as np
from envs.base import Train_Env
from utils.controller import RobotControllerPink


class Train_Env_Gathering(Train_Env):
    def __init__(self, config):
        gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        )

        scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(
                dt=1e-3,
                substeps=5,
                requires_grad=config.requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                # Skip the rigid solver's differentiable backward (legacy no-op behavior). The robot acts as a
                # non-differentiable actuator; the learning signal comes from the rope vertex gradients. Avoids
                # NaN gradients from the forward-kinematics quaternion backward.
                skip_backward=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.2, -0.5, -0.1),
                upper_bound=(0.8, 0.5, 0.9),
                grid_density=100,
            ),
            rod_options=gs.options.RODOptions(
                damping=15.0,
                angular_damping=10.0,
                n_pbd_iters=10,
            ),
            show_viewer=config.GUI,
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path='dlo-lab/exrs/brown_photostudio_02_4k.exr',
                        image_color=(0.6, 0.6, 0.6),
                        encoding='linear',
                    ),
                ),
                env_radius=15.0,
                env_euler=(0, 0, 180),
                lights=[],
            ) if config.raytracer else gs.renderers.Rasterizer(),
        )
        super().__init__(config=config, scene=scene)

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
                visualization=not self.raytracer,
            ),
        )

        if self.raytracer:
            table = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="dlo-lab/meshes/wooden_table.glb",
                    pos=(-0., -0.2, -0.799418 * 2),
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
                segment_mass=0.01,
                K=1e5,
                E=1e3,
                G=1e3,
                use_inextensible=False
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=45,
                interval=0.02,
                axis="x",
                pos=(-0.15, 0.1, 0.012),
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

        self.sphere = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=1e5,
                nu=0.3,
                von_mises_yield_stress=1e3,
            ),
            morph=gs.morphs.Sphere(
                radius=0.05,
                pos=(0.25, 0.02, 0.07),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.51, 0.77, 0.75)
            )
        )

        self.bunny = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.3,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/bunny.obj",
                scale=0.1,
                pos=(0.5, -0.05, 0.07),
            ),
            surface=gs.surfaces.Default(
                color=(0., 0.42, 0.47)
            )
        )

        self.cylinder = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.3,
            ),
            morph=gs.morphs.Cylinder(
                radius=0.05,
                height=0.05,
                pos=(0.08, -0.08, 0.025),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.93, 0.96, 0.98)
            )
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.2, -0.63, 0),
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
                pos=(0.5, -0.63, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget'],
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
        self.scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)
        print('gripper geom rigstered', self.scene.rod_solver._geom_indices)
        self.construct_extra_cameras()
        self.scene.build(n_envs=self.n_envs, env_spacing=(10, 10), center_envs_at_origin=False)

        # candidate mode grasp proposal
        self.control_idx = [1, 43]

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

        open_gap = 0.012

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_gripper_gap=open_gap,
        )

        self.c2 = RobotControllerPink(
            self.scene, self.franka2, self._ef2,
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.4, 2., 0.7), up=(0, 0, 1),
            lookat=(0.25, 0., 0), fov=30, GUI=False
        ))
        if not self.raytracer:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(1.5, 0.75, 1.5), up=(0, 0, 1),
                lookat=(0.25, -0.1, 0.), fov=30, GUI=False
            ))

        self.cameras = cameras

    @staticmethod
    def _min_dist_from_efs_to_objs(efs, objs):
        # efs: (n_envs, n_efs, 3)
        # objs: (n_envs, n_objs, 3)
        # For each ef, compute the min distance to objs
        efs_ = efs.unsqueeze(2)  # (n_envs, n_efs, 1, 3)
        objs_ = objs.unsqueeze(1)  # (n_envs, 1, n_objs, 3)
        dists = torch.norm(efs_ - objs_, dim=-1)  # (n_envs, n_efs, n_objs)
        min_dists, _ = torch.min(dists, dim=-1)  # (n_envs, n_efs)
        return min_dists  # (n_envs, n_efs)

    def reward(self):

        verts_batch = self.rope.get_all_verts_tc()  # shape: (n_envs, n_vertices, 3)

        pos1 = self.sphere.get_particles_pos()      # shape: (n_envs, n_particles, 3)
        pos1 = torch.mean(pos1, dim=1)              # shape: (n_envs, 3)
        pos2 = self.bunny.get_pos()                 # shape: (n_envs, 3)
        pos3 = self.cylinder.get_pos()              # shape: (n_envs, 3)

        d12 = torch.norm(pos1 - pos2, dim=1)  # ||p1 - p2||
        d23 = torch.norm(pos2 - pos3, dim=1)  # ||p2 - p3||
        d13 = torch.norm(pos1 - pos3, dim=1)  # ||p1 - p3||

        ef1 = self.c1.ef.get_pos()
        ef2 = self.c2.ef.get_pos()
        efs = torch.stack([ef1, ef2], dim=1)  # shape: (n_envs, 2, 3)
        objs = torch.stack([pos1, pos2, pos3], dim=1)  # shape: (n_envs, 3, 3)
        min_dists_efs_to_objs = self._min_dist_from_efs_to_objs(efs, objs)  # shape: (n_envs, 2)

        # rewards = -(d12 + d23 + d13)          # negative sum of pairwise distances

        rewards = []

        for i in range(self.n_envs):
            verts = verts_batch[i]  # (n_vertices, 3)
            verts_to_pos1 = torch.norm(verts - pos1[i], dim=1)  # (n_vertices,)
            verts_to_pos2 = torch.norm(verts - pos2[i], dim=1)
            verts_to_pos3 = torch.norm(verts - pos3[i], dim=1)

            d12_i = d12[i]
            d23_i = d23[i]
            d13_i = d13[i]
            min_dists_efs_to_objs_i = min_dists_efs_to_objs[i]
            # (n_efs,)
            min_dists_penalty = torch.maximum(0.1 - min_dists_efs_to_objs_i, torch.tensor([0.0] * len(self.control_idx)))
            min_dists_penalty = torch.sum(min_dists_penalty)

            reward = -(d12_i + d23_i + d13_i)          # negative sum of pairwise distances
            reward -= torch.min(verts_to_pos1) * 0.1   # negative min distance from rope to pos1
            reward -= torch.min(verts_to_pos2) * 0.1   # negative min distance from rope to pos2
            reward -= torch.min(verts_to_pos3) * 0.1   # negative min distance from rope to pos3
            reward -= 100 * min_dists_penalty           # penalty for ef too close to objects
            reward = torch.exp(reward)

            rewards.append(reward.item())

        return rewards             # list[float] of length n_envs

    def loss_criterion(self, state):
        verts_batch = state.pos

        pos1 = self.sphere.get_particles_pos()           # shape: (n_envs, n_particles, 3)
        pos1 = torch.mean(pos1, dim=1)                   # shape: (n_envs, 3)
        pos2 = self.bunny.get_pos()                      # shape: (n_envs, 3)
        pos3 = self.cylinder.get_pos()                   # shape: (n_envs, 3)

        d12 = torch.norm(pos1 - pos2, dim=1)  # ||p1 - p2||
        d23 = torch.norm(pos2 - pos3, dim=1)  # ||p2 - p3||
        d13 = torch.norm(pos1 - pos3, dim=1)  # ||p1 - p3||

        ef1 = verts_batch[:, self.control_idx[0], :]
        ef2 = verts_batch[:, self.control_idx[1], :]
        efs = torch.stack([ef1, ef2], dim=1)  # shape: (n_envs, 2, 3)
        objs = torch.stack([pos1, pos2, pos3], dim=1)  # shape: (n_envs, 3, 3)
        min_dists_efs_to_objs = self._min_dist_from_efs_to_objs(efs, objs)  # shape: (n_envs, 2)
        min_dists_penalty = torch.relu(0.1 - min_dists_efs_to_objs)  # (n_envs, n_efs)
        min_dists_penalty = torch.sum(min_dists_penalty, dim=1)  # (n_envs,)

        verts_to_pos1 = torch.norm(verts_batch - pos1.unsqueeze(1), dim=2)  # (n_envs, n_vertices)
        dis_1 = torch.min(verts_to_pos1, dim=1).values  # (n_envs,)
        verts_to_pos2 = torch.norm(verts_batch - pos2.unsqueeze(1), dim=2)  # (n_envs, n_vertices)
        dis_2 = torch.min(verts_to_pos2, dim=1).values  # (n_envs,)
        verts_to_pos3 = torch.norm(verts_batch - pos3.unsqueeze(1), dim=2)  # (n_envs, n_vertices)
        dis_3 = torch.min(verts_to_pos3, dim=1).values  # (n_envs,)

        loss = 0.1 * (dis_1 + dis_2 + dis_3) + 100 * min_dists_penalty  # (n_envs,)
        loss += d12 + d23 + d13

        return loss  # (n_envs,)

    def _randomize_object_positions(self, envs_idx=None):
        if not self.randomization_initialized:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        # Randomize sphere x positions from -0.025 to 0.025
        x_pos = np.random.uniform(-0.025, 0.025, size=(len(envs_idx), 1)) # [n_envs_idx, 1]
        y_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        z_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        rel_pos = np.hstack((x_pos, y_pos, z_pos))  # [n_envs_idx, 3]
        rel_pos = torch.tensor(rel_pos, dtype=gs.tc_float)
        cur_pos = self.sphere.get_particles_pos(envs_idx=envs_idx)  # [n_envs_idx, n_particles, 3]
        new_pos = cur_pos + rel_pos.unsqueeze(1)  # [n_envs_idx, n_particles, 3]
        self.sphere.set_particles_pos(new_pos, envs_idx=envs_idx)

        # Randomize cylinder x positions from -0.025 to 0.025
        x_pos = np.random.uniform(-0.025, 0.025, size=(len(envs_idx), 1)) # [n_envs_idx, 1]
        y_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        z_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        rel_pos = np.hstack((x_pos, y_pos, z_pos))  # [n_envs_idx, 3]
        rel_pos = torch.tensor(rel_pos, dtype=gs.tc_float)
        self.cylinder.set_pos(rel_pos, envs_idx=envs_idx, relative=True)

        # Randomize bunny x positions from -0.025 to 0.025
        x_pos = np.random.uniform(-0.025, 0.025, size=(len(envs_idx), 1)) # [n_envs_idx, 1]
        y_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        z_pos = np.zeros((len(envs_idx), 1)) # [n_envs_idx, 1]
        rel_pos = np.hstack((x_pos, y_pos, z_pos))  # [n_envs_idx, 3]
        rel_pos = torch.tensor(rel_pos, dtype=gs.tc_float)
        self.bunny.set_pos(rel_pos, envs_idx=envs_idx, relative=True)

    def reset(self, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)
        self._randomize_positions([self.rope], envs_idx=envs_idx)
        self._randomize_masses([self.rope], envs_idx=envs_idx)
        self._randomize_radii([self.rope], envs_idx=envs_idx)
        self._randomize_bending_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_stretching_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_twisting_stiffness([self.rope], envs_idx=envs_idx)
        self._randomize_object_positions(envs_idx=envs_idx)

        envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

        for f in [self.franka1, self.franka2]:
            f.set_qpos(
                np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                envs_idx=envs_idx_
            )

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

        self.c1.control_robot(0, 0, envs_idx=envs_idx)
        self.c2.control_robot(0, 0, envs_idx=envs_idx)
        for i in range(30):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

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

                    self.c1.draw_debug_point(dxyz1, min_z=0.03)
                    self.c2.draw_debug_point(dxyz2, min_z=0.03)
                else:
                    qpos1 = self.c1.control_robot(
                        0, 0,
                        dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.03
                    )
                    qpos2 = self.c2.control_robot(
                        0, 0,
                        dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.03
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
                # Post-step: detect collisions
                collided = self.rope._solver.vertices_collision.collided.to_numpy()  # (n_verts, n_envs)
                collided = collided.T  # (n_envs, n_vertices)
                collided_geom_idx = self.rope._solver.vertices_collision.geom_idx.to_numpy()  # (n_verts, n_envs)
                collided_geom_idx = collided_geom_idx.T  # (n_envs, n_verts)
                # check all verts
                verts_to_ignore = np.array([0, 1, 2, 3, 41, 42, 43, 44])
                verts_to_check = np.arange(self.rope.n_vertices)
                not_in_ignore = np.isin(verts_to_check, verts_to_ignore, invert=True)
                verts_to_check = verts_to_check[not_in_ignore] + self.rope._v_start
                collided_precheck = collided[:, verts_to_check]                    # (n_envs, n_verts_to_check)
                collided_geom_is_registered = np.zeros_like(collided_precheck, dtype=bool)  # (n_envs, n_verts_to_check)
                for registered_geom_idx in self.gripper_geom_indices:
                    collided_geom_is_registered |= (collided_geom_idx[:, verts_to_check] == registered_geom_idx)
                # check whether verts collided with gripper
                collided_ctrl = collided_precheck & collided_geom_is_registered
                collided_ctrl = collided_ctrl.any(axis=1)  # (n_envs,)

                newly_collided = collided_ctrl & alive
                if newly_collided.any():
                    first_fail_step[newly_collided] = np.minimum(first_fail_step[newly_collided], global_step)
                    alive[newly_collided] = False

                # Post-step: detect whether gripper lost the rod
                lost = np.ones((self.n_envs,), dtype=bool)
                for i_b in range(self.n_envs):
                    grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                    c1_retained = False
                    c2_retained = False
                    for k, v in grasp_info.items():
                        if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                            c1_retained = True
                        if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                            c2_retained = True
                    # lost either gripper
                    lost[i_b] = not (c1_retained and c2_retained)
                newly_lost = lost & alive
                if newly_lost.any():
                    first_fail_step[newly_lost] = np.minimum(first_fail_step[newly_lost], global_step)
                    alive[newly_lost] = False

                # Post-step: detect stretching failures
                if self.control_dist_init is not None:
                    # (n_envs,)
                    control_dist_now = self.rope.get_total_length()
                    # 10% stretch allowed
                    stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.10
                    newly_stretched = stretched_between_ctrl & alive
                    if newly_stretched.any():
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
                substep_rewards[~failed] = substep_rewards_pre[~failed]
                reward_accum += substep_rewards

        # Compute final state rewards (same as v2)
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

        pos1 = self.sphere.get_particles_tc()                       # shape: (n_envs, n_particles, 3)
        pos1 = torch.mean(pos1, dim=1).to(torch.float32)            # shape: (n_envs, 3)
        pos2 = self.bunny.get_pos().to(torch.float32)               # shape: (n_envs, 3)
        pos3 = self.cylinder.get_pos().to(torch.float32)            # shape: (n_envs, 3)

        vel1 = self.sphere.get_velocities_tc()                      # shape: (n_envs, n_particles, 3)
        vel1 = torch.mean(vel1, dim=1).to(torch.float32)            # shape: (n_envs, 3)
        vel2 = self.bunny.get_vel().to(torch.float32)               # shape: (n_envs, 3)
        vel3 = self.cylinder.get_vel().to(torch.float32)            # shape: (n_envs, 3)

        obs_add = torch.cat([pos1, vel1, pos2, vel2, pos3, vel3], dim=1)

        ef1_pos = self.c1.ef.get_pos().to(torch.float32)
        ef1_quat = self.c1.ef.get_quat().to(torch.float32)
        joint1_qpos = self.c1.robot.get_dofs_position(self.c1.motors_dof).to(torch.float32)
        c1_obs = torch.cat([ef1_pos, ef1_quat, joint1_qpos], dim=1)

        ef2_pos = self.c2.ef.get_pos().to(torch.float32)
        ef2_quat = self.c2.ef.get_quat().to(torch.float32)
        joint2_qpos = self.c2.robot.get_dofs_position(self.c2.motors_dof).to(torch.float32)
        c2_obs = torch.cat([ef2_pos, ef2_quat, joint2_qpos], dim=1)

        obs = torch.cat([obs_rope, obs_add, c1_obs, c2_obs], dim=1)
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
                dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.03
            )
            qpos2 = self.c2.control_robot(
                0, 0,
                dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.03
            )

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            collided = self.rope._solver.vertices_collision.collided.to_numpy()  # (n_verts, n_envs)
            collided = collided.T  # (n_envs, n_vertices)
            collided_geom_idx = self.rope._solver.vertices_collision.geom_idx.to_numpy()  # (n_verts, n_envs)
            collided_geom_idx = collided_geom_idx.T  # (n_envs, n_verts)
            # check all verts
            verts_to_ignore = np.array([0, 1, 2, 3, 41, 42, 43, 44])
            verts_to_check = np.arange(self.rope.n_vertices)
            not_in_ignore = np.isin(verts_to_check, verts_to_ignore, invert=True)
            verts_to_check = verts_to_check[not_in_ignore] + self.rope._v_start
            collided_precheck = collided[:, verts_to_check]                    # (n_envs, n_verts_to_check)
            collided_geom_is_registered = np.zeros_like(collided_precheck, dtype=bool)  # (n_envs, n_verts_to_check)
            for registered_geom_idx in self.gripper_geom_indices:
                collided_geom_is_registered |= (collided_geom_idx[:, verts_to_check] == registered_geom_idx)
            # check whether verts collided with gripper
            collided_ctrl = collided_precheck & ~collided_geom_is_registered
            collided_ctrl = collided_ctrl.any(axis=1)  # (n_envs,)

            newly_collided = collided_ctrl & alive
            if newly_collided.any():
                absorbing[newly_collided] = True
                alive[newly_collided] = False

            # Post-step: detect whether gripper lost the rod
            lost = np.ones((self.n_envs,), dtype=bool)
            for i_b in range(self.n_envs):
                grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                c1_retained = False
                c2_retained = False
                for k, v in grasp_info.items():
                    if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                        c1_retained = True
                    if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                        c2_retained = True
                # lost either gripper
                lost[i_b] = not (c1_retained and c2_retained)
            newly_lost = lost & alive
            if newly_lost.any():
                absorbing[newly_lost] = True
                alive[newly_lost] = False

            # Post-step: detect stretching failures
            if self.control_dist_init is not None:
                # (n_envs,)
                control_dist_now = self.rope.get_geodesic_distance(
                    self.control_idx[0], self.control_idx[1]
                )
                # 10% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.10
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

            # NOTE: Do not move already-failed envs
            action1[~alive, :] = 0.0
            action2[~alive, :] = 0.0

            alpha = 1 / n_steps_sub
            dxyz = [alpha * action1, alpha * action2]

            for i_g in range(len(self.control_idx)):
                controller = getattr(self, f"c{i_g+1}")
                qpos = controller.control_robot(
                    0, 0,
                    dx=dxyz[i_g][:, 0], dy=dxyz[i_g][:, 1], dz=dxyz[i_g][:, 2], min_z=0.03
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
                # 10% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.10
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
        substep_rewards[~failed] = substep_rewards_pre[~failed]

        return loss, alive, substep_rewards, state.s_global

    def train_one_iter_gd(self, it=None, max_it=None, skip_backward=False):
        self.qpos_seq = np.zeros((self._n_steps + 1, self.n_envs, len(self.control_idx) * 9))
        self.use_qpos = False

        self.reset()

        loss = 0.
        total_horizon = 30
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
                # 10% stretch allowed
                stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.10
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
