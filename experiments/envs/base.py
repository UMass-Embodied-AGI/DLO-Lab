import os
import torch
import mediapy
import warnings
import numpy as np
import genesis as gs
from omegaconf import DictConfig

from collections import defaultdict
from mushroom_rl.core import MDPInfo
from mushroom_rl.rl_utils.spaces import Box

from utils.logging import color_print
from utils.visual_tools import draw_points
from utils.gradient import (
    create_linear_array,
    create_exp_array,
    create_custom_array,
)
from utils.controller import TrajOptimController

warnings.filterwarnings(
    action='ignore',
    message='.*Template mapper caching disabled.*',
    category=UserWarning
)


class Train_Env():
    def __init__(self, config: DictConfig, scene=None):
        self.task = config.task
        self.GUI = config.get('GUI', False)
        self.raytracer = config.get('raytracer', False)
        self.n_envs = config.n_envs
        self.requires_grad = config.get('requires_grad', False)
        self.steps_interval = config.get('n_substeps_per_step', 200)
        color_print(config, "blue")

        if scene is None:
            # Initialize scene, if not provided
            if not gs._initialized:
                gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)

            viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            )

            self.scene = gs.Scene(
                viewer_options=viewer_options,
                # vis_options=gs.options.VisOptions(
                #     lights=[
                #         {"type": "directional", "dir": (1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
                #     ]
                # ),
                sim_options=gs.options.SimOptions(
                    dt=1e-3,
                    substeps=5,
                    requires_grad=self.requires_grad,
                    bptt_window=config.get('bptt_window', 0)
                ),
                rigid_options=gs.options.RigidOptions(
                    # Skip the rigid solver's differentiable backward (legacy no-op behavior). The robot acts as a
                    # non-differentiable actuator; the learning signal comes from the rope vertex gradients. Avoids
                    # NaN gradients from the forward-kinematics quaternion backward.
                    skip_backward=True,
                ),
                rod_options=gs.options.RODOptions(
                    damping=30.0,
                    angular_damping=20.0,
                    n_pbd_iters=10,
                    grad_clip=config.get('grad_clip', 0.0),
                    disable_constraint_grad=config.get('disable_constraint_grad', False)
                ),
                show_viewer=config.GUI,
                renderer=gs.renderers.RayTracer(
                    env_surface=gs.surfaces.Emission(
                        emissive_texture=gs.textures.ImageTexture(
                            image_path=config.get('exr_path', 'dlo-lab/exrs/brown_photostudio_02_4k.exr'),
                            image_color=(0.6, 0.6, 0.6),
                            encoding='linear',
                        ),
                    ),
                    env_radius=15.0,
                    env_euler=(0, 0, 180),
                    lights=[],
                ) if self.raytracer else gs.renderers.Rasterizer(),
            )
        else:
            # Use provided scene, this means genesis already initialized
            self.scene = scene

        self.create_log_dir(config.log_dir)

        self.cameras = list()
        self.frames = defaultdict(list)
        self.construct_scene(camera=config.camera)
        self._act_dim = len(self.control_idx) * 6       # n_ctrl * 6
        self.control_dist_init = None
        self.debug_point_nodes = list()

        self.cmaes_initialized = False
        self.rl_initialized = False
        self.diff_rl_initialized = False
        self.gd_initialized = False
        self.randomization_initialized = False

    def save_animation(self, save_dir):
        if len(self.frames) == 0:
            return

        for cid in self.frames:
            video_path = os.path.join(save_dir, f"view_{cid}_best.mp4")
            mediapy.write_video(video_path, self.frames[cid], fps=30, qp=18)
            color_print(f'Saved video to {video_path}', 'green')

    @staticmethod
    def draw_rope_vertices(
        camera, rope, rope_ids, env_idx=0, color=(0, 255, 0), alpha=1.0, radius=5, thickness=-1,
        annotation_args=None, text_offset=(10, -10), text_color=None, text_scale=0.5, text_thickness=1,
        use_color_palette=False, color_palette=None, rgb_img=None, return_points_3d=False, return_points_2d=False
    ):
        """
        Draw specific rope vertices on the camera image with optional text annotations.

        Parameters
        ----------
        camera : genesis.Camera
            The camera object used for rendering.
        rope : genesis.RodEntity
            The rope entity to visualize.
        rope_ids : array-like or int
            Indices of rope vertices to draw.
        env_idx : int, optional
            Environment index. Default is 0.
        color : tuple, optional
            RGB color for the points. Ignored if use_color_palette=True.
            Default is green (0, 255, 0).
        alpha : float, optional
            Transparency level [0, 1]. Default is 1.0.
        radius : int, optional
            Radius of the circles. Default is 5.
        thickness : int, optional
            Thickness of the circles. Default is -1 (filled).
        annotation_args : list of str, optional
            Text annotations for each vertex. Default is None.
        text_offset : tuple of int, optional
            2D offset for text placement. Default is (10, -10).
        text_color : tuple, optional
            RGB color for text. If None, uses same as points.
            Ignored if use_color_palette=True. Default is None.
        text_scale : float, optional
            Font scale for text. Default is 0.5.
        text_thickness : int, optional
            Thickness of text. Default is 1.
        use_color_palette : bool, optional
            If True, automatically assign colors from a palette to each vertex.
            Default is False.
        color_palette : list of tuples, optional
            Custom color palette. If None, uses default palette. Default is None.
        rgb_img : np.ndarray, optional
            Pre-rendered RGB image from the camera. If None, renders internally.
        return_points_3d : bool, optional
            If True, also return the 3D points. Default is False.
        return_points_2d : bool, optional
            If True, also return the projected 2D points. Default is False.

        Returns
        -------
        np.ndarray or tuple
            Image with drawn vertices and annotations. If return_points_3d or return_points_2d is True, also returns the 3D points and/or 2D points as numpy arrays.
        """
        if isinstance(rope_ids, int):
            points_start = rope.get_all_verts()[env_idx, 1]
            points_end = rope.get_all_verts()[env_idx, -2]
            # interpolate #rope_ids points between start and end
            points_3d = np.linspace(points_start, points_end, num=rope_ids)
        else:
            rope_ids = np.asarray(rope_ids, dtype=np.int32)
            points_3d = rope.get_all_verts()[env_idx, rope_ids]
        if rgb_img is None:
            rgb_img = camera.render()[0]
        draw_points_out = draw_points(
            camera, points_3d, rgb_img=rgb_img, color=color, alpha=alpha, radius=radius, thickness=thickness,
            annotation_args=annotation_args, text_offset=text_offset, text_color=text_color,
            text_scale=text_scale, text_thickness=text_thickness,
            use_color_palette=use_color_palette, color_palette=color_palette, return_points_2d=return_points_2d
        )

        if return_points_2d:
            img_with_points, points_2d_list = draw_points_out
        else:
            img_with_points = draw_points_out

        out = []

        if return_points_3d:
            out.append(points_3d)
        if return_points_2d:
            out.append(points_2d_list)

        if out:
            return (img_with_points, *out)

        return img_with_points

    def construct_traj_optim(self, max_ddist=0.1, max_grad_norm=1000, controller="TrajOptimController", debug=False, **kwargs):
        if not self.requires_grad:
            return

        if controller == "TrajOptimController":
            self.c = TrajOptimController(
                scene=self.scene,
                rod=self.rope,
                grasp_point_ids=self.control_idx,
                n_stages=self._n_steps,
                n_optim_dofs=3,
                max_ddist=max_ddist,
                max_grad_norm=max_grad_norm,
                debug=debug,
                **kwargs
            )

    def construct_scale_array(self, scale_method, n_steps, exp_base=1.1):
        if scale_method is None:
            scale_array = torch.ones(n_steps, dtype=gs.tc_float)
            self.scale_array = scale_array / n_steps
            print(f'Using uniform scale array:\n{self.scale_array}')
        elif scale_method == 'linear':
            self.scale_array = create_linear_array(n_steps)
            print(f'Using linear scale array:\n{self.scale_array}')
        elif scale_method == 'exp':
            self.scale_array = create_exp_array(n_steps, base=exp_base)
            print(f'Using exponential scale array (base={exp_base}):\n{self.scale_array}')
        elif scale_method == 'custom':
            self.scale_array = create_custom_array(n_steps)
            print(f'Using custom scale array:\n{self.scale_array}')
        else:
            raise ValueError(f'Unknown scale method: {scale_method}')

    def create_log_dir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

    def construct_scene(self, camera=False):
        raise NotImplementedError()

    def construct_cameras(self):
        raise NotImplementedError()

    def construct_extra_cameras(self):
        """
        Hook for adding extra cameras before scene.build().

        Override this method to attach additional cameras (e.g. wrist cameras)
        that must be created before the scene is built. Called by each env's
        construct_scene() right before self.scene.build().
        """
        pass

    def reward(self):
        raise NotImplementedError()

    def loss_criterion(self, state):
        raise NotImplementedError()

    def reset(self, debug=False, envs_idx=None):
        raise NotImplementedError()

    def init_cmaes_env(
        self,
        n_steps_sub=10,
    ):
        self._cmaes_n_steps_sub = n_steps_sub

        self.cmaes_initialized = True

    # # # # # # RL/Diff RL Utils # # # # # #
    def compute_observation(self):
        raise NotImplementedError()

    # # # # # # RL Utils # # # # # #
    def reset_all(self, env_mask, state=None):
        env_indices = torch.where(env_mask)[0]
        # Check if there are any environments to reset
        if len(env_indices) > 0:
            self.reset(envs_idx=env_indices)

        obs = self.compute_observation()
        return obs, [{}] * self.n_envs

    # # # # # # RL Utils # # # # # #
    def init_rl_env(
        self,
        n_steps = 10,
        pos_bound = 0.1,
        angle_bound = 5.0,
        n_additional_obj=0,
        steps_interval_split=2,
        l2_limit=None,
        action_magnitude=None,
        debug=False
    ):
        if l2_limit is None:
            self._l2_limit = max(pos_bound) if hasattr(pos_bound, '__len__') else pos_bound
        else:
            self._l2_limit = l2_limit
        # RL/vectorized env configuration
        self._backend = 'torch'
        if debug:
            # TODO: hack here
            if getattr(self, "c1", None) is not None:
                self.c1.debug = True
            if getattr(self, "c2", None) is not None:
                self.c2.debug = True

        # Observation / action specs
        self._obs_dim = (self.rope.n_vertices + n_additional_obj) * 6 + len(self.control_idx) * 14
        self._horizon = n_steps                                     # 10
        self._steps_per_action = self.steps_interval
        self._steps_interval_split = steps_interval_split

        # Observation/action spaces
        low_obs = torch.full((self._obs_dim,), -np.inf, dtype=torch.float32)
        high_obs = torch.full((self._obs_dim,), np.inf, dtype=torch.float32)
        observation_space = Box(low_obs, high_obs)

        half = self._act_dim // 2
        pb = list(pos_bound) if hasattr(pos_bound, '__len__') else [pos_bound] * half
        ab = list(angle_bound) if hasattr(angle_bound, '__len__') else [angle_bound] * half
        act_limit = torch.tensor(pb + ab, dtype=torch.float32)
        if action_magnitude is None:
            self._act_magnitude = act_limit
        else:
            assert len(action_magnitude) == self._act_dim, \
                f"action_magnitude length mismatch: expected {self._act_dim}, got {len(action_magnitude)}"
            self._act_magnitude = torch.tensor(action_magnitude, dtype=torch.float32)
        low_act = -torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        high_act = torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        action_space = Box(low_act, high_act)

        # Control dt approximates sim dt * internal steps
        control_dt = self.scene.sim_options.dt * self._steps_per_action
        self._mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=self._horizon, dt=control_dt, backend=self._backend)

        # Compatibility in `reset()`
        self.rl_initialized = True

    @property
    def info(self):
        return self._mdp_info

    @property
    def number(self):
        return self.n_envs

    # # # # # # RL Utils # # # # # #
    def step_all(self, env_mask, action):
        """ Used in MushroomRL """
        raise NotImplementedError()

    # # # # # # RL Utils # # # # # #
    def render_all(self, env_mask, record=False):
        """ Used in MushroomRL """
        if self.GUI:
            pass
        if record:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return None

    # # # # # # RL Utils # # # # # #
    def stop(self):
        pass

    # # # # # # Diff RL Utils # # # # # #
    def init_diff_rl_env(
        self,
        n_steps=10,
        pos_bound=0.1,
        angle_bound=5.0,
        n_additional_obj=0,
        steps_interval_split=2,
        l2_limit=None,
        action_magnitude=None,
        debug=False
    ):
        """
        Initialize environment for differentiable RL (SHAC).
        Similar to init_rl_env() but does NOT create MushroomRL MDPInfo.

        Args:
            n_steps: Horizon length (steps per episode)
            pos_bound: Position action bound
            angle_bound: Angle action bound
            n_additional_obj: Number of additional objects in observation
            steps_interval_split: Action split interval
            l2_limit: L2 movement limit (overrides pos_bound if provided)
            action_magnitude: Custom action magnitude (overrides default)
            debug: Enable debug mode
        """
        if l2_limit is None:
            self._l2_limit = pos_bound
        else:
            self._l2_limit = l2_limit

        if debug:
            # Enable debug mode for controllers
            if getattr(self, "c1", None) is not None:
                self.c1.debug = True
            if getattr(self, "c2", None) is not None:
                self.c2.debug = True

        # Observation / action specs (same as init_rl_env)
        self._obs_dim = (self.rope.n_vertices + n_additional_obj) * 6 + len(self.control_idx) * 14
        self._horizon = n_steps
        self._steps_per_action = self.steps_interval
        self._steps_interval_split = steps_interval_split

        # Action bounds
        self._act_dim = len(self.control_idx) * 3
        act_limit = [pos_bound] * self._act_dim
        act_limit = torch.tensor(act_limit, dtype=torch.float32)
        if action_magnitude is None:
            self._act_magnitude = act_limit
        else:
            assert len(action_magnitude) == self._act_dim, \
                f"action_magnitude length mismatch: expected {self._act_dim}, got {len(action_magnitude)}"
            self._act_magnitude = torch.tensor(action_magnitude, dtype=torch.float32)
        low_act = -torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        high_act = torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        self._act_low = low_act
        self._act_high = high_act

        self.diff_rl_initialized = True

    # # # # # # Diff RL Utils # # # # # #
    def initialize_trajectory(self):
        """
        Initialize a new trajectory by resetting the environment.
        This cuts off gradients between episodes in differentiable RL.

        Returns:
            obs: Initial observation tensor (n_envs, obs_dim)
        """
        self.reset()
        obs = self.compute_observation()
        return obs

    # # # # # # Diff RL Utils # # # # # #
    def step_diff_rl(self, env_mask, action):
        """
        Step the environment using differentiable RL action.
        Action is expected to be a tensor of shape (n_envs, act_dim).

        Args:
            env_mask: Boolean tensor indicating active environments (n_envs,)
            action: Action tensor (n_envs, act_dim)

        Returns:
            loss: Loss tensor (n_envs,)
            new_env_mask: Updated environment mask (n_envs,)
            rewards: Reward tensor (n_envs,)
            s_global: Current timestep (int), this is to retrieve gradients later
        """
        raise NotImplementedError()

    def loss_above_plane(self, state):
        # Required loss to make sure the vertices above the plane
        verts_batch = state.pos
        loss_abv_plane = torch.relu(
            self.rope.material.segment_radius - verts_batch[:, :, 2]
        ).sum(dim=1)                    # (n_envs,)

        return loss_abv_plane

    def init_gd_env(
        self,
        n_steps=100,
        pos_bound = 0.01,
        angle_bound = 0.1,
        min_z=0.013,
        feasible_region=None,
        scale_method=None,
        exp_base=1.1,
        lr=0.01,
        lr_min=1e-6,
        debug=False
    ):
        self._n_steps = n_steps
        self._max_ddist = pos_bound
        self._min_z = min_z
        if feasible_region is not None:
            assert len(feasible_region) == 6, "feasible_region must be a tuple of (x_min, x_max, y_min, y_max, z_min, z_max)"
            self._feasible_region = list()
            for val in feasible_region:
                if val is not None:
                    self._feasible_region.append(torch.tensor(val, dtype=gs.tc_float))
                else:
                    self._feasible_region.append(None)
        else:
            self._feasible_region = None
        self.lr = lr
        self.lr_min = lr_min

        self.construct_scale_array(
            scale_method=scale_method,
            n_steps=n_steps,
            exp_base=exp_base,
        )

        if debug:
            # TODO: hack here
            if getattr(self, "c1", None) is not None:
                self.c1.debug = True
            if getattr(self, "c2", None) is not None:
                self.c2.debug = True

        self.gd_initialized = True

    def train_one_iter_gd(self, it=None, max_it=None):
        raise NotImplementedError()

    def eval_traj(self, trajs, **kwargs):
        assert self.cmaes_initialized, "CMA-ES environment not initialized. Call init_cmaes_env() first."
        out = dict()
        return out

    def adaptive_scale(self, trajs, deltas, ratio=0.1):
        norm_trajs = np.linalg.norm(trajs, axis=-1, keepdims=True)
        norm_deltas = np.linalg.norm(deltas, axis=-1, keepdims=True)

        norm_deltas = np.clip(norm_deltas, a_min=gs.EPS, a_max=None)
        scaling_factor = (ratio * norm_trajs) / norm_deltas

        deltas_scaled = deltas * scaling_factor

        return deltas_scaled

    def init_domain_randomization(
        self,
        pos_bound=None,
        mass_list=None,
        radius_list=None,
        bending_stiffness_bound=None,
        twisting_stiffness_bound=None,
        stretching_stiffness_bound=None,
        friction_list=None,
    ):
        # Initial position randomization
        if pos_bound is not None:
            assert len(pos_bound) == 4, f"len(pos_bound) should be 4, got {len(pos_bound)}"
            self._pos_bound = torch.tensor(pos_bound, dtype=gs.tc_float)
        else:
            self._pos_bound = None

        # Mass randomization
        if mass_list is not None:
            self._mass_list = torch.tensor(mass_list, dtype=gs.tc_float)
        else:
            self._mass_list = None

        # Radius randomization
        if radius_list is not None:
            self._radius_list = torch.tensor(radius_list, dtype=gs.tc_float)
        else:
            self._radius_list = None

        # Bending stiffness randomization
        if bending_stiffness_bound is not None:
            assert len(bending_stiffness_bound) == 2, \
                f"len(bending_stiffness_bound) should be 2, got {len(bending_stiffness_bound)}"
            self._bending_stiffness_bound = torch.tensor(bending_stiffness_bound, dtype=gs.tc_float)
        else:
            self._bending_stiffness_bound = None

        # Twisting stiffness randomization
        if twisting_stiffness_bound is not None:
            assert len(twisting_stiffness_bound) == 2, \
                f"len(twisting_stiffness_bound) should be 2, got {len(twisting_stiffness_bound)}"
            self._twisting_stiffness_bound = torch.tensor(twisting_stiffness_bound, dtype=gs.tc_float)
        else:
            self._twisting_stiffness_bound = None

        # Stretching stiffness randomization
        if stretching_stiffness_bound is not None:
            assert len(stretching_stiffness_bound) == 2, \
                f"len(stretching_stiffness_bound) should be 2, got {len(stretching_stiffness_bound)}"
            self._stretching_stiffness_bound = torch.tensor(stretching_stiffness_bound, dtype=gs.tc_float)
        else:
            self._stretching_stiffness_bound = None

        # Friction coefficient randomization
        if friction_list is not None:
            self._friction_list = torch.tensor(friction_list, dtype=gs.tc_float)
        else:
            self._friction_list = None

        self.randomization_initialized = True

    def _randomize_positions(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._pos_bound is None:
            return
        if self.scene.requires_grad and envs_idx is not None:
            # We will reset the positions for all envs, which will break the gradient computation
            # Thus, we do not allow envs_idx to be specified in this case
            raise ValueError("envs_idx should be None when gradient is enabled.")
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        batched_bound_min = self._pos_bound[0:2].unsqueeze(0).repeat(len(envs_idx), 1)
        batched_bound_max = self._pos_bound[2:4].unsqueeze(0).repeat(len(envs_idx), 1)
        # uniform delta positions
        rand_pos = torch.rand((len(envs_idx), 2), dtype=gs.tc_float)
        rand_pos = batched_bound_min + rand_pos * (batched_bound_max - batched_bound_min)
        # set z to zero
        rand_pos_z = torch.zeros((len(envs_idx), 1), dtype=gs.tc_float)
        rand_pos = torch.cat([rand_pos, rand_pos_z], dim=1)      # (n_envs, 3)
        gs.logger.debug(f"Randomized positions for envs {envs_idx.tolist()}:\n{rand_pos}")

        for obj in objects:
            # retrieve current positions for all envs (n_envs, n_verts, 3)
            curr_pos = obj.get_all_verts_tc()  # after reset, is the initial position
            updated_pos = curr_pos.clone()
            updated_pos[envs_idx] += rand_pos.unsqueeze(1)
            obj.set_pos(0, updated_pos)

    def _randomize_masses(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._mass_list is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        n_masses = len(self._mass_list)
        rand_indices = torch.randint(0, n_masses, (len(envs_idx),), dtype=torch.long)
        selected_masses = self._mass_list[rand_indices]
        gs.logger.debug(f"Randomized masses for envs {envs_idx.tolist()}:\n{selected_masses.tolist()}")

        for obj in objects:
            updated_mass = selected_masses.unsqueeze(1).repeat(1, obj.n_vertices)
            obj.set_segment_mass(updated_mass, envs_idx=envs_idx)

    def _randomize_radii(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._radius_list is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        n_radii = len(self._radius_list)
        rand_indices = torch.randint(0, n_radii, (len(envs_idx),), dtype=torch.long)
        selected_radii = self._radius_list[rand_indices]
        gs.logger.debug(f"Randomized radii for envs {envs_idx.tolist()}:\n{selected_radii.tolist()}")

        for obj in objects:
            updated_radius = selected_radii.unsqueeze(1).repeat(1, obj.n_vertices)
            obj.set_segment_radius(updated_radius, envs_idx=envs_idx)

    def _randomize_bending_stiffness(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._bending_stiffness_bound is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        batched_bound_min = torch.stack([self._bending_stiffness_bound[0]] * len(envs_idx))       # (n_envs,)
        batched_bound_max = torch.stack([self._bending_stiffness_bound[1]] * len(envs_idx))       # (n_envs,)
        rand_stiffness = torch.rand((len(envs_idx),), dtype=gs.tc_float)
        selected_stiffness = batched_bound_min + rand_stiffness * (batched_bound_max - batched_bound_min)
        gs.logger.debug(f"Randomized bending stiffness for envs {envs_idx.tolist()}:\n{selected_stiffness.tolist()}")

        for obj in objects:
            obj.set_bending_stiffness(selected_stiffness, envs_idx=envs_idx)

    def _randomize_twisting_stiffness(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._twisting_stiffness_bound is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        batched_bound_min = torch.stack([self._twisting_stiffness_bound[0]] * len(envs_idx))       # (n_envs,)
        batched_bound_max = torch.stack([self._twisting_stiffness_bound[1]] * len(envs_idx))       # (n_envs,)
        rand_stiffness = torch.rand((len(envs_idx),), dtype=gs.tc_float)
        selected_stiffness = batched_bound_min + rand_stiffness * (batched_bound_max - batched_bound_min)
        gs.logger.debug(f"Randomized twisting stiffness for envs {envs_idx.tolist()}:\n{selected_stiffness.tolist()}")

        for obj in objects:
            obj.set_twisting_stiffness(selected_stiffness, envs_idx=envs_idx)

    def _randomize_stretching_stiffness(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._stretching_stiffness_bound is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        batched_bound_min = torch.stack([self._stretching_stiffness_bound[0]] * len(envs_idx))       # (n_envs,)
        batched_bound_max = torch.stack([self._stretching_stiffness_bound[1]] * len(envs_idx))       # (n_envs,)
        rand_stiffness = torch.rand((len(envs_idx),), dtype=gs.tc_float)
        selected_stiffness = batched_bound_min + rand_stiffness * (batched_bound_max - batched_bound_min)
        gs.logger.debug(f"Randomized stretching stiffness for envs {envs_idx.tolist()}:\n{selected_stiffness.tolist()}")

        for obj in objects:
            obj.set_stretching_stiffness(selected_stiffness, envs_idx=envs_idx)

    def _randomize_friction(self, objects, envs_idx=None):
        if not self.randomization_initialized or self._friction_list is None:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)

        # (N, 2) for (mu_s, mu_k)
        n_friction = len(self._friction_list)
        rand_indices = torch.randint(0, n_friction, (len(envs_idx),), dtype=torch.long)
        selected_friction = self._friction_list[rand_indices]   # (len(envs_idx), 2)
        gs.logger.debug(f"Randomized (mu_s, mu_k) for envs {envs_idx.tolist()}:\n{selected_friction.tolist()}")

        for obj in objects:
            updated_friction = selected_friction.unsqueeze(1).repeat(1, obj.n_vertices, 1)  # (n_envs, n_verts, 2)
            obj.set_mu_s(updated_friction[..., 0], envs_idx=envs_idx)
            obj.set_mu_k(updated_friction[..., 1], envs_idx=envs_idx)
