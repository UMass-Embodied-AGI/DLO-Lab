import torch
import genesis as gs
import genesis.utils.geom as gu

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
import qpsolvers

# from genesis.engine.entities import RodEntity

def rod_vertex_attached_to_gripper(rod, vert_idx, ef, envs_idx=0):
    rod.attach_to_rigid_link_with_envs_idx(ef, [vert_idx], envs_idx=envs_idx)

def rod_vertex_detached_from_gripper(rod, vert_idx, envs_idx=0):
    rod.detach_from_rigid_link_with_envs_idx([vert_idx], envs_idx=envs_idx)

def cosine_learning_rate_scheduler(base_lr, cur_iter, max_iter, min_lr=1e-6):
    if cur_iter >= max_iter:
        return min_lr
    cosine_decay = 0.5 * (1 + np.cos(np.pi * cur_iter / max_iter))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    return lr

class RobotControllerPink:
    """
    Robot controller using Pink-based inverse kinematics.

    This is a drop-in replacement for RobotController that uses Pink IK
    instead of Genesis's built-in IK solver.
    """

    def __init__(
        self,
        scene,
        robot,
        ef,
        configs=None,
        initial_pos=(0., 0., 0.),
        initial_quat=(0., 1., 0., 0.),
        initial_gripper_gap=0.03,
        n_motors_dofs=7,
        n_fingers_dofs=2,
        urdf_path=None,
        ee_frame_name=None,
        debug=False,
        # Pink-specific parameters
        pink_max_iterations=100,
        pink_dt=0.04,
        pink_solver="proxqp",
        pink_position_cost=10.0,
        pink_orientation_cost=10.0,
        pink_posture_cost=1e-3,
    ):
        """
        Initialize Pink-based robot controller.

        Parameters
        ----------
        scene : gs.Scene
            Genesis scene object
        robot : RigidEntity
            Robot entity (e.g., Franka Panda)
        ef : RigidLink
            End-effector link
        configs : optional
            Configuration dictionary (not used currently)
        initial_pos : tuple, optional
            Initial end-effector position (x, y, z)
        initial_quat : tuple, optional
            Initial end-effector quaternion (w, x, y, z)
        initial_gripper_gap : float, optional
            Initial gripper opening width
        n_motors_dofs : int, optional
            Number of motor DOFs (default: 7 for Franka)
        n_fingers_dofs : int, optional
            Number of finger DOFs (default: 2 for Franka gripper)
        urdf_path : str, optional
            Path to robot URDF file. If None, defaults to `robot.morph.file`
        ee_frame_name : str, optional
            Name of end-effector frame in Pinocchio model. If None, defaults to `ef.name`
        debug : bool, optional
            Enable debug visualization
        pink_max_iterations : int, optional
            Maximum iterations for Pink IK solver (default: 100)
        pink_dt : float, optional
            Integration timestep for Pink IK (default: 0.04)
        pink_solver : str, optional
            QP solver to use: "proxqp", "quadprog", etc. (default: "proxqp")
        pink_position_cost : float, optional
            Position task cost weight (default: 10.0)
        pink_orientation_cost : float, optional
            Orientation task cost weight (default: 10.0)
        pink_posture_cost : float, optional
            Posture regularization cost weight (default: 1e-3)
        """

        self.scene = scene
        self.robot = robot
        self.ef = ef
        self.configs = configs
        self.motors_dof = torch.arange(n_motors_dofs)
        self.fingers_dof = torch.arange(n_motors_dofs, n_motors_dofs + n_fingers_dofs)
        self.debug = debug
        self.debug_point_nodes = list()

        # Process initial position and quaternion
        self.initial_pos = None
        self.initial_quat = None
        self._process_initial_pos_quat(initial_pos, initial_quat)
        self.init_gap = initial_gripper_gap

        # Pink-specific attributes
        if urdf_path is None:
            urdf_path = robot.morph.file
        self.urdf_path = urdf_path
        if ee_frame_name is None:
            ee_frame_name = ef.name
        self.ee_frame_name = ee_frame_name

        # Get robot base position in world frame (Pinocchio assumes robot at origin)
        # We need this to convert between world and robot-local coordinates
        self.robot_base_pos = np.array(robot.morph.pos, dtype=gs.np_float) if hasattr(robot.morph, 'pos') else np.zeros(3, dtype=gs.np_float)
        self.robot_base_quat = np.array(robot.morph.quat, dtype=gs.np_float) if hasattr(robot.morph, 'quat') else np.array([1, 0, 0, 0], dtype=gs.np_float)
        self.pink_max_iterations = pink_max_iterations
        self.pink_dt = pink_dt
        self.pink_solver = pink_solver
        self.pink_position_cost = pink_position_cost
        self.pink_orientation_cost = pink_orientation_cost
        self.pink_posture_cost = pink_posture_cost

        # Load Pinocchio model and create Configuration
        self._load_pinocchio_model()

    def _process_initial_pos_quat(self, initial_pos=None, initial_quat=None):
        if initial_pos is not None:
            if isinstance(initial_pos, np.ndarray):
                if initial_pos.ndim == 1:
                    assert initial_pos.shape == (3,), "initial_pos must be of shape (3,)"
                    initial_pos = np.stack([initial_pos] * self.scene.n_envs) if self.scene.n_envs > 0 else initial_pos
                elif initial_pos.ndim == 2:
                    assert initial_pos.shape == (self.scene.n_envs, 3), "initial_pos must be of shape (n_envs, 3)"
            elif isinstance(initial_pos, (list, tuple)):
                assert len(initial_pos) == 3, "initial_pos must be a tuple/list of length 3"
                initial_pos = np.array(initial_pos, dtype=gs.np_float)
                initial_pos = np.stack([initial_pos] * self.scene.n_envs) if self.scene.n_envs > 0 else initial_pos
            else:
                raise ValueError("initial_pos must be a np.ndarray or tuple/list")
            self.initial_pos = initial_pos

        if initial_quat is not None:
            if isinstance(initial_quat, np.ndarray):
                if initial_quat.ndim == 1:
                    assert initial_quat.shape == (4,), "initial_quat must be of shape (4,)"
                    initial_quat = np.stack([initial_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else initial_quat
                elif initial_quat.ndim == 2:
                    assert initial_quat.shape == (self.scene.n_envs, 4), "initial_quat must be of shape (n_envs, 4)"
            elif isinstance(initial_quat, (list, tuple)):
                assert len(initial_quat) == 4, "initial_quat must be a tuple/list of length 4"
                initial_quat = np.array(initial_quat, dtype=gs.np_float)
                initial_quat = np.stack([initial_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else initial_quat
            else:
                raise ValueError("initial_quat must be a np.ndarray or tuple/list")
            self.initial_quat = initial_quat

    def _load_pinocchio_model(self):
        """Load the robot model with Pinocchio and create Configuration."""
        try:
            # Build model from URDF using RobotWrapper for better path handling
            import pathlib
            urdf_path_obj = pathlib.Path(self.urdf_path)

            # Use pin_robot to avoid overwriting self.robot (Genesis entity)
            self.pin_robot = pin.RobotWrapper.BuildFromURDF(
                filename=urdf_path_obj.as_posix(),
                package_dirs=[".", urdf_path_obj.parent.as_posix()],
                root_joint=None,
            )

            self.robot_model = self.pin_robot.model
            self.robot_data = self.pin_robot.data

            # Verify end-effector frame exists
            if not self.robot_model.existFrame(self.ee_frame_name):
                available_frames = [
                    self.robot_model.frames[i].name
                    for i in range(len(self.robot_model.frames))
                ]
                raise ValueError(
                    f"End-effector frame '{self.ee_frame_name}' not found in model. "
                    f"Available frames: {available_frames}"
                )

            # Create Pink Configuration object (will be updated with current qpos before each IK call)
            initial_q = pin.neutral(self.robot_model)
            self.configuration = pink.Configuration(self.robot_model, self.robot_data, initial_q)

            if self.debug:
                gs.logger.debug(f"Pink IK: Loaded Pinocchio model from {self.urdf_path}")
                gs.logger.debug(f"  Model has {self.robot_model.nq} position DOFs, {self.robot_model.nv} velocity DOFs")
                gs.logger.debug(f"  End-effector frame: {self.ee_frame_name}")
                gs.logger.debug(f"  Genesis EF link name: {self.ef.name}")
                gs.logger.debug(f"  Robot base position (world): {self.robot_base_pos}")
                gs.logger.debug(f"  Robot base quaternion (world): {self.robot_base_quat}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Pinocchio model from {self.urdf_path}: {e}\n"
                "Make sure the URDF path is correct and accessible."
            )

    def set_initial_position(self, init_pos=None, init_quat=None, envs_idx=None):
        """
        Set robot to initial end-effector position using Pink IK.

        Parameters
        ----------
        init_pos : array_like, optional
            Initial end-effector position. If None, use predefined initial position.
        init_quat : array_like, optional
            Initial end-effector orientation as a quaternion. If None, use predefined initial orientation.
        envs_idx : list or None, optional
            Indices of environments to reset. If None, reset all environments.

        Returns
        -------
        qpos : torch.Tensor
            Joint positions including gripper
        """
        # May override predefined initial position/quaternion
        self._process_initial_pos_quat(init_pos, init_quat)

        pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
        quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)

        is_batched = self.scene.n_envs > 0
        if envs_idx is None or len(envs_idx) == self.scene.n_envs:
            # Reset all environments
            self.pos_abs = pos_abs
            self.quat_abs = quat_abs
        else:
            # Only update specified environments
            self.pos_abs[envs_idx] = pos_abs[envs_idx]
            self.quat_abs[envs_idx] = quat_abs[envs_idx]

        # Get current qpos as initial guess
        current_qpos = self.robot.get_qpos()

        if is_batched:
            # Solve for each environment separately
            qpos_list = []
            env_list = range(self.scene.n_envs) if envs_idx is None else envs_idx
            for batch_idx in env_list:
                pos_np = self.pos_abs[batch_idx].cpu().numpy()
                quat_np = self.quat_abs[batch_idx].cpu().numpy()
                init_qpos_np = current_qpos[batch_idx].cpu().numpy()  # Full qpos including gripper

                qpos_full, _ = self._solve_pink_ik(pos_np, quat_np, init_qpos_np)
                qpos_list.append(torch.tensor(qpos_full, dtype=gs.tc_float))

            qpos_full = torch.stack(qpos_list)
            if envs_idx is not None:
                assert len(envs_idx) == len(qpos_list), f"len(envs_idx)={len(envs_idx)}, len(qpos_list)={len(qpos_list)}"

            # Set gripper to initial gap
            qpos_full[..., -2:] = self.init_gap
            qpos = qpos_full
        else:
            # Single environment
            # Pass full qpos (including gripper) to Pink
            qpos_full, _ = self._solve_pink_ik(
                pos=self.pos_abs.cpu().numpy(),
                quat=self.quat_abs.cpu().numpy(),
                init_qpos=current_qpos.cpu().numpy(),  # Full 9 DOFs
            )
            qpos = torch.tensor(qpos_full, dtype=gs.tc_float)

            # Set gripper to initial gap
            qpos[-2:] = self.init_gap

        self.robot.set_dofs_position(qpos, envs_idx=envs_idx)

        return qpos

    def set_initial_dofs_position(self, qpos, use_initial_pq=True, envs_idx=None):
        """
        Set initial joint positions directly.

        Parameters
        ----------
        qpos : array_like
            Joint positions
        use_initial_pq : bool, optional
            If True, use predefined initial position/quaternion.
            If False, compute from current end-effector pose.
        """
        is_batched = self.scene.n_envs > 0
        if envs_idx is None or len(envs_idx) == self.scene.n_envs:
            # Reset all environments
            if is_batched:
                qpos_full = torch.stack([qpos] * self.scene.n_envs)
            else:
                qpos_full = qpos
            self.robot.set_dofs_position(qpos_full)
        else:
            # Only update specified environments
            qpos_full = torch.stack([qpos] * len(envs_idx))
            self.robot.set_dofs_position(qpos_full, envs_idx=envs_idx)

        if use_initial_pq:
            if envs_idx is None or len(envs_idx) == self.scene.n_envs:
                pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
                quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)
                self.pos_abs = pos_abs
                self.quat_abs = quat_abs
            else:
                pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
                quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)
                self.pos_abs[envs_idx] = pos_abs[envs_idx]
                self.quat_abs[envs_idx] = quat_abs[envs_idx]
        else:
            if envs_idx is None or len(envs_idx) == self.scene.n_envs:
                self.pos_abs = self.ef.get_pos()
                self.quat_abs = self.ef.get_quat()
            else:
                self.pos_abs[envs_idx] = self.ef.get_pos()[envs_idx]
                self.quat_abs[envs_idx] = self.ef.get_quat()[envs_idx]

    def _solve_pink_ik(self, pos, quat, init_qpos, stop_threshold=1e-5):
        """
        Solve IK using Pink for a single configuration.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Target position in WORLD coordinates
        quat : np.ndarray, shape (4,)
            Target quaternion [w, x, y, z] (Genesis convention) in WORLD frame
        init_qpos : np.ndarray, shape (robot_model.nq,)
            Initial joint configuration (full, including gripper)
        stop_threshold : float, optional
            Convergence threshold for stopping the IK solver

        Returns
        -------
        qpos : np.ndarray, shape (robot_model.nq,)
            Solved joint configuration (full, including gripper)
        converged : bool
            Whether the IK solver converged
        """
        # Update configuration with current qpos
        current_q = np.clip(
            init_qpos,
            self.robot_model.lowerPositionLimit,
            self.robot_model.upperPositionLimit,
        )
        self.configuration.update(current_q)

        # Transform target from world coordinates to robot-local coordinates
        # Pinocchio assumes robot base is at origin, but Genesis robot is at robot_base_pos
        robot_base_pos_np = self.robot_base_pos
        robot_base_quat_np = self.robot_base_quat

        # Transform position from world frame to robot-local frame
        # pos_local = R_base^T @ (pos_world - base_pos)
        pos_local = gu.inv_transform_by_trans_quat(pos, robot_base_pos_np, robot_base_quat_np)

        # Transform orientation from world frame to robot-local frame
        # quat_local = quat_world * quat_base^(-1)
        quat_local = gu.transform_quat_by_quat(quat, gu.inv_quat(robot_base_quat_np))

        # Build target pose in robot-local frame
        quat_local_xyzw = np.array([quat_local[1], quat_local[2], quat_local[3], quat_local[0]])  # [w,x,y,z] -> [x,y,z,w]
        rotation = pin.Quaternion(quat_local_xyzw[3], quat_local_xyzw[0], quat_local_xyzw[1], quat_local_xyzw[2]).toRotationMatrix()
        target_pose = pin.SE3(rotation, pos_local)

        # Create frame task
        frame_task = FrameTask(
            self.ee_frame_name,
            position_cost=self.pink_position_cost,
            orientation_cost=self.pink_orientation_cost,
        )
        frame_task.set_target(target_pose)

        # Create posture task
        posture_task = PostureTask(cost=self.pink_posture_cost)
        posture_task.set_target_from_configuration(self.configuration)

        tasks = [frame_task, posture_task]

        # Check solver availability
        solver = self.pink_solver
        if solver not in qpsolvers.available_solvers:
            if "quadprog" in qpsolvers.available_solvers:
                solver = "quadprog"
            elif len(qpsolvers.available_solvers) > 0:
                solver = qpsolvers.available_solvers[0]
            else:
                print("Warning: No QP solvers available")
                return init_qpos

        # Solve IK iteratively
        converged = False

        for iteration in range(self.pink_max_iterations):
            try:
                # Solve for velocity
                velocity = pink.solve_ik(
                    self.configuration,
                    tasks,
                    self.pink_dt,
                    solver=solver,
                )

                # Integrate and clip to limits
                new_q = self.configuration.integrate(velocity, self.pink_dt)
                new_q = np.clip(
                    new_q,
                    self.robot_model.lowerPositionLimit,
                    self.robot_model.upperPositionLimit,
                )
                self.configuration.update(new_q)

                # Check convergence
                error_norm = np.linalg.norm(frame_task.compute_error(self.configuration))
                if error_norm < stop_threshold:
                    converged = True
                    if self.debug:
                        gs.logger.debug(f"  Pink IK converged in {iteration+1} iterations, error norm: {error_norm:.6f}")
                    break

            except pink.exceptions.NoSolutionFound:
                continue
            except Exception as e:
                gs.logger.warning(f"Warning: Pink IK error at iteration {iteration}: {e}")
                break

        return self.configuration.q.copy(), converged

    def set_robot(
        self, g_dof1, g_dof2,
        pos=None, quat=None,
        g_dof_use_force=False, envs_idx=None, **kwargs
    ):
        """
        Control robot end-effector to move by specified deltas using Pink IK.

        Note: Requires dxyzijk to be either all scalars or batch shape matching number of envs
            regardless of whether envs_idx is specified.

        Parameters
        ----------
        g_dof1, g_dof2 : float
            Gripper DOF commands
        dx, dy, dz : float or torch.Tensor
            Position delta in meters
        di, dj, dk : float or torch.Tensor
            Orientation delta (roll, pitch, yaw) in degrees or radians
        g_dof_use_force : bool, optional
            Use force control for gripper (default: False)
        envs_idx : list of int, optional
            Indices of environments to control (default: all)
        **kwargs : optional
            Additional arguments (e.g., min_z for ground constraint)

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        if pos is None:
            target_pos = self.pos_abs
        else:
            target_pos = torch.as_tensor(pos, dtype=gs.tc_float)
            if target_pos.ndim == 1 and self.pos_abs.ndim == 2:
                target_pos = torch.stack([target_pos] * self.scene.n_envs) if self.scene.n_envs > 0 else target_pos
            elif target_pos.ndim != self.pos_abs.ndim:
                raise ValueError("`pos` and `pos_abs` must have the same number of dimensions.")

        # Handle minimum Z constraint
        if kwargs.get('min_z', None) is not None:
            min_z = torch.zeros_like(target_pos[..., 2])
            min_z.fill_(kwargs.pop('min_z'))
            kwargs.update({'underground': (target_pos[..., 2] < min_z).any()})
            target_pos[..., 2] = torch.maximum(target_pos[..., 2], min_z)

        if quat is None:
            target_quat = self.quat_abs
        else:
            target_quat = torch.as_tensor(quat, dtype=gs.tc_float)
            if target_quat.ndim == 1 and self.quat_abs.ndim == 2:
                target_quat = torch.stack([target_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else target_quat
            elif target_quat.ndim != self.quat_abs.ndim:
                raise ValueError("`quat` and `quat_abs` must have the same number of dimensions.")

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, envs_idx=envs_idx, control=False, **kwargs)

        return qpos

    def control_robot(
        self, g_dof1, g_dof2,
        dx=0., dy=0., dz=0., di=0., dj=0., dk=0.,
        g_dof_use_force=False, degrees=True, envs_idx=None, **kwargs
    ):
        """
        Control robot end-effector to move by specified deltas using Pink IK.

        Note: Requires dxyzijk to be either all scalars or batch shape matching number of envs
            regardless of whether envs_idx is specified.

        Parameters
        ----------
        g_dof1, g_dof2 : float
            Gripper DOF commands
        dx, dy, dz : float or torch.Tensor
            Position delta in meters
        di, dj, dk : float or torch.Tensor
            Orientation delta (roll, pitch, yaw) in degrees or radians
        g_dof_use_force : bool, optional
            Use force control for gripper (default: False)
        degrees : bool, optional
            Interpret di, dj, dk as degrees (default: True)
        envs_idx : list of int, optional
            Indices of environments to control (default: all)
        **kwargs : optional
            Additional arguments (e.g., min_z for ground constraint)

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        # Compute target position
        if isinstance(dx, (float, int)) and isinstance(dy, (float, int)) and isinstance(dz, (float, int)):
            delta_pos = torch.tensor([dx, dy, dz], dtype=gs.tc_float)
        else:
            delta_pos = torch.stack([dx, dy, dz], dim=-1).to(gs.tc_float)
            assert delta_pos.shape[0] == self.scene.n_envs, \
                f"dxyz has batch size {delta_pos.shape[0]}, but scene has {self.scene.n_envs} envs."
        target_pos = self.pos_abs + delta_pos

        # Handle minimum Z constraint
        if kwargs.get('min_z', None) is not None:
            min_z = torch.zeros_like(target_pos[..., 2])
            min_z.fill_(kwargs.pop('min_z'))
            kwargs.update({'underground': (target_pos[..., 2] < min_z).any()})
            target_pos[..., 2] = torch.maximum(target_pos[..., 2], min_z)

        # Handle maximum Z constraint
        if kwargs.get('max_z', None) is not None:
            max_z = torch.zeros_like(target_pos[..., 2])
            max_z.fill_(kwargs.pop('max_z'))
            kwargs.update({'overground': (target_pos[..., 2] > max_z).any()})
            target_pos[..., 2] = torch.minimum(target_pos[..., 2], max_z)

        # Handle feasible region constraint
        if kwargs.get('feasible_region', None) is not None:
            feasible_region = kwargs.pop('feasible_region')
            x_min, x_max, y_min, y_max, z_min, z_max = feasible_region
            target_pos[..., 0] = torch.clamp(target_pos[..., 0], x_min, x_max)
            target_pos[..., 1] = torch.clamp(target_pos[..., 1], y_min, y_max)
            target_pos[..., 2] = torch.clamp(target_pos[..., 2], z_min, z_max)

        # Compute target orientation
        if isinstance(di, (float, int)) and isinstance(dj, (float, int)) and isinstance(dk, (float, int)):
            delta_orient = torch.tensor([di, dj, dk], dtype=gs.tc_float)
        else:
            delta_orient = torch.stack([di, dj, dk], dim=-1).to(gs.tc_float)
            assert delta_orient.shape[0] == self.scene.n_envs, \
                f"dijk has batch size {delta_orient.shape[0]}, but scene has {self.scene.n_envs} envs."
        delta_quat = gu.xyz_to_quat(delta_orient, rpy=True, degrees=degrees)

        if delta_quat.ndim == 1 and self.quat_abs.ndim == 2:
            delta_quat = torch.stack([delta_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else delta_quat
        elif delta_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`delta_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(delta_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, envs_idx=envs_idx, **kwargs)

        return qpos

    def rotate_around_point(
        self, g_dof1, g_dof2, center, axis, angle, pos_angle=None,
        g_dof_use_force=False, degrees=True, **kwargs
    ):
        """
        Rotate robot end-effector around a specified world-space point using Pink IK.

        Parameters
        ----------
        g_dof1, g_dof2 : float
            Gripper DOF commands
        center : array_like, shape (3,)
            Center point for rotation
        axis : array_like, shape (3,)
            Rotation axis (will be normalized)
        angle : float
            Rotation angle for orientation
        pos_angle : float, optional
            Rotation angle for position (default: same as angle)
        g_dof_use_force : bool, optional
            Use force control for gripper
        degrees : bool, optional
            Interpret angles as degrees
        **kwargs : optional
            Additional arguments

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        center_tensor = torch.as_tensor(center, dtype=gs.tc_float)
        axis_tensor = torch.as_tensor(axis, dtype=gs.tc_float)

        position_angle = angle if pos_angle is None else pos_angle

        angle_tensor = torch.tensor(angle, dtype=gs.tc_float)
        pos_angle_tensor = torch.tensor(position_angle, dtype=gs.tc_float)

        orient_angle_rad = torch.deg2rad(angle_tensor) if degrees else angle_tensor
        pos_angle_rad = torch.deg2rad(pos_angle_tensor) if degrees else pos_angle_tensor

        orient_rotation_quat = gu.axis_angle_to_quat(orient_angle_rad, axis_tensor)
        pos_rotation_quat = gu.axis_angle_to_quat(pos_angle_rad, axis_tensor)

        vec_to_pos = self.pos_abs - center_tensor
        rotated_vec = gu.transform_by_quat(vec_to_pos, pos_rotation_quat)
        target_pos = center_tensor + rotated_vec

        if orient_rotation_quat.ndim == 1 and self.quat_abs.ndim == 2:
            orient_rotation_quat = torch.stack([orient_rotation_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else orient_rotation_quat
        elif orient_rotation_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`orient_rotation_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(orient_rotation_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

        return qpos

    def _execute_ik_control(self, target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, envs_idx=None, control=True, **kwargs):
        """
        Execute inverse kinematics and send control commands using Pink.

        Parameters
        ----------
        target_pos : torch.Tensor
            Target end-effector position
        target_quat : torch.Tensor
            Target end-effector quaternion
        g_dof1, g_dof2 : float
            Gripper DOF commands
        g_dof_use_force : bool
            Use force control for gripper
        envs_idx : list or None, optional
            Indices of environments to control (default: None, control all)
        control : bool, optional
            If True, use control commands; if False, set positions directly (default: True)
        **kwargs : optional
            Additional arguments (e.g., underground flag for debug)

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        is_batched = self.scene.n_envs > 0
        underground = kwargs.pop('underground', False)

        # Handle Z rectification
        if kwargs.get('rectify_z', None) is not None:
            rectify_z = kwargs.pop('rectify_z')
            assert isinstance(rectify_z, (float, int)), "`rectify_z` must be a scalar value."
            target_pos[..., 2] += rectify_z
        else:
            rectify_z = 0.0

        # Debug visualization
        if self.debug:
            for i in self.debug_point_nodes:
                self.scene.clear_debug_object(i)
            self.debug_point_nodes = list()
            env_list = range(max(self.scene.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]
            for batch_idx in env_list:
                if is_batched:
                    offset = self.scene.envs_offset[batch_idx]
                    offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
                else:
                    offset = torch.zeros(3, dtype=target_pos.dtype, device=target_pos.device)
                color = (1.0, 1.0, 0.0, 0.6) if underground else (0.0, 1.0, 0.0, 0.6)
                self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                    pos=target_pos[batch_idx] + offset if is_batched else target_pos,
                    radius=0.01,
                    color=color
                ))

        # Get current joint positions
        current_qpos = self.robot.get_qpos()

        # Convergence state
        convergence = list()

        # Solve IK using Pink
        if is_batched:
            # Solve for each environment separately
            qpos_list = []
            env_list = range(self.scene.n_envs) if envs_idx is None else envs_idx
            for batch_idx in env_list:
                pos_np = target_pos[batch_idx].cpu().numpy()
                quat_np = target_quat[batch_idx].cpu().numpy()
                init_qpos_np = current_qpos[batch_idx].cpu().numpy()  # Full qpos including gripper

                qpos_full, converged = self._solve_pink_ik(pos_np, quat_np, init_qpos_np)
                if converged:
                    qpos_list.append(torch.tensor(qpos_full, dtype=gs.tc_float))
                else:
                    # If not converged, keep current qpos
                    qpos_list.append(current_qpos[batch_idx])
                convergence.append(converged)

            qpos = torch.stack(qpos_list)
            if envs_idx is not None:
                assert len(envs_idx) == len(qpos_list), f"len(envs_idx)={len(envs_idx)}, len(qpos_list)={len(qpos_list)}"
            convergence = np.array(convergence)

            # Override gripper DOFs with commanded values
            qpos[:, -2] = g_dof1
            qpos[:, -1] = g_dof2
        else:
            # Single environment
            pos_np = target_pos.cpu().numpy()
            quat_np = target_quat.cpu().numpy()
            init_qpos_np = current_qpos.cpu().numpy()  # Full qpos including gripper

            qpos_full, converged = self._solve_pink_ik(pos_np, quat_np, init_qpos_np)
            if converged:
                qpos = torch.tensor(qpos_full, dtype=gs.tc_float)
            else:
                qpos = current_qpos
            convergence = np.array([converged])

            # Override gripper DOFs with commanded values
            qpos[-2] = g_dof1
            qpos[-1] = g_dof2

        # Send control commands
        if control:
            self.robot.control_dofs_position(qpos[..., :-2], self.motors_dof, envs_idx=envs_idx)
        else:
            self.robot.set_dofs_position(qpos[..., :-2], self.motors_dof, envs_idx=envs_idx)

        n_envs = self.scene.n_envs if envs_idx is None else len(envs_idx)
        if g_dof_use_force:
            gripper_arg = torch.tensor([[g_dof1, g_dof2]] * n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])
            if control:
                self.robot.control_dofs_force(gripper_arg, self.fingers_dof, envs_idx=envs_idx)
            else:
                self.robot.set_dofs_force(gripper_arg, self.fingers_dof, envs_idx=envs_idx)
        else:
            gripper_arg = torch.tensor([[g_dof1, g_dof2]] * n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])
            if control:
                self.robot.control_dofs_position(gripper_arg, self.fingers_dof, envs_idx=envs_idx)
            else:
                self.robot.set_dofs_position(gripper_arg, self.fingers_dof, envs_idx=envs_idx)

        # Update internal state
        self.pos_abs = target_pos - torch.tensor([0., 0., rectify_z], dtype=gs.tc_float)
        self.quat_abs = target_quat

        if is_batched:
            self.convergence = np.ones(self.scene.n_envs, dtype=bool)
            if envs_idx is None:
                self.convergence = convergence
            else:
                idx_ = [int(i) for i in envs_idx]
                self.convergence[np.array(idx_)] = convergence
        else:
            self.convergence = convergence[0]

        return qpos

    def draw_debug_point(self, delta_pos, min_z):
        delta_pos = torch.as_tensor(delta_pos)
        target_pos = self.pos_abs + delta_pos
        self.pos_abs = target_pos
        if self.debug:
            for i in self.debug_point_nodes:
                self.scene.clear_debug_object(i)
            self.debug_point_nodes = list()
            for batch_idx in range(self.scene.n_envs if self.scene.n_envs > 0 else 1):
                if self.scene.n_envs > 0:
                    offset = self.scene.envs_offset[batch_idx]
                    offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
                else:
                    offset = torch.zeros(3, dtype=target_pos.dtype, device=target_pos.device)
                if target_pos[batch_idx, 2] < min_z:
                    color = (1.0, 1.0, 0.0, 0.6)
                    target_pos[batch_idx, 2] = min_z
                else:
                    color = (0.0, 1.0, 0.0, 0.6)
                self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                    pos=target_pos[batch_idx] + offset if self.scene.n_envs > 0 else target_pos,
                    radius=0.01,
                    color=color
                ))

class TrajOptimController:
    def __init__(
        self,
        scene,
        rod,
        grasp_point_ids,
        n_stages=10,
        n_optim_dofs=3,
        max_ddist=0.05,
        max_grad_norm=1000.,
        use_adam=False,
        adam_config=None,
        debug=False,
        # lr scheduler
        lr_scheduler=None,
    ):
        self.scene = scene
        self.rod = rod
        self.grasp_point_ids = grasp_point_ids
        self.n_grasp_points = len(grasp_point_ids)

        self.traj = torch.zeros(
            size=(self.scene.n_envs, n_stages, self.n_grasp_points, n_optim_dofs), dtype=gs.tc_float
        )

        # for Adam optimizer
        self.use_adam = use_adam
        if self.use_adam:
            self.m_buffer = torch.zeros_like(self.traj)
            self.v_buffer = torch.zeros_like(self.traj)
            if adam_config is None:
                adam_config = {
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "eps": 1e-8,
                }
            else:
                if "beta1" not in adam_config:
                    adam_config["beta1"] = 0.9
                if "beta2" not in adam_config:
                    adam_config["beta2"] = 0.99
                if "eps" not in adam_config:
                    adam_config["eps"] = 1e-8
            self.adam_config = adam_config
            print(f'Using Adam optimizer with config:\n{self.adam_config}')

        if lr_scheduler is None:
            self.lr_scheduler = None
            print('No learning rate scheduler used.')
        elif lr_scheduler == 'cosine':
            self.lr_scheduler = cosine_learning_rate_scheduler
            print('Using cosine learning rate scheduler.')
        else:
            raise ValueError(f'Unknown learning rate scheduler: {lr_scheduler}')

        self.n_stages = n_stages
        self.n_optim_dofs = n_optim_dofs
        self.max_ddist = max_ddist
        self.max_grad_norm = max_grad_norm

        self._lr = 0.

        self.debug = debug
        self.debug_point_nodes = list()

    def pre_apply_grad(self, stage_idx):
        dpos = self.traj[:, stage_idx, :, :]
        return dpos   # (n_envs, n_grasp_points, 3)

    def post_check(self, stage_idx, alive):
        # zero out the traj for dead envs from this stage onwards
        self.traj[~alive, stage_idx:, :, :] = 0.0

    def gather_grad(self, stage_idx, horizon_idx, cur_step=None, max_step=None, lr=0.01, lr_min=1e-6):
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(base_lr=lr, cur_iter=cur_step, max_iter=max_step, min_lr=lr_min)

        self._lr = lr

        grad = self.rod._queried_states[horizon_idx][0].pos.grad

        # [n_envs, n_grasp_points, 3]
        contact_grad = grad[:, self.grasp_point_ids, :]
        # replace NaN or Inf with 0
        contact_grad = torch.where(torch.isnan(contact_grad), torch.zeros_like(contact_grad), contact_grad)
        contact_grad = torch.where(torch.isinf(contact_grad), torch.zeros_like(contact_grad), contact_grad)

        # clip gradient
        grad_norm = torch.linalg.norm(contact_grad, dim=-1)
        weight = self.max_grad_norm / (grad_norm + gs.EPS)
        contact_grad = contact_grad * torch.minimum(weight, torch.ones_like(weight))[:, :, None]

        if self.use_adam:
            # Adam
            beta1 = self.adam_config["beta1"]
            beta2 = self.adam_config["beta2"]
            eps = self.adam_config["eps"]

            m_t = beta1 * self.m_buffer[:, stage_idx, :, :] + (1 - beta1) * contact_grad
            v_t = beta2 * self.v_buffer[:, stage_idx, :, :] + (1 - beta2) * (contact_grad ** 2)
            self.m_buffer[:, stage_idx, :, :] = m_t
            self.v_buffer[:, stage_idx, :, :] = v_t

            m_cap = m_t / (1 - beta1 ** (cur_step + 1))
            v_cap = v_t / (1 - beta2 ** (cur_step + 1))

            d_pos = -lr * m_cap / (torch.sqrt(v_cap) + eps)
        else:
            # SGD
            d_pos = -lr * contact_grad

        self.traj[:, stage_idx, :, :] += d_pos

        # ensure the max step distance constraint
        delta_dis = self.traj[:, stage_idx, :, :]
        ddist = torch.linalg.norm(delta_dis, dim=-1)
        weight = self.max_ddist / (ddist + gs.EPS)
        self.traj[:, stage_idx, :, :] = delta_dis * torch.minimum(weight, torch.ones_like(weight))[:, :, None]
