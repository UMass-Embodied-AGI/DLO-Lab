import numpy as np
import quadrants as qd

import torch
import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import RODEntityState
from genesis.utils.sdf import sdf_func_find_closest_vert
from genesis.utils.misc import to_gs_tensor, tensor_to_array

from .base_entity import Entity


@qd.data_oriented
class RODEntity(Entity):
    """
    A discrete linear object (DLO)-based entity for rod simulation.

    This class represents a deformable object using tetrahedral elements. It interfaces with
    the physics solver to handle state updates, checkpointing, gradients, and actuation
    for physics-based simulation in batched environments.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : Solver
        The physics solver instance used for simulation.
    material : Material
        The material properties defining elasticity, density, etc.
    morph : Morph
        The morph specification that defines the entity's shape.
    surface : Surface
        The surface mesh associated with the entity (for rendering or collision).
    idx : int
        Unique identifier of the entity within the scene.
    rod_idx : int, optional
        Index of this rod in the solver (default is 0).
    v_start : int, optional
        Starting index of this entity's vertices in the global vertex array (default is 0).
    e_start : int, optional
        Starting index of this entity's edges in the global edge array (default is 0).
    iv_start : int, optional
        Starting index of this entity's internal vertices in the global internal vertex array (default is 0).
    visualize_twist : bool, optional
        Whether to visualize twist frames applied to this (Rod) entity as arrows in the viewer and rendered images.
        Note that this will not be displayed in images rendered by camera using the `RayTracer` renderer.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        rod_idx=0,
        v_start=0,
        e_start=0,
        iv_start=0,
        visualize_twist=False,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._rod_idx = rod_idx     # index of this rod in the solver
        self._v_start = v_start     # offset for vertex index
        self._e_start = e_start     # offset for edge index
        self._iv_start = iv_start   # offset for internal vertex index
        self._step_global_added = None
        self._visualize_twist = visualize_twist

        self._surface.update_texture()

        self.sample()

        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_position(self, pos):
        """
        Set the target position(s) for the Rod entity.

        Parameters
        ----------
        pos : torch.Tensor or array-like
            The desired position(s). Can be:
            - (3,): a single COM offset vector.
            - (n_vertices, 3): per-vertex positions for all vertices.
            - (n_envs, 3): per-environment COM offsets.
            - (n_envs, n_vertices, 3): full batched per-vertex positions.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self.init_positions_COM_offset + pos
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self.n_vertices, 3):
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self.init_positions_COM_offset.unsqueeze(0) + pos.unsqueeze(1)
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_velocity(self, vel):
        """
        Set the target velocity(ies) for the Rod entity.

        Parameters
        ----------
        vel : torch.Tensor or array-like
            The desired velocity(ies). Can be:
            - (3,): a global velocity vector for all vertices.
            - (n_vertices, 3): per-vertex velocities.
            - (n_envs, 3): per-environment velocities broadcast to all vertices.
            - (n_envs, n_vertices, 3): full batched per-vertex velocities.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self.n_vertices, 3):
                self._tgt["vel"] = vel.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel.unsqueeze(1).tile((1, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def get_state(self):
        state = RODEntityState(self, self._sim.cur_step_global)
        self.get_frame(
            self._sim.cur_substep_local,
            state.pos,
            state.vel,
            state.fixed,
            state.theta,
            state.omega,
            state.edge,
            state.length,
            state.d1,
            state.d2,
            state.d3,
            state.d1_ref,
            state.d2_ref,
            state.kb,
            state.twist,
            state.kappa_rest,
            state.collided,
            state.collision_normal,
            state.collision_penetration
        )

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def get_kinematic_indices(self, envs_idx=0):
        kinematic_indices = list()
        f = self._sim.cur_substep_local
        for i_v in range(self.n_vertices):
            i_global = self._v_start + i_v
            if self.solver.vertices_ng[f, i_v, envs_idx].kinematic:
                kinematic_indices.append(i_global)
        return kinematic_indices

    # def deactivate(self):         # NOTE: Not used
    #     gs.logger.info(f"{self.__class__.__name__} <{self.id}> deactivated.")
    #     self._tgt["act"] = gs.INACTIVE
    #     self.active = False

    # def activate(self):           # NOTE: Not used
    #     gs.logger.info(f"{self.__class__.__name__} <{self.id}> activated.")
    #     self._tgt["act"] = gs.ACTIVE
    #     self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, rest_state):
        """
        Initialize Rod entity with given vertices.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        Raises
        ------
        Exception
            If no vertices are provided.
        """
        verts = verts.astype(gs.np_float, copy=False)
        n_verts = verts.shape[0]

        # rotate
        R = gu.quat_to_R(np.array(self.morph.quat, dtype=gs.np_float))
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM

        if not init_positions.shape[0] > 0:
            gs.raise_exception(f"Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.init_positions_COM_offset = self.init_positions - gs.tensor(verts_COM)

        edges = list()
        is_loop = self.is_loop
        n_edges = n_verts if is_loop else n_verts - 1
        for i in range(n_edges):
            pos_a = init_positions[i]
            pos_b = init_positions[(i + 1) % n_verts]
            edges.append(pos_b - pos_a)
        edges = np.array(edges, dtype=gs.np_float)

        self.edges = gs.tensor(edges)

        # resolve rest state
        if rest_state == "default":
            self.rest_positions = self.init_positions.clone()
        elif rest_state == "straight":
            # definitely not loop
            rest_positions = np.zeros_like(init_positions)
            lengths = np.linalg.norm(edges, axis=1)
            # create a straight line along x axis
            for i in range(n_edges):
                rest_positions[i + 1][0] = rest_positions[i][0] + lengths[i]
            self.rest_positions = gs.tensor(rest_positions)
            gs.logger.info(
                f"Entity {self.uid}({self._rod_idx}) initialized with rest state 'straight', "
                f"min_el: {lengths.min():.2e}, max_el: {lengths.max():.2e}, mean_el: {lengths.mean():.2e}."
            )
        else:
            gs.raise_exception(f"Unsupported rest state {rest_state}.")

    def _sample_rod(self, n_vertices: int, interval: float, axis: int):
        verts = list()
        for i in range(n_vertices):
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = i * interval
            verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def _sample_circle(self, n_vertices: int, radius: float, axis: int, gap: int):
        verts = list()
        for i in range(n_vertices):
            theta = 2 * np.pi * i / (n_vertices + gap)     # +1 to avoid overlap at the end
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = radius * np.cos(theta)
            vert[(axis + 1) % 3] = radius * np.sin(theta)
            verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def _sample_half_circle(self, n_vertices: int, radius: float, axis: int, gap: int):
        verts = list()
        for i in range(n_vertices + 2 * gap):
            theta = np.pi * i / (n_vertices + 2 * gap - 1)  # Adjusted to cover half circle
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = radius * np.cos(theta)
            vert[(axis + 1) % 3] = radius * np.sin(theta)
            if gap <= i < n_vertices + gap:
                verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def sample(self):
        """
        Sample mesh and elements based on the entity's morph type.

        Raises
        ------
        Exception
            If the morph type is unsupported.
        """

        file_path = getattr(self.morph, 'file', None)
        if file_path is None:
            # Parametric morph
            if self.morph.axis == "x":
                axis = 0
            elif self.morph.axis == "y":
                axis = 1
            elif self.morph.axis == "z":
                axis = 2
            else:
                gs.raise_exception(f"Unsupported axis {self.morph.axis}.")

            if self.morph.type == "rod":
                vertices = self._sample_rod(self.morph.n_vertices, self.morph.interval, axis)
            elif self.morph.type == "circle":
                vertices = self._sample_circle(self.morph.n_vertices, self.morph.radius, axis, self.morph.gap)
            elif self.morph.type == "half_circle":
                vertices = self._sample_half_circle(self.morph.n_vertices, self.morph.radius, axis, self.morph.gap)
            else:
                gs.raise_exception(f"Unsupported morph type {self.morph.type}.")
            vertices = vertices + self.morph.pos
        else:
            vertices = np.load(self.morph.file)
            assert vertices.ndim == 2, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
            assert vertices.shape[1] == 3, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
            vertices = vertices + self.morph.pos

        self.instantiate(vertices, self.morph.rest_state)

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid}({self._rod_idx}) added. class: {self.__class__.__name__}, "
                f"morph: {self.morph.__class__.__name__}, #v: {self.n_vertices}, "
                f"o: {self.is_loop}, fix: {self.morph.fixed}, material: {self.material}."
            )

        # Convert to appropriate numpy array types
        verts_np = tensor_to_array(self.init_positions, dtype=gs.np_float)
        rest_verts_np = tensor_to_array(self.rest_positions, dtype=gs.np_float)
        edges_np = tensor_to_array(self.edges, dtype=gs.np_float)

        self._solver._kernel_add_rods(
            rod_idx=self._rod_idx,
            is_loop=self.is_loop,
            use_inextensible=self.material.use_inextensible,
            stretching_stiffness=self.material.K,
            bending_stiffness=self.material.E,
            twisting_stiffness=self.material.G,
            plastic_yield=self.material.plastic_yield,
            plastic_creep=self.material.plastic_creep,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            n_verts=self.n_vertices,
        )

        self._solver._kernel_finalize_rest_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            segment_mass=self.material.segment_mass,
            segment_radius=self.material.segment_radius,
            static_friction=self.material.static_friction,
            kinetic_friction=self.material.kinetic_friction,
            restitution=self.material.restitution,
            verts_rest=rest_verts_np,
            edges_rest=edges_np,
        )

        self._solver._kernel_finalize_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            fixed=self.morph.fixed,
            verts=verts_np,
            edges=edges_np,
        )
        self.active = True

    # ------------------------------------------------------------------------------------
    # ---------------------------- checkpoint and buffer ---------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_keys(self):
        """
        Initialize the keys used in target state management.

        This defines which physical properties (e.g., position, velocity) will be tracked for checkpointing and buffering.
        """
        self._tgt_keys = [
            # vertex states
            "pos", "vel", "fixed",
            # edge states
            "edge", "length", "d1", "d2", "d3", "d1_ref", "d2_ref", "theta", "omega",
            # internal vertex states
            "kb", "twist", "kappa_rest",
        ]

    def init_tgt_vars(self):
        """
        Initialize the target state variables and their buffers.

        This sets up internal dictionaries to store per-step target values for properties like velocity, position, actuation, and activation.
        """

        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        """
        Initialize checkpoint storage for simulation state.
        """
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        """
        Save the current target state buffers to a checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to save.
        """
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {
                "_tgt_buffer": dict(),
            }

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        """
        Restore target state buffers from a previously saved checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to load.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        """
        Clear target buffers and any externally queried simulation states.

        Used before backpropagation to reset gradients.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]

        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["vel"])

        if self._tgt["fixed"] is not None:
            self._tgt["fixed"].assert_contiguous()
            self._tgt["fixed"].assert_sceneless()
            self.set_fixed(self._sim.cur_substep_local, self._tgt["fixed"])

        if self._tgt["edge"] is not None:
            self._tgt["edge"].assert_contiguous()
            self._tgt["edge"].assert_sceneless()
            self.set_edge(self._sim.cur_substep_local, self._tgt["edge"])

        if self._tgt["length"] is not None:
            self._tgt["length"].assert_contiguous()
            self._tgt["length"].assert_sceneless()
            self.set_length(self._sim.cur_substep_local, self._tgt["length"])

        if self._tgt["d1"] is not None:
            self._tgt["d1"].assert_contiguous()
            self._tgt["d1"].assert_sceneless()
            self.set_d1(self._sim.cur_substep_local, self._tgt["d1"])

        if self._tgt["d2"] is not None:
            self._tgt["d2"].assert_contiguous()
            self._tgt["d2"].assert_sceneless()
            self.set_d2(self._sim.cur_substep_local, self._tgt["d2"])

        if self._tgt["d3"] is not None:
            self._tgt["d3"].assert_contiguous()
            self._tgt["d3"].assert_sceneless()
            self.set_d3(self._sim.cur_substep_local, self._tgt["d3"])

        if self._tgt["d1_ref"] is not None:
            self._tgt["d1_ref"].assert_contiguous()
            self._tgt["d1_ref"].assert_sceneless()
            self.set_d1_ref(self._sim.cur_substep_local, self._tgt["d1_ref"])

        if self._tgt["d2_ref"] is not None:
            self._tgt["d2_ref"].assert_contiguous()
            self._tgt["d2_ref"].assert_sceneless()
            self.set_d2_ref(self._sim.cur_substep_local, self._tgt["d2_ref"])

        if self._tgt["theta"] is not None:
            self._tgt["theta"].assert_contiguous()
            self._tgt["theta"].assert_sceneless()
            self.set_theta(self._sim.cur_substep_local, self._tgt["theta"])

        if self._tgt["omega"] is not None:
            self._tgt["omega"].assert_contiguous()
            self._tgt["omega"].assert_sceneless()
            self.set_omega(self._sim.cur_substep_local, self._tgt["omega"])

        if self._tgt["kb"] is not None:
            self._tgt["kb"].assert_contiguous()
            self._tgt["kb"].assert_sceneless()
            self.set_kb(self._sim.cur_substep_local, self._tgt["kb"])

        if self._tgt["twist"] is not None:
            self._tgt["twist"].assert_contiguous()
            self._tgt["twist"].assert_sceneless()
            self.set_twist(self._sim.cur_substep_local, self._tgt["twist"])

        if self._tgt["kappa_rest"] is not None:
            self._tgt["kappa_rest"].assert_contiguous()
            self._tgt["kappa_rest"].assert_sceneless()
            self.set_kappa_rest(self._sim.cur_substep_local, self._tgt["kappa_rest"])

        for key in self._tgt_keys:
            self._tgt[key] = None

    def process_input_grad(self):
        """
        Process gradients of input states and propagate them backward.

        Notes
        -----
        Automatically applies the backward hooks for position and velocity tensors.
        Clears the gradients in the solver to avoid double accumulation.
        """

        _tgt_pos = self._tgt_buffer["pos"].pop()
        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

        _tgt_vel = self._tgt_buffer["vel"].pop()
        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        _tgt_edge = self._tgt_buffer["edge"].pop()
        if _tgt_edge is not None and _tgt_edge.requires_grad:
            _tgt_edge._backward_from_ti(self.set_edge_grad, self._sim.cur_substep_local)

        _tgt_length = self._tgt_buffer["length"].pop()
        if _tgt_length is not None and _tgt_length.requires_grad:
            _tgt_length._backward_from_ti(self.set_length_grad, self._sim.cur_substep_local)

        _tgt_d1 = self._tgt_buffer["d1"].pop()
        if _tgt_d1 is not None and _tgt_d1.requires_grad:
            _tgt_d1._backward_from_ti(self.set_d1_grad, self._sim.cur_substep_local)

        _tgt_d2 = self._tgt_buffer["d2"].pop()
        if _tgt_d2 is not None and _tgt_d2.requires_grad:
            _tgt_d2._backward_from_ti(self.set_d2_grad, self._sim.cur_substep_local)

        _tgt_d3 = self._tgt_buffer["d3"].pop()
        if _tgt_d3 is not None and _tgt_d3.requires_grad:
            _tgt_d3._backward_from_ti(self.set_d3_grad, self._sim.cur_substep_local)

        _tgt_d1_ref = self._tgt_buffer["d1_ref"].pop()
        if _tgt_d1_ref is not None and _tgt_d1_ref.requires_grad:
            _tgt_d1_ref._backward_from_ti(self.set_d1_ref_grad, self._sim.cur_substep_local)

        _tgt_d2_ref = self._tgt_buffer["d2_ref"].pop()
        if _tgt_d2_ref is not None and _tgt_d2_ref.requires_grad:
            _tgt_d2_ref._backward_from_ti(self.set_d2_ref_grad, self._sim.cur_substep_local)

        _tgt_theta = self._tgt_buffer["theta"].pop()
        if _tgt_theta is not None and _tgt_theta.requires_grad:
            _tgt_theta._backward_from_ti(self.set_theta_grad, self._sim.cur_substep_local)

        _tgt_omega = self._tgt_buffer["omega"].pop()
        if _tgt_omega is not None and _tgt_omega.requires_grad:
            _tgt_omega._backward_from_ti(self.set_omega_grad, self._sim.cur_substep_local)

        _tgt_kb = self._tgt_buffer["kb"].pop()
        if _tgt_kb is not None and _tgt_kb.requires_grad:
            _tgt_kb._backward_from_ti(self.set_kb_grad, self._sim.cur_substep_local)

        _tgt_twist = self._tgt_buffer["twist"].pop()
        if _tgt_twist is not None and _tgt_twist.requires_grad:
            _tgt_twist._backward_from_ti(self.set_twist_grad, self._sim.cur_substep_local)

        _tgt_kappa_rest = self._tgt_buffer["kappa_rest"].pop()
        if _tgt_kappa_rest is not None and _tgt_kappa_rest.requires_grad:
            _tgt_kappa_rest._backward_from_ti(self.set_kappa_rest_grad, self._sim.cur_substep_local)

        # Manually zero the grad since manually setting state breaks gradient flow
        if _tgt_vel is not None or _tgt_pos is not None:
            self._reset_grad()

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.

        Returns True if any queried-state gradient was injected at the current step (used by
        truncated BPTT to reset its window).
        """
        injected = False
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)
                injected = True
        return injected

    def distribute_output_grads(self):
        """
        Copy Taichi gradients back to PyTorch state tensors after physics backward.

        This is the reverse of collect_output_grads.
        """
        if (self._sim.cur_step_global + 1) in self._queried_states:
            # Copy gradients from Taichi to PyTorch for all states at this timestep
            for state in self._queried_states[self._sim.cur_step_global + 1]:
                self.distribute_grad_to_state(state)

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(self, f, pos):
        """
        Set vertex positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing new positions.
        """

        self._solver._kernel_set_vertices_pos(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos=pos,
        )

    def set_pos_grad(self, f, pos_grad):
        """
        Set gradient of vertex positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of positions.
        """

        self._solver._kernel_set_vertices_pos_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set vertex velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing velocities.
        """

        self._solver._kernel_set_vertices_vel(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradient of vertex velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of velocities.
        """

        self._solver._kernel_set_vertices_vel_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    def set_edge(self, f, edge):
        """
        Set edge directions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        edge : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing edge directions.
        """

        self._solver._kernel_set_edges_edge(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            edge=edge,
        )

    def set_edge_grad(self, f, edge_grad):
        """
        Set gradient of edge directions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        edge_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of edge directions.
        """

        self._solver._kernel_set_edges_edge_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            edge_grad=edge_grad,
        )

    def set_length(self, f, length):
        """
        Set edge lengths in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        length : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing edge lengths.
        """

        self._solver._kernel_set_edges_length(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            length=length,
        )

    def set_length_grad(self, f, length_grad):
        """
        Set gradient of edge lengths in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        length_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing gradients of edge lengths.
        """

        self._solver._kernel_set_edges_length_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            length_grad=length_grad,
        )

    def set_d1(self, f, d1):
        """
        Set edge material frame d1 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d1 : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing d1 vectors.
        """

        self._solver._kernel_set_edges_d1(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d1=d1,
        )

    def set_d1_grad(self, f, d1_grad):
        """
        Set gradient of edge material frame d1 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d1_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of d1 vectors.
        """

        self._solver._kernel_set_edges_d1_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d1_grad=d1_grad,
        )

    def set_d2(self, f, d2):
        """
        Set edge material frame d2 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d2 : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing d2 vectors.
        """

        self._solver._kernel_set_edges_d2(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d2=d2,
        )

    def set_d2_grad(self, f, d2_grad):
        """
        Set gradient of edge material frame d2 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d2_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of d2 vectors.
        """

        self._solver._kernel_set_edges_d2_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d2_grad=d2_grad,
        )

    def set_d3(self, f, d3):
        """
        Set edge material frame d3 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d3 : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing d3 vectors.
        """

        self._solver._kernel_set_edges_d3(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d3=d3,
        )

    def set_d3_grad(self, f, d3_grad):
        """
        Set gradient of edge material frame d3 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d3_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of d3 vectors.
        """

        self._solver._kernel_set_edges_d3_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d3_grad=d3_grad,
        )

    def set_d1_ref(self, f, d1_ref):
        """
        Set reference edge material frame d1 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d1_ref : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing reference d1 vectors.
        """

        self._solver._kernel_set_edges_d1_ref(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d1_ref=d1_ref,
        )
    
    def set_d1_ref_grad(self, f, d1_ref_grad):
        """
        Set gradient of reference edge material frame d1 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d1_ref_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of reference d1 vectors.
        """

        self._solver._kernel_set_edges_d1_ref_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d1_ref_grad=d1_ref_grad,
        )
    
    def set_d2_ref(self, f, d2_ref):
        """
        Set reference edge material frame d2 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d2_ref : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing reference d2 vectors.
        """

        self._solver._kernel_set_edges_d2_ref(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d2_ref=d2_ref,
        )
    
    def set_d2_ref_grad(self, f, d2_ref_grad):
        """
        Set gradient of reference edge material frame d2 vectors in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        d2_ref_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges, 3) containing gradients of reference d2 vectors.
        """

        self._solver._kernel_set_edges_d2_ref_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            d2_ref_grad=d2_ref_grad,
        )

    def set_theta(self, f, theta):
        """
        Set edge twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        theta : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing twist angles.
        """

        self._solver._kernel_set_edges_theta(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega=theta,
        )

    def set_theta_grad(self, f, theta_grad):
        """
        Set gradient of edge twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        theta_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing gradients of twist angles.
        """

        self._solver._kernel_set_edges_theta_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            theta_grad=theta_grad,
        )

    def set_omega(self, f, omega):
        """
        Set edge angular velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        omega : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing angular velocities.
        """

        self._solver._kernel_set_edges_omega(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega=omega,
        )

    def set_omega_grad(self, f, omega_grad):
        """
        Set gradient of edge angular velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        omega_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing gradients of angular velocities.
        """

        self._solver._kernel_set_edges_omega_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega_grad=omega_grad,
        )

    def set_kb(self, f, kb):
        """
        Set internal vertex bending curvatures in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        kb : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices, 3) containing bending curvatures.
        """

        self._solver._kernel_set_internal_vertices_kb(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            kb=kb,
        )

    def set_kb_grad(self, f, kb_grad):
        """
        Set gradient of internal vertex bending curvatures in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        kb_grad : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices, 3) containing gradients of bending curvatures.
        """

        self._solver._kernel_set_internal_vertices_kb_grad(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            kb_grad=kb_grad,
        )

    def set_twist(self, f, twist):
        """
        Set internal vertex twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        twist : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices,) containing twist angles.
        """

        self._solver._kernel_set_internal_vertices_twist(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            twist=twist,
        )

    def set_twist_grad(self, f, twist_grad):
        """
        Set gradient of internal vertex twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        twist_grad : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices,) containing gradients of twist angles.
        """

        self._solver._kernel_set_internal_vertices_twist_grad(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            twist_grad=twist_grad,
        )

    def set_kappa_rest(self, f, kappa_rest):
        """
        Set internal vertex rest curvatures in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        kappa_rest : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices, 3) containing rest curvatures.
        """

        self._solver._kernel_set_internal_vertices_kappa_rest(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            kappa_rest=kappa_rest,
        )

    def set_kappa_rest_grad(self, f, kappa_rest_grad):
        """
        Set gradient of internal vertex rest curvatures in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        kappa_rest_grad : gs.Tensor
            Tensor of shape (n_envs, n_internal_vertices, 3) containing gradients of rest curvatures.
        """

        self._solver._kernel_set_internal_vertices_kappa_rest_grad(
            f=f,
            iv_start=self._iv_start,
            n_internal_vertices=self.n_internal_vertices,
            kappa_rest_grad=kappa_rest_grad,
        )

    def set_fixed(self, f, fixed):
        """
        Set the fixed status of each vertex in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        fixed : gs.Tensor
            Tensor of shape (n_envs, n_vertices,) containing boolean fixed status for each vertex.
        """

        self._solver._kernel_set_fixed_states(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            fixed=fixed,
        )

    def _validate_param_shape(self, param, envs_idx):
        if envs_idx is None:
            assert param.shape[0] == self._sim._B, \
                f"Parameter should have shape ({self._sim._B},), got {param.shape}."
        else:
            assert param.shape[0] == len(envs_idx), \
                f"Parameter should have shape ({len(envs_idx)},), got {param.shape}."

    def _resolve_envs_idx(self, envs_idx):
        envs_idx = envs_idx if envs_idx is not None else torch.arange(self._sim._B, dtype=torch.int32)
        envs_idx = to_gs_tensor(envs_idx)
        return envs_idx

    @gs.assert_built
    def set_bending_stiffness(self, bending_stiffness, envs_idx=None):
        """
        Set the bending stiffness of the rod in the solver.

        Parameters
        ----------
        bending_stiffness : gs.Tensor
            Tensor of shape (len(envs_idx),) containing bending stiffness values.
            If envs_idx is None, shape should be (n_envs,).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set stiffness for. If None, sets for all environments.
        """

        self._validate_param_shape(bending_stiffness, envs_idx)

        self._solver._kernel_set_bending_stiffness(
            rod_idx=self._rod_idx,
            bending_stiffness=bending_stiffness.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_twisting_stiffness(self, twisting_stiffness, envs_idx=None):
        """
        Set the twisting stiffness of the rod in the solver.

        Parameters
        ----------
        twisting_stiffness : gs.Tensor
            Tensor of shape (len(envs_idx),) containing twisting stiffness values.
            If envs_idx is None, shape should be (n_envs,).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set stiffness for. If None, sets for all environments.
        """

        self._validate_param_shape(twisting_stiffness, envs_idx)

        self._solver._kernel_set_twisting_stiffness(
            rod_idx=self._rod_idx,
            twisting_stiffness=twisting_stiffness.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_stretching_stiffness(self, stretching_stiffness, envs_idx=None):
        """
        Set the stretching stiffness of the rod in the solver.

        Parameters
        ----------
        stretching_stiffness : gs.Tensor
            Tensor of shape (len(envs_idx),) containing stretching stiffness values.
            If envs_idx is None, shape should be (n_envs,).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set stiffness for. If None, sets for all environments.
        """

        self._validate_param_shape(stretching_stiffness, envs_idx)

        self._solver._kernel_set_stretching_stiffness(
            rod_idx=self._rod_idx,
            stretching_stiffness=stretching_stiffness.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_segment_mass(self, segment_mass, envs_idx=None):
        """
        Set the segment mass of the rod in the solver.

        Parameters
        ----------
        segment_mass : gs.Tensor
            Tensor of shape (len(envs_idx), n_vertices,) containing segment mass values.
            If envs_idx is None, shape should be (n_envs, n_vertices).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set mass for. If None, sets for all environments.
        """

        self._validate_param_shape(segment_mass, envs_idx)

        self._solver._kernel_set_segment_mass(
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            mass=segment_mass.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_segment_radius(self, segment_radius, envs_idx=None):
        """
        Set the segment radius of the rod in the solver.

        Parameters
        ----------
        segment_radius : gs.Tensor
            Tensor of shape (len(envs_idx), n_vertices,) containing segment radius values.
            If envs_idx is None, shape should be (n_envs, n_vertices).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set radius for. If None, sets for all environments.
        """

        self._validate_param_shape(segment_radius, envs_idx)

        self._solver._kernel_set_segment_radius(
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            radius=segment_radius.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_mu_s(self, mu_s, envs_idx=None):
        """
        Set the static friction coefficient of the rod in the solver.

        Parameters
        ----------
        mu_s : gs.Tensor
            Tensor of shape (len(envs_idx), n_vertices,) containing static friction coefficient values.
            If envs_idx is None, shape should be (n_envs, n_vertices).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set coefficient for. If None, sets for all environments.
        """

        self._validate_param_shape(mu_s, envs_idx)

        self._solver._kernel_set_mu_s(
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            mu_s=mu_s.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_mu_k(self, mu_k, envs_idx=None):
        """
        Set the kinetic friction coefficient of the rod in the solver.

        Parameters
        ----------
        mu_k : gs.Tensor
            Tensor of shape (len(envs_idx), n_vertices,) containing kinetic friction coefficient values.
            If envs_idx is None, shape should be (n_envs, n_vertices).
        envs_idx : list or np.ndarray, optional
            List of environment indices to set coefficient for. If None, sets for all environments.
        """

        self._validate_param_shape(mu_k, envs_idx)

        self._solver._kernel_set_mu_k(
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            mu_k=mu_k.contiguous(),
            envs_idx=self._resolve_envs_idx(envs_idx),
        )

    @gs.assert_built
    def set_init_vertices(self, verts_np, edges_np):
        self._solver._kernel_finalize_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            verts=verts_np,
            edges=edges_np,
        )

    @gs.assert_built
    def set_fixed_states(self, fixed_states=None, fixed_ids=None):
        """
        Set the fixed status of each vertex. This method is used to fixed vertices along the whole simulation
        before it starts.

        Parameters
        ----------
        fixed_states: list or np.ndarray
            List or array of booleans indicating fixed status for each vertex. Shape should be (n_vertices,).
        fixed_ids: list
            List of vertex indices to be fixed.
        """

        if fixed_ids is None and fixed_states is None:
            is_fixed = np.zeros(self.n_vertices, dtype=gs.np_bool)
        elif fixed_ids is None and fixed_states is not None:
            is_fixed = np.asarray(fixed_states).copy().reshape(-1).astype(gs.np_bool)
            assert is_fixed.shape[0] == self.n_vertices, \
                f"Fixed states has {is_fixed.shape[0]} vertices, but rod {self._rod_idx} has {self.n_vertices}."
        elif fixed_ids is not None:
            is_fixed = [1 if i in fixed_ids else 0 for i in range(self.n_vertices)]
            is_fixed = np.array(is_fixed, dtype=gs.np_bool)
        else:
            raise ValueError("`fixed_ids` and `fixed_states` cannot be provided at the same time.")

        is_fixed = np.tile(is_fixed, (self._sim._B, 1))  # (n_envs, n_vertices)

        # set fixed states for the first local frame
        self._solver._kernel_set_fixed_states(
            f=0,    # start from 0, then f -> f+1
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            fixed=is_fixed,
        )

    @gs.assert_built
    def attach_to_rigid_link(self, rigid_link, verts_ids):
        """
        Attaches specific vertices of the rod to a rigid link.

        Parameters
        ----------
        rigid_link : genesis.RigidLink
            The rigid link to attach to.
        verts_indices : list or np.ndarray
            A list of local vertex indices on the rod to attach.
        """
        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            assert 0 <= vert_id < self.n_vertices, \
                f"Vertex index {vert_id} out of range for rod with {self.n_vertices} vertices."

        verts_ids = np.asarray(verts_ids).copy().reshape(-1).astype(np.int32)
        if len(verts_ids) == 0:
            return

        # (B, len(verts_ids), 3)
        v_pos_world = self.get_all_verts_tc(False)[:, verts_ids, :]
        # (B, 3)
        l_pos = rigid_link.get_pos()
        # (B, 4)
        l_quat = rigid_link.get_quat()
        # (B, 4, 4)
        l_T_inv = gu.trans_quat_to_T(l_pos, l_quat).inverse()
        v_hpos_world = torch.nn.functional.pad(
            v_pos_world, (0, 1), "constant", 1.0
        )

        v_hpos_local = torch.einsum("bij,bkj->bki", l_T_inv, v_hpos_world)

        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            global_vert_id = self._v_start + vert_id
            self._solver._kernel_set_attached_states(
                i_v=global_vert_id,
                link_idx=rigid_link.idx,
                local_pos=v_hpos_local[:, i, :3].contiguous()
            )
        gs.logger.info(
            f"Rod {self.uid}({self._rod_idx}) vertices {verts_ids} attached to {rigid_link.idx}"
        )

    @gs.assert_built
    def attach_to_rigid_link_with_envs_idx(self, rigid_link, verts_ids, envs_idx):
        """
        Attaches specific vertices of the rod to a rigid link for specific environments.

        Parameters
        ----------
        rigid_link : genesis.RigidLink
            The rigid link to attach to.
        verts_indices : list or np.ndarray
            A list of local vertex indices on the rod to attach.
        envs_idx : int
            The environment index to apply the attachment.
        """
        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            assert 0 <= vert_id < self.n_vertices, \
                f"Vertex index {vert_id} out of range for rod with {self.n_vertices} vertices."

        verts_ids = np.asarray(verts_ids).copy().reshape(-1).astype(np.int32)
        if len(verts_ids) == 0:
            return

        # (1, len(verts_ids), 3)
        v_pos_world = self.get_all_verts_tc(False)[envs_idx, verts_ids, :].unsqueeze(0)
        # (1, 3)
        l_pos = rigid_link.get_pos()[envs_idx, :].unsqueeze(0)
        # (1, 4)
        l_quat = rigid_link.get_quat()[envs_idx, :].unsqueeze(0)
        # (1, 4, 4)
        l_T_inv = gu.trans_quat_to_T(l_pos, l_quat).inverse()
        v_hpos_world = torch.nn.functional.pad(
            v_pos_world, (0, 1), "constant", 1.0
        )

        v_hpos_local = torch.einsum("bij,bkj->bki", l_T_inv, v_hpos_world).squeeze(0)

        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            global_vert_id = self._v_start + vert_id
            self._solver._kernel_set_attached_states_with_envs_idx(
                i_v=global_vert_id,
                link_idx=rigid_link.idx,
                local_pos=v_hpos_local[i, :3].contiguous(),
                envs_idx=envs_idx
            )
        gs.logger.info(
            f"Rod {self.uid}({self._rod_idx}) vertices {verts_ids} attached to {rigid_link.idx} in env {envs_idx}"
        )

    @gs.assert_built
    def detach_from_rigid_link(self, verts_ids):
        """
        Detaches specific vertices of the rod from any rigid link.

        Parameters
        ----------
        verts_indices : list or np.ndarray
            A list of local vertex indices on the rod to detach.
        """

        verts_ids = np.asarray(verts_ids).copy().reshape(-1).astype(np.int32)
        if len(verts_ids) == 0:
            return

        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            assert 0 <= vert_id < self.n_vertices, \
                f"Vertex index {vert_id} out of range for rod with {self.n_vertices} vertices."
            global_vert_id = self._v_start + vert_id
            self._solver._kernel_detach_vertex(i_v=global_vert_id)

        gs.logger.info(
            f"Rod {self.uid}({self._rod_idx}) vertices {verts_ids} detached from rigid link."
        )

    @gs.assert_built
    def detach_from_rigid_link_with_envs_idx(self, verts_ids, envs_idx):
        """
        Detaches specific vertices of the rod from any rigid link for specific environments.

        Parameters
        ----------
        verts_indices : list or np.ndarray
            A list of local vertex indices on the rod to detach.
        envs_idx : int
            The environment index to apply the detachment.
        """

        verts_ids = np.asarray(verts_ids).copy().reshape(-1).astype(np.int32)
        if len(verts_ids) == 0:
            return

        for i in range(len(verts_ids)):
            vert_id = verts_ids[i]
            assert 0 <= vert_id < self.n_vertices, \
                f"Vertex index {vert_id} out of range for rod with {self.n_vertices} vertices."
            global_vert_id = self._v_start + vert_id
            self._solver._kernel_detach_vertex_with_envs_idx(
                i_v=global_vert_id,
                envs_idx=envs_idx
            )

        gs.logger.info(
            f"Rod {self.uid}({self._rod_idx}) vertices {verts_ids} detached from rigid link in env {envs_idx}."
        )

    @qd.kernel
    def _kernel_get_verts_pos(self, f: qd.i32, pos: qd.types.ndarray(), verts_idx: qd.types.ndarray()):
        # get current position of vertices
        for i_v, i_b in qd.ndrange(verts_idx.shape[0], verts_idx.shape[1]):
            i_global = verts_idx[i_v, i_b] + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]

    @qd.kernel
    def get_frame(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        fixed: qd.types.ndarray(),
        theta: qd.types.ndarray(),
        omega: qd.types.ndarray(),
        edge: qd.types.ndarray(),
        length: qd.types.ndarray(),
        d1: qd.types.ndarray(),
        d2: qd.types.ndarray(),
        d3: qd.types.ndarray(),
        d1_ref: qd.types.ndarray(),
        d2_ref: qd.types.ndarray(),
        kb: qd.types.ndarray(),
        twist: qd.types.ndarray(),
        kappa_rest: qd.types.ndarray(),
        # collision state
        collided: qd.types.ndarray(),
        collision_normal: qd.types.ndarray(),
        collision_penetration: qd.types.ndarray(),
    ):
        """
        Extract the state of particles at the given frame.

        Parameters
        ----------
        f : int
            The substep/frame index to fetch the state from.

        pos : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store positions.

        vel : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store velocities.

        fixed : np.ndarray
            Output array of shape (n_envs, n_vertices) to store fixed status.

        theta : np.ndarray
            Output array of shape (n_envs, n_edges) to store twist angles.

        omega : np.ndarray
            Output array of shape (n_envs, n_edges) to store angular velocities.

        edge : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store edge directions.

        length : np.ndarray
            Output array of shape (n_envs, n_edges) to store edge lengths.

        d1 : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store material frame d1 vectors.

        d2 : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store material frame d2 vectors.

        d3 : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store material frame d3 vectors.

        d1_ref : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store reference material frame d1 vectors.

        d2_ref : np.ndarray
            Output array of shape (n_envs, n_edges, 3) to store reference material frame d2 vectors.

        kb : np.ndarray
            Output array of shape (n_envs, n_internal_vertices, 3) to store bending curvatures.

        twist : np.ndarray
            Output array of shape (n_envs, n_internal_vertices) to store twist angles.

        kappa_rest : np.ndarray
            Output array of shape (n_envs, n_internal_vertices, 3) to store rest curvatures.

        collided : np.ndarray
            Output array of shape (n_envs, n_vertices) to store collision status.

        collision_normal : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store collision normals.

        collision_penetration : np.ndarray
            Output array of shape (n_envs, n_vertices) to store collision penetration depths.
        """

        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]
                vel[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vel[j]
                collision_normal[i_b, i_v, j] = self._solver.vertices_collision[i_global, i_b].normal[j]
            fixed[i_b, i_v] = self._solver.vertices_ng[f, i_global, i_b].fixed
            collided[i_b, i_v] = self._solver.vertices_collision[i_global, i_b].collided
            collision_penetration[i_b, i_v] = self._solver.vertices_collision[i_global, i_b].penetration

        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            length[i_b, i_e] = self._solver.edges[f, i_global, i_b].length
            theta[i_b, i_e] = self._solver.edges[f, i_global, i_b].theta
            omega[i_b, i_e] = self._solver.edges[f, i_global, i_b].omega
            for j in qd.static(range(3)):
                edge[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].edge[j]
                d1[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].d1[j]
                d2[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].d2[j]
                d3[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].d3[j]
                d1_ref[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].d1_ref[j]
                d2_ref[i_b, i_e, j] = self._solver.edges[f, i_global, i_b].d2_ref[j]

        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._sim._B):
            i_global = i_iv + self.iv_start
            for j in qd.static(range(3)):
                kb[i_b, i_iv, j] = self._solver.internal_vertices[f, i_global, i_b].kb[j]
            twist[i_b, i_iv] = self._solver.internal_vertices[f, i_global, i_b].twist
            for j in qd.static(range(2)):
                kappa_rest[i_b, i_iv, j] = self._solver.internal_vertices[f, i_global, i_b].kappa_rest[j]

    @qd.kernel
    def get_vertices_pos_kernel(
        self,
        pos: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
    ):
        for i_v, i_b_ in qd.ndrange(self.n_vertices, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[0, i_global, i_b].vert[j]

    @gs.assert_built
    def get_vertices_pos(self, envs_idx=None):
        envs_idx = self._resolve_envs_idx(envs_idx)
        base_v_shape = (envs_idx.shape[0], self.n_vertices, 3)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            # "scene": self.scene,
        }
        pos = torch.zeros(base_v_shape, **args)
        self.get_vertices_pos_kernel(pos, envs_idx)
        if self._scene.n_envs == 0:
            pos = pos[0]
        return pos

    @qd.kernel
    def get_all_verts_kernel(
        self,
        pos: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[0, i_global, i_b].vert[j]

    @gs.assert_built
    def get_all_verts(self):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        pos = np.zeros(base_v_shape, **args)
        self.get_all_verts_kernel(pos)
        return pos

    @gs.assert_built
    def get_all_verts_tc(self, requires_grad=False):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": requires_grad,
            # "scene": self.scene,
        }
        pos = torch.zeros(base_v_shape, **args)
        self.get_all_verts_kernel(pos)
        return pos

    @qd.kernel
    def get_all_edge_lengths_kernel(
        self,
        edge_lengths: qd.types.ndarray(),
    ):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            edge_lengths[i_b, i_e] = self._solver.edges[0, i_global, i_b].length

    @gs.assert_built
    def get_total_length(self):
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        edge_lengths = np.zeros((self.sim._B, self.n_edges), **args)
        self.get_all_edge_lengths_kernel(edge_lengths)
        total_length = np.sum(edge_lengths, axis=1)  # (B,)
        return total_length

    @qd.kernel
    def get_all_vels_kernel(
        self,
        vel: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                vel[i_b, i_v, j] = self._solver.vertices[0, i_global, i_b].vel[j]

    @gs.assert_built
    def get_all_vels(self):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        vel = np.zeros(base_v_shape, **args)
        self.get_all_vels_kernel(vel)
        return vel

    @gs.assert_built
    def get_all_vels_tc(self, requires_grad=False):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": requires_grad,
            # "scene": self.scene,
        }
        vel = torch.zeros(base_v_shape, **args)
        self.get_all_vels_kernel(vel)
        return vel

    @qd.kernel
    def sample_centerline_kernel(
        self,
        n_samples: qd.i32,
        cum_lens: qd.types.ndarray(),    # (B, n_edges + 1); cum_lens[:, 0] == 0
        sampled_pos: qd.types.ndarray(), # (B, n_samples, 3) – output
    ):
        """
        Sample *n_samples* positions uniformly by arc length along the rope
        centerline, parallelised over (sample, batch).

        A linear scan through *cum_lens* locates the containing edge, then
        the position is linearly interpolated between the two endpoint vertices.
        """
        for i_s, i_b in qd.ndrange(n_samples, self._sim._B):
            total_len = cum_lens[i_b, self.n_edges]
            target_s = total_len * 0.5
            if n_samples > 1:
                target_s = qd.cast(i_s, gs.ti_float) / qd.cast(n_samples - 1, gs.qd_float) * total_len

            placed = 0
            for i_e in range(self.n_edges):
                if placed == 0:
                    is_last = qd.cast(i_e == self.n_edges - 1, qd.i32)
                    if cum_lens[i_b, i_e + 1] >= target_s or is_last == 1:
                        seg_len = cum_lens[i_b, i_e + 1] - cum_lens[i_b, i_e]
                        t = (target_s - cum_lens[i_b, i_e]) / (seg_len + gs.EPS)
                        t = qd.min(qd.max(t, 0.0), 1.0)
                        g_a = i_e + self._v_start
                        g_b = (i_e + 1) % self.n_vertices + self._v_start
                        for j in qd.static(range(3)):
                            pa = self._solver.vertices[0, g_a, i_b].vert[j]
                            pb = self._solver.vertices[0, g_b, i_b].vert[j]
                            sampled_pos[i_b, i_s, j] = (1.0 - t) * pa + t * pb
                        placed = 1

    @gs.assert_built
    def sample_centerline(self, n_samples: int) -> np.ndarray:
        """
        Sample *n_samples* positions uniformly by arc length along the rope
        centerline, via linear interpolation between consecutive vertices.

        Parameters
        ----------
        n_samples : int
            Number of sample points.  ``n_samples=1`` returns the midpoint.

        Returns
        -------
        np.ndarray of shape ``(n_envs, n_samples, 3)``
        """
        seg_lens = np.zeros((self.sim._B, self.n_edges), dtype=gs.np_float)
        self.get_all_edge_lengths_kernel(seg_lens)
        cum_lens = np.zeros((self.sim._B, self.n_edges + 1), dtype=gs.np_float)
        np.cumsum(seg_lens, axis=1, out=cum_lens[:, 1:])
        out = np.zeros((self.sim._B, n_samples, 3), dtype=gs.np_float)
        self.sample_centerline_kernel(n_samples, cum_lens, out)
        return out

    @gs.assert_built
    def sample_centerline_tc(self, n_samples: int, requires_grad: bool = False) -> "torch.Tensor":
        """Torch-tensor variant of :meth:`sample_centerline`."""
        seg_lens = torch.zeros((self.sim._B, self.n_edges), dtype=gs.tc_float)
        self.get_all_edge_lengths_kernel(seg_lens)
        cum_lens = torch.zeros((self.sim._B, self.n_edges + 1), dtype=gs.tc_float)
        torch.cumsum(seg_lens, dim=1, out=cum_lens[:, 1:])
        out = torch.zeros((self.sim._B, n_samples, 3), dtype=gs.tc_float, requires_grad=requires_grad)
        self.sample_centerline_kernel(n_samples, cum_lens, out)
        return out

    @qd.kernel
    def get_all_energy_kernel(
        self,
        energy: qd.types.ndarray(),
    ):
        for i_b in qd.ndrange(self._sim._B):
            e = 0.0
            e += self._solver.rods_energy[self._rod_idx, i_b].stretching_energy
            e += self._solver.rods_energy[self._rod_idx, i_b].bending_energy
            e += self._solver.rods_energy[self._rod_idx, i_b].twisting_energy
            energy[i_b] = e

    @gs.assert_built
    def get_all_energy(self):
        base_v_shape = (self.sim._B,)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        energy = np.zeros(base_v_shape, **args)
        self.get_all_energy_kernel(energy)
        return energy

    @qd.kernel
    def get_all_stretching_force_kernel(
        self,
        stretching_force: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                stretching_force[i_b, i_v, j] = self._solver.vertices_force[i_global, i_b].f_s[j]

    @gs.assert_built
    def get_all_stretching_force(self):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        stretching_force = np.zeros(base_v_shape, **args)
        self.get_all_stretching_force_kernel(stretching_force)
        return stretching_force

    @qd.kernel
    def get_all_bending_force_kernel(
        self,
        bending_force: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                bending_force[i_b, i_v, j] = self._solver.vertices_force[i_global, i_b].f_b[j]

    @gs.assert_built
    def get_all_bending_force(self):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        bending_force = np.zeros(base_v_shape, **args)
        self.get_all_bending_force_kernel(bending_force)
        return bending_force

    @qd.kernel
    def get_geodesic_distance_kernel(
        self,
        vert_start_idx: qd.i32,
        vert_end_idx: qd.i32,
        distance: qd.types.ndarray(),
    ):
        for i_b in qd.ndrange(self._sim._B):
            d = 0.0
            for i_e in range(vert_start_idx, vert_end_idx):
                i_global = i_e + self.e_start
                edge_len = self._solver.edges[0, i_global, i_b].length
                d += edge_len
            distance[i_b] = d

    @gs.assert_built
    def get_geodesic_distance(self, vert_start_idx, vert_end_idx):
        assert 0 <= vert_start_idx < vert_end_idx <= self.n_edges, \
            f"Invalid vertex indices for geodesic distance: {vert_start_idx} to {vert_end_idx}."
        base_v_shape = (self.sim._B,)
        args = {"dtype": gs.np_float}
        distance = np.zeros(base_v_shape, **args)
        self.get_geodesic_distance_kernel(vert_start_idx, vert_end_idx, distance)
        return distance

    @gs.assert_built
    def get_geodesic_distance_tc(self, vert_start_idx, vert_end_idx, requires_grad=False):
        assert 0 <= vert_start_idx < vert_end_idx <= self.n_edges, \
            f"Invalid vertex indices for geodesic distance: {vert_start_idx} to {vert_end_idx}."
        base_v_shape = (self.sim._B,)
        args = {"dtype": gs.tc_float, "requires_grad": requires_grad}
        distance = torch.zeros(base_v_shape, **args)
        self.get_geodesic_distance_kernel(vert_start_idx, vert_end_idx, distance)
        return distance

    @qd.kernel
    def get_nearest_verts_from_rigid_geom_kernel(
        self,
        geom_idx: qd.i32,
        nearest_points: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            i_va = sdf_func_find_closest_vert(
                geoms_state=self.sim.rigid_solver.geoms_state,
                geoms_info=self.sim.rigid_solver.geoms_info,
                sdf_info=self.sim.rigid_solver.sdf._sdf_info,
                pos_world=self._solver.vertices[0, i_global, i_b].vert,
                geom_idx=geom_idx,
                i_b=i_b
            )
            g_pos = self.sim.rigid_solver.geoms_state.pos[geom_idx, i_b]
            g_quat = self.sim.rigid_solver.geoms_state.quat[geom_idx, i_b]
            n_pos = gu.qd_transform_by_trans_quat(
                self.sim.rigid_solver.verts_info.init_pos[i_va], g_pos, g_quat
            )
            for j in qd.static(range(3)):
                nearest_points[i_b, i_v, j] = n_pos[j]

    @gs.assert_built
    def get_nearest_verts_from_rigid_geom(self, geom_idx):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        nearest_points = np.zeros(base_v_shape, **args)
        self.get_nearest_verts_from_rigid_geom_kernel(geom_idx, nearest_points)
        return nearest_points

    @gs.assert_built
    def get_nearest_verts_from_rigid_geom_tc(self, geom_idx, requires_grad=False):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": requires_grad,
            # "scene": self.scene,
        }
        nearest_points = torch.zeros(base_v_shape, **args)
        self.get_nearest_verts_from_rigid_geom_kernel(geom_idx, nearest_points)
        return nearest_points

    @qd.kernel
    def get_all_stretching_stiffness_kernel(
        self,
        stretching_stiffness: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            stretching_stiffness[i_b] = self._solver.rods_stretching_stiffness[self._rod_idx, i_b]

    @gs.assert_built
    def get_all_stretching_stiffness_tc(self):
        base_v_shape = (self.sim._B,)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        stretching_stiffness = torch.zeros(base_v_shape, **args)
        self.get_all_stretching_stiffness_kernel(stretching_stiffness)
        return stretching_stiffness

    @qd.kernel
    def get_all_bending_stiffness_kernel(
        self,
        bending_stiffness: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            bending_stiffness[i_b] = self._solver.rods_bending_stiffness[self._rod_idx, i_b]

    @gs.assert_built
    def get_all_bending_stiffness_tc(self):
        base_v_shape = (self.sim._B,)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        bending_stiffness = torch.zeros(base_v_shape, **args)
        self.get_all_bending_stiffness_kernel(bending_stiffness)
        return bending_stiffness

    @qd.kernel
    def get_all_twisting_stiffness_kernel(
        self,
        twisting_stiffness: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            twisting_stiffness[i_b] = self._solver.rods_twisting_stiffness[self._rod_idx, i_b]

    @gs.assert_built
    def get_all_twisting_stiffness_tc(self):
        base_v_shape = (self.sim._B,)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        twisting_stiffness = torch.zeros(base_v_shape, **args)
        self.get_all_twisting_stiffness_kernel(twisting_stiffness)
        return twisting_stiffness

    @qd.kernel
    def get_stretching_stiffness_grad_kernel(
        self,
        grad: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            grad[i_b] = self._solver.rods_stretching_stiffness.grad[self._rod_idx, i_b]

    @qd.kernel
    def get_bending_stiffness_grad_kernel(
        self,
        grad: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            grad[i_b] = self._solver.rods_bending_stiffness.grad[self._rod_idx, i_b]

    @qd.kernel
    def get_twisting_stiffness_grad_kernel(
        self,
        grad: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            grad[i_b] = self._solver.rods_twisting_stiffness.grad[self._rod_idx, i_b]

    @gs.assert_built
    def get_stretching_stiffness_grad(self):
        """Return dL/dK for this rod as a torch tensor of shape (n_envs,).

        Valid only after a backward pass and only if the solver was created with
        ``RODOptions(requires_grad_K=True)``.
        """
        grad = torch.zeros((self.sim._B,), dtype=gs.tc_float, requires_grad=False)
        self.get_stretching_stiffness_grad_kernel(grad)
        return grad

    @gs.assert_built
    def get_bending_stiffness_grad(self):
        """Return dL/dE for this rod as a torch tensor of shape (n_envs,).

        Valid only after a backward pass and only if the solver was created with
        ``RODOptions(requires_grad_E=True)``.
        """
        grad = torch.zeros((self.sim._B,), dtype=gs.tc_float, requires_grad=False)
        self.get_bending_stiffness_grad_kernel(grad)
        return grad

    @gs.assert_built
    def get_twisting_stiffness_grad(self):
        """Return dL/dG for this rod as a torch tensor of shape (n_envs,).

        Valid only after a backward pass and only if the solver was created with
        ``RODOptions(requires_grad_G=True)``.
        """
        grad = torch.zeros((self.sim._B,), dtype=gs.tc_float, requires_grad=False)
        self.get_twisting_stiffness_grad_kernel(grad)
        return grad

    @qd.kernel
    def get_all_segment_mass_kernel(
        self,
        mass: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            mass[i_b, i_v] = self._solver.vertices_param[i_global, i_b].mass

    @gs.assert_built
    def get_all_segment_mass_tc(self):
        base_v_shape = (self.sim._B, self.n_vertices)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        mass = torch.zeros(base_v_shape, **args)
        self.get_all_segment_mass_kernel(mass)
        return mass

    @qd.kernel
    def get_all_segment_radius_kernel(
        self,
        radius: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            radius[i_b, i_v] = self._solver.vertices_param[i_global, i_b].radius

    @gs.assert_built
    def get_all_segment_radius(self):
        base_v_shape = (self.sim._B, self.n_vertices)
        args = {
            "dtype": gs.np_float,
        }
        radius = np.zeros(base_v_shape, **args)
        self.get_all_segment_radius_kernel(radius)
        return radius

    @gs.assert_built
    def get_all_segment_radius_tc(self):
        base_v_shape = (self.sim._B, self.n_vertices)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        radius = torch.zeros(base_v_shape, **args)
        self.get_all_segment_radius_kernel(radius)
        return radius

    @qd.kernel
    def get_all_mu_s_kernel(
        self,
        mu_s: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            mu_s[i_b, i_v] = self._solver.vertices_param[i_global, i_b].mu_s

    @gs.assert_built
    def get_all_mu_s_tc(self):
        base_v_shape = (self.sim._B, self.n_vertices)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        mu_s = torch.zeros(base_v_shape, **args)
        self.get_all_mu_s_kernel(mu_s)
        return mu_s

    @qd.kernel
    def get_all_mu_k_kernel(
        self,
        mu_k: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            mu_k[i_b, i_v] = self._solver.vertices_param[i_global, i_b].mu_k

    @gs.assert_built
    def get_all_mu_k_tc(self):
        base_v_shape = (self.sim._B, self.n_vertices)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
        }
        mu_k = torch.zeros(base_v_shape, **args)
        self.get_all_mu_k_kernel(mu_k)
        return mu_k

    @qd.kernel
    def _kernel_add_frame_pos_grad(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                self._solver.vertices.grad[f, i_global, i_b].vert[j] += pos_grad[i_b, i_v, j]

    @qd.kernel
    def _kernel_add_frame_vel_grad(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                self._solver.vertices.grad[f, i_global, i_b].vel[j] += vel_grad[i_b, i_v, j]

    @qd.kernel
    def _kernel_add_frame_edge_grad(self, f: qd.i32, edge_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].edge[j] += edge_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_length_grad(self, f: qd.i32, length_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].length += length_grad[i_b, i_e]

    @qd.kernel
    def _kernel_add_frame_d1_grad(self, f: qd.i32, d1_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].d1[j] += d1_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_d2_grad(self, f: qd.i32, d2_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].d2[j] += d2_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_d3_grad(self, f: qd.i32, d3_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].d3[j] += d3_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_d1_ref_grad(self, f: qd.i32, d1_ref_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].d1_ref[j] += d1_ref_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_d2_ref_grad(self, f: qd.i32, d2_ref_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.edges.grad[f, i_global, i_b].d2_ref[j] += d2_ref_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_theta_grad(self, f: qd.i32, theta_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].theta += theta_grad[i_b, i_e]

    @qd.kernel
    def _kernel_add_frame_omega_grad(self, f: qd.i32, omega_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].omega += omega_grad[i_b, i_e]

    @qd.kernel
    def _kernel_add_frame_kb_grad(self, f: qd.i32, kb_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                self._solver.internal_vertices.grad[f, i_global, i_b].kb[j] += kb_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_add_frame_twist_grad(self, f: qd.i32, twist_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.internal_vertices.grad[f, i_global, i_b].twist += twist_grad[i_b, i_e]

    @qd.kernel
    def _kernel_add_frame_kappa_rest_grad(self, f: qd.i32, kappa_rest_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(2)):
                self._solver.internal_vertices.grad[f, i_global, i_b].kappa_rest[j] += kappa_rest_grad[i_b, i_e, j]

    @qd.kernel
    def _kernel_distribute_frame_pos_grad(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos_grad[i_b, i_v, j] = self._solver.vertices.grad[f, i_global, i_b].vert[j] 

    @qd.kernel
    def _kernel_distribute_frame_vel_grad(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                vel_grad[i_b, i_v, j] = self._solver.vertices.grad[f, i_global, i_b].vel[j]

    @qd.kernel
    def _kernel_distribute_frame_edge_grad(self, f: qd.i32, edge_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                edge_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].edge[j]

    @qd.kernel
    def _kernel_distribute_frame_length_grad(self, f: qd.i32, length_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            length_grad[i_b, i_e] = self._solver.edges.grad[f, i_global, i_b].length

    @qd.kernel
    def _kernel_distribute_frame_d1_grad(self, f: qd.i32, d1_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                d1_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].d1[j]

    @qd.kernel
    def _kernel_distribute_frame_d2_grad(self, f: qd.i32, d2_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                d2_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].d2[j]

    @qd.kernel
    def _kernel_distribute_frame_d3_grad(self, f: qd.i32, d3_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                d3_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].d3[j]

    @qd.kernel
    def _kernel_distribute_frame_d1_ref_grad(self, f: qd.i32, d1_ref_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                d1_ref_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].d1_ref[j]

    @qd.kernel
    def _kernel_distribute_frame_d2_ref_grad(self, f: qd.i32, d2_ref_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            for j in qd.static(range(3)):
                d2_ref_grad[i_b, i_e, j] = self._solver.edges.grad[f, i_global, i_b].d2_ref[j]

    @qd.kernel
    def _kernel_distribute_frame_theta_grad(self, f: qd.i32, theta_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            theta_grad[i_b, i_e] = self._solver.edges.grad[f, i_global, i_b].theta

    @qd.kernel
    def _kernel_distribute_frame_omega_grad(self, f: qd.i32, omega_grad: qd.types.ndarray()):
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            omega_grad[i_b, i_e] = self._solver.edges.grad[f, i_global, i_b].omega

    @qd.kernel
    def _kernel_distribute_frame_kb_grad(self, f: qd.i32, kb_grad: qd.types.ndarray()):
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._sim._B):
            i_global = i_iv + self.iv_start
            for j in qd.static(range(3)):
                kb_grad[i_b, i_iv, j] = self._solver.internal_vertices.grad[f, i_global, i_b].kb[j]

    @qd.kernel
    def _kernel_distribute_frame_twist_grad(self, f: qd.i32, twist_grad: qd.types.ndarray()):
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._sim._B):
            i_global = i_iv + self.iv_start
            twist_grad[i_b, i_iv] = self._solver.internal_vertices.grad[f, i_global, i_b].twist

    @qd.kernel
    def _kernel_distribute_frame_kappa_rest_grad(self, f: qd.i32, kappa_rest_grad: qd.types.ndarray()):
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._sim._B):
            i_global = i_iv + self.iv_start
            for j in qd.static(range(2)):
                kappa_rest_grad[i_b, i_iv, j] = self._solver.internal_vertices.grad[f, i_global, i_b].kappa_rest[j]

    def add_grad_from_state(self, state):
        """
        Accumulate gradients from a recorded state back into the solver.

        Parameters
        ----------
        state : RODEntityState
            The state object containing gradients for physical quantities.
        """
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self._kernel_add_frame_pos_grad(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self._kernel_add_frame_vel_grad(self._sim.cur_substep_local, state.vel.grad)

        if state.edge.grad is not None:
            state.edge.assert_contiguous()
            self._kernel_add_frame_edge_grad(self._sim.cur_substep_local, state.edge.grad)

        if state.length.grad is not None:
            state.length.assert_contiguous()
            self._kernel_add_frame_length_grad(self._sim.cur_substep_local, state.length.grad)

        if state.d1.grad is not None:
            state.d1.assert_contiguous()
            self._kernel_add_frame_d1_grad(self._sim.cur_substep_local, state.d1.grad)

        if state.d2.grad is not None:
            state.d2.assert_contiguous()
            self._kernel_add_frame_d2_grad(self._sim.cur_substep_local, state.d2.grad)

        if state.d3.grad is not None:
            state.d3.assert_contiguous()
            self._kernel_add_frame_d3_grad(self._sim.cur_substep_local, state.d3.grad)

        if state.d1_ref.grad is not None:
            state.d1_ref.assert_contiguous()
            self._kernel_add_frame_d1_ref_grad(self._sim.cur_substep_local, state.d1_ref.grad)

        if state.d2_ref.grad is not None:
            state.d2_ref.assert_contiguous()
            self._kernel_add_frame_d2_ref_grad(self._sim.cur_substep_local, state.d2_ref.grad)

        if state.theta.grad is not None:
            state.theta.assert_contiguous()
            self._kernel_add_frame_theta_grad(self._sim.cur_substep_local, state.theta.grad)

        if state.omega.grad is not None:
            state.omega.assert_contiguous()
            self._kernel_add_frame_omega_grad(self._sim.cur_substep_local, state.omega.grad)

        if state.kb.grad is not None:
            state.kb.assert_contiguous()
            self._kernel_add_frame_kb_grad(self._sim.cur_substep_local, state.kb.grad)

        if state.twist.grad is not None:
            state.twist.assert_contiguous()
            self._kernel_add_frame_twist_grad(self._sim.cur_substep_local, state.twist.grad)

        if state.kappa_rest.grad is not None:
            state.kappa_rest.assert_contiguous()
            self._kernel_add_frame_kappa_rest_grad(self._sim.cur_substep_local, state.kappa_rest.grad)

    def distribute_grad_to_state(self, state):
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self._kernel_distribute_frame_pos_grad(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self._kernel_distribute_frame_vel_grad(self._sim.cur_substep_local, state.vel.grad)

        if state.edge.grad is not None:
            state.edge.assert_contiguous()
            self._kernel_distribute_frame_edge_grad(self._sim.cur_substep_local, state.edge.grad)

        if state.length.grad is not None:
            state.length.assert_contiguous()
            self._kernel_distribute_frame_length_grad(self._sim.cur_substep_local, state.length.grad)

        if state.d1.grad is not None:
            state.d1.assert_contiguous()
            self._kernel_distribute_frame_d1_grad(self._sim.cur_substep_local, state.d1.grad)

        if state.d2.grad is not None:
            state.d2.assert_contiguous()
            self._kernel_distribute_frame_d2_grad(self._sim.cur_substep_local, state.d2.grad)

        if state.d3.grad is not None:
            state.d3.assert_contiguous()
            self._kernel_distribute_frame_d3_grad(self._sim.cur_substep_local, state.d3.grad)

        if state.d1_ref.grad is not None:
            state.d1_ref.assert_contiguous()
            self._kernel_distribute_frame_d1_ref_grad(self._sim.cur_substep_local, state.d1_ref.grad)

        if state.d2_ref.grad is not None:
            state.d2_ref.assert_contiguous()
            self._kernel_distribute_frame_d2_ref_grad(self._sim.cur_substep_local, state.d2_ref.grad)

        if state.theta.grad is not None:
            state.theta.assert_contiguous()
            self._kernel_distribute_frame_theta_grad(self._sim.cur_substep_local, state.theta.grad)

        if state.omega.grad is not None:
            state.omega.assert_contiguous()
            self._kernel_distribute_frame_omega_grad(self._sim.cur_substep_local, state.omega.grad)

        if state.kb.grad is not None:
            state.kb.assert_contiguous()
            self._kernel_distribute_frame_kb_grad(self._sim.cur_substep_local, state.kb.grad)

        if state.twist.grad is not None:
            state.twist.assert_contiguous()
            self._kernel_distribute_frame_twist_grad(self._sim.cur_substep_local, state.twist.grad)

        if state.kappa_rest.grad is not None:
            state.kappa_rest.assert_contiguous()
            self._kernel_distribute_frame_kappa_rest_grad(self._sim.cur_substep_local, state.kappa_rest.grad)

    def _reset_grad(self):
        """
        Clear all gradients for particle properties.
        """
        self._reset_frame_grad(self._sim.cur_substep_local)

    @qd.kernel
    def _reset_frame_grad(self, f: qd.i32):
        """
        Zero out the gradients of position, velocity, and angular velocity for the current substep.

        Parameters
        ----------
        f : int
            The substep/frame index for which to clear gradients.

        Notes
        -----
        This method is primarily used during backward passes to manually reset gradients
        that may be corrupted by explicit state setting.
        """
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            self._solver.vertices.grad[f, i_global, i_b].vert = 0
            self._solver.vertices.grad[f, i_global, i_b].vel = 0
        for i_e, i_b in qd.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].edge = 0
            self._solver.edges.grad[f, i_global, i_b].length = 0
            self._solver.edges.grad[f, i_global, i_b].d1 = 0
            self._solver.edges.grad[f, i_global, i_b].d2 = 0
            self._solver.edges.grad[f, i_global, i_b].d3 = 0
            self._solver.edges.grad[f, i_global, i_b].d1_ref = 0
            self._solver.edges.grad[f, i_global, i_b].d2_ref = 0
            self._solver.edges.grad[f, i_global, i_b].theta = 0
            self._solver.edges.grad[f, i_global, i_b].omega = 0
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._sim._B):
            i_global = i_iv + self.iv_start
            self._solver.internal_vertices.grad[f, i_global, i_b].kb = 0
            self._solver.internal_vertices.grad[f, i_global, i_b].twist = 0
            self._solver.internal_vertices.grad[f, i_global, i_b].kappa_rest = 0

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Rod):
            return "rod_base"
        if isinstance(morph, gs.morphs.ParameterizedRod):
            return "rod_param"
        return "rod_entity"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the Rod entity."""
        return len(self.init_positions)

    @property
    def n_edges(self):
        """Number of edges in the Rod entity."""
        return len(self.edges)

    @property
    def is_loop(self):
        """Whether the rod is loop."""
        return self.morph.is_loop

    @property
    def n_internal_vertices(self):
        """Number of internal vertices in the Rod entity."""
        return len(self.init_positions) if self.is_loop else len(self.init_positions) - 2

    @property
    def n_dofs(self):
        """Number of degrees of freedom (DOFs) in the Rod entity."""
        # 3 for each vertex + 1 for each edge
        return 3 * self.n_vertices + self.n_edges

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def e_start(self):
        """Global edge index offset for this entity."""
        return self._e_start

    @property
    def iv_start(self):
        """Global internal vertex index offset for this entity."""
        return self._iv_start

    @property
    def visualize_twist(self):
        """Whether to visualize twist frames."""
        return self._visualize_twist

    @property
    def material(self):
        """Material properties of the Rod entity."""
        return self._material
