# pylint: disable=no-value-for-parameter

from typing import TYPE_CHECKING, Iterable

import numpy as np
from math import pi
import quadrants as qd
import quadrants.math as qm
import torch

import genesis as gs
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.rod_entity import RODEntity
from genesis.engine.states.solvers import RODSolverState
from genesis.utils.geom import qd_transform_by_trans_quat
from genesis.utils.array_class import LinksState

from .base_solver import Solver

if TYPE_CHECKING:
    pass

EPS = 1e-14


@qd.func
def get_perpendicular_vector(vector):
    """
    Returns a *unit* vector perpendicular to the input vector.
    """
    # Pick axis least aligned with vector
    abs_vector = qd.abs(vector)

    a = qd.Vector([0.0, 0.0, 0.0])
    if abs_vector.x <= abs_vector.y and abs_vector.x <= abs_vector.z:
        a = qd.Vector([1.0, 0.0, 0.0])
    elif abs_vector.y <= abs_vector.z:
        a = qd.Vector([0.0, 1.0, 0.0])
    else:
        a = qd.Vector([0.0, 0.0, 1.0])
    return qm.cross(vector, a).normalized()

@qd.func
def parallel_transport_normalized(t0, t1, v):
    """
    Transport vector :math:`v` from edge with tangent vector :math:`e0` to edge with tangent
    vector :math:`e1` (edge tangent vectors are normalized)
    """
    sin_theta_axis = qm.cross(t0, t1)
    cos_theta = qm.dot(t0, t1)
    den = 1 + cos_theta # denominator

    vprime = qd.Vector([0.0, 0.0, 0.0])
    if qd.abs(den) < EPS:
        vprime = v

    elif qd.abs(t0.x - t1.x) < EPS and qd.abs(t0.y - t1.y) < EPS and qd.abs(t0.z - t1.z) < EPS:
        vprime = v

    else:
        vprime = cos_theta * v + qm.cross(sin_theta_axis, v) + (qm.dot(sin_theta_axis, v) / den) * sin_theta_axis
    return vprime

@qd.func
def curvature_binormal(e0, e1):
    """
    Compute the curvature binormal for a vertex between two edges with tangents
    :math:`e0` and :math:`e1`, respectively (edge tangent vectors *not* necessarily normalized)
    """
    return 2.0 * qm.cross(e0, e1) / (qm.length(e0) * qm.length(e1) + qm.dot(e0, e1))

@qd.func
def get_updated_material_frame(prev_d3, d3, ref_d1, ref_d2, theta):
    """
    Parallel transport the reference frame vectors :math:`ref_d1` and :math:`ref_d2` from
    the previous edge to the new tangent vector :math:`d3` to get the updated reference frame.
    Then, rotate them by the twist angle :math:`theta` to get the updated material frame.
    """
    ref_d1 = parallel_transport_normalized(prev_d3, d3, ref_d1)
    ref_d2 = parallel_transport_normalized(prev_d3, d3, ref_d2)
    d1 = qd.cos(theta) * ref_d1 + qd.sin(theta) * ref_d2
    d2 = -qd.sin(theta) * ref_d1 + qd.cos(theta) * ref_d2
    return d1, d2, ref_d1, ref_d2

@qd.func
def get_angle(a, vec1, vec2):
    """
    Get the signed angle from :math:`vec1` to :math:`vec2` around axis :math:`a`; 
    ccw angles are positive. Assumes all vectors are *normalized* and *perpendicular* to :math:`a`
    Output in the range :math:`[-pi, pi]`
    """
    s = qd.max(-1.0, qd.min(1.0, qm.cross(vec1, vec2).dot(a)))
    c = qd.max(-1.0, qd.min(1.0, qm.dot(vec1, vec2)))
    return qm.atan2(s, c)

@qd.func
def get_updated_reference_twist(ref_d1_im1, ref_d1, d3_im1, d3):
    """
    Get the reference twist angle for the current edge based on the previous edge's
    reference director :math:`ref_d1_im1`, the current edge's reference director 
    :math:`ref_d1`, and the previous and current edge's tangent vectors :math:`d3_im1` 
    and :math:`d3`. Assumes all vectors are *normalized*.
    """
    # Finite rotation angle needed to take the parallel transported copy
    # of the previous edge's reference director to the current edge's
    # reference director.
    vec1 = parallel_transport_normalized(d3_im1, d3, ref_d1_im1)
    vec2 = ref_d1
    reference_twist = get_angle(d3, vec1, vec2)
    return reference_twist

@qd.func
def quat_rotate(q: qm.vec4, v: qm.vec3) -> qm.vec3:
    """
    Rotate vector `v` by quaternion `q`.
    """
    qvec = qd.Vector([q[1], q[2], q[3]])
    uv = qm.cross(qvec, v)
    uuv = qm.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


@qd.data_oriented
class RODSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._adjacent_gap = options.adjacent_gap
        self._damping = options.damping
        self._angular_damping = options.angular_damping
        self._n_pbd_iters = options.n_pbd_iters
        self._grad_clip = options.grad_clip
        self._req_grad_K = options.requires_grad_K
        self._req_grad_E = options.requires_grad_E
        self._req_grad_G = options.requires_grad_G
        self._disable_constraint_grad = options.disable_constraint_grad
        self._max_collision_grad_norm = 0.1

        # properties
        self._geom_indices = np.array([], dtype=gs.np_int)

        # boundary
        self.setup_boundary()

        # lazy initialization of vertex constraints (attachment)
        self._constraints_initialized = False

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif isinstance(shape, (list, tuple)):
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_rod_fields(self):
        # rod stiffness parameters. Each field's `needs_grad` comes from the solver-wide
        # RODOptions.requires_grad_{K,E,G} flags (read in __init__).
        # whether any rod uses the inextensibility constraint (gates its differentiable bookkeeping)
        self._any_inextensible = any(getattr(e.material, "use_inextensible", False) for e in self._entities)

        # rod information (parametric)
        struct_rod_param = qd.types.struct(
            # material properties
            use_inextensible=gs.qd_bool,
            plastic_yield=gs.qd_float,
            plastic_creep=gs.qd_float,
        )

        # rod information (structural)
        struct_rod_info = qd.types.struct(
            # indices
            first_vert_idx=gs.qd_int,           # index of the first vertex of this rod
            first_edge_idx=gs.qd_int,           # index of the first edge of this rod
            first_internal_vert_idx=gs.qd_int,  # index of the first internal vertex of this rod
            n_verts=gs.qd_int,                  # number of vertices in this rod

            # is loop
            is_loop=gs.qd_bool,
        )

        # rod energy (w/o time dimension)
        struct_rod_energy = qd.types.struct(
            stretching_energy=gs.qd_float,
            bending_energy=gs.qd_float,
            twisting_energy=gs.qd_float,
        )

        self.rods_stretching_stiffness = qd.field(
            dtype=gs.qd_float, needs_grad=self._req_grad_K,
            shape=self._batch_shape(self._n_rods)
        )
        self.rods_bending_stiffness = qd.field(
            dtype=gs.qd_float, needs_grad=self._req_grad_E,
            shape=self._batch_shape(self._n_rods)
        )
        self.rods_twisting_stiffness = qd.field(
            dtype=gs.qd_float, needs_grad=self._req_grad_G,
            shape=self._batch_shape(self._n_rods)
        )

        self.rods_param = struct_rod_param.field(
            shape=self._batch_shape(self._n_rods),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        self.rods_info = struct_rod_info.field(
            shape=self._n_rods, layout=qd.Layout.SOA
        )

        self.rods_energy = struct_rod_energy.field(
            shape=self._batch_shape(self._n_rods),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        # keep track of gradients for time-stepping
        self.gradients = qd.field(
            dtype=gs.qd_float, needs_grad=True,
            shape=self._batch_shape(self._n_dofs)
        )

        # Per-batch accumulator for the L2 norm of the recurrent adjoint state (used by
        # per-substep norm-based gradient clipping, RODOptions.grad_clip)
        self.adjoint_normsq = qd.field(dtype=gs.qd_float, needs_grad=False, shape=(self._B,))

        # Read-only snapshots of the incoming velocity/position adjoint, used by the friction /
        # collision backward so that contact pairs sharing a vertex read a consistent value and
        # accumulate with atomic_add (deterministic, race-free)
        self.friction_vel_grad_snapshot = qd.field(
            dtype=gs.qd_vec3, needs_grad=False, shape=self._batch_shape(self._n_vertices)
        )
        self.collision_vert_grad_snapshot = qd.field(
            dtype=gs.qd_vec3, needs_grad=False, shape=self._batch_shape(self._n_vertices)
        )

        # Pre-projection vertex positions saved per (substep, PBD iteration) during the forward
        # inextensibility projection. The inextensibility constraint is applied in place on
        # vertices[f+1], so its pre-projection input is otherwise overwritten; the backward needs
        # it to linearize the (nonlinear) length constraint at the correct point. Unlike the
        # sparse collision constraint, inextensibility acts on every edge every substep, so reading
        # the post-projection geometry (as collision/friction do) gives a badly biased gradient.
        # Indexed [f, iter, vertex, batch].
        if self._sim.requires_grad and self._any_inextensible and not self._disable_constraint_grad:
            self.inext_preproj_vert = qd.field(
                dtype=gs.qd_vec3, needs_grad=False,
                shape=self._batch_shape((self.sim.substeps_local + 1, self._n_pbd_iters, self._n_vertices)),
            )

    def init_vertex_fields(self):
        # vertex information (parametric)
        struct_vertex_param = qd.types.struct(
            mass=gs.qd_float,
            radius=gs.qd_float,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            restitution=gs.qd_float,    # coefficient of restitution for self-collision
        )

        # vertex information (structural)
        struct_vertex_info = qd.types.struct(
            rod_idx=gs.qd_int,          # index of the rod this vertex belongs to
        )

        # vertex state (dynamic)
        struct_vertex_state = qd.types.struct(
            vert=gs.qd_vec3,            # current position
            vel=gs.qd_vec3,
        )

        # vertex force (w/o time dimension)
        struct_vertex_force = qd.types.struct(
            f_s=gs.qd_vec3,             # stretching force
            f_b=gs.qd_vec3,             # bending force
            f_t=gs.qd_vec3,             # twisting force
        )

        struct_vertex_state_ng = qd.types.struct(
            fixed=gs.qd_bool,           # is the vertex fixed
            kinematic=gs.qd_bool,       # is the vertex kinematic
        )

        struct_vertex_state_collision = qd.types.struct(
            collided=gs.qd_bool,        # has the vertex collided in this step
            normal=gs.qd_vec3,          # collision normal
            penetration=gs.qd_float,    # penetration depth
            geom_idx=gs.qd_int,         # index of the geometry collided with
        )

        self.vertices_param = struct_vertex_param.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        self.vertices_info = struct_vertex_info.field(
            shape=self._n_vertices, layout=qd.Layout.SOA
        )

        self.vertices = struct_vertex_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=True,
            layout=qd.Layout.SOA
        )

        self.vertices_force = struct_vertex_force.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        self.vertices_ng = struct_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        # for policy learning
        self.vertices_collision = struct_vertex_state_collision.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA
        )

        # for visualization
        self.vertices_render = qd.Vector.field(
            3, dtype=gs.qd_float, needs_grad=False,
            shape=self._batch_shape(self._n_vertices)
        )

    def init_vertex_constraints(self):

        vertex_constraint_info = qd.types.struct(
            constrained=gs.qd_bool,
            link_idx=gs.qd_int,
            target_pos=gs.qd_vec3,
            local_pos=gs.qd_vec3,
        )

        self.vertex_constraints = vertex_constraint_info.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=qd.Layout.AOS
        )

        self._constraints_initialized = True

    def init_edge_fields(self):
        # edge information (static)
        struct_edge_info = qd.types.struct(
            edge_rest=gs.qd_vec3,
            length_rest=gs.qd_float,
            d1_rest=gs.qd_vec3,         # material frame direction 1 in rest state
            d2_rest=gs.qd_vec3,         # material frame direction 2 in rest state
            d3_rest=gs.qd_vec3,         # material frame direction 3 in rest state (tangent)
            vert_idx=gs.qd_int,         # index of the starting vertex of this edge
        )

        # edge state (dynamic)
        struct_edge_state = qd.types.struct(
            edge=gs.qd_vec3,        # current edge vector
            length=gs.qd_float,     # current edge length
            d1=gs.qd_vec3,          # material frame direction 1
            d2=gs.qd_vec3,          # material frame direction 2
            d3=gs.qd_vec3,          # material frame direction 3 (tangent)
            d1_ref=gs.qd_vec3,      # reference material frame direction 1
            d2_ref=gs.qd_vec3,      # reference material frame direction 2
            theta=gs.qd_float,      # twist angle
            omega=gs.qd_float,      # twist rate (angular velocity)
        )

        self.edges_info = struct_edge_info.field(
            shape=self._n_edges, layout=qd.Layout.SOA
        )

        self.edges = struct_edge_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_edges)),
            needs_grad=True,
            layout=qd.Layout.SOA
        )

    def init_internal_vertex_fields(self):
        # internal vertex information (static)
        struct_internal_vertex_info = qd.types.struct(
            twist_rest=gs.qd_float,     # rest twist
            edge_idx=gs.qd_int,         # index of the starting edge of this internal vertex
        )

        struct_internal_vertex_state_ng = qd.types.struct(
            kb=gs.qd_vec3,          # current curvature binormal
            twist=gs.qd_float,      # current twist
            kappa_rest=gs.qd_vec2,  # rest curvature,
        )

        self.internal_vertices_info = struct_internal_vertex_info.field(
            shape=self._n_internal_vertices, layout=qd.Layout.SOA
        )

        self.internal_vertices = struct_internal_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_internal_vertices)),
            needs_grad=True,
            layout=qd.Layout.SOA
        )

    def init_constraints(self):
        # NOTE: call this after call `_kernel_add_rods`
        valid_edge_pairs = list()
        for i in range(self._n_vertices):
            for j in range(i + 1, self._n_vertices):
                rod_id_i = self.vertices_info[i].rod_idx
                local_id_i = i - self.rods_info[rod_id_i].first_vert_idx
                rod_id_j = self.vertices_info[j].rod_idx
                local_id_j = j - self.rods_info[rod_id_j].first_vert_idx

                # filtering
                # 1. ensure i and j can actually start an edge
                is_loop_i = self.rods_info[rod_id_i].is_loop
                is_loop_j = self.rods_info[rod_id_j].is_loop

                if not is_loop_i and local_id_i >= self.rods_info[rod_id_i].n_verts - 1:
                    continue
                if not is_loop_j and local_id_j >= self.rods_info[rod_id_j].n_verts - 1:
                    continue

                # 2. ignore adjacent edges on the same rod
                if rod_id_i == rod_id_j:
                    if is_loop_i:
                        n_verts_in_rod = self.rods_info[rod_id_i].n_verts
                        dist_forward = local_id_j - local_id_i
                        dist_backward = (local_id_i + n_verts_in_rod) - local_id_j

                        if dist_forward < self._adjacent_gap + 1 or dist_backward < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the loop.
                    else:
                        if abs(local_id_j - local_id_i) < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the chain.

                valid_edge_pairs.append((i, j))

        valid_edge_pairs = np.array(valid_edge_pairs, dtype=gs.np_int)
        self._n_valid_edge_pairs = valid_edge_pairs.shape[0]

        # constraint for rod-rod collision
        struct_rr_info = qd.types.struct(
            valid_pair=qd.types.vector(2, gs.qd_int),
        )

        struct_rr_state = qd.types.struct(
            normal=gs.qd_vec3,
            penetration=gs.qd_float,
        )

        self.rr_constraint_info = struct_rr_info.field(
            shape=self._n_valid_edge_pairs, layout=qd.Layout.SOA
        )

        self.rr_constraints = struct_rr_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_valid_edge_pairs)),
            needs_grad=True,
            layout=qd.Layout.AOS
        )

        self.rr_constraint_info.valid_pair.from_numpy(valid_edge_pairs)

    def register_gripper_geom_indices(self, geom_indices: Iterable[int]=()):
        """
        Register the geometry indices of the gripper for collision handling.
        Needs to be called before building the scene.
        """
        geom_indices = np.asarray(geom_indices, dtype=gs.np_int)
        self._geom_indices = geom_indices

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.vertices.grad.fill(0)
        self.edges.grad.fill(0)
        self.internal_vertices.grad.fill(0)
        self.gradients.grad.fill(0)
        self.rr_constraints.grad.fill(0)

        # stiffness param grads accumulate across the whole backward-through-time; they are
        # only zeroed here (once per `scene.reset()`, i.e. once per optimization iteration).
        if self._req_grad_K:
            self.rods_stretching_stiffness.grad.fill(0)
        if self._req_grad_E:
            self.rods_bending_stiffness.grad.fill(0)
        if self._req_grad_G:
            self.rods_twisting_stiffness.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._n_rods = self.n_rods
        self._n_vertices = self.n_vertices
        self._n_edges = self.n_edges
        self._n_internal_vertices = self.n_internal_vertices
        self._n_dofs = self.n_dofs

        # rendering
        self.envs_offset = qd.Vector.field(3, dtype=qd.f32, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(np.float32))

        if self.is_active:
            self.init_rod_fields()
            self.init_vertex_fields()
            self.init_vertex_constraints()
            self.init_edge_fields()
            self.init_internal_vertex_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_constraints()

        # Overwrite gravity because only field is supported for now
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def add_entity(self, idx, material, morph, surface, visualize_twist, name: str | None = None):

        # create entity
        entity = RODEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            rod_idx=self.n_rods,
            v_start=self.n_vertices,
            e_start=self.n_edges,
            iv_start=self.n_internal_vertices,
            visualize_twist=visualize_twist,
            name=name,
        )

        self._entities.append(entity)
        return entity

    @property
    def is_active(self):
        return self._n_vertices > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ logging --------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def get_rod_length(self, f: qd.i32, i_r: qd.i32, length: qd.types.ndarray()):
        n_verts = self.rods_info[i_r].n_verts
        first_edge_idx = self.rods_info[i_r].first_edge_idx
        for i_e, i_b in qd.ndrange(n_verts - 1, self._B):
            edge_idx = first_edge_idx + i_e
            length[i_b] += self.edges[f, edge_idx, i_b].length

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_clear_energy(self):
        for i_r, i_b in qd.ndrange(self._n_rods, self._B):
            self.rods_energy[i_r, i_b].stretching_energy = 0.0
            self.rods_energy[i_r, i_b].bending_energy = 0.0
            self.rods_energy[i_r, i_b].twisting_energy = 0.0

    @qd.func
    def _func_clear_force(self):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices_force[i_v, i_b].f_s = qd.Vector.zero(gs.qd_float, 3)
            self.vertices_force[i_v, i_b].f_b = qd.Vector.zero(gs.qd_float, 3)
            self.vertices_force[i_v, i_b].f_t = qd.Vector.zero(gs.qd_float, 3)

    @qd.func
    def _func_clear_gradients(self):
        self.gradients.fill(0.0)

    @qd.kernel
    def update_centerline_positions(self, f: qd.i32):      # Differential
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            if (
                not self.vertices_ng[f, i_v, i_b].fixed and 
                not self.vertex_constraints[i_v, i_b].constrained
            ):
                # self.vertices[f + 1, i_v, i_b].vert += self.vertices[f + 1, i_v, i_b].vel * self.substep_dt
                self.vertices[f + 1, i_v, i_b].vert = (
                    self.vertices[f + 1, i_v, i_b].vel * self.substep_dt + self.vertices[f, i_v, i_b].vert
                )

    @qd.kernel
    def update_centerline_velocities(self, f: qd.i32):       # Differential
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            mass = self.vertices_param[i_v, i_b].mass
            if (
                not self.vertices_ng[f, i_v, i_b].fixed and
                not self.vertex_constraints[i_v, i_b].constrained
            ):
                gradient = qd.Vector([
                    self.gradients[3 * i_v + 0, i_b],
                    self.gradients[3 * i_v + 1, i_b],
                    self.gradients[3 * i_v + 2, i_b],
                ])
                self.vertices[f + 1, i_v, i_b].vel -= gradient / mass * self.substep_dt

                # apply damping if enabled
                self.vertices[f + 1, i_v, i_b].vel *= qd.exp(-self.substep_dt * self.damping)
                # self.vertices[f, i_v, i_b].vel *= (1.0 - self.damping)
                # add gravity (avoiding damping on gravity)
                self.vertices[f + 1, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @qd.kernel
    def update_angular_velocities(self, f: qd.i32):      # Differential
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            if (not self.vertices_ng[f, v_s, i_b].fixed and not self.vertex_constraints[v_s, i_b].constrained) or \
               (not self.vertices_ng[f, v_e, i_b].fixed and not self.vertex_constraints[v_e, i_b].constrained):
                theta_dof_idx = 3 * self._n_vertices + i_e
                gradient = self.gradients[theta_dof_idx, i_b]
                inertia = 1.0
                self.edges[f + 1, i_e, i_b].omega -= gradient / inertia * self.substep_dt
                self.edges[f + 1, i_e, i_b].omega *= qd.exp(-self.substep_dt * self.angular_damping)

    @qd.kernel
    def update_centerline_edges(self, f: qd.i32):    # Differential
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            self.edges[f + 1, i_e, i_b].edge = self.vertices[f + 1, v_e, i_b].vert - self.vertices[f + 1, v_s, i_b].vert
            self.edges[f + 1, i_e, i_b].length = qm.length(self.edges[f + 1, i_e, i_b].edge)

    @qd.kernel
    def update_frame_thetas(self, f: qd.i32):      # Differential
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            if (not self.vertices_ng[f, v_s, i_b].fixed and not self.vertex_constraints[v_s, i_b].constrained) or \
               (not self.vertices_ng[f, v_e, i_b].fixed and not self.vertex_constraints[v_e, i_b].constrained):
                # self.edges[f + 1, i_e, i_b].theta -= self.gradients[3 * self._n_vertices + i_e, i_b] * self.substep_dt
                self.edges[f + 1, i_e, i_b].theta = (
                    self.edges[f + 1, i_e, i_b].omega * self.substep_dt + self.edges[f, i_e, i_b].theta
                )

    @qd.kernel
    def update_material_states(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges[f + 1, i_e, i_b].d3 = self.edges[f + 1, i_e, i_b].edge.normalized()

            d1, d2, d1_ref, d2_ref = get_updated_material_frame(
                self.edges[f, i_e, i_b].d3,         # prev d3
                self.edges[f + 1, i_e, i_b].d3,     # curr d3
                self.edges[f, i_e, i_b].d1_ref,
                self.edges[f, i_e, i_b].d2_ref,
                self.edges[f + 1, i_e, i_b].theta,
            )
            self.edges[f + 1, i_e, i_b].d1 = d1
            self.edges[f + 1, i_e, i_b].d2 = d2
            self.edges[f + 1, i_e, i_b].d1_ref = d1_ref
            self.edges[f + 1, i_e, i_b].d2_ref = d2_ref

        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)

            self.internal_vertices[f + 1, i_iv, i_b].kb = curvature_binormal(
                self.edges[f + 1, e_s, i_b].d3, self.edges[f + 1, e_e, i_b].d3
            )
            twist_ref = get_updated_reference_twist(
                self.edges[f + 1, e_s, i_b].d1_ref, self.edges[f + 1, e_e, i_b].d1_ref,
                self.edges[f + 1, e_s, i_b].d3, self.edges[f + 1, e_e, i_b].d3
            )
            self.internal_vertices[f + 1, i_iv, i_b].twist = self.edges[f + 1, e_e, i_b].theta - self.edges[f + 1, e_s, i_b].theta + twist_ref

    @qd.kernel
    def update_velocities_after_projection(self, f: qd.i32):   # Differential
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            if (not self.vertices_ng[f, i_v, i_b].fixed and not self.vertex_constraints[i_v, i_b].constrained):
                self.vertices[f + 1, i_v, i_b].vel = (self.vertices[f + 1, i_v, i_b].vert - self.vertices[f, i_v, i_b].vert) / self.substep_dt

    @qd.kernel
    def transfer_vertex_states(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices_ng[f + 1, i_v, i_b].fixed = self.vertices_ng[f, i_v, i_b].fixed

    @qd.kernel
    def init_pos_and_vel(self, f: qd.i32):  # Differential
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices[f + 1, i_v, i_b].vert = self.vertices[f, i_v, i_b].vert
            self.vertices[f + 1, i_v, i_b].vel = self.vertices[f, i_v, i_b].vel

    @qd.kernel
    def init_theta_and_omega(self, f: qd.i32):     # Differential
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges[f + 1, i_e, i_b].theta = self.edges[f, i_e, i_b].theta
            self.edges[f + 1, i_e, i_b].omega = self.edges[f, i_e, i_b].omega

    @qd.kernel
    def compute_stretching_energy(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check stretching enabled
            K = self.rods_stretching_stiffness[rod_id, i_b]
            if K > 0.:
                r = (self.vertices_param[v_s, i_b].radius + self.vertices_param[v_e, i_b].radius) * 0.5
                a, b = r, r
                A = pi * a * b  # cross-sectional area

                strain_i = (self.edges[f, i_e, i_b].length / self.edges_info[i_e].length_rest) - 1.0

                self.rods_energy[rod_id, i_b].stretching_energy += 0.5 * K * A * qd.pow(strain_i, 2) * self.edges_info[i_e].length_rest

                # -------------------------------- gradients --------------------------------

                gradient_magnitude = K * A * strain_i

                gradient_dx_i   = - gradient_magnitude * self.edges[f, i_e, i_b].d3
                gradient_dx_ip1 =   gradient_magnitude * self.edges[f, i_e, i_b].d3

                for k in qd.static(range(3)):
                    qd.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    qd.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_ip1[k])

                    qd.atomic_add(self.vertices_force[v_s, i_b].f_s[k], -gradient_dx_i[k])
                    qd.atomic_add(self.vertices_force[v_e, i_b].f_s[k], -gradient_dx_ip1[k])

    @qd.kernel
    def compute_bending_energy(self, f: qd.i32):
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check bending enabled
            E = self.rods_bending_stiffness[rod_id, i_b]
            if E > 0.:
                r = self.vertices_param[v_m, i_b].radius
                a, b = r, r
                A = pi * a * b  # cross-sectional area
                B11 = E * A * qd.pow(a, 2) / 4.0
                B22 = E * A * qd.pow(b, 2) / 4.0

                kb = self.internal_vertices[f, i_iv, i_b].kb
                l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5

                kappa1_i =   0.5 * qm.dot(kb, self.edges[f, e_s, i_b].d2 + self.edges[f, e_e, i_b].d2)
                kappa2_i = - 0.5 * qm.dot(kb, self.edges[f, e_s, i_b].d1 + self.edges[f, e_e, i_b].d1)

                # bending plasticity
                kappa1_rest_i = self.internal_vertices[f, i_iv, i_b].kappa_rest[0]
                kappa2_rest_i = self.internal_vertices[f, i_iv, i_b].kappa_rest[1]
                curr_kappa = qd.Vector([kappa1_i, kappa2_i])
                rest_kappa = qd.Vector([kappa1_rest_i, kappa2_rest_i])

                elastic_kappa = curr_kappa - rest_kappa

                yield_thres = self.rods_param[rod_id, i_b].plastic_yield
                creep_rate = self.rods_param[rod_id, i_b].plastic_creep

                elastic_kappa_sq = qm.dot(elastic_kappa, elastic_kappa)  # ||elastic_kappa||^2
                yield_thres_sq = yield_thres * yield_thres

                # when elastic_kappa_sq < yield_thres_sq: no plasticity
                # when elastic_kappa_sq > yield_thres_sq: apply plasticity
                excess_sq = qd.max(0.0, elastic_kappa_sq - yield_thres_sq)

                scale = creep_rate * excess_sq / (elastic_kappa_sq + EPS)
                delta_rest_kappa = scale * elastic_kappa

                self.internal_vertices[f + 1, i_iv, i_b].kappa_rest = (
                    delta_rest_kappa + self.internal_vertices[f, i_iv, i_b].kappa_rest
                )

                kappa1_rest_i = self.internal_vertices[f + 1, i_iv, i_b].kappa_rest[0]
                kappa2_rest_i = self.internal_vertices[f + 1, i_iv, i_b].kappa_rest[1]

                self.rods_energy[rod_id, i_b].bending_energy += 0.5 * (
                    B11 * qd.pow(kappa1_i - kappa1_rest_i, 2) +
                    B22 * qd.pow(kappa2_i - kappa2_rest_i, 2)
                ) / l_i

                # -------------------------------- gradients --------------------------------

                gradient_kappa1_i_x_i = qd.Vector.zero(dt=gs.qd_float, n=9)
                gradient_kappa2_i_x_i = qd.Vector.zero(dt=gs.qd_float, n=9)

                chi = 1. + qm.dot(self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3)
                d1_tilde = (self.edges[f, e_s, i_b].d1 + self.edges[f, e_e, i_b].d1) / chi
                d2_tilde = (self.edges[f, e_s, i_b].d2 + self.edges[f, e_e, i_b].d2) / chi
                d3_tilde = (self.edges[f, e_s, i_b].d3 + self.edges[f, e_e, i_b].d3) / chi

                dkappa1_i_de_im1 = qm.cross(d2_tilde, -self.edges[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                    kappa1_i * d3_tilde / self.edges_info[e_s].length_rest
                dkappa1_i_de_i = qm.cross(d2_tilde, self.edges[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                    kappa1_i * d3_tilde / self.edges_info[e_e].length_rest
                dkappa2_i_de_im1 = qm.cross(d1_tilde, self.edges[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                    kappa2_i * d3_tilde / self.edges_info[e_s].length_rest
                dkappa2_i_de_i = qm.cross(d1_tilde, -self.edges[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                    kappa2_i * d3_tilde / self.edges_info[e_e].length_rest

                gradient_kappa1_i_x_i[0:3] = dkappa1_i_de_im1 * (- 1.0)
                gradient_kappa1_i_x_i[3:6] = dkappa1_i_de_im1 * (  1.0) + dkappa1_i_de_i * (- 1.0)
                gradient_kappa1_i_x_i[6:9] = dkappa1_i_de_i   * (  1.0)
                gradient_kappa2_i_x_i[0:3] = dkappa2_i_de_im1 * (- 1.0)
                gradient_kappa2_i_x_i[3:6] = dkappa2_i_de_im1 * (  1.0) + dkappa2_i_de_i * (- 1.0)
                gradient_kappa2_i_x_i[6:9] = dkappa2_i_de_i   * (  1.0)

                gradient_dx_i = (
                    B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_x_i + \
                    B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_x_i
                ) / l_i
                for k in qd.static(range(3)):
                    qd.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    qd.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                    qd.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                    qd.atomic_add(self.vertices_force[v_s, i_b].f_b[k], -gradient_dx_i[k])
                    qd.atomic_add(self.vertices_force[v_m, i_b].f_b[k], -gradient_dx_i[k + 3])
                    qd.atomic_add(self.vertices_force[v_e, i_b].f_b[k], -gradient_dx_i[k + 6])

                gradient_kappa1_i_theta_i = - qd.Vector([
                    qm.dot(kb, self.edges[f, e_s, i_b].d1) * 0.5,
                    qm.dot(kb, self.edges[f, e_e, i_b].d1) * 0.5
                ])
                gradient_kappa2_i_theta_i = - qd.Vector([
                    qm.dot(kb, self.edges[f, e_s, i_b].d2) * 0.5,
                    qm.dot(kb, self.edges[f, e_e, i_b].d2) * 0.5
                ])

                gradient_dtheta_i = (
                    B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_theta_i + \
                    B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_theta_i
                ) / l_i
                theta_dof_s_idx = 3 * self._n_vertices + e_s
                theta_dof_e_idx = 3 * self._n_vertices + e_e
                qd.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
                qd.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @qd.kernel
    def compute_twisting_energy(self, f: qd.i32):
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check twisting enabled
            G = self.rods_twisting_stiffness[rod_id, i_b]
            if G > 0.:
                r = self.vertices_param[v_m, i_b].radius
                a, b = r, r
                A = pi * a * b  # cross-sectional area
                beta = G * A * (qd.pow(a, 2) + qd.pow(b, 2)) / 4.0

                kb = self.internal_vertices[f, i_iv, i_b].kb
                l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5
                m_i = self.internal_vertices[f, i_iv, i_b].twist
                m_i_rest = self.internal_vertices_info[i_iv].twist_rest

                self.rods_energy[rod_id, i_b].twisting_energy += 0.5 * beta * qd.pow(m_i - m_i_rest, 2) / l_i

                # -------------------------------- gradients --------------------------------

                gradient_m_i_dx_i = qd.Vector.zero(dt=gs.qd_float, n=9)
                gradient_m_i_dx_i[0:3] = - kb / (2.0 * self.edges[f, e_s, i_b].length)
                gradient_m_i_dx_i[3:6] =   kb / (2.0 * self.edges[f, e_s, i_b].length) - kb / (2.0 * self.edges[f, e_e, i_b].length)
                gradient_m_i_dx_i[6:9] =   kb / (2.0 * self.edges[f, e_e, i_b].length)
                gradient_dx_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dx_i
                for k in qd.static(range(3)):
                    qd.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    qd.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                    qd.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                    qd.atomic_add(self.vertices_force[v_s, i_b].f_t[k], -gradient_dx_i[k])
                    qd.atomic_add(self.vertices_force[v_m, i_b].f_t[k], -gradient_dx_i[k + 3])
                    qd.atomic_add(self.vertices_force[v_e, i_b].f_t[k], -gradient_dx_i[k + 6])

                gradient_m_i_dtheta_i = qd.Vector([-1.0, 1.0])
                gradient_dtheta_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dtheta_i
                theta_dof_s_idx = 3 * self._n_vertices + e_s
                theta_dof_e_idx = 3 * self._n_vertices + e_e
                qd.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
                qd.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @qd.kernel
    def clear_energy_and_gradients(self):
        # clear energy and gradients
        self._func_clear_force()
        self._func_clear_energy()
        self._func_clear_gradients()

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        # clear kinematic states
        self._kernel_clear_kinematic_states_all_substeps()
        # clear contact states
        self._kernel_clear_contact_states_all_substeps()
        # clear collision states
        self._kernel_clear_collision_states()

        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()
            entity.distribute_output_grads()

        self._kernel_clear_collision_states.grad()
        self._kernel_clear_contact_states_all_substeps.grad()
        self._kernel_clear_kinematic_states_all_substeps.grad()

    def substep_pre_coupling(self, f):
        if self.is_active:
            self.init_pos_and_vel(f)
            self.init_theta_and_omega(f)
            self.clear_energy_and_gradients()
            self.compute_stretching_energy(f)
            self.compute_bending_energy(f)
            self.compute_twisting_energy(f)
            self.update_centerline_velocities(f)
            self.update_angular_velocities(f)

    def substep_pre_coupling_grad(self, f):
        if self.is_active:
            self.update_angular_velocities.grad(f)
            self.update_centerline_velocities.grad(f)
            self.compute_twisting_energy.grad(f)
            self.compute_bending_energy.grad(f)
            self.compute_stretching_energy.grad(f)
            self.clear_energy_and_gradients.grad()
            self.init_theta_and_omega.grad(f)
            self.init_pos_and_vel.grad(f)

            # per-substep adjoint clipping: norm clipping
            if self._grad_clip > 0.0:
                self.adjoint_normsq.fill(0.0)
                self._kernel_accum_adjoint_normsq(f)
                self._kernel_scale_adjoint(f, self._grad_clip)

    @qd.kernel
    def _kernel_accum_adjoint_normsq(self, f: qd.i32):
        # per-batch sum of squares of every component of the recurrent adjoint at frame f.
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for k in qd.static(range(3)):
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.vertices.grad[f, i_v, i_b].vert[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.vertices.grad[f, i_v, i_b].vel[k], 2))
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].edge[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].d1[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].d2[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].d3[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].d1_ref[k], 2))
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].d2_ref[k], 2))
            qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].length, 2))
            qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].theta, 2))
            qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.edges.grad[f, i_e, i_b].omega, 2))
        for i_iv, i_b in qd.ndrange(self._n_internal_vertices, self._B):
            for k in qd.static(range(3)):
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.internal_vertices.grad[f, i_iv, i_b].kb[k], 2))
            qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.internal_vertices.grad[f, i_iv, i_b].twist, 2))
            for k in qd.static(range(2)):
                qd.atomic_add(self.adjoint_normsq[i_b], qd.pow(self.internal_vertices.grad[f, i_iv, i_b].kappa_rest[k], 2))

    @qd.kernel
    def _kernel_scale_adjoint(self, f: qd.i32, clip: qd.f64):
        # scale frame f's adjoint by min(1, clip / ||adjoint||) per batch (direction-preserving).
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            factor = qd.min(1.0, clip / (qd.pow(self.adjoint_normsq[i_b], 0.5) + 1e-12))
            for k in qd.static(range(3)):
                self.vertices.grad[f, i_v, i_b].vert[k] *= factor
                self.vertices.grad[f, i_v, i_b].vel[k] *= factor
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            factor = qd.min(1.0, clip / (qd.pow(self.adjoint_normsq[i_b], 0.5) + 1e-12))
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].edge[k] *= factor
                self.edges.grad[f, i_e, i_b].d1[k] *= factor
                self.edges.grad[f, i_e, i_b].d2[k] *= factor
                self.edges.grad[f, i_e, i_b].d3[k] *= factor
                self.edges.grad[f, i_e, i_b].d1_ref[k] *= factor
                self.edges.grad[f, i_e, i_b].d2_ref[k] *= factor
            self.edges.grad[f, i_e, i_b].length *= factor
            self.edges.grad[f, i_e, i_b].theta *= factor
            self.edges.grad[f, i_e, i_b].omega *= factor
        for i_iv, i_b in qd.ndrange(self._n_internal_vertices, self._B):
            factor = qd.min(1.0, clip / (qd.pow(self.adjoint_normsq[i_b], 0.5) + 1e-12))
            for k in qd.static(range(3)):
                self.internal_vertices.grad[f, i_iv, i_b].kb[k] *= factor
            self.internal_vertices.grad[f, i_iv, i_b].twist *= factor
            for k in qd.static(range(2)):
                self.internal_vertices.grad[f, i_iv, i_b].kappa_rest[k] *= factor

    def substep_post_coupling(self, f):
        if self.is_active:
            self.update_centerline_positions(f)
            self.update_frame_thetas(f)
            for i in qd.static(range(self._n_pbd_iters)):
                self.inextensibility_forward(f, i)
                self.collision_forward(f, i)
            self.update_centerline_edges(f)
            self.update_material_states(f)
            self.update_velocities_after_projection(f)
            self.friction_forward(f)
            self.transfer_vertex_states(f)   # f -> f+1

            if self._constraints_initialized:
                self.apply_hard_constraints(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active:
            self.transfer_vertex_states.grad(f)
            self.friction_forward.grad(self, f)
            self.update_velocities_after_projection.grad(f)
            self.update_material_states.grad(f)
            self.update_centerline_edges.grad(f)
            if not self._disable_constraint_grad:
                for i in range(self._n_pbd_iters - 1, -1, -1):
                    self.collision_forward.grad(self, f, i)
                    if self._any_inextensible:
                        self.inextensibility_forward.grad(self, f, i)
            self.update_frame_thetas.grad(f)
            self.update_centerline_positions.grad(f)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices[target, i_v, i_b].vert = self.vertices[source, i_v, i_b].vert
            self.vertices[target, i_v, i_b].vel = self.vertices[source, i_v, i_b].vel

            self.vertices_ng[target, i_v, i_b].fixed = self.vertices_ng[source, i_v, i_b].fixed

        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges[target, i_e, i_b].edge = self.edges[source, i_e, i_b].edge
            self.edges[target, i_e, i_b].length = self.edges[source, i_e, i_b].length
            self.edges[target, i_e, i_b].d1 = self.edges[source, i_e, i_b].d1
            self.edges[target, i_e, i_b].d2 = self.edges[source, i_e, i_b].d2
            self.edges[target, i_e, i_b].d3 = self.edges[source, i_e, i_b].d3
            self.edges[target, i_e, i_b].d1_ref = self.edges[source, i_e, i_b].d1_ref
            self.edges[target, i_e, i_b].d2_ref = self.edges[source, i_e, i_b].d2_ref
            self.edges[target, i_e, i_b].theta = self.edges[source, i_e, i_b].theta
            self.edges[target, i_e, i_b].omega = self.edges[source, i_e, i_b].omega

        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            self.internal_vertices[target, i_iv, i_b].kb = self.internal_vertices[source, i_iv, i_b].kb
            self.internal_vertices[target, i_iv, i_b].twist = self.internal_vertices[source, i_iv, i_b].twist
            self.internal_vertices[target, i_iv, i_b].kappa_rest = self.internal_vertices[source, i_iv, i_b].kappa_rest

    @qd.kernel
    def copy_grad(self, source: qd.i32, target: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices.grad[target, i_v, i_b].vert = self.vertices.grad[source, i_v, i_b].vert
            self.vertices.grad[target, i_v, i_b].vel = self.vertices.grad[source, i_v, i_b].vel

            self.vertices_ng[target, i_v, i_b].fixed = self.vertices_ng[source, i_v, i_b].fixed

        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges.grad[target, i_e, i_b].edge = self.edges.grad[source, i_e, i_b].edge
            self.edges.grad[target, i_e, i_b].length = self.edges.grad[source, i_e, i_b].length
            self.edges.grad[target, i_e, i_b].d1 = self.edges.grad[source, i_e, i_b].d1
            self.edges.grad[target, i_e, i_b].d2 = self.edges.grad[source, i_e, i_b].d2
            self.edges.grad[target, i_e, i_b].d3 = self.edges.grad[source, i_e, i_b].d3
            self.edges.grad[target, i_e, i_b].d1_ref = self.edges.grad[source, i_e, i_b].d1_ref
            self.edges.grad[target, i_e, i_b].d2_ref = self.edges.grad[source, i_e, i_b].d2_ref
            self.edges.grad[target, i_e, i_b].theta = self.edges.grad[source, i_e, i_b].theta
            self.edges.grad[target, i_e, i_b].omega = self.edges.grad[source, i_e, i_b].omega

        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            self.internal_vertices.grad[target, i_iv, i_b].kb = self.internal_vertices.grad[source, i_iv, i_b].kb
            self.internal_vertices.grad[target, i_iv, i_b].twist = self.internal_vertices.grad[source, i_iv, i_b].twist
            self.internal_vertices.grad[target, i_iv, i_b].kappa_rest = self.internal_vertices.grad[source, i_iv, i_b].kappa_rest

    @qd.kernel
    def reset_grad_till_frame(self, f: qd.i32):
        # zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
        for i_f, i_v, i_b in qd.ndrange(f, self._n_vertices, self._B):
            self.vertices.grad[i_f, i_v, i_b].vert = qd.Vector.zero(gs.qd_float, 3)
            self.vertices.grad[i_f, i_v, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_f, i_e, i_b in qd.ndrange(f, self._n_edges, self._B):
            self.edges.grad[i_f, i_e, i_b].edge = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].length = 0.0
            self.edges.grad[i_f, i_e, i_b].d1 = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].d2 = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].d3 = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].d1_ref = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].d2_ref = qd.Vector.zero(gs.qd_float, 3)
            self.edges.grad[i_f, i_e, i_b].theta = 0.0
            self.edges.grad[i_f, i_e, i_b].omega = 0.0

        for i_f, i_iv, i_b in qd.ndrange(f, self.n_internal_vertices, self._B):
            self.internal_vertices.grad[i_f, i_iv, i_b].kb = qd.Vector.zero(gs.qd_float, 3)
            self.internal_vertices.grad[i_f, i_iv, i_b].twist = 0.0
            self.internal_vertices.grad[i_f, i_iv, i_b].kappa_rest = qd.Vector.zero(gs.qd_float, 2)

    def truncate_adjoint(self):
        # Truncated BPTT: detach the recurrent adjoint by zeroing the whole grad window (all frames),
        # so no gradient carries to earlier steps.
        if self.is_active and self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local + 1)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states. Returns True if anything was injected.
        """
        injected = False
        for entity in self._entities:
            if entity.collect_output_grads():
                injected = True
        return injected

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.edge.grad is not None:
                state.edge.assert_contiguous()
                self.add_grad_from_edge(self._sim.cur_substep_local, state.edge.grad)

            if state.length.grad is not None:
                state.length.assert_contiguous()
                self.add_grad_from_length(self._sim.cur_substep_local, state.length.grad)

            if state.d1.grad is not None:
                state.d1.assert_contiguous()
                self.add_grad_from_d1(self._sim.cur_substep_local, state.d1.grad)

            if state.d2.grad is not None:
                state.d2.assert_contiguous()
                self.add_grad_from_d2(self._sim.cur_substep_local, state.d2.grad)

            if state.d3.grad is not None:
                state.d3.assert_contiguous()
                self.add_grad_from_d3(self._sim.cur_substep_local, state.d3.grad)

            if state.d1_ref.grad is not None:
                state.d1_ref.assert_contiguous()
                self.add_grad_from_d1_ref(self._sim.cur_substep_local, state.d1_ref.grad)

            if state.d2_ref.grad is not None:
                state.d2_ref.assert_contiguous()
                self.add_grad_from_d2_ref(self._sim.cur_substep_local, state.d2_ref.grad)

            if state.theta.grad is not None:
                state.theta.assert_contiguous()
                self.add_grad_from_theta(self._sim.cur_substep_local, state.theta.grad)

            if state.omega.grad is not None:
                state.omega.assert_contiguous()
                self.add_grad_from_omega(self._sim.cur_substep_local, state.omega.grad)

            if state.kb.grad is not None:
                state.kb.assert_contiguous()
                self.add_grad_from_kb(self._sim.cur_substep_local, state.kb.grad)

            if state.twist.grad is not None:
                state.twist.assert_contiguous()
                self.add_grad_from_twist(self._sim.cur_substep_local, state.twist.grad)

            if state.kappa_rest.grad is not None:
                state.kappa_rest.assert_contiguous()
                self.add_grad_from_kappa_rest(self._sim.cur_substep_local, state.kappa_rest.grad)

    @qd.kernel
    def add_grad_from_pos(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        # pos_grad shape: [B, n_vertices, 3]
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for k in qd.static(range(3)):
                self.vertices.grad[f, i_v, i_b].vert[k] += pos_grad[i_b, i_v, k]

    @qd.kernel
    def add_grad_from_vel(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        # vel_grad shape: [B, n_vertices, 3]
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for k in qd.static(range(3)):
                self.vertices.grad[f, i_v, i_b].vel[k] += vel_grad[i_b, i_v, k]

    @qd.kernel
    def add_grad_from_edge(self, f: qd.i32, edge_grad: qd.types.ndarray()):
        # edge_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].edge[k] += edge_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_length(self, f: qd.i32, length_grad: qd.types.ndarray()):
        # length_grad shape: [B, n_edges]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges.grad[f, i_e, i_b].length += length_grad[i_b, i_e]

    @qd.kernel
    def add_grad_from_d1(self, f: qd.i32, d1_grad: qd.types.ndarray()):
        # d1_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].d1[k] += d1_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_d2(self, f: qd.i32, d2_grad: qd.types.ndarray()):
        # d2_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].d2[k] += d2_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_d3(self, f: qd.i32, d3_grad: qd.types.ndarray()):
        # d3_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].d3[k] += d3_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_d1_ref(self, f: qd.i32, d1_ref_grad: qd.types.ndarray()):
        # d1_ref_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].d1_ref[k] += d1_ref_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_d2_ref(self, f: qd.i32, d2_ref_grad: qd.types.ndarray()):
        # d2_ref_grad shape: [B, n_edges, 3]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            for k in qd.static(range(3)):
                self.edges.grad[f, i_e, i_b].d2_ref[k] += d2_ref_grad[i_b, i_e, k]

    @qd.kernel
    def add_grad_from_theta(self, f: qd.i32, theta_grad: qd.types.ndarray()):
        # theta_grad shape: [B, n_edges]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges.grad[f, i_e, i_b].theta += theta_grad[i_b, i_e]

    @qd.kernel
    def add_grad_from_omega(self, f: qd.i32, omega_grad: qd.types.ndarray()):
        # omega_grad shape: [B, n_edges]
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            self.edges.grad[f, i_e, i_b].omega += omega_grad[i_b, i_e]

    @qd.kernel
    def add_grad_from_kb(self, f: qd.i32, kb_grad: qd.types.ndarray()):
        # kb_grad shape: [B, n_internal_vertices, 3]
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            for k in qd.static(range(3)):
                self.internal_vertices.grad[f, i_iv, i_b].kb[k] += kb_grad[i_b, i_iv, k]

    @qd.kernel
    def add_grad_from_twist(self, f: qd.i32, twist_grad: qd.types.ndarray()):
        # twist_grad shape: [B, n_internal_vertices]
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            self.internal_vertices.grad[f, i_iv, i_b].twist += twist_grad[i_b, i_iv]

    @qd.kernel
    def add_grad_from_kappa_rest(self, f: qd.i32, kappa_rest_grad: qd.types.ndarray()):
        # kappa_rest_grad shape: [B, n_internal_vertices, 2]
        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            for k in qd.static(range(2)):
                self.internal_vertices.grad[f, i_iv, i_b].kappa_rest[k] += kappa_rest_grad[i_b, i_iv, k]


    def save_ckpt(self, ckpt_name):
        if self.is_active and self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["vel"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["fixed"] = torch.zeros(
                    self._batch_shape((self.n_vertices,), first_dim=True), dtype=gs.tc_bool
                )
                self._ckpt[ckpt_name]["theta"] = torch.zeros(
                    self._batch_shape((self.n_edges,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["omega"] = torch.zeros(
                    self._batch_shape((self.n_edges,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["edge"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["length"] = torch.zeros(
                    self._batch_shape((self.n_edges,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d1"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d2"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d3"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d1_ref"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d2_ref"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["kb"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["twist"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["kappa_rest"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices, 2), first_dim=True), dtype=gs.tc_float
                )

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["fixed"],
                self._ckpt[ckpt_name]["theta"],
                self._ckpt[ckpt_name]["omega"],
                self._ckpt[ckpt_name]["edge"],
                self._ckpt[ckpt_name]["length"],
                self._ckpt[ckpt_name]["d1"],
                self._ckpt[ckpt_name]["d2"],
                self._ckpt[ckpt_name]["d3"],
                self._ckpt[ckpt_name]["d1_ref"],
                self._ckpt[ckpt_name]["d2_ref"],
                self._ckpt[ckpt_name]["kb"],
                self._ckpt[ckpt_name]["twist"],
                self._ckpt[ckpt_name]["kappa_rest"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        if self.is_active:
            # restart from frame 0 in memory
            self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        if self.is_active:
            self.copy_frame(0, self._sim.substeps_local)
            self.copy_grad(0, self._sim.substeps_local)

        if self.is_active and self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._scene._envs_idx,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["fixed"],
                self._ckpt[ckpt_name]["theta"],
                self._ckpt[ckpt_name]["omega"],
                self._ckpt[ckpt_name]["edge"],
                self._ckpt[ckpt_name]["length"],
                self._ckpt[ckpt_name]["d1"],
                self._ckpt[ckpt_name]["d2"],
                self._ckpt[ckpt_name]["d3"],
                self._ckpt[ckpt_name]["d1_ref"],
                self._ckpt[ckpt_name]["d2_ref"],
                self._ckpt[ckpt_name]["kb"],
                self._ckpt[ckpt_name]["twist"],
                self._ckpt[ckpt_name]["kappa_rest"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            self._kernel_set_state(
                f, envs_idx, state.pos, state.vel, state.fixed, 
                state.theta, state.omega,
                state.edge, state.length,
                state.d1, state.d2, state.d3,
                state.d1_ref, state.d2_ref,
                state.kb, state.twist, state.kappa_rest,
            )

    def get_state(self, f):
        if self.is_active:
            state = RODSolverState(self._scene)
            self._kernel_get_state(
                f, state.pos, state.vel, state.fixed, 
                state.theta, state.omega,
                state.edge, state.length,
                state.d1, state.d2, state.d3,
                state.d1_ref, state.d2_ref,
                state.kb, state.twist, state.kappa_rest,
            )
        else:
            state = None
        return state

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        vertices = self.vertices_render
        radii = self.vertices_param.radius

        return vertices, radii

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        raise NotImplementedError()

    @qd.kernel
    def _kernel_add_rods(
        self,
        rod_idx: qd.i32,
        is_loop: qd.u1,
        use_inextensible: qd.u1,
        stretching_stiffness: qd.f64,
        bending_stiffness: qd.f64,
        twisting_stiffness: qd.f64,
        plastic_yield: qd.f64,
        plastic_creep: qd.f64,
        v_start: qd.i32,
        e_start: qd.i32,
        iv_start: qd.i32,
        n_verts: qd.i32,
    ):
        # info (parametric)
        for i_b in range(self._B):
            self.rods_param[rod_idx, i_b].use_inextensible = use_inextensible
            self.rods_stretching_stiffness[rod_idx, i_b] = stretching_stiffness
            self.rods_bending_stiffness[rod_idx, i_b] = bending_stiffness
            self.rods_twisting_stiffness[rod_idx, i_b] = twisting_stiffness
            self.rods_param[rod_idx, i_b].plastic_yield = plastic_yield
            self.rods_param[rod_idx, i_b].plastic_creep = plastic_creep

            self.rods_energy[rod_idx, i_b].stretching_energy = 0.0
            self.rods_energy[rod_idx, i_b].bending_energy = 0.0
            self.rods_energy[rod_idx, i_b].twisting_energy = 0.0

        self.rods_info[rod_idx].is_loop = is_loop

        # -------------------------------- build indices --------------------------------

        # info (structural)
        self.rods_info[rod_idx].first_vert_idx = v_start
        self.rods_info[rod_idx].first_edge_idx = e_start
        self.rods_info[rod_idx].first_internal_vert_idx = iv_start
        self.rods_info[rod_idx].n_verts = n_verts

        # rod id of verts
        for i_v in range(n_verts):
            vert_idx = i_v + v_start
            self.vertices_info[vert_idx].rod_idx = rod_idx

        # vert id of edges
        n_edges = n_verts if is_loop else n_verts - 1
        for i_e in range(n_edges):
            vert_idx = i_e + v_start
            edge_idx = i_e + e_start
            self.edges_info[edge_idx].vert_idx = vert_idx

        # edge id of internal verts
        n_internal_verts = n_verts - (0 if is_loop else 2)
        for i_iv in range(n_internal_verts):
            edge_idx = -1
            if is_loop:
                edge_idx = qm.mod(i_iv - 1, n_internal_verts) + e_start
            else:
                edge_idx = i_iv + e_start
            iv_idx = i_iv + iv_start
            self.internal_vertices_info[iv_idx].edge_idx = edge_idx

    @qd.kernel
    def _kernel_finalize_rest_states(
        self,
        f: qd.i32,
        rod_idx: qd.i32,
        v_start: qd.i32,
        e_start: qd.i32,
        iv_start: qd.i32,
        segment_mass: qd.f64,        # NOTE: we can use array
        segment_radius: qd.f64,      # NOTE: we can use array
        static_friction: qd.f64,     # NOTE: we can use array
        kinetic_friction: qd.f64,    # NOTE: we can use array
        restitution: qd.f64,         # NOTE: we can use array
        verts_rest: qd.types.ndarray(dtype=qm.vec3, ndim=1),
        edges_rest: qd.types.ndarray(dtype=qm.vec3, ndim=1),
    ):  
        n_verts_local = verts_rest.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start

            # info (parametric)
            self.vertices_param[i_global, i_b].mass = segment_mass
            self.vertices_param[i_global, i_b].radius = segment_radius
            self.vertices_param[i_global, i_b].mu_s = static_friction
            self.vertices_param[i_global, i_b].mu_k = kinetic_friction
            self.vertices_param[i_global, i_b].restitution = restitution

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            # info (structural)
            self.vertices_info[i_global].rod_idx = rod_idx

            # finalize rest vertices    # not used
            # self.vertices_info[i_global].vert_rest = verts_rest[i_v]

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        qd.loop_config(serialize=True)
        for i_e in range(n_edges_local):
            i_global = i_e + e_start
            # v_s, v_e = self.get_edge_vertices(i_global)

            # finalize rest edges

            # self.edges_info[i_global].edge_rest = self.vertices_info[v_e].vert_rest - self.vertices_info[v_s].vert_rest
            self.edges_info[i_global].edge_rest = edges_rest[i_e]
            self.edges_info[i_global].length_rest = qm.length(self.edges_info[i_global].edge_rest)
            self.edges_info[i_global].d3_rest = self.edges_info[i_global].edge_rest.normalized()

            # finalize rest material frame (d1, d2, d3)

            if i_e == 0: # first edge
                self.edges_info[i_global].d1_rest = get_perpendicular_vector(self.edges_info[i_global].d3_rest)
            else:
                self.edges_info[i_global].d1_rest = parallel_transport_normalized(
                    self.edges_info[i_global - 1].d3_rest,
                    self.edges_info[i_global].d3_rest,
                    self.edges_info[i_global - 1].d1_rest,
                )
            self.edges_info[i_global].d2_rest = qm.cross(self.edges_info[i_global].d3_rest, self.edges_info[i_global].d1_rest)

        # deal with loop topology

        if self.rods_info[rod_idx].is_loop:
            e_end = e_start + n_edges_local - 1

            d1_final_transport = parallel_transport_normalized(
                self.edges_info[e_end].d3_rest,
                self.edges_info[e_start].d3_rest,
                self.edges_info[e_end].d1_rest,
            )

            total_holonomy_angle = get_angle(
                self.edges_info[e_start].d3_rest,
                d1_final_transport,
                self.edges_info[e_start].d1_rest,
            )

            for i_e in range(n_edges_local):
                i_global = i_e + e_start

                correction_angle = - total_holonomy_angle * (i_e / n_edges_local)
                d1_uncorrected = self.edges_info[i_global].d1_rest
                d2_uncorrected = self.edges_info[i_global].d2_rest
                c, s = qd.cos(correction_angle), qd.sin(correction_angle)
                self.edges_info[i_global].d1_rest = c * d1_uncorrected + s * d2_uncorrected
                self.edges_info[i_global].d2_rest = -s * d1_uncorrected + c * d2_uncorrected

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in qd.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # finalize rest curvature binormal

            rest_kbs = curvature_binormal(self.edges_info[e_s].d3_rest, self.edges_info[e_e].d3_rest)
            self.internal_vertices[f, i_iv, i_b].kappa_rest = qd.Vector([
                  0.5 * qm.dot(rest_kbs, self.edges_info[e_s].d2_rest + self.edges_info[e_e].d2_rest),
                - 0.5 * qm.dot(rest_kbs, self.edges_info[e_s].d1_rest + self.edges_info[e_e].d1_rest),
            ])
            self.internal_vertices_info[i_global].twist_rest = 0.0  # assume no initial twist

    @qd.kernel
    def _kernel_finalize_states(
        self,
        f: qd.i32,
        rod_idx: qd.i32,
        v_start: qd.i32,
        e_start: qd.i32,
        iv_start: qd.i32,
        fixed: qd.u1,
        verts: qd.types.ndarray(dtype=qm.vec3, ndim=1),
        edges: qd.types.ndarray(dtype=qm.vec3, ndim=1),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start

            # state (dynamic)
            self.vertices[f, i_global, i_b].vert = verts[i_v]
            self.vertices[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

            # state (dynamic w/o grad)
            self.vertices_ng[f, i_global, i_b].fixed = fixed
            self.vertices_ng[f, i_global, i_b].kinematic = False

            # vertex constraints
            self.vertex_constraints[i_global, i_b].constrained = False
            self.vertex_constraints[i_global, i_b].link_idx = -1
            self.vertex_constraints[i_global, i_b].target_pos = qd.Vector.zero(gs.qd_float, 3)
            self.vertex_constraints[i_global, i_b].local_pos = qd.Vector.zero(gs.qd_float, 3)

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        for i_b in range(self._B):
            for i_e in range(n_edges_local):
                i_global = i_e + e_start
                # v_s, v_e = self.get_edge_vertices(i_global)

                # state (dynamic)

                # self.edges[f, i_global, i_b].edge = self.vertices[f, v_e, i_b].vert - self.vertices[f, v_s, i_b].vert
                self.edges[f, i_global, i_b].edge = edges[i_e]
                self.edges[f, i_global, i_b].length = qm.length(self.edges[f, i_global, i_b].edge)
                self.edges[f, i_global, i_b].d3 = self.edges[f, i_global, i_b].edge.normalized()

                if i_e == 0: # first edge
                    self.edges[f, i_global, i_b].d1 = get_perpendicular_vector(self.edges[f, i_global, i_b].d3)
                else:
                    self.edges[f, i_global, i_b].d1 = parallel_transport_normalized(
                        self.edges[f, i_global - 1, i_b].d3,
                        self.edges[f, i_global, i_b].d3,
                        self.edges[f, i_global - 1, i_b].d1,
                    )
                self.edges[f, i_global, i_b].d1_ref = self.edges[f, i_global, i_b].d1

                self.edges[f, i_global, i_b].d2 = qm.cross(self.edges[f, i_global, i_b].d3, self.edges[f, i_global, i_b].d1)
                self.edges[f, i_global, i_b].d2_ref = self.edges[f, i_global, i_b].d2

                self.edges[f, i_global, i_b].theta = 0.0  # assume no initial twist
                self.edges[f, i_global, i_b].omega = 0.0  # assume no initial twist rate

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in qd.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # state (dynamic)

            self.internal_vertices[f, i_global, i_b].kb = curvature_binormal(
                self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3
            )
            self.internal_vertices[f, i_global, i_b].twist = 0.0    # assume no initial twist

    @qd.kernel
    def _kernel_set_vertices_pos(
        self,
        f: qd.i32,
        v_start: qd.i32,
        n_vertices: qd.i32,
        pos: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.vertices[f, i_global, i_b].vert[j] = pos[i_b, i_v, j]

    @qd.kernel
    def _kernel_set_vertices_pos_grad(
        self,
        f: qd.i32,
        v_start: qd.i32,
        n_vertices: qd.i32,
        pos_grad: qd.types.ndarray(),
    ):  
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                pos_grad[i_b, i_v, j] = self.vertices.grad[f, i_global, i_b].vert[j]

    @qd.kernel
    def _kernel_set_vertices_vel(
        self,
        f: qd.i32,
        v_start: qd.i32,
        n_vertices: qd.i32,
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.vertices[f, i_global, i_b].vel[j] = vel[i_b, i_v, j]

    @qd.kernel
    def _kernel_set_vertices_vel_grad(
        self,
        f: qd.i32,
        v_start: qd.i32,
        n_vertices: qd.i32,
        vel_grad: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                vel_grad[i_b, i_v, j] = self.vertices.grad[f, i_global, i_b].vel[j]

    @qd.kernel
    def _kernel_set_edges_edge(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        edge: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].edge[j] = edge[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_edge_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        edge_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                edge_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].edge[j]

    @qd.kernel
    def _kernel_set_edges_length(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        length: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            self.edges[f, i_global, i_b].length = length[i_b, i_e]

    @qd.kernel
    def _kernel_set_edges_length_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        length_grad: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            length_grad[i_b, i_e] = self.edges.grad[f, i_global, i_b].length

    @qd.kernel
    def _kernel_set_edges_d1(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d1: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].d1[j] = d1[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_d1_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d1_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                d1_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].d1[j]

    @qd.kernel
    def _kernel_set_edges_d2(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d2: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].d2[j] = d2[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_d2_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d2_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                d2_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].d2[j]

    @qd.kernel
    def _kernel_set_edges_d3(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d3: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].d3[j] = d3[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_d3_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d3_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                d3_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].d3[j]

    @qd.kernel
    def _kernel_set_edges_d1_ref(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d1_ref: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].d1_ref[j] = d1_ref[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_d1_ref_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d1_ref_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                d1_ref_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].d1_ref[j]

    @qd.kernel
    def _kernel_set_edges_d2_ref(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d2_ref: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                self.edges[f, i_global, i_b].d2_ref[j] = d2_ref[i_b, i_e, j]

    @qd.kernel
    def _kernel_set_edges_d2_ref_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        d2_ref_grad: qd.types.ndarray(),  # shape [B, n_edges, 3]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            for j in qd.static(range(3)):
                d2_ref_grad[i_b, i_e, j] = self.edges.grad[f, i_global, i_b].d2_ref[j]

    @qd.kernel
    def _kernel_set_edges_theta(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        theta: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            self.edges[f, i_global, i_b].theta = theta[i_b, i_e]

    @qd.kernel
    def _kernel_set_edges_theta_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        theta_grad: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            theta_grad[i_b, i_e] = self.edges.grad[f, i_global, i_b].theta

    @qd.kernel
    def _kernel_set_edges_omega(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        omega: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            self.edges[f, i_global, i_b].omega = omega[i_b, i_e]

    @qd.kernel
    def _kernel_set_edges_omega_grad(
        self,
        f: qd.i32,
        e_start: qd.i32,
        n_edges: qd.i32,
        omega_grad: qd.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in qd.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            omega_grad[i_b, i_e] = self.edges.grad[f, i_global, i_b].omega

    @qd.kernel
    def _kernel_set_internal_vertices_kb(
        self,
        f: qd.i32,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        kb: qd.types.ndarray(),  # shape [B, n_internal_vertices, 3]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            for j in qd.static(range(3)):
                self.internal_vertices[f, i_global, i_b].kb[j] = kb[i_b, i_iv, j]

    @qd.kernel
    def _kernel_set_internal_vertices_kb_grad(
        self,
        f: qd.i32,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        kb_grad: qd.types.ndarray(),  # shape [B, n_internal_vertices, 3]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            for j in qd.static(range(3)):
                kb_grad[i_b, i_iv, j] = self.internal_vertices.grad[f, i_global, i_b].kb[j]

    @qd.kernel
    def _kernel_set_internal_vertices_twist(
        self,
        f: qd.i32,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        twist: qd.types.ndarray(),  # shape [B, n_internal_vertices]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            self.internal_vertices[f, i_global, i_b].twist = twist[i_b, i_iv]

    @qd.kernel
    def _kernel_set_internal_vertices_twist_grad(
        self,
        f: qd.i32,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        twist_grad: qd.types.ndarray(),  # shape [B, n_internal_vertices]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            twist_grad[i_b, i_iv] = self.internal_vertices.grad[f, i_global, i_b].twist

    @qd.kernel
    def _kernel_set_internal_vertices_kappa_rest(
        self,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        kappa_rest: qd.types.ndarray(),  # shape [B, n_internal_vertices, 2]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            for j in qd.static(range(2)):
                self.internal_vertices_info[i_global].kappa_rest[j] = kappa_rest[i_b, i_iv, j]

    @qd.kernel
    def _kernel_set_internal_vertices_kappa_rest_grad(
        self,
        iv_start: qd.i32,
        n_internal_vertices: qd.i32,
        kappa_rest_grad: qd.types.ndarray(),  # shape [B, n_internal_vertices, 2]
    ):
        for i_iv, i_b in qd.ndrange(n_internal_vertices, self._B):
            i_global = i_iv + iv_start
            for j in qd.static(range(2)):
                kappa_rest_grad[i_b, i_iv, j] = self.internal_vertices_info.grad[i_global].kappa_rest[j]

    @qd.kernel
    def _kernel_set_fixed_states(
        self,
        f: qd.i32,
        v_start: qd.i32,
        n_vertices: qd.i32,
        fixed: qd.types.ndarray(),  # shape [B, n_vertices]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            self.vertices_ng[f, i_global, i_b].fixed = fixed[i_b, i_v]

    @qd.kernel
    def _kernel_set_bending_stiffness(
        self,
        rod_idx: qd.i32,
        bending_stiffness: qd.types.ndarray(),  # shape [len(envs_idx)]
        envs_idx: qd.types.ndarray(),
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.rods_bending_stiffness[rod_idx, i_b] = bending_stiffness[i_b_]

    @qd.kernel
    def _kernel_set_twisting_stiffness(
        self,
        rod_idx: qd.i32,
        twisting_stiffness: qd.types.ndarray(),  # shape [len(envs_idx)]
        envs_idx: qd.types.ndarray(),
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.rods_twisting_stiffness[rod_idx, i_b] = twisting_stiffness[i_b_]

    @qd.kernel
    def _kernel_set_stretching_stiffness(
        self,
        rod_idx: qd.i32,
        stretching_stiffness: qd.types.ndarray(),  # shape [len(envs_idx)]
        envs_idx: qd.types.ndarray(),
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.rods_stretching_stiffness[rod_idx, i_b] = stretching_stiffness[i_b_]

    @qd.kernel
    def _kernel_set_segment_mass(
        self,
        v_start: qd.i32,
        n_vertices: qd.i32,
        mass: qd.types.ndarray(),  # shape [len(envs_idx), n_vertices]
        envs_idx: qd.types.ndarray(),
    ):
        for i_v, i_b_ in qd.ndrange(n_vertices, envs_idx.shape[0]):
            i_global = i_v + v_start
            i_b = envs_idx[i_b_]
            self.vertices_param[i_global, i_b].mass = mass[i_b_, i_v]

    @qd.kernel
    def _kernel_set_segment_radius(
        self,
        v_start: qd.i32,
        n_vertices: qd.i32,
        radius: qd.types.ndarray(),  # shape [len(envs_idx), n_vertices]
        envs_idx: qd.types.ndarray(),
    ):
        for i_v, i_b_ in qd.ndrange(n_vertices, envs_idx.shape[0]):
            i_global = i_v + v_start
            i_b = envs_idx[i_b_]
            self.vertices_param[i_global, i_b].radius = radius[i_b_, i_v]

    @qd.kernel
    def _kernel_set_mu_s(
        self,
        v_start: qd.i32,
        n_vertices: qd.i32,
        mu_s: qd.types.ndarray(),  # shape [len(envs_idx), n_vertices]
        envs_idx: qd.types.ndarray(),
    ):
        for i_v, i_b_ in qd.ndrange(n_vertices, envs_idx.shape[0]):
            i_global = i_v + v_start
            i_b = envs_idx[i_b_]
            self.vertices_param[i_global, i_b].mu_s = mu_s[i_b_, i_v]

    @qd.kernel
    def _kernel_set_mu_k(
        self,
        v_start: qd.i32,
        n_vertices: qd.i32,
        mu_k: qd.types.ndarray(),  # shape [len(envs_idx), n_vertices]
        envs_idx: qd.types.ndarray(),
    ):
        for i_v, i_b_ in qd.ndrange(n_vertices, envs_idx.shape[0]):
            i_global = i_v + v_start
            i_b = envs_idx[i_b_]
            self.vertices_param[i_global, i_b].mu_k = mu_k[i_b_, i_v]

    @qd.kernel
    def _kernel_set_attached_states(
        self,
        i_v: qd.i32,
        link_idx: qd.i32,
        local_pos: qd.types.ndarray(),  # shape [B, 3]
    ):
        for i_b in range(self._B):
            self.vertex_constraints[i_v, i_b].constrained = True
            self.vertex_constraints[i_v, i_b].link_idx = link_idx
            for j in qd.static(range(3)):
                self.vertex_constraints[i_v, i_b].local_pos[j] = local_pos[i_b, j]

    @qd.kernel
    def _kernel_set_attached_states_with_envs_idx(
        self,
        i_v: qd.i32,
        link_idx: qd.i32,
        local_pos: qd.types.ndarray(),  # shape [3]
        envs_idx: qd.i32,
    ):
        self.vertex_constraints[i_v, envs_idx].constrained = True
        self.vertex_constraints[i_v, envs_idx].link_idx = link_idx
        for j in qd.static(range(3)):
            self.vertex_constraints[i_v, envs_idx].local_pos[j] = local_pos[j]

    @qd.kernel
    def _kernel_detach_vertex(
        self,
        i_v: qd.i32,
    ):
        for i_b in range(self._B):
            self.vertex_constraints[i_v, i_b].constrained = False
            self.vertex_constraints[i_v, i_b].link_idx = -1
            self.vertex_constraints[i_v, i_b].target_pos = qd.Vector.zero(gs.qd_float, 3)
            self.vertex_constraints[i_v, i_b].local_pos = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_detach_vertex_with_envs_idx(
        self,
        i_v: qd.i32,
        envs_idx: qd.i32,
    ):
        self.vertex_constraints[i_v, envs_idx].constrained = False
        self.vertex_constraints[i_v, envs_idx].link_idx = -1
        self.vertex_constraints[i_v, envs_idx].target_pos = qd.Vector.zero(gs.qd_float, 3)
        self.vertex_constraints[i_v, envs_idx].local_pos = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_update_attached_verts(
        self,
        links_state: LinksState,
    ):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.constrained and vc.link_idx >= 0:
                i_l = vc.link_idx
                l_pos = links_state.pos[i_l, i_b]
                l_quat = links_state.quat[i_l, i_b]

                # transform the stored local position to world space
                v_pos_local = vc.local_pos
                v_pos_world = qd_transform_by_trans_quat(v_pos_local, l_pos, l_quat)
                for j in qd.static(range(3)):
                    self.vertex_constraints[i_v, i_b].target_pos[j] = v_pos_world[j]

    @qd.kernel
    def apply_hard_constraints(self, f: qd.i32):
        """Apply hard constraints by directly overriding positions and velocities."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.constrained and vc.link_idx >= 0:
                self.vertices[f + 1, i_v, i_b].vert = vc.target_pos
                self.vertices[f + 1, i_v, i_b].vel.fill(0.0)

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),        # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),        # shape [B, n_vertices, 3]
        fixed: qd.types.ndarray(),      # shape [B, n_vertices]
        theta: qd.types.ndarray(),      # shape [B, n_edges]
        omega: qd.types.ndarray(),      # shape [B, n_edges]
        edge: qd.types.ndarray(),       # shape [B, n_edges, 3]
        length: qd.types.ndarray(),     # shape [B, n_edges]
        d1: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d2: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d3: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d1_ref: qd.types.ndarray(),     # shape [B, n_edges, 3]
        d2_ref: qd.types.ndarray(),     # shape [B, n_edges, 3]
        kb: qd.types.ndarray(),             # shape [B, n_internal_vertices, 3]
        twist: qd.types.ndarray(),          # shape [B, n_internal_vertices]
        kappa_rest: qd.types.ndarray(),     # shape [B, n_internal_vertices, 2]
    ):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self.vertices[f, i_v, i_b].vert[j]
                vel[i_b, i_v, j] = self.vertices[f, i_v, i_b].vel[j]
            fixed[i_b, i_v] = self.vertices_ng[f, i_v, i_b].fixed

        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            length[i_b, i_e] = self.edges[f, i_e, i_b].length
            theta[i_b, i_e] = self.edges[f, i_e, i_b].theta
            omega[i_b, i_e] = self.edges[f, i_e, i_b].omega
            for j in qd.static(range(3)):
                edge[i_b, i_e, j] = self.edges[f, i_e, i_b].edge[j]
                d1[i_b, i_e, j] = self.edges[f, i_e, i_b].d1[j]
                d2[i_b, i_e, j] = self.edges[f, i_e, i_b].d2[j]
                d3[i_b, i_e, j] = self.edges[f, i_e, i_b].d3[j]
                d1_ref[i_b, i_e, j] = self.edges[f, i_e, i_b].d1_ref[j]
                d2_ref[i_b, i_e, j] = self.edges[f, i_e, i_b].d2_ref[j]

        for i_iv, i_b in qd.ndrange(self.n_internal_vertices, self._B):
            for j in qd.static(range(3)):
                kb[i_b, i_iv, j] = self.internal_vertices[f, i_iv, i_b].kb[j]
            twist[i_b, i_iv] = self.internal_vertices[f, i_iv, i_b].twist
            for j in qd.static(range(2)):
                kappa_rest[i_b, i_iv, j] = self.internal_vertices[f, i_iv, i_b].kappa_rest[j]

    @qd.kernel
    def get_state_render_kernel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.vertices[f, i_v, i_b].vert[j], qd.f32)
                self.vertices_render[i_v, i_b][j] = pos_j + self.envs_offset[i_b][j]

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        envs_idx: qd.types.ndarray(),
        pos: qd.types.ndarray(),        # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),        # shape [B, n_vertices, 3]
        fixed: qd.types.ndarray(),      # shape [B, n_vertices]
        theta: qd.types.ndarray(),      # shape [B, n_edges]
        omega: qd.types.ndarray(),      # shape [B, n_edges]
        edge: qd.types.ndarray(),       # shape [B, n_edges, 3]
        length: qd.types.ndarray(),     # shape [B, n_edges]
        d1: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d2: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d3: qd.types.ndarray(),         # shape [B, n_edges, 3]
        d1_ref: qd.types.ndarray(),     # shape [B, n_edges, 3]
        d2_ref: qd.types.ndarray(),     # shape [B, n_edges, 3]
        kb: qd.types.ndarray(),             # shape [B, n_internal_vertices, 3]
        twist: qd.types.ndarray(),          # shape [B, n_internal_vertices]
        kappa_rest: qd.types.ndarray(),     # shape [B, n_internal_vertices, 2]
    ):
        for i_v, i_b_ in qd.ndrange(self._n_vertices, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for j in qd.static(range(3)):
                self.vertices[f, i_v, i_b].vert[j] = pos[i_b, i_v, j]
                self.vertices[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]
            self.vertices_ng[f, i_v, i_b].fixed = fixed[i_b, i_v]

        for i_e, i_b_ in qd.ndrange(self._n_edges, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.edges[f, i_e, i_b].length = length[i_b, i_e]
            self.edges[f, i_e, i_b].theta = theta[i_b, i_e]
            self.edges[f, i_e, i_b].omega = omega[i_b, i_e]

            for j in qd.static(range(3)):
                self.edges[f, i_e, i_b].edge[j] = edge[i_b, i_e, j]
                self.edges[f, i_e, i_b].d1[j] = d1[i_b, i_e, j]
                self.edges[f, i_e, i_b].d2[j] = d2[i_b, i_e, j]
                self.edges[f, i_e, i_b].d3[j] = d3[i_b, i_e, j]
                self.edges[f, i_e, i_b].d1_ref[j] = d1_ref[i_b, i_e, j]
                self.edges[f, i_e, i_b].d2_ref[j] = d2_ref[i_b, i_e, j]

        for i_iv, i_b_ in qd.ndrange(self.n_internal_vertices, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for j in qd.static(range(3)):
                self.internal_vertices[f, i_iv, i_b].kb[j] = kb[i_b, i_iv, j]
            self.internal_vertices[f, i_iv, i_b].twist = twist[i_b, i_iv]
            for j in qd.static(range(2)):
                self.internal_vertices[f, i_iv, i_b].kappa_rest[j] = kappa_rest[i_b, i_iv, j]

    # ------------------------------------------------------------------------------------
    # --------------------------------- index utilities -----------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def get_edge_vertices(self, i_e: qd.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_end = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            next_local_idx = qd.cast(qm.mod(local_v_start + 1, n_verts), qd.i32)
            v_end = first_vert_idx + next_local_idx
        else:
            v_end = v_start + 1

        return v_start, v_end

    @qd.func
    def get_hinge_edges(self, i_iv: qd.i32):
        e_start = self.internal_vertices_info[i_iv].edge_idx
        v_start_of_e_start = self.edges_info[e_start].vert_idx
        rod_id = self.vertices_info[v_start_of_e_start].rod_idx

        e_end = -1
        if self.rods_info[rod_id].is_loop:
            first_edge_idx = self.rods_info[rod_id].first_edge_idx
            n_verts = self.rods_info[rod_id].n_verts
            n_edges = n_verts - 1   # normal case
            if self.rods_info[rod_id].is_loop:
                n_edges = n_verts

            local_e_start = e_start - first_edge_idx
            next_local_idx = qd.cast(qm.mod(local_e_start + 1, n_edges), qd.i32)
            e_end = first_edge_idx + next_local_idx
        else:
            e_end = e_start + 1

        return e_start, e_end

    @qd.func
    def get_hinge_vertices(self, i_e: qd.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_middle, v_end = -1, -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            local_v_middle = qd.cast(qm.mod(local_v_start + 1, n_verts), qd.i32)
            local_v_end = qd.cast(qm.mod(local_v_start + 2, n_verts), qd.i32)

            v_middle = first_vert_idx + local_v_middle
            v_end = first_vert_idx + local_v_end
        else:
            v_middle = v_start + 1
            v_end = v_start + 2

        return v_start, v_middle, v_end

    @qd.func
    def get_next_vertex_of_edge(self, i_v: qd.i32):
        rod_id = self.vertices_info[i_v].rod_idx

        ip1_v = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_i_v = i_v - first_vert_idx
            next_local_idx = qd.cast(qm.mod(local_i_v + 1, n_verts), qd.i32)
            ip1_v = first_vert_idx + next_local_idx
        else:
            ip1_v = i_v + 1

        return ip1_v

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height
    
    @property
    def damping(self):
        return self._damping

    @property
    def angular_damping(self):
        return self._angular_damping

    @property
    def n_dofs(self):
        return sum([entity.n_dofs for entity in self._entities])
    
    @property
    def n_rods(self):
        return len(self._entities)

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_edges(self):
        return sum([entity.n_edges for entity in self._entities])

    @property
    def n_internal_vertices(self):
        return sum([entity.n_internal_vertices for entity in self._entities])

    @property
    def geom_indices(self):
        return self._geom_indices

    # ------------------------------------------------------------------------------------
    # -------------------------------- pbd constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_get_inverse_mass(self, f: qd.i32, i_v: qd.i32, i_b: qd.i32):
        mass = self.vertices_param[i_v, i_b].mass
        inv_mass = 0.0
        if (
            self.vertices_ng[f, i_v, i_b].fixed or 
            self.vertices_ng[f, i_v, i_b].kinematic or 
            mass <= 0.
        ):
            inv_mass = 0.0
        else:
            inv_mass = 1.0 / mass
        return inv_mass

    @qd.kernel
    def _kernel_clear_contact_states(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_valid_edge_pairs, self._B):
            for j in qd.static(range(3)):
                self.rr_constraints[f, i_p, i_b].normal[j] = 0.0
            self.rr_constraints[f, i_p, i_b].penetration = 0.0

    @qd.kernel
    def _kernel_clear_contact_states_all_substeps(self):
        for i_f, i_p, i_b in qd.ndrange(self._sim.substeps_local, self._n_valid_edge_pairs, self._B):
            for j in qd.static(range(3)):
                self.rr_constraints[i_f, i_p, i_b].normal[j] = 0.0
            self.rr_constraints[i_f, i_p, i_b].penetration = 0.0

    @qd.kernel
    def _kernel_clear_kinematic_states(self, f: qd.i32):
        # TODO: do we need to clear kinematic states?
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices_ng[f, i_v, i_b].kinematic = False

    @qd.kernel
    def _kernel_clear_kinematic_states_all_substeps(self):
        for i_f, i_v, i_b in qd.ndrange(self._sim.substeps_local, self._n_vertices, self._B):
            self.vertices_ng[i_f, i_v, i_b].kinematic = False

    @qd.kernel
    def _kernel_clear_collision_states(self):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.vertices_collision[i_v, i_b].collided = False
            self.vertices_collision[i_v, i_b].penetration = 0.0
            self.vertices_collision[i_v, i_b].geom_idx = -1
            for j in qd.static(range(3)):
                self.vertices_collision[i_v, i_b].normal[j] = 0.0

    @qd.kernel
    def _kernel_apply_inextensibility_constraints(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check inextensibility enabled
            if self.rods_param[rod_id, i_b].use_inextensible:
                inv_mass_s = self._func_get_inverse_mass(f, v_s, i_b)
                inv_mass_e = self._func_get_inverse_mass(f, v_e, i_b)
                inv_mass_sum = inv_mass_s + inv_mass_e

                if inv_mass_sum > EPS:
                    p_s, p_e = self.vertices[f + 1, v_s, i_b].vert, self.vertices[f + 1, v_e, i_b].vert

                    edge_vec = p_e - p_s
                    dist = qm.length(edge_vec)

                    constraint_error = dist - self.edges_info[i_e].length_rest

                    if dist > EPS:
                        normal = edge_vec / dist
                        lambda_ = constraint_error / inv_mass_sum
                        delta_p_s = lambda_ * inv_mass_s * normal
                        delta_p_e = -lambda_ * inv_mass_e * normal

                        # apply corrections
                        self.vertices[f + 1, v_s, i_b].vert += delta_p_s
                        self.vertices[f + 1, v_e, i_b].vert += delta_p_e

    @qd.ad.grad_replaced
    def inextensibility_forward(self, f, iter_idx):
        # Save the pre-projection positions for this PBD iteration so the backward can linearize the
        # nonlinear length constraint at the correct point (the in-place kernel overwrites them).
        # Skipped when the constraint backward is disabled (the buffer is then never allocated/read).
        if self._sim.requires_grad and self._any_inextensible and not self._disable_constraint_grad:
            self._kernel_store_inext_preproj(f, iter_idx)
        self._kernel_apply_inextensibility_constraints(f)

    @qd.ad.grad_for(inextensibility_forward)
    def inextensibility_backward(self, f, iter_idx):
        # Snapshot the incoming position adjoint (frame f+1) so the grad kernel reads a consistent,
        # immutable value while accumulating contributions race-free (deterministic). The collision
        # backward runs immediately before this in the reverse loop and reuses the same snapshot
        # buffer; re-snapshotting here captures the adjoint as left by collision's backward.
        self._kernel_snapshot_vert_grad(f)
        self._kernel_apply_inextensibility_constraints_grad(f, iter_idx)

    @qd.kernel
    def _kernel_store_inext_preproj(self, f: qd.i32, iter_idx: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.inext_preproj_vert[f, iter_idx, i_v, i_b] = self.vertices[f + 1, i_v, i_b].vert

    @qd.kernel
    def _kernel_apply_inextensibility_constraints_grad(self, f: qd.i32, iter_idx: qd.i32):
        # Custom adjoint of `_kernel_apply_inextensibility_constraints`. The forward applies the
        # in-place edge-length correction
        #     p_s_out = p_s + a_s * D,   p_e_out = p_e + a_e * D
        # where  a_s = w_s/(w_s+w_e),  a_e = -w_e/(w_s+w_e),  w = inverse mass,
        #        D = edge_vec - L_rest * n,   edge_vec = p_e - p_s,   n = edge_vec/dist.
        # With M = dD/d(edge_vec) = I - (L_rest/dist)(I - n n^T)  (symmetric), the input adjoints
        # gain (on top of the identity pass-through already in the buffer):
        #     g_{p_s} += -M G,   g_{p_e} += +M G,   G = a_s * g_s_out + a_e * g_e_out,
        # where g_*_out are the incoming output adjoints read from the immutable snapshot.
        # Geometry (p_s, p_e) is read from the saved pre-projection positions, not vertices[f+1],
        # so the linearization point matches the forward.
        for i_e, i_b in qd.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            if self.rods_param[rod_id, i_b].use_inextensible:
                inv_mass_s = self._func_get_inverse_mass(f, v_s, i_b)
                inv_mass_e = self._func_get_inverse_mass(f, v_e, i_b)
                inv_mass_sum = inv_mass_s + inv_mass_e

                if inv_mass_sum > EPS:
                    p_s = self.inext_preproj_vert[f, iter_idx, v_s, i_b]
                    p_e = self.inext_preproj_vert[f, iter_idx, v_e, i_b]
                    edge_vec = p_e - p_s
                    dist = qm.length(edge_vec)

                    if dist > EPS:
                        L_rest = self.edges_info[i_e].length_rest
                        normal = edge_vec / dist

                        a_s = inv_mass_s / inv_mass_sum
                        a_e = -inv_mass_e / inv_mass_sum

                        # Incoming output adjoints from the immutable snapshot (race-free reads).
                        g_s = self.collision_vert_grad_snapshot[v_s, i_b]
                        g_e = self.collision_vert_grad_snapshot[v_e, i_b]

                        g_combined = a_s * g_s + a_e * g_e
                        # M G = G - (L_rest/dist)(G - (n.G) n)
                        m_g = g_combined - (L_rest / dist) * (g_combined - normal.dot(g_combined) * normal)

                        # Accumulate the Jacobian contribution race-free (snapshot reads + atomic_add).
                        for _k in qd.static(range(3)):
                            qd.atomic_add(self.vertices.grad[f + 1, v_s, i_b].vert[_k], -m_g[_k])
                            qd.atomic_add(self.vertices.grad[f + 1, v_e, i_b].vert[_k], m_g[_k])

    @qd.kernel
    def _kernel_apply_rod_collision_constraints(self, f: qd.i32, iter_idx: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_valid_edge_pairs, self._B):
            idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
            idx_a2 = self.get_next_vertex_of_edge(idx_a1)
            idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
            idx_b2 = self.get_next_vertex_of_edge(idx_b1)

            p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
            p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

            radius_a = (self.vertices_param[idx_a1, i_b].radius + self.vertices_param[idx_a2, i_b].radius) * 0.5
            radius_b = (self.vertices_param[idx_b1, i_b].radius + self.vertices_param[idx_b2, i_b].radius) * 0.5

            # compute closest points (t, u) and distance
            e1, e2 = p_a2 - p_a1, p_b2 - p_b1
            e12 = p_b1 - p_a1
            d1, d2 = e1.dot(e1), e2.dot(e2)
            r = e1.dot(e2)
            s1, s2 = e1.dot(e12), e2.dot(e12)
            den = d1 * d2 - r * r

            t = 0.0
            if den > EPS:
                t = (s1 * d2 - s2 * r) / den
            t = qm.clamp(t, 0.0, 1.0)

            u_unclamped = 0.0
            if d2 > EPS:
                u_unclamped = (t * r - s2) / d2
            u = qm.clamp(u_unclamped, 0.0, 1.0)

            # re-compute t if u was clamped
            if qd.abs(u - u_unclamped) > EPS:
                if d1 > EPS:
                    t = (u * r + s1) / d1
                t = qm.clamp(t, 0.0, 1.0)

            # check for penetration
            closest_p_a = p_a1 + t * e1
            closest_p_b = p_b1 + u * e2
            dist_vec = closest_p_a - closest_p_b
            dist = qm.length(dist_vec)

            penetration = radius_a + radius_b - dist
            if penetration > 0.:
                normal = dist_vec.normalized() if dist > EPS else qd.Vector([0.0, 0.0, 1.0])

                w = qd.Vector([1.0 - t, t, 1.0 - u, u])
                im = qd.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = qm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    lambda_ = penetration / w_sum_sq_inv_mass

                    self.vertices[f + 1, idx_a1, i_b].vert += lambda_ * im[0] * w[0] * normal
                    self.vertices[f + 1, idx_a2, i_b].vert += lambda_ * im[1] * w[1] * normal
                    self.vertices[f + 1, idx_b1, i_b].vert -= lambda_ * im[2] * w[2] * normal
                    self.vertices[f + 1, idx_b2, i_b].vert -= lambda_ * im[3] * w[3] * normal

                if iter_idx == 0:
                    self.rr_constraints[f, i_p, i_b].normal = normal
                    self.rr_constraints[f, i_p, i_b].penetration = penetration

                if iter_idx == self._n_pbd_iters - 1 and penetration > 1e-6:
                    # after the last iteration, collision remains, then record the collision info
                    self.vertices_collision[idx_a1, i_b].collided = True
                    self.vertices_collision[idx_a1, i_b].penetration = penetration
                    self.vertices_collision[idx_a1, i_b].normal = normal
                    self.vertices_collision[idx_a2, i_b].collided = True
                    self.vertices_collision[idx_a2, i_b].penetration = penetration
                    self.vertices_collision[idx_a2, i_b].normal = normal
                    self.vertices_collision[idx_b1, i_b].collided = True
                    self.vertices_collision[idx_b1, i_b].penetration = penetration
                    self.vertices_collision[idx_b1, i_b].normal = -normal
                    self.vertices_collision[idx_b2, i_b].collided = True
                    self.vertices_collision[idx_b2, i_b].penetration = penetration
                    self.vertices_collision[idx_b2, i_b].normal = -normal

    @qd.kernel
    def _kernel_apply_rod_collision_constraints_grad(self, f: qd.i32, iter_idx: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_valid_edge_pairs, self._B):
            idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
            idx_a2 = self.get_next_vertex_of_edge(idx_a1)
            idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
            idx_b2 = self.get_next_vertex_of_edge(idx_b1)

            p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
            p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

            radius_a = (self.vertices_param[idx_a1, i_b].radius + self.vertices_param[idx_a2, i_b].radius) * 0.5
            radius_b = (self.vertices_param[idx_b1, i_b].radius + self.vertices_param[idx_b2, i_b].radius) * 0.5

            # compute closest points (t, u) and distance
            e1, e2 = p_a2 - p_a1, p_b2 - p_b1
            e12 = p_b1 - p_a1
            d1, d2 = e1.dot(e1), e2.dot(e2)
            r = e1.dot(e2)
            s1, s2 = e1.dot(e12), e2.dot(e12)
            den = d1 * d2 - r * r

            t = 0.0
            if den > EPS:
                t = (s1 * d2 - s2 * r) / den
            t = qm.clamp(t, 0.0, 1.0)

            u_unclamped = 0.0
            if d2 > EPS:
                u_unclamped = (t * r - s2) / d2
            u = qm.clamp(u_unclamped, 0.0, 1.0)

            # re-compute t if u was clamped
            if qd.abs(u - u_unclamped) > EPS:
                if d1 > EPS:
                    t = (u * r + s1) / d1
                t = qm.clamp(t, 0.0, 1.0)

            # check for penetration
            closest_p_a = p_a1 + t * e1
            closest_p_b = p_b1 + u * e2
            dist_vec = closest_p_a - closest_p_b
            dist = qm.length(dist_vec)

            penetration = radius_a + radius_b - dist
            if penetration > 0.:
                # Read the incoming position adjoint from the immutable snapshot so pairs sharing
                # a vertex all see a consistent value (race-free, deterministic).
                g_p_a1 = self.collision_vert_grad_snapshot[idx_a1, i_b]
                g_p_a2 = self.collision_vert_grad_snapshot[idx_a2, i_b]
                g_p_b1 = self.collision_vert_grad_snapshot[idx_b1, i_b]
                g_p_b2 = self.collision_vert_grad_snapshot[idx_b2, i_b]

                normal = dist_vec.normalized() if dist > EPS else qd.Vector([0.0, 0.0, 1.0])
                w = qd.Vector([1.0 - t, t, 1.0 - u, u])
                im = qd.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = qm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    g_displacement_vec = (
                        im[0] * w[0] * g_p_a1 + im[1] * w[1] * g_p_a2 -
                        im[2] * w[2] * g_p_b1 - im[3] * w[3] * g_p_b2
                    )

                    g_lambda = normal.dot(g_displacement_vec)
                    g_penetration = g_lambda / w_sum_sq_inv_mass

                    if iter_idx == 0:
                        g_penetration += self.rr_constraints.grad[f, i_p, i_b].penetration

                    g_dist = -g_penetration
                    g_dist_vec = g_dist * normal

                    g_closest_p_a = g_dist_vec
                    g_closest_p_b = -g_dist_vec

                    # ratio-preserve distribution
                    g_v_a1 = (1.0 - t) * g_closest_p_a
                    g_v_a2 = t * g_closest_p_a
                    g_v_b1 = (1.0 - u) * g_closest_p_b
                    g_v_b2 = u * g_closest_p_b

                    total_mag_sq = (
                        g_v_a1.dot(g_v_a1) + g_v_a2.dot(g_v_a2) +
                        g_v_b1.dot(g_v_b1) + g_v_b2.dot(g_v_b2)
                    )
                    scale = 1.0
                    if total_mag_sq > self._max_collision_grad_norm ** 2:
                        total_mag = qm.sqrt(total_mag_sq)
                        if total_mag > EPS:
                            scale = self._max_collision_grad_norm / total_mag

                    # Accumulate race-free (reads came from the snapshot, writes via atomic_add).
                    for _k in qd.static(range(3)):
                        qd.atomic_add(self.vertices.grad[f + 1, idx_a1, i_b].vert[_k], g_v_a1[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_a2, i_b].vert[_k], g_v_a2[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_b1, i_b].vert[_k], g_v_b1[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_b2, i_b].vert[_k], g_v_b2[_k] * scale)

    @qd.kernel
    def _kernel_apply_rod_friction(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_valid_edge_pairs, self._B):
            penetration = self.rr_constraints[f, i_p, i_b].penetration
            if penetration > 0.0:
                idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
                idx_a2 = self.get_next_vertex_of_edge(idx_a1)
                idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
                idx_b2 = self.get_next_vertex_of_edge(idx_b1)

                p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
                p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

                # compute closest points (t, u) and distance
                e1, e2 = p_a2 - p_a1, p_b2 - p_b1
                e12 = p_b1 - p_a1
                d1, d2 = e1.dot(e1), e2.dot(e2)
                r = e1.dot(e2)
                s1, s2 = e1.dot(e12), e2.dot(e12)
                den = d1 * d2 - r * r

                t = 0.0
                if den > EPS:
                    t = (s1 * d2 - s2 * r) / den
                t = qm.clamp(t, 0.0, 1.0)

                u_unclamped = 0.0
                if d2 > EPS:
                    u_unclamped = (t * r - s2) / d2
                u = qm.clamp(u_unclamped, 0.0, 1.0)

                # Re-compute t if u was clamped
                if qd.abs(u - u_unclamped) > EPS:
                    if d1 > EPS:
                        t = (u * r + s1) / d1
                    t = qm.clamp(t, 0.0, 1.0)

                v_a1, v_a2 = self.vertices[f + 1, idx_a1, i_b].vel, self.vertices[f + 1, idx_a2, i_b].vel
                v_b1, v_b2 = self.vertices[f + 1, idx_b1, i_b].vel, self.vertices[f + 1, idx_b2, i_b].vel

                v_a = (1 - t) * v_a1 + t * v_a2
                v_b = (1 - u) * v_b1 + u * v_b2
                v_rel = v_a - v_b

                normal = self.rr_constraints[f, i_p, i_b].normal
                v_normal_mag = v_rel.dot(normal)
                v_tangent = v_rel - v_normal_mag * normal
                v_tangent_norm = qm.length(v_tangent)

                w = qd.Vector([1.0 - t, t, 1.0 - u, u])
                im = qd.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = qm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    normal_vel_mag = penetration / self._substep_dt

                    mu_s = (self.vertices_param[idx_a1, i_b].mu_s + self.vertices_param[idx_a2, i_b].mu_s + self.vertices_param[idx_b1, i_b].mu_s + self.vertices_param[idx_b2, i_b].mu_s) * 0.25
                    mu_k = (self.vertices_param[idx_a1, i_b].mu_k + self.vertices_param[idx_a2, i_b].mu_k + self.vertices_param[idx_b1, i_b].mu_k + self.vertices_param[idx_b2, i_b].mu_k) * 0.25

                    delta_v_tangent = qd.Vector.zero(gs.qd_float, 3)
                    if v_tangent_norm < mu_s * normal_vel_mag:
                        delta_v_tangent = -v_tangent
                    else:
                        delta_v_tangent = -v_tangent.normalized() * mu_k * normal_vel_mag

                    lambda_ = delta_v_tangent / w_sum_sq_inv_mass
                    self.vertices[f + 1, idx_a1, i_b].vel += lambda_ * im[0] * w[0]
                    self.vertices[f + 1, idx_a2, i_b].vel += lambda_ * im[1] * w[1]
                    self.vertices[f + 1, idx_b1, i_b].vel -= lambda_ * im[2] * w[2]
                    self.vertices[f + 1, idx_b2, i_b].vel -= lambda_ * im[3] * w[3]

    @qd.kernel
    def _kernel_apply_rod_friction_grad(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_valid_edge_pairs, self._B):
            if self.rr_constraints[f, i_p, i_b].penetration > 0.0:
                idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
                idx_a2 = self.get_next_vertex_of_edge(idx_a1)
                idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
                idx_b2 = self.get_next_vertex_of_edge(idx_b1)

                p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
                p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

                # compute closest points (t, u) and distance
                e1, e2 = p_a2 - p_a1, p_b2 - p_b1
                e12 = p_b1 - p_a1
                d1, d2 = e1.dot(e1), e2.dot(e2)
                r = e1.dot(e2)
                s1, s2 = e1.dot(e12), e2.dot(e12)
                den = d1 * d2 - r * r

                t = 0.0
                if den > EPS:
                    t = (s1 * d2 - s2 * r) / den
                t = qm.clamp(t, 0.0, 1.0)

                u_unclamped = 0.0
                if d2 > EPS:
                    u_unclamped = (t * r - s2) / d2
                u = qm.clamp(u_unclamped, 0.0, 1.0)

                # Re-compute t if u was clamped
                if qd.abs(u - u_unclamped) > EPS:
                    if d1 > EPS:
                        t = (u * r + s1) / d1
                    t = qm.clamp(t, 0.0, 1.0)

                # Read the incoming velocity adjoint from the immutable snapshot so that pairs
                # sharing a vertex all see a consistent value (race-free, deterministic).
                g_v_a1_out = self.friction_vel_grad_snapshot[idx_a1, i_b]
                g_v_a2_out = self.friction_vel_grad_snapshot[idx_a2, i_b]
                g_v_b1_out = self.friction_vel_grad_snapshot[idx_b1, i_b]
                g_v_b2_out = self.friction_vel_grad_snapshot[idx_b2, i_b]

                w = qd.Vector([1.0 - t, t, 1.0 - u, u])
                im = qd.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b), self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b), self._func_get_inverse_mass(f, idx_b2, i_b),
                ])
                w_sum_sq_inv_mass = qm.dot(w * w, im)

                if w_sum_sq_inv_mass > EPS:
                    g_lambda_vec = (
                        g_v_a1_out * im[0] * w[0] + g_v_a2_out * im[1] * w[1] -
                        g_v_b1_out * im[2] * w[2] - g_v_b2_out * im[3] * w[3]
                    )

                    g_delta_v_tangent = g_lambda_vec / w_sum_sq_inv_mass

                    v_a1, v_a2 = self.vertices[f + 1, idx_a1, i_b].vel, self.vertices[f + 1, idx_a2, i_b].vel
                    v_b1, v_b2 = self.vertices[f + 1, idx_b1, i_b].vel, self.vertices[f + 1, idx_b2, i_b].vel
                    v_rel = ((1 - t) * v_a1 + t * v_a2) - ((1 - u) * v_b1 + u * v_b2)
                    normal = self.rr_constraints[f, i_p, i_b].normal
                    v_tangent = v_rel - v_rel.dot(normal) * normal
                    v_tangent_norm = qm.length(v_tangent)

                    penetration = self.rr_constraints[f, i_p, i_b].penetration
                    normal_vel_mag = penetration / self._substep_dt
                    mu_s = (self.vertices_param[idx_a1, i_b].mu_s + self.vertices_param[idx_a2, i_b].mu_s + self.vertices_param[idx_b1, i_b].mu_s + self.vertices_param[idx_b2, i_b].mu_s) * 0.25
                    mu_k = (self.vertices_param[idx_a1, i_b].mu_k + self.vertices_param[idx_a2, i_b].mu_k + self.vertices_param[idx_b1, i_b].mu_k + self.vertices_param[idx_b2, i_b].mu_k) * 0.25

                    g_v_tangent = qd.Vector.zero(gs.qd_float, 3)
                    g_normal_vel_mag = 0.0

                    if v_tangent_norm < mu_s * normal_vel_mag:
                        g_v_tangent -= g_delta_v_tangent
                    else: # differentiate through the kinetic friction case
                        n_t = v_tangent.normalized()
                        F_k = mu_k * normal_vel_mag
                        g_F_k = -n_t.dot(g_delta_v_tangent)
                        g_n_t = -F_k * g_delta_v_tangent
                        inv_norm = 1.0 / qm.max(v_tangent_norm, EPS)
                        g_v_tangent += (g_n_t - n_t.dot(g_n_t) * n_t) * inv_norm
                        g_normal_vel_mag += g_F_k * mu_k
                    
                    self.rr_constraints.grad[f, i_p, i_b].penetration += g_normal_vel_mag / self._substep_dt
                    
                    g_v_rel = g_v_tangent - normal.dot(g_v_tangent) * normal
                    
                    g_v_a = g_v_rel
                    g_v_b = -g_v_rel

                    # ratio-preserve distribution
                    g_v_a1 = (1.0 - t) * g_v_a
                    g_v_a2 = t * g_v_a
                    g_v_b1 = (1.0 - u) * g_v_b
                    g_v_b2 = u * g_v_b

                    total_mag_sq = (
                        g_v_a1.dot(g_v_a1) + g_v_a2.dot(g_v_a2) +
                        g_v_b1.dot(g_v_b1) + g_v_b2.dot(g_v_b2)
                    )
                    scale = 1.0
                    if total_mag_sq > self._max_collision_grad_norm ** 2:
                        total_mag = qm.sqrt(total_mag_sq)
                        if total_mag > EPS:
                            scale = self._max_collision_grad_norm / total_mag

                    # Accumulate the friction Jacobian contribution race-free. Reads came from the
                    # snapshot, so atomic_add here is order-only non-deterministic on bounded
                    # (clipped) terms -> reproducible.
                    for _k in qd.static(range(3)):
                        qd.atomic_add(self.vertices.grad[f + 1, idx_a1, i_b].vel[_k], g_v_a1[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_a2, i_b].vel[_k], g_v_a2[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_b1, i_b].vel[_k], g_v_b1[_k] * scale)
                        qd.atomic_add(self.vertices.grad[f + 1, idx_b2, i_b].vel[_k], g_v_b2[_k] * scale)

    @qd.ad.grad_replaced
    def collision_forward(self, f, iter_idx):
        self._kernel_apply_rod_collision_constraints(f, iter_idx)

    @qd.ad.grad_for(collision_forward)
    def collision_backward(self, f, iter_idx):
        # snapshot the incoming position adjoint (frame f+1) so the grad kernel reads a consistent,
        # immutable value while accumulating contributions race-free (deterministic).
        self._kernel_snapshot_vert_grad(f)
        self._kernel_apply_rod_collision_constraints_grad(f, iter_idx)

    @qd.kernel
    def _kernel_snapshot_vert_grad(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.collision_vert_grad_snapshot[i_v, i_b] = self.vertices.grad[f + 1, i_v, i_b].vert

    @qd.ad.grad_replaced
    def friction_forward(self, f):
        self._kernel_apply_rod_friction(f)

    @qd.ad.grad_for(friction_forward)
    def friction_backward(self, f):
        # snapshot the incoming velocity adjoint (frame f+1) so the grad kernel reads a consistent,
        # immutable value while accumulating contributions race-free (deterministic).
        self._kernel_snapshot_vel_grad(f)
        self._kernel_apply_rod_friction_grad(f)

    @qd.kernel
    def _kernel_snapshot_vel_grad(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.friction_vel_grad_snapshot[i_v, i_b] = self.vertices.grad[f + 1, i_v, i_b].vel
