from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils import array_class
from genesis.utils.array_class import LinksState
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@qd.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "LegacyCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options

        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver
        self.rod_solver = self.sim.rod_solver

    def build(self) -> None:
        self._rigid_mpm = self.rigid_solver.is_active and self.mpm_solver.is_active and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active and self.sph_solver.is_active and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active and self.pbd_solver.is_active and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active and self.fem_solver.is_active and self.options.rigid_fem
        self._rigid_rod = self.rigid_solver.is_active and self.rod_solver.is_active and self.options.rigid_rod
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph
        self._rod_mpm = self.rod_solver.is_active and self.mpm_solver.is_active and self.options.rod_mpm

        if (self._rigid_mpm or self._rigid_sph or self._rigid_pbd or self._rigid_fem or self._rigid_rod) and any(
            geom.needs_coup for geom in self.rigid_solver.geoms
        ):
            self.rigid_solver.collider._sdf.activate()

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = qd.field(gs.qd_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = qd.types.struct(
                link_idx=gs.qd_int,
                local_pos=gs.qd_vec3,
            )

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=qd.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        if self._rigid_rod:
            self.rod_rigid_gripper_geom_indices = qd.field(
                dtype=gs.qd_int, needs_grad=False, shape=(self.rigid_solver.n_geoms, self.rod_solver._B)
            )

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

        if self._rigid_rod:
            gripper_n_geoms = self.rod_solver.geom_indices.shape[0]
            # -2: uninitialized; -1: gripper not contact; >=0: gripper contact with rod vertex index
            if envs_idx is None:
                self.rod_rigid_gripper_geom_indices.fill(-2)
                if gripper_n_geoms > 0:
                    self.init_rod_rigid_gripper_geom_indices(gripper_n_geoms, self.rod_solver.geom_indices)
            else:
                self._kernel_reset_rod(envs_idx)
                if gripper_n_geoms > 0:
                    self.init_rod_rigid_gripper_geom_indices_with_envs_idx(gripper_n_geoms, self.rod_solver.geom_indices, envs_idx)
            gs.logger.info(f"Registered {self.rod_solver.geom_indices.shape[0]} gripper geometries for rod collision handling.")
            gs.logger.info(f"Geom indices: {self.rod_solver.geom_indices}")

    @qd.kernel
    def _kernel_reset_mpm(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.kernel
    def _kernel_reset_sph(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.kernel
    def _kernel_reset_rod(self, envs_idx: qd.types.ndarray()):
        for i_v, i_b_ in qd.ndrange(self.rod_solver._n_vertices, envs_idx.shape[0]):
            self.rod_rigid_gripper_geom_indices[i_v, envs_idx[i_b_]] = -2

    @qd.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_global_info=rigid_global_info,
                    sdf_info=sdf_info,
                    collider_static_config=collider_static_config,
                )
        return vel

    @qd.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        return vel

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        normal_rigid = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @qd.func
    def _func_collide_with_rigid_geom_rod(
        self,
        f,
        i,
        pos_world,
        vel,
        mass,
        radius,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        signed_dist -= radius

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom_rod(
                f, i,
                pos_world,
                vel,
                mass,
                normal_rigid,
                influence,
                geom_idx,
                batch_idx,
                geoms_info,
                links_state,
                rigid_global_info,
            )

            # for RL training
            if signed_dist < -1e-6 and geom_idx != 0:
                self.rod_solver.vertices_collision[i, batch_idx].collided = True
                self.rod_solver.vertices_collision[i, batch_idx].normal = normal_rigid
                self.rod_solver.vertices_collision[i, batch_idx].penetration = -signed_dist
                self.rod_solver.vertices_collision[i, batch_idx].geom_idx = geom_idx

        return vel

    @qd.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=geoms_info.link_idx[geom_idx],
            i_b=i_b,
            links_state=links_state,
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_global_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                i_b,
                links_state,
            )

        return vel

    @qd.func
    def _func_collide_in_rigid_geom_rod(
        self,
        f,
        i,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=geoms_info.link_idx[geom_idx],
            i_b=i_b,
            links_state=links_state,
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            # make the vertex kinematic during rod-rigid contact
            if self.rod_rigid_gripper_geom_indices[geom_idx, i_b] >= -1:
                # mark contact with rod vertex i
                self.rod_rigid_gripper_geom_indices[geom_idx, i_b] = i
                self.rod_solver.vertices_ng[f, i, i_b].kinematic = True
                rvel_tan = rvel_tan * (1 - influence * geoms_info.coup_friction[geom_idx])
            else:
                rvel_tan_norm = rvel_tan.norm(gs.EPS)
                rvel_tan = (
                    rvel_tan
                    / rvel_tan_norm
                    * qd.max(
                        0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx]
                    )
                )

            # normal component after collision
            rvel_normal = (
                -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]
            )

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_global_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                i_b,
                links_state,
            )

        return vel

    @qd.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        for entity in qd.static(self.tool_solver.entities):
            if qd.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @qd.func
    def _func_mpm_surface_normal(self, f, pos, i_b):
        # find the base grid node for the given position
        mpm_base = qd.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.ti_int)

        # calculate the mass gradient using central differences
        mass_grad = qd.Vector.zero(gs.ti_float, 3)
        for d in qd.static(range(3)):
            p_node = mpm_base - self.mpm_solver.grid_offset
            n_node = mpm_base - self.mpm_solver.grid_offset

            p_node[d] += 1
            n_node[d] -= 1

            mass_p = 0.0
            # check if the positive-side node is within grid bounds
            if p_node[d] >= 0 and p_node[d] < self.mpm_solver.grid_res[d]:
                mass_p = self.mpm_solver.grid[f, p_node, i_b].mass

            mass_n = 0.0
            # check if the negative-side node is within grid bounds
            if n_node[d] >= 0 and n_node[d] < self.mpm_solver.grid_res[d]:
                mass_n = self.mpm_solver.grid[f, n_node, i_b].mass

            # gradient component = (mass_positive - mass_negative) / (2 * grid_spacing)
            mass_grad[d] = (mass_p - mass_n) / (2 * self.mpm_solver.dx)

        # calculate the final normal from the gradient
        mass_grad_norm = mass_grad.norm()
        normal = qd.Vector([0.0, 0.0, 1.0]) # Default upward normal
        if mass_grad_norm > gs.EPS:
            # the normal points opposite to the direction of steepest mass increase
            normal = -mass_grad / mass_grad_norm

        return normal, mass_grad_norm

    @qd.func
    def _func_mpm_collide_with_rod(self, f, pos_grid, vel_grid, mass_grid, i_b):
        # vertex collisions
        for i_v in range(self.rod_solver.n_vertices):
            pos_rod = self.rod_solver.vertices[f, i_v, i_b].vert
            vel_rod = self.rod_solver.vertices[f, i_v, i_b].vel
            radius_rod = self.rod_solver.vertices_param[i_v, i_b].radius
            inv_mass_rod = self.rod_solver._func_get_inverse_mass(f, i_v, i_b)
            rest_rod = self.rod_solver.vertices_param[i_v, i_b].restitution
            inv_mass_grid = 1.0 / (mass_grid + gs.EPS)

            collision_dist = self.mpm_solver.dx + radius_rod

            dist_vec = pos_grid - pos_rod
            if dist_vec.norm() < collision_dist:
                normal = dist_vec.normalized()
                v_rel = vel_grid - vel_rod
                v_rel_normal_mag = v_rel.dot(normal)

                if v_rel_normal_mag < 0:
                    impulse_mag = -(1 + rest_rod) * v_rel_normal_mag / (inv_mass_rod + inv_mass_grid + gs.EPS)
                    vel_grid += impulse_mag * normal * inv_mass_grid

                    rod_reaction_dv = -impulse_mag * normal * inv_mass_rod
                    self.rod_solver.vertices[f + 1, i_v, i_b].vel += rod_reaction_dv

        # edge collisions
        for i_e in range(self.rod_solver.n_edges):
            v_idx_1 = self.rod_solver.edges_info[i_e].vert_idx
            v_idx_2 = self.rod_solver.get_next_vertex_of_edge(v_idx_1)

            inv_mass_1 = self.rod_solver._func_get_inverse_mass(f, v_idx_1, i_b)
            inv_mass_2 = self.rod_solver._func_get_inverse_mass(f, v_idx_2, i_b)

            inv_mass_rod_avg = (inv_mass_1 + inv_mass_2) * 0.5

            rest_rod = (
                self.rod_solver.vertices_param[v_idx_1, i_b].restitution +
                self.rod_solver.vertices_param[v_idx_2, i_b].restitution
            ) * 0.5
            inv_mass_grid = 1.0 / (mass_grid + gs.EPS)

            pos_1 = self.rod_solver.vertices[f, v_idx_1, i_b].vert
            pos_2 = self.rod_solver.vertices[f, v_idx_2, i_b].vert

            vel_1 = self.rod_solver.vertices[f, v_idx_1, i_b].vel
            vel_2 = self.rod_solver.vertices[f, v_idx_2, i_b].vel

            radius_rod = (
                self.rod_solver.vertices_param[v_idx_1, i_b].radius +
                self.rod_solver.vertices_param[v_idx_2, i_b].radius
            ) * 0.5

            # increase collision thickness slightly for edges
            collision_dist = self.mpm_solver.dx + radius_rod

            # find closest point t on the rod segment
            seg_dir = pos_2 - pos_1
            seg_len_sq = seg_dir.dot(seg_dir)

            t = (pos_grid - pos_1).dot(seg_dir) / (seg_len_sq + 1e-9)
            t = qd.max(0.0, qd.min(1.0, t))

            closest_pos_on_rod = pos_1 + t * seg_dir
            dist_vec = pos_grid - closest_pos_on_rod

            if dist_vec.norm() < collision_dist:
                # interpolate rod velocity at contact point
                vel_rod_at_contact = (1.0 - t) * vel_1 + t * vel_2

                normal = dist_vec.normalized()
                v_rel = vel_grid - vel_rod_at_contact
                v_rel_normal_mag = v_rel.dot(normal)

                if v_rel_normal_mag < 0:
                    # calculate impulse magnitude
                    impulse_mag = -(1 + rest_rod) * v_rel_normal_mag / (inv_mass_rod_avg + inv_mass_grid + gs.EPS)

                    # apply to MPM Grid
                    vel_grid += impulse_mag * normal * inv_mass_grid

                    # apply reaction to vertices
                    # distribute the impulse to the two vertices based on linear interpolation (t)
                    # Impulse_1 = (1-t) * Impulse, Impulse_2 = t * Impulse
                    rod_reaction_dv_1 = -impulse_mag * normal * (1.0 - t) * inv_mass_1
                    rod_reaction_dv_2 = -impulse_mag * normal * t * inv_mass_2

                    self.rod_solver.vertices[f + 1, v_idx_1, i_b].vel += rod_reaction_dv_1
                    self.rod_solver.vertices[f + 1, v_idx_2, i_b].vel += rod_reaction_dv_2

        return vel_grid

    @qd.kernel
    def mpm_grid_op(
        self,
        f: qd.i32,
        t: qd.f32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for ii, jj, kk, i_b in qd.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in qd.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if qd.static(self.tool_solver.is_active):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                #################### MPM <-> Rigid ####################
                if qd.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                        i_b,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_global_info=rigid_global_info,
                        sdf_info=sdf_info,
                        collider_static_config=collider_static_config,
                    )

                #################### MPM <-> Rod ####################
                if qd.static(self._rod_mpm):
                    vel_mpm = self._func_mpm_collide_with_rod(f, pos, vel_mpm, mass_mpm, i_b)

                #################### MPM <-> SPH ####################
                if qd.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if qd.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @qd.kernel
    def mpm_surface_to_particle(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf.sdf_func_normal_world(
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            rigid_global_info=rigid_global_info,
                            collider_static_config=collider_static_config,
                            sdf_info=sdf_info,
                            pos_world=self.mpm_solver.particles[f, i_p, i_b].pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active:
            self.fem_solver._kernel_update_linked_vertex_constraints(self.rigid_solver.links_state)

    @qd.kernel
    def fem_surface_force(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in qd.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = qd.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if qd.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if qd.static(self._fem_mpm):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                        # follow MPM p2g scheme
                        vel_mpm = qd.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = qd.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.qd_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.qd_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in qd.static(qd.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.qd_float(1.0)
                            for d in qd.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if qd.static(self._fem_sph):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size  # self._dx
                        stencil_size = 2  # self._stencil_size

                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        # ---------- SPH -> FEM ----------
                        sph_vel = qd.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = (
                                vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                            )  # exclude tangential velocity

                            # ---------- FEM -> SPH ----------
                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                            for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # boundary condition
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: qd.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @qd.kernel
    def rod_vertex_force(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_v, i_b in qd.ndrange(self.rod_solver._n_vertices, self.rod_solver._B):
            # ROD <-> Rigid
            if not self.rod_solver.vertex_constraints[i_v, i_b].constrained:
                for i_g in qd.ndrange(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        vel_rod = self._func_collide_with_rigid_geom_rod(
                            f,
                            i_v,
                            self.rod_solver.vertices[f, i_v, i_b].vert,
                            self.rod_solver.vertices[f + 1, i_v, i_b].vel,
                            self.rod_solver.vertices_param[i_v, i_b].mass,
                            self.rod_solver.vertices_param[i_v, i_b].radius,
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )
                        self.rod_solver.vertices[f + 1, i_v, i_b].vel = vel_rod

            # vel_rod_prime = self.rod_solver.boundary.impose_vel(
            #     self.rod_solver.vertices[f, i_v, i_b].vert,
            #     self.rod_solver.vertices[f + 1, i_v, i_b].vel,
            #     self.rod_solver.vertices_param[i_v, i_b].radius,
            # )
            # self.rod_solver.vertices[f + 1, i_v, i_b].vel = vel_rod_prime

    def rod_rigid_link_constraints(self):
        if self.rigid_solver.is_active:
            self.rod_solver._kernel_update_attached_verts(self.rigid_solver.links_state)

    @qd.kernel
    def init_rod_rigid_gripper_geom_indices(self, n_geoms: qd.i32, geom_indices: qd.types.ndarray()):
        for i, i_b in qd.ndrange(n_geoms, self.rod_solver._B):
            i_g = geom_indices[i]
            self.rod_rigid_gripper_geom_indices[i_g, i_b] = -1

    @qd.kernel
    def init_rod_rigid_gripper_geom_indices_with_envs_idx(self, n_geoms: qd.i32, geom_indices: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i, i_b_ in qd.ndrange(n_geoms, envs_idx.shape[0]):
            i_g = geom_indices[i]
            self.rod_rigid_gripper_geom_indices[i_g, envs_idx[i_b_]] = -1

    @qd.kernel
    def clear_rod_rigid_gripper_geom_indices(self):
        for i_g, i_b in qd.ndrange(self.rigid_solver.n_geoms, self.rod_solver._B):
            # clear previous gripper geom indices
            if self.rod_rigid_gripper_geom_indices[i_g, i_b] >= -1:
                self.rod_rigid_gripper_geom_indices[i_g, i_b] = -1

    def get_rod_rigid_gripper_contact_info(self, envs_idx):
        # vertex idx -> geom idx
        out = dict()
        for i in range(self.rigid_solver.n_geoms):
            if self.rod_rigid_gripper_geom_indices[i, envs_idx] >= 0:
                out[self.rod_rigid_gripper_geom_indices[i, envs_idx]] = i
        return out

    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_global_info: array_class.RigidGlobalInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_global_info,
                            sdf_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_global_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_attach_pbd_to_rigid_link(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        link_idx: qd.i32,
        links_state: LinksState,
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = qd_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @qd.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = qd.math.vec3([0.0, 0.0, 0.0])

    @qd.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: qd.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in qd.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @qd.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        collider_static_config: qd.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        contact_normal = sdf.sdf_func_normal_world(
            geoms_state=geoms_state,
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_coupling_force(
                pos_world,
                force,
                geoms_info.link_idx[geom_idx],
                batch_idx,
                links_state,
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._collider_static_config,
            )

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            self.mpm_grid_op(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

            # 1-way: animate particles by links
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.links_state)

        if self.fem_solver.is_active:
            self.fem_surface_force(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
            self.fem_rigid_link_constraints()

        # Rod <-> Rigid
        if self._rigid_rod:
            self.clear_rod_rigid_gripper_geom_indices()
            self.rod_vertex_force(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
            self.rod_rigid_link_constraints()

    def couple_grad(self, f):
        if self.fem_solver.is_active:
            self.fem_surface_force.grad(
                f,
                self.rigid_solver.geoms_state,
                self.rigid_solver.geoms_info,
                self.rigid_solver.links_state,
                self.rigid_solver._rigid_global_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                links_state=self.rigid_solver.links_state,
                rigid_global_info=self.rigid_solver._rigid_global_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
