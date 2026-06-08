import genesis as gs
from genesis.repr_base import RBC


class ToolEntityState:
    """
    Dynamic state queried from a genesis ToolEntity.
    """

    def __init__(self, entity, s_global):
        self.entity = entity
        self.s_global = s_global

        args = {
            "dtype": gs.tc_float,
            "requires_grad": self.entity.scene.requires_grad,
            "scene": self.entity.scene,
        }
        self.pos = gs.zeros((self.entity.sim._B, 3), **args)
        self.quat = gs.zeros((self.entity.sim._B, 4), **args)
        self.vel = gs.zeros((self.entity.sim._B, 3), **args)
        self.ang = gs.zeros((self.entity.sim._B, 3), **args)

    def serializable(self):
        self.entity = None

        self.pos = self.pos.detach()
        self.quat = self.quat.detach()
        self.vel = self.vel.detach()
        self.ang = self.ang.detach()

    # def __repr__(self):
    #     return f'{self.__repr_name__()}\n' \
    #            f'entity : {_repr(self.entity)}\n' \
    #            f'pos    : {_repr(self.pos)}\n' \
    #            f'quat   : {_repr(self.quat)}\n' \
    #            f'vel    : {_repr(self.vel)}\n' \
    #            f'ang    : {_repr(self.ang)}'


class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": self._entity.scene.requires_grad,
            "scene": self._entity.scene,
        }
        self._pos = gs.zeros(base_shape + (3,), **args)
        self._vel = gs.zeros(base_shape + (3,), **args)
        self._C = gs.zeros(base_shape + (3, 3), **args)
        self._F = gs.zeros(base_shape + (3, 3), **args)
        self._Jp = gs.zeros(base_shape, **args)

        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros(base_shape, **args)

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._C = self._C.detach()
        self._F = self._F.detach()
        self._Jp = self._Jp.detach()
        self._active = self._active.detach()

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHEntityState(RBC):
    """
    Dynamic state queried from a genesis SPHEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": self._entity.scene,
        }

        self._pos = gs.zeros(base_shape + (3,), **args)
        self._vel = gs.zeros(base_shape + (3,), **args)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel


class FEMEntityState:
    """
    Dynamic state queried from a genesis FEMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_vertices, 3)

        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": self.entity.scene,
        }
        self._pos = gs.zeros(base_shape, **args)
        self._vel = gs.zeros(base_shape, **args)

        args["dtype"] = int
        args["requires_grad"] = False
        self._active = gs.zeros((self.entity.sim._B, self.entity.n_elements), **args)

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class RigidEntityState(RBC):
    """
    Dynamic state queried from a genesis RigidEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global

        num_batch = self._entity._solver._B
        requires_grad = self._entity.scene.requires_grad
        scene = self._entity.scene
        self._pos = gs.zeros((num_batch, 3), dtype=float, requires_grad=requires_grad, scene=scene)
        self._quat = gs.zeros((num_batch, 4), dtype=float, requires_grad=requires_grad, scene=scene)

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._quat = self._quat.detach()

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat


class RODEntityState:
    """
    Dynamic state queried from a genesis RODEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_v_shape = (self.entity.sim._B, self.entity.n_vertices, 3)
        base_e_shape = (self.entity.sim._B, self.entity.n_edges)
        base_iv_shape = (self.entity.sim._B, self.entity.n_internal_vertices)

        args = {
            "dtype": gs.tc_float,
            "requires_grad": self.entity.scene.requires_grad,
            "scene": self.entity.scene,
        }

        # vertex states
        self._pos = gs.zeros(base_v_shape, **args)
        self._vel = gs.zeros(base_v_shape, **args)

        # edge states
        self._edge = gs.zeros(base_e_shape + (3,), **args)
        self._length = gs.zeros(base_e_shape, **args)
        self._d1 = gs.zeros(base_e_shape + (3,), **args)
        self._d2 = gs.zeros(base_e_shape + (3,), **args)
        self._d3 = gs.zeros(base_e_shape + (3,), **args)
        self._d1_ref = gs.zeros(base_e_shape + (3,), **args)
        self._d2_ref = gs.zeros(base_e_shape + (3,), **args)
        self._theta = gs.zeros(base_e_shape, **args)
        self._omega = gs.zeros(base_e_shape, **args)

        # internal vertex states
        self._kb = gs.zeros(base_iv_shape + (3,), **args)
        self._twist = gs.zeros(base_iv_shape, **args)
        self._kappa_rest = gs.zeros(base_iv_shape + (2,), **args)

        args["dtype"] = int
        args["requires_grad"] = False
        self._fixed = gs.zeros((self.entity.sim._B, self.entity.n_vertices), **args)

        # collision state (used for policy learning)
        self._collided = gs.zeros(
            (self.entity.sim._B, self.entity.n_vertices),
            dtype=gs.tc_bool, requires_grad=False, scene=self.entity.scene
        )
        self._collision_normal = gs.zeros(
            (self.entity.sim._B, self.entity.n_vertices, 3),
            dtype=gs.tc_float, requires_grad=False, scene=self.entity.scene
        )
        self._collision_penetration = gs.zeros(
            (self.entity.sim._B, self.entity.n_vertices),
            dtype=gs.tc_float, requires_grad=False, scene=self.entity.scene
        )

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._fixed = self._fixed.detach()
        self._edge = self._edge.detach()
        self._length = self._length.detach()
        self._d1 = self._d1.detach()
        self._d2 = self._d2.detach()
        self._d3 = self._d3.detach()
        self._d1_ref = self._d1_ref.detach()
        self._d2_ref = self._d2_ref.detach()
        self._theta = self._theta.detach()
        self._omega = self._omega.detach()
        self._kb = self._kb.detach()
        self._twist = self._twist.detach()
        self._kappa_rest = self._kappa_rest.detach()

        self._collided = self._collided.detach()
        self._collision_normal = self._collision_normal.detach()
        self._collision_penetration = self._collision_penetration.detach()

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def fixed(self):
        return self._fixed

    @property
    def edge(self):
        return self._edge

    @property
    def length(self):
        return self._length

    @property
    def d1(self):
        return self._d1

    @property
    def d2(self):
        return self._d2

    @property
    def d3(self):
        return self._d3

    @property
    def d1_ref(self):
        return self._d1_ref

    @property
    def d2_ref(self):
        return self._d2_ref

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    @property
    def kb(self):
        return self._kb

    @property
    def twist(self):
        return self._twist

    @property
    def kappa_rest(self):
        return self._kappa_rest

    @property
    def collided(self):
        return self._collided

    @property
    def collision_normal(self):
        return self._collision_normal

    @property
    def collision_penetration(self):
        return self._collision_penetration
