import quadrants as qd
from ..base import Material


@qd.data_oriented
class Base(Material):
    """
    The base class of ROD materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    K: float, optional
        Stretching stiffness. Default is 0. Non-positive values indicate disable stretching.
    E: float, optional
        Bending stiffness. Default is 1e6. Non-positive values indicate disable bending.
    G: float, optional
        Twisting stiffness. Default is 1e6. Non-positive values indicate disable twisting.
    plastic_yield: float, optional
        Plastic yield threshold. Default is 10.0 (should be no plasticity).
    plastic_creep: float, optional
        Plastic creep rate. Default is 1.0.
    restitution: float, optional
        Coefficient of restitution for self-collision. Default is 0.0 (perfectly inelastic).
    static_friction: float, optional
        Static friction coefficient. Default is 0.3.
    kinetic_friction: float, optional
        Kinetic friction coefficient. Default is 0.25.
    use_inextensible: bool, optional
        If True, use inextensible rods. Default is True.
    segment_mass: float, optional
        Mass of each rod segment. Default is 0.02.
    segment_radius: float, optional
        Radius of each rod segment. Default is 0.01.
    """

    def __init__(
        self,
        K=0.0,
        E=1.0e6,
        G=1.0e6,
        plastic_yield=10.0,
        plastic_creep=1.0,
        static_friction=0.3,
        kinetic_friction=0.25,
        restitution=0.0,
        use_inextensible=True,
        segment_mass=0.02,
        segment_radius=0.01,
    ):
        super().__init__()

        self._K = K
        self._E = E
        self._G = G
        self._plastic_yield = plastic_yield
        self._plastic_creep = plastic_creep
        self._static_friction = static_friction
        self._kinetic_friction = kinetic_friction
        self._restitution = restitution
        self._use_inextensible = use_inextensible
        self._segment_mass = segment_mass
        self._segment_radius = segment_radius

        # will be set when added to solver
        self._idx = None

    def build(self, rod_solver):
        pass

    @property
    def idx(self):
        return self._idx

    @property
    def K(self):
        return self._K

    @property
    def E(self):
        return self._E

    @property
    def G(self):
        return self._G

    @property
    def plastic_yield(self):
        return self._plastic_yield

    @property
    def plastic_creep(self):
        return self._plastic_creep

    @property
    def static_friction(self):
        return self._static_friction

    @property
    def kinetic_friction(self):
        return self._kinetic_friction

    @property
    def restitution(self):
        return self._restitution

    @property
    def use_inextensible(self):
        return self._use_inextensible

    @property
    def segment_mass(self):
        return self._segment_mass

    @property
    def segment_radius(self):
        return self._segment_radius
