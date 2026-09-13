"""One read-only, full TCP power budget for the outer QP.

Environment-on-probe wrench and TCP twist share tool components and reference
point. Assurance labels describe declared mathematical assumptions; no object
here verifies hardware bounds or settles a physical energy ledger.
"""
from dataclasses import dataclass, field, replace
import math
import numpy as np
from .types import positive, vector


@dataclass(frozen=True)
class PortEnergyConstraint:
    wrench_environment: np.ndarray
    available_j: float
    hold_s: float
    beta: float = 1.
    wrench_error: np.ndarray = field(default_factory=lambda: np.zeros(6))
    wrench_rate: np.ndarray = field(default_factory=lambda: np.zeros(6))
    contact_map: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    damping: np.ndarray = field(default_factory=lambda: np.empty(0))
    contact_speed_bound: np.ndarray = field(default_factory=lambda: np.empty(0))
    tracking_error: np.ndarray = field(default_factory=lambda: np.zeros(6))
    assurance: str = 'command_model'
    bounds_version: str = 'unverified'
    frame: str = 'tcp_tool'
    task_power_w: float = 0.

    def __post_init__(self):
        if self.frame != 'tcp_tool' or self.assurance not in ('command_model', 'two_port_command_model', 'declared_bound', 'monitor'):
            raise ValueError('explicit TCP/tool energy frame and assurance required')
        if not self.bounds_version or (self.assurance == 'declared_bound' and self.bounds_version == 'unverified'):
            raise ValueError('declared bounds require an explicit version')
        for name in ('available_j', 'hold_s', 'beta', 'task_power_w'):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(name+' must be numeric, not boolean')
        object.__setattr__(self, 'available_j', positive(self.available_j, 'available_j', zero=True))
        object.__setattr__(self, 'hold_s', positive(self.hold_s, 'hold_s'))
        object.__setattr__(self, 'beta', positive(self.beta, 'beta'))
        object.__setattr__(self, 'task_power_w', positive(self.task_power_w, 'task_power_w', zero=True))
        if self.task_power_w and self.assurance != 'two_port_command_model':
            raise ValueError('task power requires explicit two_port_command_model assurance')
        if self.beta > 1.:
            raise ValueError('beta must be at most one')
        for name in ('wrench_environment', 'wrench_error', 'wrench_rate', 'tracking_error'):
            value = vector(getattr(self, name), (6,), name=name)
            if name != 'wrench_environment' and np.any(value < 0):
                raise ValueError(name+' must be nonnegative')
            object.__setattr__(self, name, value)
        pc = np.asarray(self.contact_map)
        if pc.ndim != 2 or pc.shape[1] != 6:
            raise ValueError('contact_map must have shape (m,6)')
        object.__setattr__(self, 'contact_map', vector(pc, pc.shape, name='contact_map'))
        for name in ('damping', 'contact_speed_bound'):
            value=vector(getattr(self, name), (len(pc),), name=name)
            if np.any(value < 0):
                raise ValueError(name+' must be nonnegative')
            object.__setattr__(self, name, value)
        # Reject arithmetic overflow before it can create a false credit.
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                quantities=np.r_[self.uncertainty, self.damping_coefficient,
                    self.tracking_contact_error, self.tracking_cost_w,
                    self.task_power_w+self.beta*self.available_j/self.hold_s]
            if not np.isfinite(quantities).all():
                raise ValueError('nonfinite derived power bounds')
        except FloatingPointError as exc:
            raise ValueError('overflowed energy bounds') from exc

    @property
    def uncertainty(self):
        return self.wrench_error+.5*self.hold_s*self.wrench_rate

    @property
    def damping_coefficient(self):
        return self.damping*self.contact_speed_bound

    @property
    def tracking_contact_error(self):
        return abs(self.contact_map) @ self.tracking_error

    @property
    def tracking_cost_w(self):
        return float((abs(self.wrench_environment)+self.uncertainty) @ self.tracking_error
                     + self.damping_coefficient @ self.tracking_contact_error)

    def velocity_rows(self):
        active=self.damping > 0.
        remaining=self.contact_speed_bound[active]-self.tracking_contact_error[active]
        return self.contact_map[active], -remaining, remaining

    def lower_power_w(self, velocity):
        v=vector(velocity,(6,),name='velocity')
        with np.errstate(over='ignore',invalid='ignore'):
            result=float(self.wrench_environment @ v-self.uncertainty @ abs(v)
                         -self.damping_coefficient @ abs(self.contact_map @ v)-self.tracking_cost_w)
        return result if math.isfinite(result) else -math.inf

    def margin_power_w(self, velocity):
        # Task authorization changes affordability, never the external port work.
        return self.lower_power_w(velocity)+self.task_power_w+self.beta*self.available_j/self.hold_s

    def margin_work_j(self, velocity):
        return self.hold_s*self.margin_power_w(velocity)

    def aged(self, elapsed_s):
        """Bounds at publication, retaining a complete subsequent hold horizon.

        The caller declares W to be bounded at the QP snapshot time. Wrench
        source age must already be included in that snapshot's wrench_error.
        """
        age=float(elapsed_s)
        if not math.isfinite(age) or age < 0:
            raise ValueError('finite nonnegative snapshot age required')
        with np.errstate(over='ignore',invalid='ignore'):
            error=self.wrench_error+age*self.wrench_rate
        return replace(self,wrench_error=error)

    def lower_work_j(self, velocity, elapsed_s):
        """Conservative full-port input work minus declared damping, any prefix."""
        t=float(elapsed_s)
        if not math.isfinite(t) or not 0 <= t <= self.hold_s:
            raise ValueError('prefix must lie inside the declared hold')
        v=vector(velocity,(6,),name='velocity')
        if t == 0.:
            return 0.
        with np.errstate(over='ignore',invalid='ignore'):
            uncertainty=self.wrench_error+.5*t*self.wrench_rate
            work=float(t*(self.wrench_environment @ v-uncertainty @ abs(v)
                -(abs(self.wrench_environment)+uncertainty) @ self.tracking_error
                -self.damping_coefficient @ (abs(self.contact_map @ v)+self.tracking_contact_error)))
        if not math.isfinite(work):
            raise ValueError('nonfinite prefix work cannot be settled')
        return work

    def admissible(self, velocity, *, tolerance_w=1e-8, velocity_tolerance=1e-8):
        try:
            if (not math.isfinite(tolerance_w) or not math.isfinite(velocity_tolerance)
                    or tolerance_w < 0 or velocity_tolerance < 0):
                return False
            v=vector(velocity,(6,),name='velocity')
            a,lo,hi=self.velocity_rows()
            return bool(np.all(a @ v >= lo-velocity_tolerance)
                        and np.all(a @ v <= hi+velocity_tolerance)
                        and self.margin_power_w(v) >= -tolerance_w)
        except (TypeError,ValueError,OverflowError):
            return False
