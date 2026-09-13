"""Ingress identity and elapsed-time checks; no hardware or controller state."""
from dataclasses import dataclass
import math
from numbers import Real


def _positive(value,name,zero=False):
    if isinstance(value,bool):raise ValueError(name+' must be numeric, not boolean')
    value=float(value)
    if not math.isfinite(value) or value<0 or (not zero and value==0):
        raise ValueError(name+' must be finite and positive')
    return value


@dataclass(frozen=True)
class SourceGapContext:
    """Admission snapshot of the *existing* command lease, never a renewal.

    The caller captures this before accepting the resumed force sample. Both
    ingress and the force observer check their own previous-source watermark.
    ``max_gap_s`` is the configured command-lease budget, not an age relaxation.
    """
    source_id: str
    previous_source_t_s: float
    source_t_s: float
    lease_id: int
    lease_committed_s: float
    lease_expires_s: float
    admitted_at_s: float
    max_gap_s: float
    policy: str = 'lease_fresh_foh_v1'


@dataclass(frozen=True)
class SourceStep:
    source_id: str
    source_t_s: float
    source_wall_time_ns: int
    fresh: bool
    source_dt_s: float
    age_s: float
    gap_recovered: bool = False
    gap_context: SourceGapContext | None = None

    @property
    def sample_id(self):return self.source_id,self.source_t_s


class SourceClock:
    """A repeated UDP snapshot is a held observation, never a new sample.

    The nominal ingress period fixes filter poles. The optional variable-step
    timebase advances those states using actual fresh-sample intervals, bounded
    independently from source age; fixed-period behavior remains the default.
    """
    def __init__(self,period_s,*,jitter_fraction=0.,max_age_s,timebase="fixed_period_v1",max_interval_s=None,
                 gap_policy='strict_v1',max_recovery_interval_s=None):
        self.period_s=_positive(period_s,'source period')
        self.jitter_fraction=_positive(jitter_fraction,'source jitter',True)
        self.max_age_s=_positive(max_age_s,'source maximum age')
        if self.jitter_fraction>=1 or self.max_age_s<self.period_s:
            raise ValueError('jitter must be <1 and maximum age must cover one source period')
        if timebase not in ('fixed_period_v1','variable_step_bilinear_v1'):
            raise ValueError('unsupported source timebase')
        self.timebase=timebase
        self.max_interval_s=(None if max_interval_s is None else _positive(max_interval_s,'maximum source interval'))
        if timebase=='variable_step_bilinear_v1':
            if self.max_interval_s is None or not self.period_s<=self.max_interval_s<=self.max_age_s:
                raise ValueError('variable timebase requires period <= max_interval_s <= max_age_s')
        elif self.max_interval_s is not None:
            raise ValueError('max_interval_s requires the variable-step timebase')
        if gap_policy not in ('strict_v1','lease_fresh_foh_v1'):
            raise ValueError('unsupported source gap policy')
        self.gap_policy=gap_policy
        self.max_recovery_interval_s=(None if max_recovery_interval_s is None else
            _positive(max_recovery_interval_s,'maximum recovery interval'))
        if gap_policy=='lease_fresh_foh_v1':
            if (timebase!='variable_step_bilinear_v1' or self.max_recovery_interval_s is None or
                    self.max_recovery_interval_s<self.max_interval_s):
                raise ValueError('lease gap recovery requires variable timebase and command-lease interval budget')
        elif max_recovery_interval_s is not None:
            raise ValueError('recovery interval requires lease gap policy')
        self.last=None;self.last_now=None;self.source_updates=0

    def begin_epoch(self):
        """Drop the previous ingress watermark. The next sample starts a new run."""
        self.last=None
        self.last_now=None

    def _admit_gap(self,context,source_id,t,now,delta):
        if self.gap_policy!='lease_fresh_foh_v1' or not isinstance(context,SourceGapContext):
            raise ValueError('source interval exceeds declared maximum; lease gap context required')
        if context.policy!=self.gap_policy:
            raise ValueError('source gap policy mismatch')
        if context.source_id!=source_id or context.previous_source_t_s!=self.last.source_t_s or context.source_t_s!=t:
            raise ValueError('source gap context watermark or epoch mismatch')
        if type(context.lease_id) is not int or context.lease_id<=0:
            raise ValueError('source gap requires a committed command lease identity')
        for name in ('previous_source_t_s','source_t_s','lease_committed_s','lease_expires_s','admitted_at_s','max_gap_s'):
            value=getattr(context,name)
            if not isinstance(value,Real):raise ValueError('source gap '+name+' must be numeric')
            _positive(value,'source gap '+name,True)
        if context.admitted_at_s!=now:
            raise ValueError('source gap admission timestamp mismatch')
        if not context.lease_committed_s<t<=now<context.lease_expires_s:
            raise ValueError('source gap original command lease inactive or expired')
        if (context.max_gap_s!=self.max_recovery_interval_s or delta>context.max_gap_s+1e-12 or
                context.lease_expires_s-context.lease_committed_s>context.max_gap_s+1e-12):
            raise ValueError('source gap exceeds command-lease interval budget')

    def observe(self,source_id,source_t_s,wall_time_ns,*,now_s,gap_context=None):
        t=_positive(source_t_s,'source timestamp',True)
        now=_positive(now_s,'current timestamp',True)
        if not isinstance(source_id,str) or not source_id:
            raise ValueError('source identity required')
        if type(wall_time_ns) is not int or wall_time_ns<0:
            raise ValueError('source wall timestamp must be a nonnegative integer')
        if self.last_now is not None and now<self.last_now:raise ValueError('control clock reversed')
        age=now-t
        if not 0<=age<=self.max_age_s:raise ValueError('source unavailable or beyond declared hold age')
        fresh=self.last is None or t!=self.last.source_t_s
        delta=self.period_s;gap_recovered=False
        if self.last is not None:
            if source_id!=self.last.source_id:raise ValueError('source epoch changed')
            if t<self.last.source_t_s:raise ValueError('source timestamp reversed')
            if fresh:
                delta=t-self.last.source_t_s
                if self.timebase=='variable_step_bilinear_v1':
                    if delta>self.max_interval_s+1e-12:
                        self._admit_gap(gap_context,source_id,t,now,delta)
                        gap_recovered=True
                elif abs(delta-self.period_s)>self.period_s*self.jitter_fraction+1e-12:
                    raise ValueError('source cadence outside declared filter timebase')
            else:
                delta=self.last.source_dt_s
                if wall_time_ns!=self.last.source_wall_time_ns:
                    raise ValueError('repeated source timestamp has conflicting wall provenance')
        if gap_context is not None and not gap_recovered:
            raise ValueError('source gap context requires a fresh recovery interval')
        result=SourceStep(source_id,t,wall_time_ns,fresh,delta,age,gap_recovered,gap_context)
        self.last_now=now
        if fresh:self.last=result;self.source_updates+=1
        return result
