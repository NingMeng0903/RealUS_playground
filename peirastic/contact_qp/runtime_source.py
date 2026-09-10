"""Ingress identity and elapsed-time checks; no hardware or controller state."""
from dataclasses import dataclass
import math


def _positive(value,name,zero=False):
    if isinstance(value,bool):raise ValueError(name+' must be numeric, not boolean')
    value=float(value)
    if not math.isfinite(value) or value<0 or (not zero and value==0):
        raise ValueError(name+' must be finite and positive')
    return value


@dataclass(frozen=True)
class SourceStep:
    source_id: str
    source_t_s: float
    source_wall_time_ns: int
    fresh: bool
    source_dt_s: float
    age_s: float

    @property
    def sample_id(self):return self.source_id,self.source_t_s


class SourceClock:
    """A repeated UDP snapshot is a held observation, never a new sample.

    The nominal ingress period fixes filter poles. The optional variable-step
    timebase advances those states using actual fresh-sample intervals, bounded
    independently from source age; fixed-period behavior remains the default.
    """
    def __init__(self,period_s,*,jitter_fraction=0.,max_age_s,timebase="fixed_period_v1",max_interval_s=None):
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
        self.last=None;self.last_now=None;self.source_updates=0

    def observe(self,source_id,source_t_s,wall_time_ns,*,now_s):
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
        delta=self.period_s
        if self.last is not None:
            if source_id!=self.last.source_id:raise ValueError('source epoch changed')
            if t<self.last.source_t_s:raise ValueError('source timestamp reversed')
            if fresh:
                delta=t-self.last.source_t_s
                if self.timebase=='variable_step_bilinear_v1':
                    if delta>self.max_interval_s+1e-12:raise ValueError('source interval exceeds declared maximum')
                elif abs(delta-self.period_s)>self.period_s*self.jitter_fraction+1e-12:
                    raise ValueError('source cadence outside declared filter timebase')
            else:
                delta=self.last.source_dt_s
                if wall_time_ns!=self.last.source_wall_time_ns:
                    raise ValueError('repeated source timestamp has conflicting wall provenance')
        result=SourceStep(source_id,t,wall_time_ns,fresh,delta,age)
        self.last_now=now
        if fresh:self.last=result;self.source_updates+=1
        return result
