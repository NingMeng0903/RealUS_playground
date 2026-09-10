"""Fixed-pole Butterworth states with measured-interval trapezoidal updates.

The analog frequency is prewarped ONCE at the nominal period. At that exact
period these states have the original digital Butterworth transfer function.
Irregular sampling remains a declared interpolation model, not new bandwidth.
"""
import math
import numpy as np


def _frequency(cutoff_hz, nominal_s):
    if not math.isfinite(cutoff_hz) or not math.isfinite(nominal_s) or not 0<2*cutoff_hz*nominal_s<1:
        raise ValueError('cutoff must be representable at nominal sample period')
    return 2/nominal_s*math.tan(math.pi*cutoff_hz*nominal_s)


def _step(dt_s):
    if isinstance(dt_s,bool) or not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError('positive finite measured sample interval required')
    return float(dt_s)


class VariableLowpass1:
    def __init__(self, cutoff_hz, nominal_s, output, previous_input):
        self.omega=_frequency(cutoff_hz,nominal_s)
        self.output=np.array(output,dtype=float,copy=True)
        self.previous_input=np.array(previous_input,dtype=float,copy=True)

    def update(self,value,dt_s):
        dt=_step(dt_s);value=np.asarray(value,dtype=float)
        if value.shape!=self.output.shape or not np.isfinite(value).all():raise ValueError('invalid lowpass input')
        wh=self.omega*dt
        result=((2-wh)*self.output+wh*(self.previous_input+value))/(2+wh)
        if not np.isfinite(result).all():raise ValueError('nonfinite lowpass state')
        self.output=result;self.previous_input=value.copy()
        return result.copy()

    def digital_state(self,b,a):
        # scipy lfilter DFII state AFTER the latest input sample.
        return (float(b[1])*self.previous_input-float(a[1])*self.output)[None,:]


class VariableHighpass2:
    def __init__(self,cutoff_hz,nominal_s):
        self.omega=_frequency(cutoff_hz,nominal_s)
        self.reset(0.)

    def reset(self,value):
        if not math.isfinite(value):raise ValueError('invalid highpass steady input')
        self.state=np.array([0.,float(value)/(self.omega*self.omega)])
        self.previous_input=float(value)

    def update(self,value,dt_s):
        dt=_step(dt_s);value=float(value)
        if not math.isfinite(value):raise ValueError('invalid highpass input')
        w=self.omega;d=math.sqrt(2)*w;h=.5*dt
        velocity,position=self.state
        # Solve the fixed continuous second-order system trapezoidal step.
        rhs0=(1-h*d)*velocity-h*w*w*position+h*(self.previous_input+value)
        rhs1=h*velocity+position
        velocity_new=(rhs0-h*w*w*rhs1)/(1+h*d+h*h*w*w)
        position_new=rhs1+h*velocity_new
        result=value-d*velocity_new-w*w*position_new
        if not all(math.isfinite(x) for x in (velocity_new,position_new,result)):
            raise ValueError('nonfinite highpass state')
        self.state[:]=velocity_new,position_new;self.previous_input=value
        return result
