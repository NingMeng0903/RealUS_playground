"""Reproduce default-law equality against the saved pre-seam source."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from peirastic.tests.test_contact_qp_nominal import make_law, sample, RAW, assert_state_equal
BASE=Path('MD/contact_qp/two_clock_before/source')
def old_module(label,relative):
    spec=importlib.util.spec_from_file_location('_contact_before_'+label,BASE/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
root='rm75_control/rm75_control/control/admittance_common/'
oldctrl=old_module('controller',root+'controller.py')
for stem,names in [('adaptive_ke',('AdaptiveKeConfig','EnvironmentStiffnessEstimator')),
                   ('fast_retract_guard',('FastRetractGuardConfig','FastRetractGuard')),
                   ('tdpa',('TdpaConfig','TimeDomainPassivityObserver'))]:
    module=old_module(stem,root+stem+'.py')
    for name in names:setattr(oldctrl,name,getattr(module,name))
legacy=old_module('legacy','peirastic/realman8dof/force/legacy.py')
tilt=old_module('tilt','peirastic/realman8dof/force/torque_tilt.py')
old=tilt.LegacyForceWithTilt(legacy.LegacyForceLaw(oldctrl.AdmittanceController(.005,oldctrl.AdmittanceConfig.from_dict(RAW))),
                           tilt.TorqueTilt(tilt.TorqueTiltConfig.from_dict(RAW)))
new=make_law();maximum=0.
for i in range(100000):
    kw=sample(i)
    if i and i%10000==0:
        for controller in (old,new):controller.reset(pose=kw['pose'],f_ext=kw['f_ext'])
    a=old.update(**kw);b=new.prepare(measurement_id=i,**kw)
    maximum=max(maximum,float(np.max(abs(a.v_force-b.v_force))))
    np.testing.assert_array_equal(a.v_force,b.v_force)
    new.commit_applied(b.v_force,final_full_twist=b.telemetry['v_cmd'])
    if i%1000==0:assert_state_equal(old,new)
assert_state_equal(old,new)
print({'ticks':100000,'max_abs_error':maximum,'baseline':'archived pre-two-clock source','status':'PASS'})
