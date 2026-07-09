# param_model — 参数化模型生产

YAML spec → slider/rail URDF + world calibration math. No Genesis runtime dependency.

| module | role |
| --- | --- |
| `generator.py` | parametric URDF (frame, rail, slider, arm mount) |
| `placement.py` | `world_calib` @ rail_y=0 → entity pose |
| `urdf_prepare.py` | mesh paths + visual cache for Genesis |
| `paths.py` | `config/`, `assets/`, generated URDF paths |

Edit geometry in `../config/slider_rail.yaml`, then:

```bash
python -m rm75_control.control.joint_admittance_8dof.param_model \
  --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml \
  --out rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf
```

The Genesis viewer (`../viewer/`) loads this spec by default.
