"""Validate a real study configuration or explicitly submit an ICRA scan.

The default operation is offline. --execute uses the existing public HFPC API;
it neither starts a controller nor homes or positions the robot.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='study YAML (baseline/shadow/active)')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--validate-only',action='store_true',help='offline validation; also the default')
    action.add_argument('--execute',action='store_true',help='explicitly send to an already running controller')
    parser.add_argument('--path-spec',help='existing taught icra_path_v1 JSON or YAML; required to execute')
    parser.add_argument('--prefix',default='',help='existing controller IPC prefix')
    args=parser.parse_args(argv)
    from peirastic.contact_qp.runtime_config import load_study_config,validate_study_config
    try:
        config=load_study_config(args.config)
        report=validate_study_config(config)
        report['nominal_profile']='icra_contact_payload (original tool-axis 4 N, 0.010 m/s normal and seek limits)'
        spec=None
        if args.path_spec:
            import yaml
            from peirastic.scan_path import ForearmReference
            spec=yaml.safe_load(Path(args.path_spec).expanduser().read_text())
            reference=ForearmReference(spec)
            report['path_duration_s']=reference.duration_s
        if args.execute and spec is None:raise ValueError('--execute requires --path-spec from taught poses')
    except (ValueError,TypeError,KeyError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report,indent=2))
    if not args.execute:return 0
    from peirastic.api.arm import PeirasticArm
    from peirastic.scan_path import SCAN_FORCE_AXES
    from peirastic.realman8dof.force.contact_nominal import icra_contact_payload
    arm=PeirasticArm(prefix=args.prefix)
    arm.set_force_control(control_frame='tool',max_vz_tool_m_s=.010)
    arm.set_force_raw_override(icra_contact_payload())
    return int(arm.hfpc(reference='icra_path',path_spec=spec,contact_qp=config,
        law='contact_qp' if config.get('mode')=='active' else 'tff',
        force=[0,0,4,0,0,0],force_axes=SCAN_FORCE_AXES,
        wait_for_contact=True,scan_contact_n=4.,scan_contact_s=.1,
        label='contact_study_'+config.get('mode','baseline')))


if __name__=='__main__':raise SystemExit(main())
