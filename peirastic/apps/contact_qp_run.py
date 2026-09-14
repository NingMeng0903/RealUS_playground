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
        nominal_frame = ('contact-face' if (config.get('qp') or {}).get('allocation_policy')
                         == 'delay_kf_cop_v1' else 'tool-axis')
        from peirastic.contact_qp.runtime_config import motion_settings
        motion=motion_settings(config)
        normal=motion.get('normal_max_m_s',.010);seek=motion.get('seek_m_s',.010)
        report['nominal_profile']=(f'icra_contact_payload ({nominal_frame} 4 N, '
                                   f'{normal:.3f} m/s normal, {seek:.3f} m/s seek limits)')
        report['force_age_limit_s']=(config.get('source') or {}).get('max_age_s')
        report['candidate_lifetime_s']=(config.get('qp') or {}).get('certificate_horizon_s',.01)
        report['command_max_interval_s']=(config.get('command') or {}).get('max_interval_s')
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
    arm.set_force_control(control_frame='tool',max_vz_tool_m_s=normal)
    arm.set_force_raw_override(dict(icra_contact_payload(),max_vz_tool_m_s=normal,v_seek_free_m_s=seek))
    return int(arm.hfpc(reference='icra_path',path_spec=spec,contact_qp=config,
        law='contact_qp' if config.get('mode')=='active' else 'tff',
        force=[0,0,4,0,0,0],force_axes=SCAN_FORCE_AXES,
        wait_for_contact=True,scan_contact_n=4.,scan_contact_s=.1,
        label='contact_study_'+config.get('mode','baseline')))


if __name__=='__main__':raise SystemExit(main())
