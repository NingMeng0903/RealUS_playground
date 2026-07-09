#!/usr/bin/env python3
"""Deprecated launcher — prefer: python -m ...viewer.demo"""

from rm75_control.control.joint_admittance_8dof.viewer.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
