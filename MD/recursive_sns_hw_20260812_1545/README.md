# Recursive SNS HW dump 20260812_1545

## Diagnosis (from CSV)
- Mid-scan stop: working/dexterity CBF made qdot=0 infeasible; ProxQP timeout then latched final_qdot_violates_p0 / feedback_stale.
- joint_working_cbf:4 frequently active (avoidance working).
- qp1/qp2 still occasionally >5–15 ms (post-hoc budget).

## Fix just applied (in tree + this snapshot)
- Working CBF / dexterity: outside envelope blocks intrusion but keeps zero feasible.
- QP1 budget fail: skip chained ProxQP retry; analytic P0 projection first.
- finish gate: project before latch; do not overwrite solver fault reason.

## CSV files
total 67060
drwxrwxr-x 2 camp camp     4096 Aug 12 15:47 .
drwxrwxr-x 4 camp camp     4096 Aug 12 15:47 ..
-rw-rw-r-- 1 camp camp   233692 Aug 12 15:05 csv_rail_20260812_150518.csv
-rw-rw-r-- 1 camp camp 17180910 Aug 12 15:23 csv_rail_20260812_150549.csv
-rw-rw-r-- 1 camp camp   519179 Aug 12 15:39 csv_rail_20260812_153905.csv
-rw-rw-r-- 1 camp camp   489256 Aug 12 15:40 csv_rail_20260812_153949.csv
-rw-rw-r-- 1 camp camp   278821 Aug 12 15:40 csv_rail_20260812_154032.csv
-rw-rw-r-- 1 camp camp   169218 Aug 12 15:41 csv_rail_20260812_154056.csv
-rw-rw-r-- 1 camp camp   133395 Aug 12 15:41 csv_rail_20260812_154113.csv
-rw-rw-r-- 1 camp camp   139815 Aug 12 15:41 csv_rail_20260812_154126.csv
-rw-rw-r-- 1 camp camp  4324886 Aug 12 15:47 csv_rail_20260812_154150.csv
-rw-rw-r-- 1 camp camp  5169587 Aug 12 15:05 csv_run_20260812_150518.csv
-rw-rw-r-- 1 camp camp  5358849 Aug 12 15:06 csv_run_20260812_150549.csv
-rw-rw-r-- 1 camp camp  8903479 Aug 12 15:39 csv_run_20260812_153905.csv
-rw-rw-r-- 1 camp camp  3601376 Aug 12 15:40 csv_run_20260812_153949.csv
-rw-rw-r-- 1 camp camp  4693960 Aug 12 15:40 csv_run_20260812_154032.csv
-rw-rw-r-- 1 camp camp  2172849 Aug 12 15:41 csv_run_20260812_154056.csv
-rw-rw-r-- 1 camp camp  2109828 Aug 12 15:41 csv_run_20260812_154113.csv
-rw-rw-r-- 1 camp camp  1408068 Aug 12 15:41 csv_run_20260812_154126.csv
-rw-rw-r-- 1 camp camp 11733618 Aug 12 15:42 csv_run_20260812_154150.csv

## Code files
MD/recursive_sns_hw_20260812_1545/code/rm75_control/apps/joint_admittance_8dof/run_joint_admittance.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/configs/joint_admittance_8dof.yaml
MD/recursive_sns_hw_20260812_1545/code/rm75_control/rm75_control/control/joint_admittance_8dof/config.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/rm75_control/control/joint_admittance_8dof/generic_runtime.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/rm75_control/control/joint_admittance_8dof/hw/rail_servo.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/rm75_control/control/joint_admittance_8dof/loop.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/rm75_control/control/joint_admittance_8dof/solver/two_level_qpik.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_generic_qpik_config.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_generic_qpik_telemetry.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_generic_runtime.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_rail_reference.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_rail_scan_replay.py
MD/recursive_sns_hw_20260812_1545/code/rm75_control/tests/test_two_level_qpik.py
