#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
"$PYTHON" a_response.py > a_run.log 2>&1
"$PYTHON" build_graph.py > graph_run.log 2>&1
"$PYTHON" b_planner.py > b_run.log 2>&1
"$PYTHON" ab_integrated_extension.py > extension_run.log 2>&1
"$PYTHON" prior_response_factorial.py > prior_factorial_run.log 2>&1
"$PYTHON" diagnostics.py > diagnostics_run.log 2>&1
