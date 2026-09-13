#!/usr/bin/env bash
# Compatibility entry: the shared default now uses the no-tank outer controller.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_icra_outer.sh" "$@"
