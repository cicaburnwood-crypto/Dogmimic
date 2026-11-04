#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
G1_PYTHON=${G1_PYTHON:-/usr/bin/python3}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/deployments/g1_actions/vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206${PYTHONPATH:+:$PYTHONPATH}"
# Offline unless an interface was configured. No action runs on service startup.
if [[ -n "${G1_INTERFACE:-}" ]]; then
  exec "$G1_PYTHON" "$ROOT/apps/g1-console/server.py" --backend backend:create_backend --port "${G1_CONSOLE_PORT:-8767}"
else
  exec "$G1_PYTHON" "$ROOT/apps/g1-console/server.py" --port "${G1_CONSOLE_PORT:-8767}"
fi
