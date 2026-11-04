#!/usr/bin/env bash
set -euo pipefail
B2_CONSOLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B2_PROJECT_ROOT=$(cd "$B2_CONSOLE_ROOT/../.." && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B2_ARGS=(--port "${B2_CONSOLE_PORT:-8765}" --bundle "${B2_BUNDLE:-$B2_PROJECT_ROOT/deployments/b2_192_168_123_164_20261101}")
if [[ -n "${B2_INTERFACE:-}" ]]; then
  B2_ARGS+=(--backend sdk --interface "$B2_INTERFACE")
else
  B2_ARGS+=(--backend offline)
fi
exec "${B2_PYTHON:-/usr/bin/python3}" "$B2_CONSOLE_ROOT/server.py" "${B2_ARGS[@]}"
