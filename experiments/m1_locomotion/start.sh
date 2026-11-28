#!/usr/bin/env bash
set -euo pipefail

app_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=${M1_PYTHON:-"$app_dir/../../.venv-v21/bin/python"}
export PYTHONPATH="$app_dir/.runtime/deps:/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export LP_NUM_THREADS=${LP_NUM_THREADS:-3}

exec "$python_bin" "$app_dir/dashboard.py" "$@"
