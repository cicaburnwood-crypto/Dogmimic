#!/usr/bin/env bash
set -euo pipefail
cd /home/liudiwen/projects/supermimic_vibe
mkdir -p logs data/raw/lafan1_g1 data/processed
if [[ ! -x .venv/bin/python ]]; then
  /home/liudiwen/.local/bin/uv venv --python 3.11 .venv
fi
if [[ -s requirements-super5090.lock.txt ]]; then
  UV_HTTP_TIMEOUT=120 /home/liudiwen/.local/bin/uv pip install --python .venv/bin/python -r requirements-super5090.lock.txt
else
  UV_HTTP_TIMEOUT=120 /home/liudiwen/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt huggingface-hub
fi
/home/liudiwen/.local/bin/uv pip check --python .venv/bin/python
/home/liudiwen/.local/bin/uv pip freeze --python .venv/bin/python > requirements-super5090.lock.txt
