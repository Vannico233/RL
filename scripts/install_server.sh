#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import sys, platform; assert platform.system() == "Linux", "Training environment requires Linux"; assert sys.version_info[:2] == (3, 11), "Activate the Python 3.11 chart-rl Conda environment"'
python -m pip install --timeout 60 --retries 5 -r requirements-server.txt
python -m pip install --timeout 60 --retries 5 -e .
python -m pip check
mkdir -p runs/environment
python -m pip freeze > runs/environment/pip-freeze.txt
python scripts/run_cpu_tests.py
printf '%s\n' 'Installation and CPU tests finished. GPU training is not validated until the server preflight and G1 smoke run pass.'
