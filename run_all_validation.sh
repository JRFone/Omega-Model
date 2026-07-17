#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
python -m compileall omega_desktop.py omega_self_check.py stock_model tests
python run_all_validation.py
