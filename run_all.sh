#!/usr/bin/env bash
# Reproduces every number, table entry and figure of Section VI.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 MPLBACKEND=Agg
mkdir -p results figures
python3 validate.py        | tee results/validate.log
python3 exp_mechanism.py   | tee results/mechanism.log
python3 exp_dimension.py   | tee results/dimension.log
python3 exp_landscape.py   | tee results/landscape.log
python3 exp_breakdown.py   | tee results/breakdown.log
echo "done"
