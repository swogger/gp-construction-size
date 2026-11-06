#!/bin/sh
# D034 post hoc sensitivity analyses (specified and committed before any run): the penalty for undefined values
# scaled to the target (pen_gp, pen_nox, pen_es) and headless-chicken crossover (hc_gp); then the analysis.
#   nohup scripts/chain_d034.sh > data/runs/chain_d034.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for PH in pen_gp pen_nox hc_gp pen_es; do
    caffeinate -is $PY -m reach.runner run $PH --workers 6
    caffeinate -is $PY -m reach.runner run $PH --workers 6 --retry-errors
done
$PY -W ignore paper/scripts/sensitivity.py
echo "chain_d034 finished $(date)"
