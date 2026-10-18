#!/bin/sh
# D038 post hoc analyses (specified and committed before any run): more E0 seeds on set A (restarts against long
# runs) and the random-sampling baseline; then the analyses.
#   nohup scripts/chain_d038.sh > data/runs/chain_d038.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for PH in restart_seeds rand_sample; do
    caffeinate -is $PY -m reach.runner run $PH --workers 6
    caffeinate -is $PY -m reach.runner run $PH --workers 6 --retry-errors
done
$PY -W ignore paper/scripts/restarts.py
$PY -W ignore paper/scripts/random_baseline.py
echo "chain_d038 finished $(date)"
