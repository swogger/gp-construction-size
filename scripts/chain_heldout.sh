#!/bin/sh
# Held-out extension (OSF 2jqhr, registered commit 3182bf1): H6/H7 then H8, unattended and resumable.
#   nohup scripts/chain_heldout.sh > data/runs/chain_heldout.log 2>&1 &
# As registered: runs ending in the error state are rerun once; then the registered analysis.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
caffeinate -is $PY -m reach.runner run heldout_gp --workers 6
caffeinate -is $PY -m reach.runner run heldout_gp --workers 6 --retry-errors
$PY -W ignore -m reach.analysis.heldout_analysis > /dev/null || true
caffeinate -is $PY -m reach.runner run nox_gp --workers 6
caffeinate -is $PY -m reach.runner run nox_gp --workers 6 --retry-errors
$PY -W ignore -m reach.analysis.heldout_analysis
echo "chain_heldout finished $(date)"
