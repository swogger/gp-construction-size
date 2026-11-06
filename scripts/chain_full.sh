#!/bin/sh
# Full EML-family study (D023), unattended and resumable. Primary hypotheses first.
#   nohup scripts/chain_full.sh > data/runs/chain_full.log 2>&1 &
# Enumeration (enum_family2v) and targets (full_prepare) were completed and frozen before the OSF registration.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
caffeinate -is $PY -m reach.runner run full_gp --workers 6
$PY -W ignore -m reach.analysis.family_full > /dev/null || true
caffeinate -is $PY -m reach.runner run full_es --workers 6
caffeinate -is $PY -m reach.runner run full_budget --workers 6
caffeinate -is $PY -m reach.runner run full_budget64 --workers 6
caffeinate -is $PY -m reach.runner run full_pysr --workers 3
$PY -W ignore -m reach.analysis.family_full
echo "chain_full finished $(date)"
