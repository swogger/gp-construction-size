#!/bin/sh
# EML-family round 2 (D022), unattended and resumable: targets_fam2 -> GP (stop on exact) -> predictors -> reports.
#   nohup scripts/chain_family2.sh > data/runs/chain_family2.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
caffeinate -is $PY -m reach.runner run family_prepare2 --workers 2
caffeinate -is $PY -m reach.runner run family_predictors2 --workers 2
caffeinate -is $PY -m reach.runner run scout_family2 --workers 6
$PY -W ignore -m reach.analysis.family_report --phase scout_family2 --out docs/results/eml_family_round2_report.md
$PY -W ignore -m reach.analysis.family_predictors --phase scout_family2 --suffix _v2 --out docs/results/eml_family_round2_predictors.md
echo "chain_family2 finished $(date)"
