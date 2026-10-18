#!/bin/sh
# EML-family feasibility pilot, unattended: enumerate -> prepare -> GP scout -> report.
# Every step is a resumable queue phase; rerunning this script continues where it stopped.
#   nohup scripts/chain_family.sh > data/runs/chain_family.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
caffeinate -is $PY -m reach.runner run enum_family --workers 2
caffeinate -is $PY -m reach.runner run family_prepare --workers 2
caffeinate -is $PY -m reach.runner run scout_family --workers 6
$PY -W ignore -m reach.analysis.family_report
echo "chain_family finished $(date)"
