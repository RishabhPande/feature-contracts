#!/bin/sh
# Frozen-configuration confirmation on IEEE-CIS and Bank Account Fraud.
# Records code hashes BEFORE running. Change nothing before or during the run.
set -e
mkdir -p results
python3 -m fcontracts.config
( date -u +%FT%TZ; shasum -a 256 frozen_config.json fcontracts/*.py *.py 2>/dev/null || sha256sum frozen_config.json fcontracts/*.py *.py ) > results/code_hashes.txt
for ds in ieee baf; do
  if python3 -c "from fcontracts.data import _path; _path('$ds/' + {'ieee':'train_transaction.csv','baf':'Base.csv'}['$ds'])" 2>/dev/null; then
    python3 run_experiments.py --dataset $ds
    python3 decision.py --dataset $ds --models hgb lr
    python3 sweeps.py --dataset $ds --seeds 0 1 2 --prevalence 0.1
  else
    echo "skipping $ds: data file not found"
  fi
done
python3 analysis.py --datasets ieee baf
