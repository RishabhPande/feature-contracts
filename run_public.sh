#!/bin/sh
# Public datasets: cc, ccv, default. About 1 to 2 hours on a laptop.
set -e
mkdir -p results
python3 -m fcontracts.config
for ds in default cc ccv; do
  python3 run_experiments.py --dataset $ds
  python3 decision.py --dataset $ds
  python3 sweeps.py --dataset $ds
done
python3 latency.py
python3 analysis.py --datasets cc ccv default
