# Feature contracts: reproduction package (v2, clean-room rebuild)

Rebuilt from the manuscript text on public data only. Reproduces the paper's
tables on the credit card (cc), credit card plus velocity (ccv) and default of
credit card clients (default) datasets, and runs the frozen-configuration
confirmation on IEEE-CIS and Bank Account Fraud.

## Data (never commit data files)

Put the files under `data/`:

| Dataset | File | Source |
| --- | --- | --- |
| cc, ccv | `data/creditcard.csv` | OpenML or Kaggle `mlg-ulb/creditcardfraud` (284,807 rows, 492 frauds) |
| default | `data/default of credit card clients.xls` | UCI dataset 350 (30,000 rows, 6,636 defaults); `UCI_Credit_Card.csv` also works |
| ieee | `data/ieee/train_transaction.csv` | Kaggle competition `ieee-fraud-detection` (590,540 rows) |
| baf | `data/baf/Base.csv` | Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022` (CC BY-NC-ND 4.0) |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib xlrd
```

Tested with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9. IEEE-CIS needs
about 6 GB of memory; the others under 2 GB.

## Run

```bash
sh run_public.sh      # cc, ccv, default: benchmark, decision impact, sweeps, latency, tables
sh run_kaggle.sh      # ieee, baf: frozen-configuration confirmation (hashes code first)
python3 analysis.py --datasets cc ccv default ieee baf
```

## Files

| File | What it does | Paper |
| --- | --- | --- |
| `fcontracts/contracts.py` | Learning, calibration (grid or conformal), online check, rate checks | 4.1-4.5 |
| `fcontracts/paging.py` | Per-feature Shiryaev-Roberts and CUSUM tests, exact and event-driven | 4.6 |
| `fcontracts/inject.py` | Eight corruption operators plus a NaN fill | Table 1 |
| `fcontracts/baselines.py` | Schema, min-max, auto-profiled, robust z, isolation forest, PCA, conformance, PSI | 5.3 |
| `fcontracts/data.py` | Loaders, splits, velocity and per-card features | 5.1 |
| `fcontracts/config.py` | Frozen configuration (written to `frozen_config.json`) | 5, 7.1 |
| `run_experiments.py` | Fault-injection benchmark, clean-stream false flags and pages | 6.1-6.5 |
| `decision.py` | Clean, corrupted, masked, masked with mask-specific threshold, fallback | 6.6, 6.8 |
| `sweeps.py` | Budget sweep (Figure 3) and severity tier (Table 9) | 6.2, 6.7 |
| `latency.py` | Per-request cost at 100, 1,000 and 10,000 features | 6.4 |
| `analysis.py` | Tables, Wilcoxon with Holm, bootstrap intervals | 6, Table 11 |
| `tools_conformal_and_dropout.py` | Conformal margin, realized-rate bound, dropout model helpers | 4.3 (v2) |

## Differences from the paper's original code

This package was rebuilt from the manuscript, so details the text leaves open
were filled in and are listed here: window start per seed (one random start per
seed in the test stream after a 5,000-request burn-in); scale factor 1,000;
inconsistency noise of one reference standard deviation; Wilson bounds at 95%;
unseen categories get an evidence exceedance of 10; PCA reconstruction keeps
90% of variance; rate thresholds are the larger of the binomial tail and a
margin times the largest fitted window count, chosen per feature.
