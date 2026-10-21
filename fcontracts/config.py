"""Frozen configuration. Written to frozen_config.json before any run and hashed.

Change nothing here after the hash is recorded (see run_kaggle.sh).
"""
import json

KAPPA_GRID = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
T_GRID = [3, 4, 5, 6, 8, 10, 15, 25, 40, 80, 160]
RATE_MARGINS = [1.25, 1.5, 2, 3, 5, 8]

CONFIG = {
    "version": "3.0.0",
    "splits": {"reference": 0.60, "fit_share_of_reference": 0.80},
    "discrete_max_values": 50,
    "kappa_grid": KAPPA_GRID,
    "t_grid": T_GRID,
    "calibration": "grid",          # "grid" (paper) or "conformal" (v2 option)
    "rate": {"window": 500, "tail": 1e-6, "margins": RATE_MARGINS},
    "paging": {"A": 1e7, "p1_floor": 0.005, "p1_mult": 10.0, "p0_floor": 1e-6,
               "wilson_z": 1.96, "severity_h": 1.0, "residual_hard_mult": 2.0,
               "test": "sr", "default_tier": "hard",
               "extra_A": [1e3, 1e4]},
    "alpha": {"cc": 1e-3, "ccv": 1e-3, "default": 1e-2, "ieee": 1e-3, "baf": 1e-4,
              "synthetic": 1e-3},
    "window": {"cc": 10000, "ccv": 10000, "default": 3000, "ieee": 10000,
               "baf": 10000, "synthetic": 5000},
    "burn_in": 5000,
    "faults": ["magnitude", "scale", "encoding", "default_fill", "stale",
               "misjoin", "clipping", "inconsistency", "nan_fill", "stale_lag", "misjoin_ref"],
    "paper_faults": ["magnitude", "scale", "encoding", "default_fill", "stale",
                     "misjoin", "clipping", "inconsistency"],
    "prevalence": [0.01, 0.10, 1.0],
    "breadth": [1, 5],
    "seeds": [0, 1, 2, 3, 4],
    "scale_factor": 1000.0,
    "clip_quantile": 0.90,
    "noise_sd": 1.0,
    "min_changed": 5,
    "psi": {"window": 1000, "bins": 10, "level": 0.25},
    "decision": {"fpr": 0.01, "faults": ["magnitude", "scale", "encoding", "misjoin"],
                 "prevalence": [0.10, 1.0], "breadth": [1, 5], "seeds": [0, 1, 2]},
    "default_shuffle_seed": 0,
    # v3 additions, fixed before the Bank Account Fraud confirmation
    "drift_screen": {"spearman_abs": 0.3, "sample": 20000},
    "burst": {"min_run": 20, "run_mult": 2.0},
}


def dump(path="frozen_config.json"):
    with open(path, "w") as f:
        json.dump(CONFIG, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    dump()
