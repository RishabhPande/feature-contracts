"""Check reproduction entry points and summary inputs without importing ML code.

This is a file-presence check, not a scientific reproduction or hash verifier.
Exit 1 means required package files are missing. No files are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Entry points documented by this package, including known missing v3 drivers.
ENTRY_POINTS = (
    "run_public.sh", "run_v3.sh", "paging_sim.py", "sweeps.py", "refresh.py",
    "latency.py", "analysis_v3.py", "run_v3_ext.sh", "run_v3_ext2.sh",
    "adaptive.py", "overhead.py", "make_tables.py", "make_tables_ext.py",
    "make_figs.py", "sensitivity.py", "temporal.py",
)
FREEZE_MANIFESTS = (
    "results/code_hashes_ieee_freeze.txt",
    "results_v3/code_hashes_baf_freeze.txt",
)
DATASETS = ("cc", "ccv", "default", "ieee", "baf")


def check(root: Path) -> dict:
    """Inspect paths only; do not load datasets, results, or scientific modules."""
    missing = []
    for name in ENTRY_POINTS + FREEZE_MANIFESTS:
        if not (root / name).is_file():
            missing.append({"path": name, "reason": "documented package file"})
    checked_datasets = []
    # analysis_v3 skips datasets with no run CSV. For each CSV it does process,
    # these three companion files are read unconditionally.
    for dataset in DATASETS:
        if not (root / "results_v3" / f"{dataset}_runs.csv").is_file():
            continue
        checked_datasets.append(dataset)
        for suffix in ("clean.json", "clean_flags.npz", "clean_pages.json"):
            name = f"results_v3/{dataset}_{suffix}"
            if not (root / name).is_file():
                missing.append({"path": name, "reason": "analysis_v3 input"})
    return {"complete": not missing, "checked_summary_datasets": checked_datasets,
            "missing": missing,
            "scope": "File presence only; no data, dependency, hash, or result validation."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", help="emit a JSON inventory")
    args = parser.parse_args(argv)
    if not args.repo.is_dir():
        parser.error(f"repository directory does not exist: {args.repo}")
    report = check(args.repo)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report["scope"])
        if report["missing"]:
            print("Package incomplete. Missing files:")
            for item in report["missing"]:
                print(f"  {item['path']} ({item['reason']})")
        else:
            print("All inventoried files are present. This does not prove reproduction.")
        if not report["checked_summary_datasets"]:
            print("No v3 run CSVs found; no per-dataset summary inputs were checked.")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
