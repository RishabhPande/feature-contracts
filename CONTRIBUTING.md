# Contributing

This repo is a reproduction package for the feature-contracts
manuscript. The most useful contributions are:

* Bug reports: the script that failed, the dataset, and the full
  error
* Notes on dataset versions or sources that no longer match the
  README
* Documentation fixes and clearer run instructions

## Ground rules

1. **Never commit data files.** Everything under `data/` stays
   local.
2. **Pinned dependencies.** Changes must run against
   requirements.txt as pinned (Python 3.13).
3. **Reproducible.** A script change should produce the same tables
   and figures, or the PR must say exactly which numbers move and
   why.

## How to contribute

Open an issue for anything affecting the reproduced numbers. For
fixes, fork the repo, branch from `main`, re-run the affected
script on the smallest dataset (cc), and open a pull request with
the before/after output.
