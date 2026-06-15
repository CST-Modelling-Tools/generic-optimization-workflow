#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gow.postprocess import merge_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple GOW runs into a synthetic run results.jsonl")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--source-run-id", action="append", required=True)
    args = parser.parse_args()
    path = merge_runs(outdir=args.results_dir, target_run_id=args.target_run_id, source_run_ids=args.source_run_id)
    print(path)


if __name__ == "__main__":
    main()
