#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gow.candidate_ids import parse_candidate_id
from gow.layout import run_root
from gow.postprocess import archive_generation_workdirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive one completed generation into a single tar.gz")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-id", type=int, required=True)
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    run_dir = run_root(args.results_dir, args.run_id)
    candidate_ids: list[str] = []
    if run_dir.exists():
        for candidate_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in {"launchers", "generations", "archives"}):
            parts = parse_candidate_id(candidate_dir.name)
            if parts is not None and parts.generation_id == args.generation_id:
                candidate_ids.append(candidate_dir.name)

    path = archive_generation_workdirs(
        outdir=args.results_dir,
        run_id=args.run_id,
        generation_id=args.generation_id,
        candidate_ids=candidate_ids,
        delete_source=args.delete_source,
    )
    print(path)


if __name__ == "__main__":
    main()
