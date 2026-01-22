from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _norm(p: PathLike) -> Path:
    """
    Normalize a path consistently across OSes:
      - expand '~'
      - resolve to an absolute path
    """
    return Path(p).expanduser().resolve()


def run_root(outdir: PathLike, run_id: str) -> Path:
    """
    Root directory for a run.

    Layout:
      <outdir>/runs/<run_id>/
    """
    return _norm(outdir) / "runs" / str(run_id)


def candidate_workdir(outdir: PathLike, run_id: str, candidate_id: str) -> Path:
    """
    Working directory for a candidate evaluation.

    Layout:
      <outdir>/runs/<run_id>/<candidate_id>/
    """
    return run_root(outdir, run_id) / str(candidate_id)


def run_launchers_dir(outdir: PathLike, run_id: str) -> Path:
    """
    Directory for FireWorks launcher_* directories, scoped per run.

    Layout:
      <outdir>/runs/<run_id>/launchers/

    Scoping launchers by run avoids collisions when multiple runs share the same
    <outdir>, and is safer for parallel / cluster execution.
    """
    return run_root(outdir, run_id) / "launchers"