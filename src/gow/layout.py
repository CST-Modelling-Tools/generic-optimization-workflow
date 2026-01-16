from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def run_root(outdir: PathLike, run_id: str) -> Path:
    """
    Root directory for a run, e.g. <outdir>/runs/<run_id>.
    """
    outdir_p = Path(outdir)
    return outdir_p / "runs" / run_id


def candidate_workdir(outdir: PathLike, run_id: str, candidate_id: str) -> Path:
    """
    Working directory for a candidate evaluation, e.g.
    <outdir>/runs/<run_id>/<candidate_id>.
    """
    return run_root(outdir, run_id) / candidate_id


def launchers_dir(outdir: PathLike) -> Path:
    """
    Default directory for FireWorks launchers when using GOW's fw helpers, e.g.
    <outdir>/launchers.
    """
    return Path(outdir) / "launchers"