from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

from gow.config import ProblemConfig
from gow.evaluator import ExternalRunResult, run_external_evaluator


def evaluate_candidate(
    problem: ProblemConfig,
    *,
    run_id: str,
    candidate_id: str,
    candidate_params: Dict[str, Any],
    workdir: str | Path,
    context_override: Optional[Dict[str, Any]] = None,
) -> ExternalRunResult:
    """
    Evaluate a candidate parameter set for a given problem configuration.

    - candidate_params: values for (some or all) parameters.
      Missing parameters fall back to problem.parameters[*].value.
    - context_override: optional extra/override entries merged into problem.context.
    """
    # Start from the configured baseline values
    params = problem.runtime_params()
    # Override with candidate values
    params.update(candidate_params)

    ctx = dict(problem.context)
    if context_override:
        ctx.update(context_override)

    ev = problem.evaluator
    cmd: List[str] = list(ev.command)

    # Resolve relative path tokens in the command relative to the config file location (if available).
    # We only rewrite tokens when the resolved path actually exists to avoid breaking things like
    # "python", "mpirun", etc.
    if problem.source_path is not None:
        base = problem.source_path.parent
        resolved: List[str] = []
        for tok in cmd:
            p = Path(tok)
            if not p.is_absolute():
                candidate = base / p
                if candidate.exists():
                    tok = str(candidate.resolve())
            resolved.append(tok)
        cmd = resolved

    return run_external_evaluator(
        command=cmd,
        workdir=workdir,
        run_id=run_id,
        candidate_id=candidate_id,
        params=params,
        context=ctx,
        timeout_s=ev.timeout_s,
        extra_args=ev.extra_args,
        env=ev.env,
    )