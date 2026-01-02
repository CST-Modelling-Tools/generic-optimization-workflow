from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
    exe = ev.executable

    # If executable is a relative path, resolve it relative to the config file location
    exe_path = Path(exe)
    if not exe_path.is_absolute():
        if problem.source_path is not None:
            exe_path = (problem.source_path.parent / exe_path).resolve()
        else:
            exe_path = exe_path.resolve()


    return run_external_evaluator(
        executable=str(exe_path),
        workdir=workdir,
        run_id=run_id,
        candidate_id=candidate_id,
        params=params,
        context=ctx,
        timeout_s=ev.timeout_s,
        extra_args=ev.extra_args,
        env=ev.env,
    )