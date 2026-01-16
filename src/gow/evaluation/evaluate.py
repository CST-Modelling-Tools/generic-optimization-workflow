from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

from gow.config import ProblemConfig
from gow.evaluator import ExternalRunResult, run_external_evaluator


def _looks_like_hfe_command(cmd: List[str]) -> bool:
    """
    Heuristic: treat as HFE if the first token is 'hfe'/'hfe.exe',
    or if any token ends with those names.
    """
    if not cmd:
        return False
    first = Path(cmd[0]).name.lower()
    if first in ("hfe", "hfe.exe"):
        return True
    for tok in cmd:
        name = Path(tok).name.lower()
        if name in ("hfe", "hfe.exe"):
            return True
    return False


def _has_params_flag(cmd: List[str]) -> bool:
    return ("--params-file" in cmd) or ("--params-json" in cmd)


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

    GOW contract support (generic):
      - If the evaluator command appears to be HFE (heliostat-field-evaluator)
        and does not already specify --params-file/--params-json, we will:
          1) write <workdir>/params.json with the merged params
          2) append: --params-file <workdir>/params.json
    """
    # Candidate workdir
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Start from the configured baseline values
    params = problem.runtime_params()
    # Override with candidate values
    params.update(candidate_params)

    ctx = dict(problem.context)
    if context_override:
        ctx.update(context_override)

    ev = problem.evaluator

    # Start from configured command
    cmd: List[str] = list(ev.command)

    # Replace {python} placeholder with the current interpreter
    cmd = [sys.executable if tok == "{python}" else tok for tok in cmd]

    # Resolve relative path tokens relative to the config file location (if available)
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

    # --- Generic "contract" behavior for HFE: ensure params-file is provided ---
    if _looks_like_hfe_command(cmd) and not _has_params_flag(cmd):
        params_path = workdir / "params.json"
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        cmd = cmd + ["--params-file", str(params_path)]

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