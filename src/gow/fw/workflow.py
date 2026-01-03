from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gow.config import load_problem_config


def _ensure_fireworks_imports():
    try:
        from fireworks import Firework, Workflow  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


_ensure_fireworks_imports()
from fireworks import Firework, Workflow  # type: ignore  # noqa: E402

from .tasks import AppendResultJsonlTask, EvaluateCandidateTask, _to_jsonable  # noqa: E402


def default_run_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class SingleEvalSpec:
    # Path to the problem YAML/JSON
    problem_config: Path

    # Base results directory (NOT runs dir). The workflow will create:
    #   <outdir>/<problem_id>/runs/...
    #   <outdir>/<problem_id>/results/...
    outdir: Path

    run_id: str
    candidate_id: str
    candidate_params: Dict[str, Any]
    context_override: Optional[Dict[str, Any]] = None


def build_single_evaluate_workflow(spec: SingleEvalSpec) -> Workflow:
    """
    Build a FireWorks Workflow that evaluates a single candidate and appends it to results.jsonl.

    Target layout:
      <base_results>/<problem_id>/
        results/
          results.jsonl
          summary.json
        runs/
          <run_id>/<candidate_id>/{input.json,output.json,stdout.txt,stderr.txt,result.json}
    """
    # Normalize paths
    problem_config_abs = Path(spec.problem_config).expanduser().resolve()
    base_results_abs = Path(spec.outdir).expanduser().resolve()

    # Load to get problem_id and validate config
    problem = load_problem_config(problem_config_abs)

    # Problem root (first folder is problem id)
    problem_root = base_results_abs / problem.id

    # --- FireWork 1: evaluate candidate (writes result.json under runs/<run_id>/<candidate_id>/)
    eval_task_params: Dict[str, Any] = {
        "problem_config": str(problem_config_abs),
        # IMPORTANT: outdir is the *problem_root* (not base_results, not runs/)
        "outdir": str(problem_root),
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "candidate_params": _to_jsonable(spec.candidate_params),
    }
    if spec.context_override:
        eval_task_params["context_override"] = _to_jsonable(spec.context_override)

    fw_eval = Firework(
        [EvaluateCandidateTask(eval_task_params)],
        name=f"evaluate:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={
            "problem_id": problem.id,
            "run_id": spec.run_id,
            "candidate_id": spec.candidate_id,
        },
    )

    # --- FireWork 2: append to results/results.jsonl (idempotent + locked)
    append_task_params: Dict[str, Any] = {
        # IMPORTANT: outdir is the *problem_root*
        "outdir": str(problem_root),
        "problem_id": problem.id,
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        # filenames/locking kept as defaults in the task
    }

    fw_append = Firework(
        [AppendResultJsonlTask(append_task_params)],
        name=f"append-results:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={
            "problem_id": problem.id,
            "run_id": spec.run_id,
            "candidate_id": spec.candidate_id,
        },
        parents=[fw_eval],
    )

    wf_name = f"gow-single-eval:{problem.id}:{spec.run_id}:{spec.candidate_id}"
    return Workflow([fw_eval, fw_append], name=wf_name)