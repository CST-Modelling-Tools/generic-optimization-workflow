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

from .tasks import AppendResultJsonlTask, EvaluateCandidateTask  # noqa: E402


def default_run_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class SingleEvalSpec:
    problem_config: Path
    outdir: Path
    run_id: str
    candidate_id: str
    candidate_params: Dict[str, Any]
    context_override: Optional[Dict[str, Any]] = None


def build_single_evaluate_workflow(spec: SingleEvalSpec) -> Workflow:
    """
    Build a FireWorks Workflow that evaluates a single candidate AND appends it to results.jsonl.

    Artifacts:
      <outdir>/<problem_id>/<run_id>/<candidate_id>/{input.json,output.json,stdout.txt,stderr.txt,result.json}
      <outdir>/<problem_id>/<run_id>/results.jsonl
    """
    problem = load_problem_config(spec.problem_config)

    # FireWork 1: evaluate candidate and write result.json
    eval_task_params: Dict[str, Any] = {
        "problem_config": str(spec.problem_config),
        "outdir": str(spec.outdir),
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "candidate_params": spec.candidate_params,
    }
    if spec.context_override:
        eval_task_params["context_override"] = spec.context_override

    fw_eval = Firework(
        [EvaluateCandidateTask(eval_task_params)],
        name=f"evaluate:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={
            "problem_id": problem.id,
            "run_id": spec.run_id,
            "candidate_id": spec.candidate_id,
        },
    )

    # FireWork 2: append result.json into run-level results.jsonl (safe with lock)
    append_task_params: Dict[str, Any] = {
        "outdir": str(Path(spec.outdir).expanduser().resolve()),
        "problem_id": problem.id,
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
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