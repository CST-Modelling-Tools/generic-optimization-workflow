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
    problem_config: Path         # path to optimization_specs.yaml
    outdir: Path                 # flattened results root
    run_id: str
    candidate_id: str
    candidate_params: Dict[str, Any]
    context_override: Optional[Dict[str, Any]] = None


def build_single_evaluate_workflow(spec: SingleEvalSpec) -> Workflow:
    """
    Workflow:
      1) EvaluateCandidateTask -> writes runs/<run_id>/<candidate_id>/result.json
      2) AppendResultJsonlTask -> appends to <outdir>/results.jsonl and runs/<run_id>/results.jsonl
    """
    problem_config_abs = Path(spec.problem_config).expanduser().resolve()
    outdir_abs = Path(spec.outdir).expanduser().resolve()

    # Load just for validation + problem_id
    problem = load_problem_config(problem_config_abs)

    eval_task_params: Dict[str, Any] = {
        "problem_config": str(problem_config_abs),
        "outdir": str(outdir_abs),
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "candidate_params": _to_jsonable(spec.candidate_params),
    }
    if spec.context_override:
        eval_task_params["context_override"] = _to_jsonable(spec.context_override)

    fw_eval = Firework(
        [EvaluateCandidateTask(eval_task_params)],
        name=f"evaluate:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={"problem_id": problem.id, "run_id": spec.run_id, "candidate_id": spec.candidate_id},
    )

    append_task_params: Dict[str, Any] = {
        "outdir": str(outdir_abs),
        "problem_id": problem.id,
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
    }

    fw_append = Firework(
        [AppendResultJsonlTask(append_task_params)],
        name=f"append-results:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={"problem_id": problem.id, "run_id": spec.run_id, "candidate_id": spec.candidate_id},
        parents=[fw_eval],
    )

    wf_name = f"gow-single-eval:{problem.id}:{spec.run_id}:{spec.candidate_id}"
    return Workflow([fw_eval, fw_append], name=wf_name)