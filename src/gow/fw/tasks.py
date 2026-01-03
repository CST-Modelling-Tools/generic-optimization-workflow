from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from gow.config import load_problem_config
from gow.evaluation import evaluate_candidate


def _to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable values for FireWorks stored_data.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # fallback
    return str(obj)


def _ensure_fireworks_imports():
    """
    Import FireWorks lazily so the core package stays usable without FireWorks installed.
    """
    try:
        from fireworks import FiretaskBase, FWAction, explicit_serialize  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


def _ensure_filelock_import():
    """
    filelock gives us a cross-platform lock to safely append results.jsonl in parallel.
    """
    try:
        from filelock import FileLock  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "filelock is not installed. Install with: pip install filelock"
        ) from e


# Ensure optional deps are present only when this module is actually used.
_ensure_fireworks_imports()
_ensure_filelock_import()

from fireworks import FWAction, FiretaskBase, explicit_serialize  # type: ignore  # noqa: E402
from filelock import FileLock  # type: ignore  # noqa: E402


@explicit_serialize
class EvaluateCandidateTask(FiretaskBase):
    """
    FireWorks task that evaluates a single candidate using gow's evaluator pipeline.

    Required params (passed via FireWork spec):
      - problem_config: str (path to YAML/JSON)
      - run_id: str
      - candidate_id: str
      - candidate_params: dict
      - outdir: str (base output directory, e.g. "runs")

    Optional:
      - context_override: dict
    """

    required_params = ["problem_config", "run_id", "candidate_id", "candidate_params", "outdir"]
    optional_params = ["context_override"]

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        import json

        problem_config = Path(self["problem_config"]).expanduser()
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]
        candidate_params: Dict[str, Any] = dict(self["candidate_params"])
        outdir = Path(self["outdir"]).expanduser().resolve()
        context_override: Optional[Dict[str, Any]] = self.get("context_override")

        problem = load_problem_config(problem_config)

        # Workdir layout consistent with CLI/local runner:
        # <outdir>/<problem_id>/<run_id>/<candidate_id>/
        workdir = outdir / problem.id / run_id / candidate_id
        workdir.mkdir(parents=True, exist_ok=True)

        res = evaluate_candidate(
            problem,
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_params=candidate_params,
            workdir=workdir,
            context_override=context_override,
        )

        stored = {
            "problem_id": problem.id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "params": _to_jsonable({**problem.runtime_params(), **candidate_params}),
            "fitness": _to_jsonable(res.fitness.model_dump()),
            "returncode": res.returncode,
            "wall_time_s": res.wall_time_s,
            "workdir": str(workdir),
            "stdout_path": str(res.stdout_path),
            "stderr_path": str(res.stderr_path),
            "input_path": str(res.input_path),
            "output_path": str(res.output_path),
        }

        # Write a single JSON artifact per candidate
        artifact_path = workdir / "result.json"
        artifact_path.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")

        return FWAction(stored_data=stored, update_spec={"last_result": stored})


@explicit_serialize
class AppendResultJsonlTask(FiretaskBase):
    """
    Append <workdir>/result.json as one line to:
      <outdir>/<problem_id>/<run_id>/results.jsonl

    Uses a cross-platform file lock to prevent corruption under parallel execution.

    Required params:
      - outdir: str
      - problem_id: str
      - run_id: str
      - candidate_id: str

    Optional:
      - workdir: str  (if not provided, derived from outdir/problem_id/run_id/candidate_id)
      - result_filename: str (default: "result.json")
      - results_filename: str (default: "results.jsonl")
      - lock_filename: str (default: "results.jsonl.lock")
    """

    required_params = ["outdir", "problem_id", "run_id", "candidate_id"]
    optional_params = ["workdir", "result_filename", "results_filename", "lock_filename"]

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        import json

        outdir = Path(self["outdir"]).expanduser().resolve()
        problem_id: str = self["problem_id"]
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]

        result_filename = str(self.get("result_filename", "result.json"))
        results_filename = str(self.get("results_filename", "results.jsonl"))
        lock_filename = str(self.get("lock_filename", f"{results_filename}.lock"))

        # Derive workdir if not explicitly passed
        if self.get("workdir"):
            workdir = Path(str(self["workdir"])).expanduser().resolve()
        else:
            workdir = outdir / problem_id / run_id / candidate_id

        result_path = workdir / result_filename
        if not result_path.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_path}")

        run_root = outdir / problem_id / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        results_path = run_root / results_filename
        lock_path = run_root / lock_filename

        record = json.loads(result_path.read_text(encoding="utf-8"))

        # Safe append under lock
        lock = FileLock(str(lock_path))
        with lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return FWAction(
            stored_data={
                "appended_to": str(results_path),
                "candidate_id": candidate_id,
            }
        )