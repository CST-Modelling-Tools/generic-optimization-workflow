from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from gow.config import load_problem_config
from gow.evaluation import evaluate_candidate


def _to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable values for FireWorks stored_data
    and task parameters.
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
    return str(obj)


def _ensure_fireworks_imports():
    try:
        from fireworks import FiretaskBase, FWAction, explicit_serialize  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


def _ensure_filelock_import():
    try:
        from filelock import FileLock  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("filelock is not installed. Install with: pip install filelock") from e


_ensure_fireworks_imports()
_ensure_filelock_import()

from fireworks import FWAction, FiretaskBase, explicit_serialize  # type: ignore  # noqa: E402
from filelock import FileLock  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def _problem_root(problem_root: Path) -> Path:
    """
    problem_root is already:
      <base_results>/<problem_id>/
    """
    return problem_root


def _results_dir(problem_root: Path) -> Path:
    return _problem_root(problem_root) / "results"


def _runs_dir(problem_root: Path) -> Path:
    return _problem_root(problem_root) / "runs"


def _run_root(problem_root: Path, run_id: str) -> Path:
    return _runs_dir(problem_root) / run_id


def _candidate_workdir(problem_root: Path, run_id: str, candidate_id: str) -> Path:
    return _run_root(problem_root, run_id) / candidate_id


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------

@explicit_serialize
class EvaluateCandidateTask(FiretaskBase):
    """
    FireWorks task that evaluates a single candidate using gow's evaluator pipeline.

    IMPORTANT:
      - `outdir` is the *problem root* directory:
            <base_results>/<problem_id>/

    Target layout:
      <outdir>/
        results/
          results.jsonl
          summary.json  (not written here)
        runs/
          <run_id>/<candidate_id>/{input.json,output.json,stdout.txt,stderr.txt,result.json}
    """

    required_params = ["problem_config", "run_id", "candidate_id", "candidate_params", "outdir"]
    optional_params = ["context_override"]

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        import json

        problem_config = Path(self["problem_config"]).expanduser()
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]
        candidate_params: Dict[str, Any] = dict(self["candidate_params"])
        problem_root = Path(self["outdir"]).expanduser().resolve()
        context_override: Optional[Dict[str, Any]] = self.get("context_override")

        problem = load_problem_config(problem_config)

        # outdir is already <base_results>/<problem_id>
        # so candidate workdir is <outdir>/runs/<run_id>/<candidate_id>/
        workdir = _candidate_workdir(problem_root, run_id, candidate_id)
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

        (workdir / "result.json").write_text(
            json.dumps(stored, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return FWAction(stored_data=stored, update_spec={"last_result": stored})


@explicit_serialize
class AppendResultJsonlTask(FiretaskBase):
    """
    Append <workdir>/result.json as one line to:
      1) <outdir>/results/results.jsonl                     (problem-level canonical file)
      2) <outdir>/runs/<run_id>/results.jsonl               (optional convenience)

    Uses file locks to prevent corruption under parallel execution.
    Idempotent by candidate_id.
    """

    required_params = ["outdir", "problem_id", "run_id", "candidate_id"]
    optional_params = [
        "workdir",
        "result_filename",
        "results_filename",
        "lock_filename",
        "skip_if_exists",
        "append_run_level",  # bool, default True
    ]

    def _already_appended(self, results_path: Path, candidate_id: str) -> bool:
        try:
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = __import__("json").loads(line)
                    except Exception:
                        continue
                    if obj.get("candidate_id") == candidate_id:
                        return True
        except FileNotFoundError:
            return False
        return False

    def _append_one(
        self,
        results_path: Path,
        lock_path: Path,
        record: Dict[str, Any],
        candidate_id: str,
        skip_if_exists: bool,
    ) -> bool:
        lock = FileLock(str(lock_path))
        with lock:
            if skip_if_exists and self._already_appended(results_path, candidate_id):
                return False
            with results_path.open("a", encoding="utf-8") as f:
                f.write(__import__("json").dumps(record) + "\n")
        return True

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        import json

        problem_root = Path(self["outdir"]).expanduser().resolve()
        # problem_id is in params for bookkeeping/validation, but paths use problem_root directly
        problem_id: str = self["problem_id"]
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]

        result_filename = str(self.get("result_filename", "result.json"))
        results_filename = str(self.get("results_filename", "results.jsonl"))
        lock_filename = str(self.get("lock_filename", f"{results_filename}.lock"))
        skip_if_exists = bool(self.get("skip_if_exists", True))
        append_run_level = bool(self.get("append_run_level", True))

        # Derive workdir if not explicitly passed
        if self.get("workdir"):
            workdir = Path(str(self["workdir"])).expanduser().resolve()
        else:
            workdir = _candidate_workdir(problem_root, run_id, candidate_id)

        result_path = workdir / result_filename
        if not result_path.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_path}")

        record = json.loads(result_path.read_text(encoding="utf-8"))

        # Basic sanity check: avoid mixing wrong problem roots
        rec_pid = record.get("problem_id")
        if rec_pid and rec_pid != problem_id:
            raise RuntimeError(
                f"Record problem_id={rec_pid!r} does not match task problem_id={problem_id!r}"
            )

        # Ensure directories exist
        results_dir = _results_dir(problem_root)
        results_dir.mkdir(parents=True, exist_ok=True)

        run_root = _run_root(problem_root, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        # 1) problem-level canonical file
        problem_results_path = results_dir / results_filename
        problem_lock_path = results_dir / lock_filename
        appended_problem = self._append_one(
            problem_results_path, problem_lock_path, record, candidate_id, skip_if_exists
        )

        # 2) per-run convenience file (optional)
        appended_run = None
        run_results_path = run_root / results_filename
        run_lock_path = run_root / lock_filename
        if append_run_level:
            appended_run = self._append_one(
                run_results_path, run_lock_path, record, candidate_id, skip_if_exists
            )

        return FWAction(
            stored_data={
                "candidate_id": candidate_id,
                "problem_id": problem_id,
                "problem_results": str(problem_results_path),
                "run_results": str(run_results_path),
                "appended_problem": appended_problem,
                "appended_run": appended_run,
            }
        )