from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gow.config import load_problem_config
from gow.evaluation import evaluate_candidate


def _to_jsonable(obj: Any) -> Any:
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


def _jsonl_dumps(obj: Dict[str, Any]) -> str:
    """
    Deterministic JSON for JSONL:
    - stable key ordering across platforms/backends
    - compact representation
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


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


# ---------------------------------------------------------------------
# Path helpers (flattened)
# ---------------------------------------------------------------------

def _runs_dir(outdir: Path) -> Path:
    return outdir / "runs"


def _run_root(outdir: Path, run_id: str) -> Path:
    return _runs_dir(outdir) / run_id


def _candidate_workdir(outdir: Path, run_id: str, candidate_id: str) -> Path:
    return _run_root(outdir, run_id) / candidate_id


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _unique_key(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Idempotency key for results.jsonl lines.
    Prefer (run_id, candidate_id). If missing, returns (None, candidate_id).
    """
    rid = record.get("run_id")
    cid = record.get("candidate_id")
    rid = str(rid) if rid is not None else None
    cid = str(cid) if cid is not None else None
    return rid, cid


# ---------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------

@explicit_serialize
class EvaluateCandidateTask(FiretaskBase):
    """
    Evaluate one candidate and write:
      <outdir>/runs/<run_id>/<candidate_id>/{input.json,output.json,stdout.txt,stderr.txt,result.json}

    `outdir` is the flattened problem root (same as CLI --outdir).
    """
    required_params = ["problem_config", "run_id", "candidate_id", "candidate_params", "outdir"]
    optional_params = ["context_override", "generation_id", "candidate_index"]

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        problem_config = Path(self["problem_config"]).expanduser()
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]
        candidate_params: Dict[str, Any] = dict(self["candidate_params"])
        outdir = Path(self["outdir"]).expanduser().resolve()
        context_override: Optional[Dict[str, Any]] = self.get("context_override")

        generation_id = self.get("generation_id")
        candidate_index = self.get("candidate_index")

        problem = load_problem_config(problem_config)

        workdir = _candidate_workdir(outdir, run_id, candidate_id)
        workdir.mkdir(parents=True, exist_ok=True)

        res = evaluate_candidate(
            problem,
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_params=candidate_params,
            workdir=workdir,
            context_override=context_override,
        )

        stored: Dict[str, Any] = {
            "problem_id": problem.id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            # include generation_id in record if provided (no folder-per-generation)
            "generation_id": generation_id,
            "candidate_index": candidate_index,
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
      1) <outdir>/results.jsonl
      2) <outdir>/runs/<run_id>/results.jsonl  (optional)

    Uses file locks to prevent corruption under parallel execution.
    Idempotent by (run_id, candidate_id).

    HARD SAFETY:
      - will refuse to append if result.json's candidate_id != task candidate_id
      - will refuse to append if result.json's run_id != task run_id
    """
    required_params = ["outdir", "problem_id", "run_id", "candidate_id"]
    optional_params = [
        "workdir",
        "result_filename",
        "results_filename",
        "lock_filename",
        "skip_if_exists",
        "append_run_level",
        "generation_id",
        "candidate_index",
    ]

    def _already_appended(self, results_path: Path, *, run_id: str, candidate_id: str) -> bool:
        try:
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rid, cid = _unique_key(obj)
                    if rid == run_id and cid == candidate_id:
                        return True
        except FileNotFoundError:
            return False
        return False

    def _append_one(
        self,
        results_path: Path,
        lock_path: Path,
        record: Dict[str, Any],
        *,
        run_id: str,
        candidate_id: str,
        skip_if_exists: bool,
    ) -> bool:
        lock = FileLock(str(lock_path))
        with lock:
            if skip_if_exists and self._already_appended(results_path, run_id=run_id, candidate_id=candidate_id):
                return False
            with results_path.open("a", encoding="utf-8") as f:
                f.write(_jsonl_dumps(record) + "\n")
        return True

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        outdir = Path(self["outdir"]).expanduser().resolve()
        problem_id: str = self["problem_id"]
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]

        result_filename = str(self.get("result_filename", "result.json"))
        results_filename = str(self.get("results_filename", "results.jsonl"))
        lock_filename = str(self.get("lock_filename", f"{results_filename}.lock"))
        skip_if_exists = bool(self.get("skip_if_exists", True))
        append_run_level = bool(self.get("append_run_level", True))

        if self.get("workdir"):
            workdir = Path(str(self["workdir"])).expanduser().resolve()
        else:
            workdir = _candidate_workdir(outdir, run_id, candidate_id)

        result_path = workdir / result_filename
        if not result_path.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_path}")

        record = json.loads(result_path.read_text(encoding="utf-8"))

        # --- HARD CONSISTENCY CHECKS (this prevents the mixed naming issue) ---
        rec_pid = record.get("problem_id")
        if rec_pid and rec_pid != problem_id:
            raise RuntimeError(
                f"Record problem_id={rec_pid!r} does not match task problem_id={problem_id!r}"
            )

        rec_rid = record.get("run_id")
        if rec_rid is not None and str(rec_rid) != str(run_id):
            raise RuntimeError(
                f"Record run_id={rec_rid!r} does not match task run_id={run_id!r} "
                f"(workdir={workdir})"
            )

        rec_cid = record.get("candidate_id")
        if rec_cid is not None and str(rec_cid) != str(candidate_id):
            raise RuntimeError(
                f"Record candidate_id={rec_cid!r} does not match task candidate_id={candidate_id!r} "
                f"(workdir={workdir})"
            )

        # If workflow passed generation metadata, ensure it is present (doesn't change folder layout)
        if self.get("generation_id") is not None and record.get("generation_id") is None:
            record["generation_id"] = self.get("generation_id")
        if self.get("candidate_index") is not None and record.get("candidate_index") is None:
            record["candidate_index"] = self.get("candidate_index")

        outdir.mkdir(parents=True, exist_ok=True)

        run_root = _run_root(outdir, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        # 1) problem-level canonical file (flat)
        problem_results_path = outdir / results_filename
        problem_lock_path = outdir / lock_filename
        appended_problem = self._append_one(
            problem_results_path,
            problem_lock_path,
            record,
            run_id=run_id,
            candidate_id=candidate_id,
            skip_if_exists=skip_if_exists,
        )

        # 2) per-run convenience file (optional)
        run_results_path = run_root / results_filename
        run_lock_path = run_root / lock_filename
        appended_run = None
        if append_run_level:
            appended_run = self._append_one(
                run_results_path,
                run_lock_path,
                record,
                run_id=run_id,
                candidate_id=candidate_id,
                skip_if_exists=skip_if_exists,
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