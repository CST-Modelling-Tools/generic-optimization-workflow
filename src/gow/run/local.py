from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from gow.config import ProblemConfig
from gow.evaluation import evaluate_candidate
from gow.optimizer import make_optimizer


def _default_run_id() -> str:
    return str(uuid.uuid4())


def run_local_optimization(
    problem: ProblemConfig,
    *,
    outdir: str | Path = "results",
    run_id: Optional[str] = None,
) -> Path:
    """
    Run a simple local optimization loop (no FireWorks).

    Layout (A+B):
      <outdir>/<problem_id>/
        results/
          results.jsonl
          summary.json
        runs/
          <run_id>/
            results.jsonl
            summary.json
            c000000/...
            c000001/...
            ...

    Notes:
      - We write both a run-scoped results.jsonl and a canonical problem-scoped results.jsonl.
      - By default, the canonical problem-scoped results.jsonl APPENDS across runs.
        If later you want "canonical == latest run only", we can switch to copying
        run_results_path -> problem_results_path at the end.

    Returns:
      Path to <outdir>/<problem_id>/results/results.jsonl
    """
    outdir = Path(outdir).expanduser().resolve()
    run_id_val = run_id or _default_run_id()

    # Root per problem
    problem_root = outdir / problem.id
    results_root = problem_root / "results"
    runs_root = problem_root / "runs"

    # Root per run
    run_root = runs_root / run_id_val

    results_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    # Canonical (problem-level) + run-scoped files
    problem_results_path = results_root / "results.jsonl"
    run_results_path = run_root / "results.jsonl"

    opt_cfg = problem.optimizer
    optimizer = make_optimizer(opt_cfg.name, seed=opt_cfg.seed)

    direction = problem.objective.direction  # "minimize" or "maximize"
    maximize = direction == "maximize"

    best: Optional[Dict[str, Any]] = None  # objective + candidate_id + params

    n_done = 0
    while n_done < opt_cfg.max_evaluations:
        n_batch = min(opt_cfg.batch_size, opt_cfg.max_evaluations - n_done)
        candidates = optimizer.ask(problem, n_batch)

        fitness_dicts = []
        for i, cand in enumerate(candidates):
            candidate_id = f"c{n_done + i:06d}"
            workdir = run_root / candidate_id

            res = evaluate_candidate(
                problem,
                run_id=run_id_val,
                candidate_id=candidate_id,
                candidate_params=cand,
                workdir=workdir,
            )

            fit = res.fitness.model_dump()
            record = {
                "problem_id": problem.id,
                "run_id": run_id_val,
                "candidate_id": candidate_id,
                "params": {**problem.runtime_params(), **cand},
                "fitness": fit,
                "returncode": res.returncode,
                "wall_time_s": res.wall_time_s,
                "workdir": str(workdir),
                "stdout_path": str(res.stdout_path),
                "stderr_path": str(res.stderr_path),
                "input_path": str(res.input_path),
                "output_path": str(res.output_path),
            }

            line = json.dumps(record)

            # Append to run-scoped file
            with run_results_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Append to problem-scoped canonical file (history across runs)
            with problem_results_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            fitness_dicts.append(fit)

            # Update best according to objective direction
            obj = fit.get("objective", None)
            if obj is not None and fit.get("status") == "ok":
                if best is None:
                    best = {"objective": obj, "candidate_id": candidate_id, "params": record["params"]}
                else:
                    if maximize:
                        if obj > best["objective"]:
                            best = {"objective": obj, "candidate_id": candidate_id, "params": record["params"]}
                    else:
                        if obj < best["objective"]:
                            best = {"objective": obj, "candidate_id": candidate_id, "params": record["params"]}

        optimizer.tell(candidates, fitness_dicts)
        n_done += n_batch

    # Summaries: run-scoped + canonical problem-scoped
    run_summary_path = run_root / "summary.json"
    problem_summary_path = results_root / "summary.json"

    summary = {
        "problem_id": problem.id,
        "run_id": run_id_val,
        "max_evaluations": opt_cfg.max_evaluations,
        "objective": {"direction": direction},
        "best": best,
        "results_file": str(problem_results_path),
        "run_results_file": str(run_results_path),
        "run_root": str(run_root),
    }

    run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    problem_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return problem_results_path