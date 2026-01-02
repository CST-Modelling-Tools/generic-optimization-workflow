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
    outdir: str | Path = "runs",
    run_id: Optional[str] = None,
) -> Path:
    """
    Run a simple local optimization loop (no FireWorks):
    - propose candidates in batches
    - evaluate each candidate using the configured external evaluator
    - append results to results.jsonl
    Returns the path to results.jsonl.
    """
    outdir = Path(outdir)
    run_id = run_id or _default_run_id()

    run_root = outdir / problem.id / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    results_path = run_root / "results.jsonl"

    opt_cfg = problem.optimizer
    optimizer = make_optimizer(opt_cfg.name, seed=opt_cfg.seed)

    direction = problem.objective.direction  # "minimize" or "maximize"
    maximize = direction == "maximize"

    best = None  # store dict with objective + candidate_id + params

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
                run_id=run_id,
                candidate_id=candidate_id,
                candidate_params=cand,
                workdir=workdir,
            )

            fit = res.fitness.model_dump()
            record = {
                "problem_id": problem.id,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "params": {**problem.runtime_params(), **cand},
                "fitness": fit,
                "returncode": res.returncode,
                "wall_time_s": res.wall_time_s,
                "workdir": str(workdir),
            }

            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            fitness_dicts.append(fit)

            # Update best according to objective direction
            obj = fit.get("objective", None)
            if obj is not None and fit.get("status") == "ok":
                if best is None:
                    best = {
                        "objective": obj,
                        "candidate_id": candidate_id,
                        "params": record["params"],
                    }
                else:
                    if maximize:
                        if obj > best["objective"]:
                            best = {
                                "objective": obj,
                                "candidate_id": candidate_id,
                                "params": record["params"],
                            }
                    else:
                        if obj < best["objective"]:
                            best = {
                                "objective": obj,
                                "candidate_id": candidate_id,
                                "params": record["params"],
                            }

        optimizer.tell(candidates, fitness_dicts)
        n_done += n_batch

    # Write a summary
    summary_path = run_root / "summary.json"
    summary = {
        "problem_id": problem.id,
        "run_id": run_id,
        "max_evaluations": opt_cfg.max_evaluations,
        "objective": {"direction": direction},
        "best": best,
        "results_file": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return results_path