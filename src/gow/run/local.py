from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from gow.candidate_ids import format_candidate_id
from gow.config import ProblemConfig
from gow.evaluation import evaluate_candidate
from gow.optimizer import make_optimizer
from gow.output.jsonl import append_jsonl_line


def _optimizer_kwargs(problem: ProblemConfig) -> Dict[str, Any]:
    """Convert ProblemConfig.optimizer into kwargs for make_optimizer().

    We intentionally keep this logic local (rather than in config parsing) because
    different runners/backends may want slightly different defaults.

    Rules:
      - Start from optimizer.settings (must be a dict).
      - Merge in any extra fields on optimizer config (excluding standard keys).
      - For Differential Evolution, default population_size to batch_size.
      - Always provide seed (defaulting from optimizer.seed).
    """

    opt_cfg = problem.optimizer

    # Extract a plain dict representation of the optimizer config.
    if hasattr(opt_cfg, "model_dump"):
        data: Dict[str, Any] = opt_cfg.model_dump()
    elif hasattr(opt_cfg, "dict"):
        data = opt_cfg.dict()  # type: ignore[assignment]
    else:
        data = dict(getattr(opt_cfg, "__dict__", {}) or {})

    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"optimizer.settings must be a dict, got {type(settings)}")

    # Remove known top-level keys that are not constructor kwargs.
    for k in ("name", "seed", "max_evaluations", "batch_size", "settings"):
        data.pop(k, None)

    # Merge: extra top-level keys (rare) + settings.
    out: Dict[str, Any] = {k: v for k, v in data.items() if not str(k).startswith("_")}
    out.update(settings)
    out = {k: v for k, v in out.items() if not str(k).startswith("_")}

    # Common convenience: if population_size not set for DE, default it to batch_size.
    name_norm = str(opt_cfg.name).lower().strip()
    if name_norm in {"differential_evolution", "de"}:
        out.setdefault("population_size", opt_cfg.batch_size)

    # Always pass seed through kwargs (call sites can still pop it if needed).
    out.setdefault("seed", opt_cfg.seed)
    return out


def _default_run_id() -> str:
    return str(uuid.uuid4())


def run_local_optimization(
    problem: ProblemConfig,
    *,
    outdir: str | Path = "results",
    run_id: Optional[str] = None,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    run_id_val = run_id or _default_run_id()

    runs_root = outdir / "runs"
    run_root = runs_root / run_id_val

    outdir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    problem_results_path = outdir / "results.jsonl"
    run_results_path = run_root / "results.jsonl"

    opt_cfg = problem.optimizer
    opt_kwargs = _optimizer_kwargs(problem)

    name_norm = str(opt_cfg.name).lower().strip()
    if name_norm in {"differential_evolution", "de"}:
        if opt_cfg.max_evaluations % opt_cfg.batch_size != 0:
            raise ValueError(
                "Differential Evolution requires max_evaluations to be a multiple of batch_size "
                f"(got {opt_cfg.max_evaluations}, batch_size={opt_cfg.batch_size})"
            )

    # Avoid passing seed twice if make_optimizer also takes seed= explicitly.
    seed = opt_kwargs.pop("seed", opt_cfg.seed)
    optimizer = make_optimizer(opt_cfg.name, seed=seed, **opt_kwargs)

    direction = problem.objective.direction
    maximize = direction == "maximize"

    best: Optional[Dict[str, Any]] = None

    n_done = 0
    while n_done < opt_cfg.max_evaluations:
        n_batch = min(opt_cfg.batch_size, opt_cfg.max_evaluations - n_done)
        generation_id = n_done // opt_cfg.batch_size
        candidates = optimizer.ask(problem, n_batch)

        fitness_dicts = []
        for i, cand in enumerate(candidates):
            candidate_index = n_done + i
            candidate_id = format_candidate_id(
                generation_id=generation_id,
                candidate_index=candidate_index,
            )

            workdir = run_root / candidate_id
            workdir.mkdir(parents=True, exist_ok=True)

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
                "generation_id": generation_id,
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

            append_jsonl_line(run_results_path, record)
            append_jsonl_line(problem_results_path, record)

            fitness_dicts.append(fit)

            obj = fit.get("objective", None)
            if obj is not None and fit.get("status") == "ok":
                if best is None or (maximize and obj > best["objective"]) or (not maximize and obj < best["objective"]):
                    best = {
                        "objective": obj,
                        "candidate_id": candidate_id,
                        "generation_id": generation_id,
                        "params": record["params"],
                    }

        optimizer.tell(candidates, fitness_dicts)

        # -----------------------------
        # Optional diagnostics logging
        # -----------------------------
        if hasattr(optimizer, "_n_status_failed"):
            n_failed = optimizer._n_status_failed
            n_missing = optimizer._n_missing_score
            n_non_numeric = optimizer._n_non_numeric
            n_non_finite = optimizer._n_non_finite

            if n_failed or n_missing or n_non_numeric or n_non_finite:
                print(
                    f"[DE diagnostics | gen={generation_id}] "
                    f"failed={n_failed}, "
                    f"missing_score={n_missing}, "
                    f"non_numeric={n_non_numeric}, "
                    f"non_finite={n_non_finite}"
                )

        n_done += n_batch

    summary = {
        "problem_id": problem.id,
        "run_id": run_id_val,
        "max_evaluations": opt_cfg.max_evaluations,
        "objective": {"direction": direction},
        "best": best,
        "results_file": str(problem_results_path),
        "run_results_file": str(run_results_path),
        "run_root": str(run_root),
        "outdir": str(outdir),
    }

    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return problem_results_path