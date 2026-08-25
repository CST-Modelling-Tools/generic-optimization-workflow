from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from gow.candidate_ids import format_attempt_id, format_candidate_id, format_candidate_local_id
from gow.checkpoint import CheckpointStore
from gow.config import ProblemConfig
from gow.evaluation import evaluate_candidate
from gow.optimizer import make_optimizer
from gow.output.jsonl import append_jsonl_line
from gow.fw.tasks import rebuild_problem_results_jsonl, rebuild_run_results_jsonl, verify_run_results_complete
from gow.postprocess import archive_generation_workdirs, finalize_generation
from gow.run.control import acknowledge_pause_request, read_pause_request


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


def _load_existing_run_state(
    results_path: Path,
    *,
    maximize: bool,
) -> tuple[int, Optional[Dict[str, Any]]]:
    """Return persisted result count and best-so-far for a resumed run."""

    if not results_path.is_file():
        return 0, None

    count = 0
    best: Optional[Dict[str, Any]] = None

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                continue

            count += 1

            fit = record.get("fitness")

            if (
                not isinstance(fit, dict)
                or fit.get("status") != "ok"
            ):
                continue

            obj = fit.get("objective")

            try:
                objective = float(obj)
            except (TypeError, ValueError):
                continue

            if (
                best is None
                or (
                    maximize
                    and objective > best["objective"]
                )
                or (
                    not maximize
                    and objective < best["objective"]
                )
            ):
                best = {
                    "objective": objective,
                    "candidate_id": record.get(
                        "candidate_id"
                    ),
                    "candidate_local_id": record.get(
                        "candidate_local_id"
                    ),
                    "attempt_id": record.get(
                        "attempt_id"
                    ),
                    "generation_id": record.get(
                        "generation_id"
                    ),
                    "params": record.get(
                        "params"
                    ),
                }

    return count, best


def run_local_optimization(
    problem: ProblemConfig,
    *,
    outdir: str | Path = "results",
    run_id: Optional[str] = None,
    archive_generations: bool = False,
    delete_archived_workdirs: bool = False,
    resume_from_checkpoint: bool = False,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    if resume_from_checkpoint and run_id is None:
        raise ValueError(
            "Resuming a local run requires an explicit run_id"
        )

    run_id_val = run_id or _default_run_id()

    runs_root = outdir / "runs"
    run_root = runs_root / run_id_val

    outdir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    checkpoint_store = CheckpointStore(run_root)

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

    if resume_from_checkpoint:
        resume_optimizer_aliases = {
            "differential_evolution": {
                "differential_evolution",
                "de",
            },
            "de": {
                "differential_evolution",
                "de",
            },
            "acor": {
                "acor",
            },
        }

        accepted_checkpoint_names = (
            resume_optimizer_aliases.get(
                name_norm
            )
        )

        if accepted_checkpoint_names is None:
            raise ValueError(
                "Checkpoint resume is currently supported "
                "only for Differential Evolution and ACOR"
            )

        loaded_checkpoint = checkpoint_store.load()

        manifest = loaded_checkpoint.manifest

        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                "Unsupported checkpoint schema_version"
            )

        if manifest.get("status") != "paused":
            raise RuntimeError(
                "Only checkpoints with status='paused' "
                "can be resumed"
            )

        if manifest.get("run_id") != run_id_val:
            raise RuntimeError(
                "Checkpoint run_id does not match requested run_id"
            )

        if manifest.get("problem_id") != problem.id:
            raise RuntimeError(
                "Checkpoint problem_id does not match current problem"
            )

        checkpoint_optimizer = str(
            manifest.get("optimizer", "")
        ).lower().strip()

        if (
            checkpoint_optimizer
            not in accepted_checkpoint_names
        ):
            raise RuntimeError(
                "Checkpoint optimizer does not match "
                "the current optimizer configuration"
            )

        checkpoint_max_evaluations = manifest.get(
            "max_evaluations"
        )

        if (
            isinstance(
                checkpoint_max_evaluations,
                bool,
            )
            or checkpoint_max_evaluations
            != opt_cfg.max_evaluations
        ):
            raise RuntimeError(
                "Checkpoint max_evaluations does not match "
                "current problem configuration"
            )

        evaluations_done = manifest.get(
            "evaluations_done"
        )

        if (
            isinstance(evaluations_done, bool)
            or not isinstance(
                evaluations_done,
                int,
            )
        ):
            raise RuntimeError(
                "Checkpoint evaluations_done must be an integer"
            )

        if (
            evaluations_done <= 0
            or evaluations_done
            >= opt_cfg.max_evaluations
        ):
            raise RuntimeError(
                "Paused checkpoint evaluations_done is out of range"
            )

        if (
            evaluations_done
            % opt_cfg.batch_size
            != 0
        ):
            raise RuntimeError(
                "Optimizer checkpoint is not "
                "at a complete-generation boundary"
            )

        expected_generations = (
            evaluations_done
            // opt_cfg.batch_size
        )

        if (
            manifest.get(
                "completed_generations"
            )
            != expected_generations
        ):
            raise RuntimeError(
                "Checkpoint completed_generations is inconsistent "
                "with evaluations_done"
            )

        if (
            manifest.get(
                "next_generation"
            )
            != expected_generations
        ):
            raise RuntimeError(
                "Checkpoint next_generation is inconsistent "
                "with evaluations_done"
            )

        # Rebuild the partial run-level result stream from persisted
        # generation shards before trusting the execution cursor.
        run_results_path = rebuild_run_results_jsonl(
            outdir,
            run_id_val,
        )

        existing_count, best = (
            _load_existing_run_state(
                run_results_path,
                maximize=maximize,
            )
        )

        if existing_count != evaluations_done:
            raise RuntimeError(
                "Checkpoint/result mismatch: "
                f"checkpoint has {evaluations_done} evaluations "
                f"but persisted results contain {existing_count}"
            )

        # The optimizer object is freshly constructed.
        # All algorithmic state must now come from disk.
        #
        # Differential Evolution restores its population and RNG.
        # ACOR restores its archive, metadata, generation and RNG.
        optimizer.load_state_dict(
            loaded_checkpoint.optimizer_state
        )

        n_done = evaluations_done

    while n_done < opt_cfg.max_evaluations:
        n_batch = min(opt_cfg.batch_size, opt_cfg.max_evaluations - n_done)
        generation_id = n_done // opt_cfg.batch_size
        candidates = optimizer.ask(problem, n_batch)

        fitness_dicts = []
        candidate_ids: list[str] = []
        for i, cand in enumerate(candidates):
            candidate_index = n_done + i
            candidate_local_id = format_candidate_local_id(
                generation_id=generation_id,
                candidate_index=candidate_index,
            )
            candidate_id = format_candidate_id(
                generation_id=generation_id,
                candidate_index=candidate_index,
                run_id=run_id_val,
            )
            candidate_ids.append(candidate_id)
            attempt_index = 0
            attempt_id = format_attempt_id(candidate_id, attempt_index)

            workdir = run_root / candidate_id
            workdir.mkdir(parents=True, exist_ok=True)

            res = evaluate_candidate(
                problem,
                run_id=run_id_val,
                candidate_id=candidate_id,
                candidate_local_id=candidate_local_id,
                attempt_id=attempt_id,
                candidate_params=cand,
                workdir=workdir,
            )

            fit = res.fitness.model_dump()
            record = {
                "problem_id": problem.id,
                "run_id": run_id_val,
                "generation_id": generation_id,
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "candidate_local_id": candidate_local_id,
                "attempt_id": attempt_id,
                "attempt_index": attempt_index,
                "params": {**problem.runtime_params(), **cand},
                "fitness": fit,
                "failure_kind": fit.get("failure_kind"),
                "returncode": res.returncode,
                "wall_time_s": res.wall_time_s,
                "started_at": res.started_at,
                "finished_at": res.finished_at,
                "evaluator": res.evaluator,
                "workdir": str(workdir),
                "stdout_path": str(res.stdout_path),
                "stderr_path": str(res.stderr_path),
                "input_path": str(res.input_path),
                "output_path": str(res.output_path),
            }

            (workdir / "result.json").write_text(
                json.dumps(record, indent=2, sort_keys=True),
                encoding="utf-8",
            )


            fitness_dicts.append(fit)

            obj = fit.get("objective", None)
            if obj is not None and fit.get("status") == "ok":
                if best is None or (maximize and obj > best["objective"]) or (not maximize and obj < best["objective"]):
                    best = {
                        "objective": obj,
                        "candidate_id": candidate_id,
                        "candidate_local_id": candidate_local_id,
                        "attempt_id": attempt_id,
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

        finalize_generation(
            outdir=outdir,
            run_id=run_id_val,
            problem_id=problem.id,
            generation_id=generation_id,
            candidate_ids=candidate_ids,
            max_evaluations=opt_cfg.max_evaluations,
            direction=direction,
            completed_generations=(n_done + n_batch + opt_cfg.batch_size - 1) // opt_cfg.batch_size,
            final=(n_done + n_batch) >= opt_cfg.max_evaluations,
        )
        if archive_generations:
            archive_generation_workdirs(
                outdir=outdir,
                run_id=run_id_val,
                generation_id=generation_id,
                candidate_ids=candidate_ids,
                delete_source=delete_archived_workdirs,
            )

        n_done += n_batch

        completed_generations = (
            n_done + opt_cfg.batch_size - 1
        ) // opt_cfg.batch_size

        # Pause requests are honoured only at a completed-generation boundary.
        # Completion takes precedence if max_evaluations has already been reached.
        pause_request = None

        if n_done < opt_cfg.max_evaluations:
            pause_request = read_pause_request(
                run_root
            )

        try:
            optimizer_state = optimizer.state_dict()
        except NotImplementedError:
            optimizer_state = None

        if pause_request is not None and optimizer_state is None:
            raise RuntimeError(
                "Pause requested, but optimizer "
                f"{opt_cfg.name!r} does not support checkpoint persistence."
            )

        checkpoint_status = (
            "completed"
            if n_done >= opt_cfg.max_evaluations
            else (
                "paused"
                if pause_request is not None
                else "running"
            )
        )

        if optimizer_state is not None:
            checkpoint_store.save(
                manifest={
                    "schema_version": 1,
                    "run_id": run_id_val,
                    "problem_id": problem.id,
                    "status": checkpoint_status,
                    "optimizer": str(opt_cfg.name),
                    "evaluations_done": n_done,
                    "completed_generations": completed_generations,
                    "next_generation": completed_generations,
                    "max_evaluations": opt_cfg.max_evaluations,
                },
                optimizer_state=optimizer_state,
            )

        if pause_request is not None:
            # Materialize a consistent partial run-level results.jsonl from
            # the generation shards already finalized on disk.
            run_results_path = rebuild_run_results_jsonl(
                outdir,
                run_id_val,
            )

            # The acknowledgement is emitted only after both the optimizer
            # checkpoint and partial run results have been persisted.
            acknowledge_pause_request(
                run_root,
                pause_request,
                evaluations_done=n_done,
                completed_generations=completed_generations,
            )

            # A paused run is intentionally incomplete.
            # Do not execute final completeness verification or global rebuild.
            return run_results_path

    ok, actual = verify_run_results_complete(outdir, run_id_val, opt_cfg.max_evaluations)
    if not ok:
        raise RuntimeError(f"Run {run_id_val} incomplete: expected {opt_cfg.max_evaluations} results, found {actual}")
    run_results_path = rebuild_run_results_jsonl(outdir, run_id_val)
    problem_results_path = rebuild_problem_results_jsonl(outdir)

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

def resume_local_optimization(
    problem: ProblemConfig,
    *,
    outdir: str | Path = "results",
    run_id: str,
    archive_generations: bool = False,
    delete_archived_workdirs: bool = False,
) -> Path:
    """Resume a paused local optimization run from its checkpoint."""

    return run_local_optimization(
        problem,
        outdir=outdir,
        run_id=run_id,
        archive_generations=archive_generations,
        delete_archived_workdirs=delete_archived_workdirs,
        resume_from_checkpoint=True,
    )
