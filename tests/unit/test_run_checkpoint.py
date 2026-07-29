from __future__ import annotations

from pathlib import Path

from gow.checkpoint import CheckpointStore
from gow.config.models import ProblemConfig
from gow.run.local import run_local_optimization


def _build_problem() -> ProblemConfig:
    evaluator_path = (
        Path.cwd()
        / "tests"
        / "toy_eval.py"
    ).resolve()

    return ProblemConfig.model_validate(
        {
            "id": "run-checkpoint-test",
            "objective": {
                "direction": "minimize",
            },
            "parameters": {
                "x": {
                    "type": "real",
                    "value": 0.0,
                    "bounds": [-1.0, 1.0],
                },
                "y": {
                    "type": "real",
                    "value": 0.0,
                    "bounds": [-1.0, 1.0],
                },
            },
            "evaluator": {
                "command": [
                    "{python}",
                    str(evaluator_path),
                ],
                "timeout_s": 30,
            },
            "optimizer": {
                "name": "random_search",
                "seed": 123,
                "max_evaluations": 4,
                "batch_size": 2,
            },
        }
    )


def test_local_run_writes_completed_checkpoint(
    tmp_path: Path,
) -> None:
    problem = _build_problem()
    run_id = "checkpoint-run"

    run_local_optimization(
        problem,
        outdir=tmp_path,
        run_id=run_id,
    )

    run_dir = (
        tmp_path
        / "runs"
        / run_id
    )

    checkpoint = CheckpointStore(run_dir).load()

    assert checkpoint.manifest == {
        "schema_version": 1,
        "run_id": run_id,
        "problem_id": problem.id,
        "status": "completed",
        "optimizer": "random_search",
        "evaluations_done": 4,
        "completed_generations": 2,
        "next_generation": 2,
        "max_evaluations": 4,
    }

    assert checkpoint.optimizer_state["schema_version"] == 1
    assert "rng_state" in checkpoint.optimizer_state