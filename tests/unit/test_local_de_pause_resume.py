from __future__ import annotations

import json
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.run import (
    resume_local_optimization,
    run_local_optimization,
)
from gow.run.control import pause_request_path


POPULATION_SIZE = 4
MAX_EVALUATIONS = 12
RUN_ID = "de-resume-determinism"


def _read_jsonl(
    path: Path,
) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def _trajectory(
    rows: list[dict],
) -> list[dict]:
    """Comparable mathematical trajectory, excluding runtime metadata."""

    return [
        {
            "generation_id": row["generation_id"],
            "candidate_index": row["candidate_index"],
            "candidate_id": row["candidate_id"],
            "candidate_local_id": row["candidate_local_id"],
            "attempt_id": row["attempt_id"],
            "params": row["params"],
            "fitness": row["fitness"],
        }
        for row in rows
    ]


def test_de_pause_resume_matches_continuous_run_exactly(
    tmp_path: Path,
) -> None:

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    evaluator = project_dir / "eval.py"

    evaluator.write_text(
        '''
import json
from pathlib import Path

payload = json.loads(
    Path("input.json").read_text(
        encoding="utf-8"
    )
)

x = float(payload["params"]["x"])
y = float(payload["params"]["y"])

objective = (
    x * x
    + 3.0 * y * y
    + 0.25 * x * y
)

Path("output.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "metrics": {
                "objective": objective,
            },
            "objective": objective,
            "constraints": {},
            "artifacts": {},
            "error": None,
        }
    ),
    encoding="utf-8",
)
'''.lstrip(),
        encoding="utf-8",
    )

    config = {
        "id": "de-resume-determinism-test",

        "objective": {
            "direction": "minimize",
        },

        "parameters": {
            "x": {
                "type": "real",
                "value": 0.0,
                "bounds": [-4.0, 4.0],
            },

            "y": {
                "type": "real",
                "value": 0.0,
                "bounds": [-3.0, 3.0],
            },
        },

        "evaluator": {
            "command": [
                "{python}",
                str(evaluator.resolve()),
            ],
            "timeout_s": 30,
        },

        "optimizer": {
            "name": "differential_evolution",
            "seed": 13579,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": POPULATION_SIZE,

            "settings": {
                "mutation_factor": 0.72,
                "crossover_rate": 0.83,
                "max_generations": 20,
            },
        },
    }

    config_path = project_dir / "problem.yaml"

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    problem = load_problem_config(
        config_path
    )

    # ============================================================
    # A. CONTINUOUS RUN
    # ============================================================

    continuous_out = (
        tmp_path
        / "continuous"
    )

    run_local_optimization(
        problem,
        outdir=continuous_out,
        run_id=RUN_ID,
    )

    continuous_run_root = (
        continuous_out
        / "runs"
        / RUN_ID
    )

    continuous_rows = _read_jsonl(
        continuous_run_root
        / "results.jsonl"
    )

    assert len(continuous_rows) == MAX_EVALUATIONS

    continuous_checkpoint = CheckpointStore(
        continuous_run_root
    ).load()

    assert (
        continuous_checkpoint.manifest["status"]
        == "completed"
    )

    # ============================================================
    # B. PAUSE AFTER GENERATION ZERO
    # ============================================================

    resumed_out = (
        tmp_path
        / "paused-resumed"
    )

    resumed_run_root = (
        resumed_out
        / "runs"
        / RUN_ID
    )

    request_path = pause_request_path(
        resumed_run_root
    )

    request_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "pause",
                "request_id": "pause-before-resume-test",
            }
        ),
        encoding="utf-8",
    )

    paused_result = run_local_optimization(
        problem,
        outdir=resumed_out,
        run_id=RUN_ID,
    )

    assert paused_result == (
        resumed_run_root
        / "results.jsonl"
    )

    paused_rows = _read_jsonl(
        paused_result
    )

    assert len(paused_rows) == POPULATION_SIZE

    paused_checkpoint = CheckpointStore(
        resumed_run_root
    ).load()

    assert (
        paused_checkpoint.manifest["status"]
        == "paused"
    )

    assert (
        paused_checkpoint.manifest["evaluations_done"]
        == POPULATION_SIZE
    )

    assert (
        paused_checkpoint.manifest["next_generation"]
        == 1
    )

    # ============================================================
    # C. NEW RUNNER INVOCATION -> LOAD FROM DISK -> RESUME
    # ============================================================

    resume_local_optimization(
        problem,
        outdir=resumed_out,
        run_id=RUN_ID,
    )

    resumed_rows = _read_jsonl(
        resumed_run_root
        / "results.jsonl"
    )

    assert len(resumed_rows) == MAX_EVALUATIONS

    # No duplicate/reused indexes after resume.
    assert [
        row["candidate_index"]
        for row in resumed_rows
    ] == list(
        range(MAX_EVALUATIONS)
    )

    assert [
        row["generation_id"]
        for row in resumed_rows
    ] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    # ============================================================
    # D. MATHEMATICAL TRAJECTORY MUST BE IDENTICAL
    # ============================================================

    assert (
        _trajectory(resumed_rows)
        == _trajectory(continuous_rows)
    )

    # ============================================================
    # E. FINAL OPTIMIZER STATE MUST ALSO BE IDENTICAL
    # ============================================================

    resumed_checkpoint = CheckpointStore(
        resumed_run_root
    ).load()

    assert (
        resumed_checkpoint.manifest["status"]
        == "completed"
    )

    assert (
        resumed_checkpoint.manifest["evaluations_done"]
        == MAX_EVALUATIONS
    )

    assert (
        resumed_checkpoint.manifest["completed_generations"]
        == 3
    )

    assert (
        resumed_checkpoint.optimizer_state
        == continuous_checkpoint.optimizer_state
    )

    # ============================================================
    # F. FINAL BEST MUST BE IDENTICAL TOO
    # ============================================================

    continuous_summary = json.loads(
        (
            continuous_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    resumed_summary = json.loads(
        (
            resumed_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        resumed_summary["best"]
        == continuous_summary["best"]
    )
