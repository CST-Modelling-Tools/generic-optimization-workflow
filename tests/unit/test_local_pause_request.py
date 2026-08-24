from __future__ import annotations

import json
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.optimizer.differential_evolution import (
    DifferentialEvolutionOptimizer,
)
from gow.run import run_local_optimization
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


POPULATION_SIZE = 4
MAX_EVALUATIONS = 12


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


def test_de_pause_request_stops_at_safe_generation_boundary(
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
    Path("input.json").read_text(encoding="utf-8")
)

x = float(payload["params"]["x"])
y = float(payload["params"]["y"])

objective = x * x + y * y

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
        "id": "pause-de-test",

        "objective": {
            "direction": "minimize",
        },

        "parameters": {
            "x": {
                "type": "real",
                "value": 0.0,
                "bounds": [-2.0, 2.0],
            },

            "y": {
                "type": "real",
                "value": 0.0,
                "bounds": [-2.0, 2.0],
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
            "seed": 24680,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": POPULATION_SIZE,

            "settings": {
                "mutation_factor": 0.7,
                "crossover_rate": 0.9,
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

    outdir = tmp_path / "out"
    run_id = "pause-de-run"

    run_root = (
        outdir
        / "runs"
        / run_id
    )

    # ============================================================
    # Simulate an external Monitor pause request
    # ============================================================

    request_path = pause_request_path(
        run_root
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
                "request_id": "pause-request-0001",
                "requested_at": "2026-08-24T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    returned = run_local_optimization(
        problem,
        outdir=outdir,
        run_id=run_id,
    )

    # ============================================================
    # Partial run results must exist and contain one whole batch.
    # ============================================================

    results_path = (
        run_root
        / "results.jsonl"
    )

    assert returned == results_path
    assert results_path.is_file()

    rows = _read_jsonl(
        results_path
    )

    assert len(rows) == POPULATION_SIZE

    assert {
        row["generation_id"]
        for row in rows
    } == {0}

    assert [
        row["candidate_index"]
        for row in rows
    ] == list(
        range(POPULATION_SIZE)
    )

    # ============================================================
    # Generation shard must exist.
    # ============================================================

    generation_path = (
        run_root
        / "generations"
        / "g000000.jsonl"
    )

    assert generation_path.is_file()

    generation_rows = _read_jsonl(
        generation_path
    )

    assert len(generation_rows) == POPULATION_SIZE

    # ============================================================
    # Request consumed and acknowledgement persisted.
    # ============================================================

    assert not request_path.exists()

    ack_path = pause_ack_path(
        run_root
    )

    assert ack_path.is_file()

    ack = json.loads(
        ack_path.read_text(
            encoding="utf-8",
        )
    )

    assert ack["schema_version"] == 1
    assert ack["action"] == "pause"
    assert ack["request_id"] == "pause-request-0001"
    assert ack["status"] == "paused"

    assert (
        ack["evaluations_done"]
        == POPULATION_SIZE
    )

    assert (
        ack["completed_generations"]
        == 1
    )

    assert "acknowledged_at" in ack

    # ============================================================
    # Optimizer checkpoint must be PAUSED and restorable.
    # ============================================================

    checkpoint = CheckpointStore(
        run_root
    ).load()

    manifest = checkpoint.manifest
    state = checkpoint.optimizer_state

    assert manifest["schema_version"] == 1
    assert manifest["status"] == "paused"

    assert (
        manifest["evaluations_done"]
        == POPULATION_SIZE
    )

    assert (
        manifest["completed_generations"]
        == 1
    )

    assert (
        manifest["next_generation"]
        == 1
    )

    assert (
        manifest["max_evaluations"]
        == MAX_EVALUATIONS
    )

    assert state["optimizer"] == "differential_evolution"
    assert state["generation"] == 1
    assert state["last_targets"] == []

    restored = DifferentialEvolutionOptimizer(
        population_size=POPULATION_SIZE,
        mutation_factor=0.7,
        crossover_rate=0.9,
        max_generations=20,

        # Deliberately different seed.
        seed=999999,
    )

    restored.load_state_dict(
        state
    )

    assert restored.state_dict() == state

    # ============================================================
    # summary.json IS expected during a paused run.
    #
    # finalize_generation() is live/post-generation telemetry.
    # "finalized": false distinguishes it from a completed run.
    # ============================================================

    run_summary_path = (
        run_root
        / "summary.json"
    )

    problem_summary_path = (
        outdir
        / "summary.json"
    )

    assert run_summary_path.is_file()
    assert problem_summary_path.is_file()

    run_summary = json.loads(
        run_summary_path.read_text(
            encoding="utf-8",
        )
    )

    problem_summary = json.loads(
        problem_summary_path.read_text(
            encoding="utf-8",
        )
    )

    for summary in (
        run_summary,
        problem_summary,
    ):
        assert summary["run_id"] == run_id
        assert summary["problem_id"] == problem.id

        assert summary["finalized"] is False

        assert (
            summary["evaluations_done"]
            == POPULATION_SIZE
        )

        assert (
            summary["completed_generations"]
            == 1
        )

        assert (
            summary["max_evaluations"]
            == MAX_EVALUATIONS
        )
