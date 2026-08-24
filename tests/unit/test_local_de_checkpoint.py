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


POPULATION_SIZE = 4
MAX_EVALUATIONS = 8


def _write_text(path: Path, text: str) -> None:
    path.write_text(
        text,
        encoding="utf-8",
    )


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_local_runner_creates_restorable_de_checkpoint(
    tmp_path: Path,
) -> None:
    """Run a real local DE campaign and validate its final checkpoint."""

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # ------------------------------------------------------------
    # External evaluator used by GOW
    # ------------------------------------------------------------

    evaluator_path = project_dir / "de_checkpoint_eval.py"

    _write_text(
        evaluator_path,
        r'''
import json
from pathlib import Path


input_payload = json.loads(
    Path("input.json").read_text(
        encoding="utf-8",
    )
)

params = input_payload.get("params", {})

x = float(params["x"])
y = float(params["y"])

objective = (
    x * x
    + 2.0 * y * y
)

output_payload = {
    "status": "ok",
    "metrics": {
        "objective": objective,
    },
    "objective": objective,
    "constraints": {},
    "artifacts": {},
    "error": None,
}

Path("output.json").write_text(
    json.dumps(output_payload),
    encoding="utf-8",
)
'''.lstrip(),
    )

    # ------------------------------------------------------------
    # GOW problem
    # ------------------------------------------------------------

    config = {
        "id": "local-de-checkpoint-test",

        "objective": {
            "direction": "minimize",
        },

        "parameters": {
            "x": {
                "type": "real",
                "value": 0.0,
                "bounds": [-2.0, 2.0],
                "optimizable": True,
            },

            "y": {
                "type": "real",
                "value": 0.0,
                "bounds": [-3.0, 3.0],
                "optimizable": True,
            },
        },

        "evaluator": {
            "command": [
                "{python}",
                str(evaluator_path.resolve()),
            ],
            "timeout_s": 30,
        },

        "optimizer": {
            "name": "differential_evolution",
            "seed": 424242,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": POPULATION_SIZE,

            "settings": {
                "mutation_factor": 0.70,
                "crossover_rate": 0.85,
                "max_generations": 20,
            },
        },
    }

    config_path = project_dir / "problem.yaml"
    _write_yaml(
        config_path,
        config,
    )

    problem = load_problem_config(config_path)

    # ------------------------------------------------------------
    # Real local GOW execution
    # ------------------------------------------------------------

    outdir = tmp_path / "out"
    run_id = "de-checkpoint-runner-test"

    result_path = run_local_optimization(
        problem,
        outdir=outdir,
        run_id=run_id,
    )

    assert result_path.exists()

    run_root = (
        outdir
        / "runs"
        / run_id
    )

    assert run_root.is_dir()

    # ------------------------------------------------------------
    # Checkpoint artifacts
    # ------------------------------------------------------------

    checkpoint_dir = (
        run_root
        / "checkpoint"
    )

    manifest_path = (
        checkpoint_dir
        / "manifest.json"
    )

    optimizer_state_path = (
        checkpoint_dir
        / "optimizer_state.bin"
    )

    checksum_path = (
        checkpoint_dir
        / "checkpoint.sha256"
    )

    assert checkpoint_dir.is_dir()

    assert manifest_path.is_file()
    assert optimizer_state_path.is_file()
    assert checksum_path.is_file()

    # ------------------------------------------------------------
    # Load through the real CheckpointStore
    # ------------------------------------------------------------

    store = CheckpointStore(run_root)

    loaded = store.load()

    manifest = loaded.manifest
    state = loaded.optimizer_state

    # ------------------------------------------------------------
    # Validate run-level checkpoint state
    # ------------------------------------------------------------

    assert manifest["schema_version"] == 1

    assert manifest["run_id"] == run_id
    assert manifest["problem_id"] == problem.id

    assert manifest["optimizer"] == "differential_evolution"

    assert manifest["status"] == "completed"

    assert manifest["evaluations_done"] == MAX_EVALUATIONS
    assert manifest["max_evaluations"] == MAX_EVALUATIONS

    assert manifest["completed_generations"] == 2
    assert manifest["next_generation"] == 2

    # ------------------------------------------------------------
    # Validate optimizer-level checkpoint state
    # ------------------------------------------------------------

    assert state["schema_version"] == 1
    assert state["optimizer"] == "differential_evolution"

    assert state["initialized"] is True
    assert state["generation"] == 2

    assert state["last_targets"] == []

    assert len(state["population"]) == POPULATION_SIZE
    assert len(state["fitness"]) == POPULATION_SIZE

    assert state["configuration"] == {
        "population_size": POPULATION_SIZE,
        "mutation_factor": 0.70,
        "crossover_rate": 0.85,
        "max_generations": 20,
    }

    assert state["direction"] == "minimize"

    assert state["param_names"] == [
        "x",
        "y",
    ]

    assert state["rng_state"] is not None

    # ------------------------------------------------------------
    # Restore into a completely new DE instance
    # ------------------------------------------------------------

    restored = DifferentialEvolutionOptimizer(
        population_size=POPULATION_SIZE,
        mutation_factor=0.70,
        crossover_rate=0.85,
        max_generations=20,

        # Deliberately different seed.
        # The checkpoint must replace its RNG state.
        seed=999999,
    )

    restored.load_state_dict(state)

    restored_state = restored.state_dict()

    # Exact round-trip of the optimizer state.
    assert restored_state == state

    # ------------------------------------------------------------
    # Results sanity check
    # ------------------------------------------------------------

    run_results_path = (
        run_root
        / "results.jsonl"
    )

    assert run_results_path.is_file()

    rows = []

    for line in run_results_path.read_text(
        encoding="utf-8",
    ).splitlines():
        line = line.strip()

        if line:
            rows.append(
                json.loads(line)
            )

    assert len(rows) == MAX_EVALUATIONS

    assert {
        row["generation_id"]
        for row in rows
    } == {
        0,
        1,
    }

    assert [
        row["candidate_index"]
        for row in rows
    ] == list(
        range(MAX_EVALUATIONS)
    )
