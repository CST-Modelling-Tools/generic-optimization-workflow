from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.optimizer.differential_evolution import (
    DifferentialEvolutionOptimizer,
)
from gow.run import (
    resume_local_optimization,
    run_local_optimization,
)
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


POPULATION_SIZE = 4
MAX_EVALUATIONS = 12

RUN_ID = "de-portability-a-to-b"
REQUEST_ID = "de-portability-pause-0001"

MUTATION_FACTOR = 0.68
CROSSOVER_RATE = 0.87
MAX_GENERATIONS = 20
SEED = 97531


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
    """
    Return only deterministic mathematical/provenance fields.

    Filesystem paths and timestamps are deliberately ignored because the
    execution is physically moved between installations.
    """

    keys = (
        "generation_id",
        "candidate_index",
        "candidate_id",
        "candidate_local_id",
        "attempt_id",
        "params",
        "fitness",
    )

    return [
        {
            key: row[key]
            for key in keys
        }
        for row in rows
    ]


def _write_project(
    root: Path,
) -> Path:
    """
    Create a completely independent DE project installation.

    Machine A, Machine B and the continuous reference each receive their
    own evaluator and their own problem.yaml.
    """

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator = root / "eval.py"

    evaluator.write_text(
        """
import json
from pathlib import Path


payload = json.loads(
    Path("input.json").read_text(
        encoding="utf-8"
    )
)

x = float(
    payload["params"]["x"]
)

y = float(
    payload["params"]["y"]
)

objective = (
    2.0 * x * x
    + 0.5 * y * y
    - 0.3 * x * y
    + 0.1 * x
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
""".lstrip(),
        encoding="utf-8",
    )

    config = {
        "id": "de-portability-test",

        "objective": {
            "direction": "minimize",
        },

        "parameters": {
            "x": {
                "type": "real",
                "value": 0.0,
                "bounds": [-5.0, 5.0],
            },

            "y": {
                "type": "real",
                "value": 0.0,
                "bounds": [-4.0, 4.0],
            },
        },

        "evaluator": {
            "command": [
                "{python}",
                str(
                    evaluator.resolve()
                ),
            ],
            "timeout_s": 30,
        },

        "optimizer": {
            "name": "differential_evolution",
            "seed": SEED,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": POPULATION_SIZE,

            "settings": {
                "mutation_factor": MUTATION_FACTOR,
                "crossover_rate": CROSSOVER_RATE,
                "max_generations": MAX_GENERATIONS,
            },
        },
    }

    config_path = root / "problem.yaml"

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def _new_de(
    seed: int,
) -> DifferentialEvolutionOptimizer:
    """
    Construct a fresh DE instance.

    The seed is intentionally caller-controlled. load_state_dict() must
    replace its RNG state with the checkpoint RNG state.
    """

    return DifferentialEvolutionOptimizer(
        population_size=POPULATION_SIZE,
        mutation_factor=MUTATION_FACTOR,
        crossover_rate=CROSSOVER_RATE,
        max_generations=MAX_GENERATIONS,
        seed=seed,
    )


def test_local_de_checkpoint_is_portable_between_installations(
    tmp_path: Path,
) -> None:
    """
    Prove run-level portability for Differential Evolution.

    A campaign is paused on Machine A, only runs/<run_id> is transported
    to Machine B, Machine A is physically deleted, and Machine B resumes
    the campaign.

    The transported execution must be mathematically identical to an
    uninterrupted reference execution.
    """

    # ============================================================
    # A. CONTINUOUS REFERENCE
    # ============================================================

    reference_machine = (
        tmp_path
        / "machine_reference"
    )

    reference_config = _write_project(
        reference_machine
        / "project"
    )

    reference_problem = load_problem_config(
        reference_config
    )

    reference_outdir = (
        reference_machine
        / "results"
    )

    run_local_optimization(
        reference_problem,
        outdir=reference_outdir,
        run_id=RUN_ID,
    )

    reference_run_root = (
        reference_outdir
        / "runs"
        / RUN_ID
    )

    reference_rows = _read_jsonl(
        reference_run_root
        / "results.jsonl"
    )

    assert (
        len(reference_rows)
        == MAX_EVALUATIONS
    )

    assert [
        row["generation_id"]
        for row in reference_rows
    ] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    reference_checkpoint = CheckpointStore(
        reference_run_root
    ).load()

    assert (
        reference_checkpoint.manifest["status"]
        == "completed"
    )

    assert (
        reference_checkpoint.manifest[
            "evaluations_done"
        ]
        == MAX_EVALUATIONS
    )

    assert (
        reference_checkpoint.manifest[
            "completed_generations"
        ]
        == 3
    )

    assert (
        reference_checkpoint.optimizer_state[
            "optimizer"
        ]
        == "differential_evolution"
    )

    # ============================================================
    # B. MACHINE A
    # ============================================================

    machine_a = (
        tmp_path
        / "machine_A"
    )

    machine_a_config = _write_project(
        machine_a
        / "project"
    )

    machine_a_problem = load_problem_config(
        machine_a_config
    )

    machine_a_outdir = (
        machine_a
        / "results"
    )

    machine_a_run_root = (
        machine_a_outdir
        / "runs"
        / RUN_ID
    )

    # Request PAUSE before launch.
    #
    # GOW must not stop before a safe generation boundary. Therefore
    # exactly one complete DE generation will be evaluated.
    request = pause_request_path(
        machine_a_run_root
    )

    request.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "pause",
                "request_id": REQUEST_ID,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    paused_result = run_local_optimization(
        machine_a_problem,
        outdir=machine_a_outdir,
        run_id=RUN_ID,
    )

    paused_rows = _read_jsonl(
        paused_result
    )

    assert (
        len(paused_rows)
        == POPULATION_SIZE
    )

    assert [
        row["generation_id"]
        for row in paused_rows
    ] == [
        0,
        0,
        0,
        0,
    ]

    paused_checkpoint = CheckpointStore(
        machine_a_run_root
    ).load()

    assert (
        paused_checkpoint.manifest["status"]
        == "paused"
    )

    assert (
        paused_checkpoint.manifest[
            "evaluations_done"
        ]
        == POPULATION_SIZE
    )

    assert (
        paused_checkpoint.manifest[
            "completed_generations"
        ]
        == 1
    )

    assert (
        paused_checkpoint.manifest[
            "next_generation"
        ]
        == 1
    )

    assert (
        paused_checkpoint.optimizer_state[
            "optimizer"
        ]
        == "differential_evolution"
    )

    assert (
        not pause_request_path(
            machine_a_run_root
        ).exists()
    )

    assert (
        pause_ack_path(
            machine_a_run_root
        ).is_file()
    )

    # ============================================================
    # C. MACHINE B - INDEPENDENT INSTALLATION
    # ============================================================

    machine_b = (
        tmp_path
        / "machine_B"
    )

    machine_b_config = _write_project(
        machine_b
        / "project"
    )

    machine_b_problem = load_problem_config(
        machine_b_config
    )

    machine_b_outdir = (
        machine_b
        / "results"
    )

    machine_b_runs_root = (
        machine_b_outdir
        / "runs"
    )

    machine_b_runs_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    machine_b_run_root = (
        machine_b_runs_root
        / RUN_ID
    )

    # These are genuinely different project locations.
    assert (
        machine_a_config.resolve()
        != machine_b_config.resolve()
    )

    assert (
        (
            machine_a
            / "project"
            / "eval.py"
        ).resolve()
        != (
            machine_b
            / "project"
            / "eval.py"
        ).resolve()
    )

    # ============================================================
    # D. TRANSPORT ONLY runs/<run_id>
    # ============================================================

    shutil.copytree(
        machine_a_run_root,
        machine_b_run_root,
    )

    copied_checkpoint = CheckpointStore(
        machine_b_run_root
    ).load()

    assert (
        copied_checkpoint.manifest
        == paused_checkpoint.manifest
    )

    assert (
        copied_checkpoint.optimizer_state
        == paused_checkpoint.optimizer_state
    )

    # ============================================================
    # E. DESTROY MACHINE A COMPLETELY
    # ============================================================

    shutil.rmtree(
        machine_a
    )

    assert not machine_a.exists()

    assert (
        machine_b_run_root.is_dir()
    )

    assert (
        (
            machine_b_run_root
            / "checkpoint"
            / "manifest.json"
        ).is_file()
    )

    assert (
        (
            machine_b_run_root
            / "checkpoint"
            / "optimizer_state.bin"
        ).is_file()
    )

    assert (
        (
            machine_b_run_root
            / "checkpoint"
            / "checkpoint.sha256"
        ).is_file()
    )

    # ============================================================
    # F. RESUME ON MACHINE B
    # ============================================================

    resume_local_optimization(
        machine_b_problem,
        outdir=machine_b_outdir,
        run_id=RUN_ID,
    )

    machine_b_rows = _read_jsonl(
        machine_b_run_root
        / "results.jsonl"
    )

    assert (
        len(machine_b_rows)
        == MAX_EVALUATIONS
    )

    assert [
        row["generation_id"]
        for row in machine_b_rows
    ] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    # ============================================================
    # G. EXACT MATHEMATICAL TRAJECTORY
    # ============================================================

    assert (
        _trajectory(machine_b_rows)
        == _trajectory(reference_rows)
    )

    # ============================================================
    # H. EXACT FINAL CHECKPOINT
    # ============================================================

    machine_b_checkpoint = CheckpointStore(
        machine_b_run_root
    ).load()

    assert (
        machine_b_checkpoint.manifest["status"]
        == "completed"
    )

    assert (
        machine_b_checkpoint.manifest[
            "evaluations_done"
        ]
        == MAX_EVALUATIONS
    )

    assert (
        machine_b_checkpoint.manifest[
            "completed_generations"
        ]
        == 3
    )

    # Strong DE property:
    #
    # population, fitness, generation, parameter metadata, diagnostics
    # and complete random.Random state must all be bit-for-bit equivalent.
    assert (
        machine_b_checkpoint.optimizer_state
        == reference_checkpoint.optimizer_state
    )

    # ============================================================
    # I. STRONGEST FUTURE-STATE TEST
    #
    # Reconstruct two completely fresh DE objects using deliberately
    # different constructor seeds. load_state_dict() must overwrite those
    # seeds with the exact RNG state stored in each checkpoint.
    #
    # The NEXT population must therefore be identical.
    # ============================================================

    reference_de = _new_de(
        seed=111111,
    )

    transported_de = _new_de(
        seed=999999,
    )

    reference_de.load_state_dict(
        reference_checkpoint.optimizer_state
    )

    transported_de.load_state_dict(
        machine_b_checkpoint.optimizer_state
    )

    assert (
        reference_de.state_dict()
        == transported_de.state_dict()
    )

    reference_next = reference_de.ask(
        reference_problem,
        POPULATION_SIZE,
    )

    transported_next = transported_de.ask(
        machine_b_problem,
        POPULATION_SIZE,
    )

    assert (
        transported_next
        == reference_next
    )