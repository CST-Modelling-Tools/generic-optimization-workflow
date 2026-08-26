from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.optimizer.acor import ACOROptimizer
from gow.run import (
    resume_local_optimization,
    run_local_optimization,
)
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


BATCH_SIZE = 4
MAX_EVALUATIONS = 12

RUN_ID = "acor-portability-a-to-b"
REQUEST_ID = "acor-portability-pause-0001"

Q = 0.15
XI = 0.75
MAX_GENERATIONS = 20
MIN_SIGMA = 1e-12
BOUND_STRATEGY = "clip"
SEED = 13579


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
    Keep only deterministic mathematical/provenance data.

    Runtime timestamps and filesystem paths are intentionally ignored,
    because those are expected to differ after transportation.
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
    Create an independent ACOR project installation.

    Reference, Machine A and Machine B each receive their own physical
    evaluator and problem.yaml.
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

count = int(
    payload["params"]["count"]
)

objective = (
    1.75 * x * x
    + 0.65 * y * y
    - 0.22 * x * y
    + 0.08 * x
    + 0.015 * count
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
        "id": "acor-portability-test",

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

            "count": {
                "type": "int",
                "value": 2,
                "bounds": [0, 7],
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
            "name": "acor",
            "seed": SEED,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": BATCH_SIZE,

            "settings": {
                "q": Q,
                "xi": XI,
                "max_generations": MAX_GENERATIONS,
                "min_sigma": MIN_SIGMA,
                "bound_strategy": BOUND_STRATEGY,
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


def _new_acor(
    seed: int,
) -> ACOROptimizer:
    """
    Construct a fresh ACOR instance.

    Different constructor seeds are intentional. load_state_dict() must
    overwrite the RNG state using the persisted checkpoint.
    """

    return ACOROptimizer(
        batch_size=BATCH_SIZE,
        q=Q,
        xi=XI,
        max_generations=MAX_GENERATIONS,
        min_sigma=MIN_SIGMA,
        bound_strategy=BOUND_STRATEGY,
        seed=seed,
    )


def test_local_acor_checkpoint_is_portable_between_installations(
    tmp_path: Path,
) -> None:
    """
    Prove run-level ACOR portability.

    Machine A pauses after one complete generation. Only runs/<run_id> is
    copied to Machine B. Machine A is then physically deleted.

    Machine B must resume the campaign and remain mathematically identical
    to an uninterrupted reference execution.
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
        == "acor"
    )

    assert (
        reference_checkpoint.optimizer_state[
            "generation"
        ]
        == 3
    )

    assert (
        reference_checkpoint.optimizer_state[
            "awaiting_tell"
        ]
        is False
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

    # Request pause before launch.
    #
    # ACOR ask() consumes RNG, so GOW must wait until tell() completes.
    # Therefore the checkpoint must contain one COMPLETE generation.
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
        == BATCH_SIZE
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
        == BATCH_SIZE
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

    paused_state = (
        paused_checkpoint.optimizer_state
    )

    assert (
        paused_state["optimizer"]
        == "acor"
    )

    assert (
        paused_state["generation"]
        == 1
    )

    assert (
        paused_state["awaiting_tell"]
        is False
    )

    assert (
        paused_state["rng_state"]
        is not None
    )

    assert (
        len(paused_state["archive"])
        == BATCH_SIZE
    )

    assert (
        paused_state["archive_size"]
        == BATCH_SIZE
    )

    assert (
        paused_state["ants"]
        == BATCH_SIZE
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
        row["candidate_index"]
        for row in machine_b_rows
    ] == list(
        range(MAX_EVALUATIONS)
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

    assert (
        machine_b_checkpoint.manifest[
            "next_generation"
        ]
        == 3
    )

    assert (
        machine_b_checkpoint.optimizer_state
        == reference_checkpoint.optimizer_state
    )

    # ============================================================
    # I. BEST RESULT
    # ============================================================

    reference_summary = json.loads(
        (
            reference_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    machine_b_summary = json.loads(
        (
            machine_b_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        machine_b_summary["best"]
        == reference_summary["best"]
    )

    # ============================================================
    # J. STRONGEST FUTURE-STATE TEST
    #
    # Construct ACOR instances with DIFFERENT initial seeds.
    # load_state_dict() must restore archive + RNG from disk.
    #
    # If the checkpoint is truly deterministic, their NEXT ask()
    # must generate the exact same candidate population.
    # ============================================================

    reference_acor = _new_acor(
        seed=111111,
    )

    transported_acor = _new_acor(
        seed=999999,
    )

    reference_acor.load_state_dict(
        reference_checkpoint.optimizer_state
    )

    transported_acor.load_state_dict(
        machine_b_checkpoint.optimizer_state
    )

    assert (
        reference_acor.state_dict()
        == transported_acor.state_dict()
    )

    reference_next = reference_acor.ask(
        reference_problem,
        BATCH_SIZE,
    )

    transported_next = transported_acor.ask(
        machine_b_problem,
        BATCH_SIZE,
    )

    assert (
        transported_next
        == reference_next
    )