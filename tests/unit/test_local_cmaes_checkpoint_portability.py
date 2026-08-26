from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.run import (
    resume_local_optimization,
    run_local_optimization,
)
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


BATCH_SIZE = 6
MAX_EVALUATIONS = 18
RUN_ID = "cmaes-portability-a-to-b"
REQUEST_ID = "cmaes-portability-pause-0001"


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
    Return only deterministic mathematical data.

    Timestamps, wall-clock information and filesystem paths are
    deliberately excluded because they are expected to differ after
    transporting a run to another installation.
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
    Create an independent project installation.

    Every machine gets its own evaluator and its own problem.yaml.
    The mathematical problem is identical, but the evaluator path is
    physically different.
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
    1.7 * x * x
    + 0.55 * y * y
    - 0.21 * x * y
    + 0.04 * x
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
        "id": "cmaes-portability-test",
        "objective": {
            "direction": "minimize",
        },
        "parameters": {
            "x": {
                "type": "real",
                "value": 0.20,
                "bounds": [-4.0, 4.0],
            },
            "y": {
                "type": "real",
                "value": -0.15,
                "bounds": [-3.0, 3.0],
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
            "name": "cmaes",
            "seed": 24680,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": BATCH_SIZE,
            "settings": {
                "sigma0": 0.18,
                "max_generations": 20,
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


def test_local_cmaes_checkpoint_is_portable_between_installations(
    tmp_path: Path,
) -> None:
    """
    Prove that a paused CMA-ES local run can be physically moved to a
    different installation and resumed without changing its mathematical
    trajectory.

    The source installation is deleted before RESUME so no accidental
    dependency on its filesystem can survive.
    """

    # ============================================================
    # A. Independent continuous reference
    # ============================================================

    reference_root = (
        tmp_path
        / "machine_reference"
    )

    reference_config = _write_project(
        reference_root
        / "project"
    )

    reference_problem = load_problem_config(
        reference_config
    )

    reference_outdir = (
        reference_root
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

    reference_checkpoint = CheckpointStore(
        reference_run_root
    ).load()

    assert (
        reference_checkpoint.manifest["status"]
        == "completed"
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

    # Request PAUSE before launch. The cooperative runner will honor
    # it only after finishing the first complete generation.
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
    # C. MACHINE B - independent installation
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

    # Sanity check: the two project configurations really belong to
    # physically different installations.
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
    # D. Transport the execution
    #
    # Deliberately copy only runs/<run_id>, not the complete outdir.
    # This establishes the smallest run-level transport unit.
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
    # E. Destroy MACHINE A completely.
    #
    # After this point RESUME cannot possibly read the original
    # evaluator, checkpoint, results, workdirs or configuration.
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

    # ============================================================
    # F. RESUME on MACHINE B
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

    # ============================================================
    # G. Exact mathematical trajectory
    # ============================================================

    assert (
        _trajectory(machine_b_rows)
        == _trajectory(reference_rows)
    )

    # ============================================================
    # H. Final checkpoint equivalence
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

    reference_state = dict(
        reference_checkpoint.optimizer_state
    )

    machine_b_state = dict(
        machine_b_checkpoint.optimizer_state
    )

    assert (
        set(machine_b_state)
        == set(reference_state)
    )

    reference_pickle = reference_state.pop(
        "es_pickle"
    )

    machine_b_pickle = machine_b_state.pop(
        "es_pickle"
    )

    # All GOW-owned state must be identical.
    assert (
        machine_b_state
        == reference_state
    )

    assert isinstance(
        reference_pickle,
        bytes,
    )

    assert isinstance(
        machine_b_pickle,
        bytes,
    )

    reference_es = pickle.loads(
        reference_pickle
    )

    machine_b_es = pickle.loads(
        machine_b_pickle
    )

    assert (
        int(machine_b_es.countiter)
        == int(reference_es.countiter)
    )

    assert (
        int(machine_b_es.countevals)
        == int(reference_es.countevals)
    )

    assert (
        float(machine_b_es.sigma)
        == float(reference_es.sigma)
    )

    assert (
        [
            float(value)
            for value in machine_b_es.mean
        ]
        == [
            float(value)
            for value in reference_es.mean
        ]
    )

    # Strongest continuation criterion:
    # both final states must generate the exact same NEXT population.
    reference_next = [
        [
            float(value)
            for value in candidate
        ]
        for candidate in reference_es.ask(
            number=BATCH_SIZE
        )
    ]

    machine_b_next = [
        [
            float(value)
            for value in candidate
        ]
        for candidate in machine_b_es.ask(
            number=BATCH_SIZE
        )
    ]

    assert (
        machine_b_next
        == reference_next
    )