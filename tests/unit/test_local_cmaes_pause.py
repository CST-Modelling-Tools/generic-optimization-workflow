from __future__ import annotations

import json
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.run import run_local_optimization
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


BATCH_SIZE = 6
MAX_EVALUATIONS = 18
RUN_ID = "cmaes-pause-checkpoint"
REQUEST_ID = "cmaes-pause-request-0001"


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


def test_local_cmaes_pause_creates_safe_checkpoint(
    tmp_path: Path,
) -> None:

    # ============================================================
    # External deterministic evaluator
    # ============================================================

    project_dir = (
        tmp_path
        / "project"
    )

    project_dir.mkdir()

    evaluator = (
        project_dir
        / "eval.py"
    )

    evaluator.write_text(
        '''
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
'''.lstrip(),
        encoding="utf-8",
    )

    # ============================================================
    # CMA-ES problem
    # ============================================================

    config = {
        "id": "cmaes-local-pause-test",

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
            "max_evaluations":
                MAX_EVALUATIONS,
            "batch_size":
                BATCH_SIZE,

            "settings": {
                "sigma0": 0.18,
                "max_generations": 20,
            },
        },
    }

    config_path = (
        project_dir
        / "problem.yaml"
    )

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
    # Prepare pause request BEFORE starting the run.
    #
    # The runner must not stop immediately. It must first:
    #
    # ask()
    # evaluate entire population
    # tell()
    # state_dict()
    # checkpoint
    # ACK
    # return
    #
    # ============================================================

    outdir = (
        tmp_path
        / "output"
    )

    run_root = (
        outdir
        / "runs"
        / RUN_ID
    )

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
                "request_id": REQUEST_ID,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Execute
    # ============================================================

    result_path = (
        run_local_optimization(
            problem,
            outdir=outdir,
            run_id=RUN_ID,
        )
    )

    assert result_path.is_file()

    # ============================================================
    # Results must contain exactly one COMPLETE CMA generation.
    # ============================================================

    rows = _read_jsonl(
        result_path
    )

    assert len(rows) == BATCH_SIZE

    assert [
        row["candidate_index"]
        for row in rows
    ] == list(
        range(BATCH_SIZE)
    )

    assert [
        row["generation_id"]
        for row in rows
    ] == [
        0
        for _ in range(
            BATCH_SIZE
        )
    ]

    # ============================================================
    # Checkpoint
    # ============================================================

    checkpoint = CheckpointStore(
        run_root
    ).load()

    manifest = checkpoint.manifest

    assert (
        manifest["schema_version"]
        == 1
    )

    assert (
        manifest["status"]
        == "paused"
    )

    assert (
        str(
            manifest["optimizer"]
        ).lower()
        == "cmaes"
    )

    assert (
        manifest["evaluations_done"]
        == BATCH_SIZE
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

    # ============================================================
    # CMA-ES optimizer state
    # ============================================================

    state = checkpoint.optimizer_state

    assert state["schema_version"] == 1
    assert state["optimizer"] == "cmaes"

    assert state["initialized"] is True
    assert state["generation"] == 1

    # Critical generation-boundary guarantee.
    assert state["last_xs"] == []

    assert (
        state["configuration"][
            "batch_size"
        ]
        == BATCH_SIZE
    )

    assert (
        state["configuration"][
            "population_size"
        ]
        == BATCH_SIZE
    )

    assert (
        state["es_countiter"]
        == 1
    )

    assert (
        state["es_countevals"]
        == BATCH_SIZE
    )

    assert isinstance(
        state["es_pickle"],
        bytes,
    )

    assert state["es_pickle"]

    # ============================================================
    # Pause request lifecycle
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

    assert (
        ack["status"]
        == "paused"
    )

    assert (
        ack["request_id"]
        == REQUEST_ID
    )

    assert (
        ack["evaluations_done"]
        == BATCH_SIZE
    )

    # ============================================================
    # Summary must explicitly represent a non-finalized run.
    # ============================================================

    summary_path = (
        run_root
        / "summary.json"
    )

    assert summary_path.is_file()

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8",
        )
    )

    assert (
        summary["finalized"]
        is False
    )
