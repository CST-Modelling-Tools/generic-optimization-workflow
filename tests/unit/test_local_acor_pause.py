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


BATCH_SIZE = 4
MAX_EVALUATIONS = 12
RUN_ID = "acor-local-pause"


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


def test_acor_local_runner_can_pause_at_generation_boundary(
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
    + 1.5 * y * y
    + 0.2 * x * y
)

Path("output.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "objective": objective,
            "metrics": {
                "objective": objective,
            },
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
        "id": "acor-local-pause-test",

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
            "name": "acor",
            "seed": 24680,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": BATCH_SIZE,

            "settings": {
                "q": 0.18,
                "xi": 0.82,
                "max_generations": 10,
                "min_sigma": 1e-10,
                "bound_strategy": "resample",
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

    outdir = tmp_path / "results"

    run_root = (
        outdir
        / "runs"
        / RUN_ID
    )

    request = pause_request_path(
        run_root
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
                "request_id": "acor-pause-0001",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result_path = run_local_optimization(
        problem,
        outdir=outdir,
        run_id=RUN_ID,
    )

    # Pause returns a partial run-level results stream.
    assert result_path == (
        run_root
        / "results.jsonl"
    )

    rows = _read_jsonl(
        result_path
    )

    assert len(rows) == BATCH_SIZE

    assert [
        row["generation_id"]
        for row in rows
    ] == [
        0,
        0,
        0,
        0,
    ]

    assert [
        row["candidate_index"]
        for row in rows
    ] == [
        0,
        1,
        2,
        3,
    ]

    checkpoint = CheckpointStore(
        run_root
    ).load()

    assert (
        checkpoint.manifest["status"]
        == "paused"
    )

    assert (
        checkpoint.manifest["optimizer"]
        == "acor"
    )

    assert (
        checkpoint.manifest["evaluations_done"]
        == BATCH_SIZE
    )

    assert (
        checkpoint.manifest["completed_generations"]
        == 1
    )

    assert (
        checkpoint.manifest["next_generation"]
        == 1
    )

    state = checkpoint.optimizer_state

    assert state["optimizer"] == "acor"
    assert state["generation"] == 1
    assert state["awaiting_tell"] is False

    assert len(
        state["archive"]
    ) == BATCH_SIZE

    # Request consumed only after checkpoint persistence.
    assert not request.exists()

    ack_path = pause_ack_path(
        run_root
    )

    assert ack_path.is_file()

    ack = json.loads(
        ack_path.read_text(
            encoding="utf-8",
        )
    )

    assert ack["status"] == "paused"

    assert (
        ack["request_id"]
        == "acor-pause-0001"
    )

    assert (
        ack["evaluations_done"]
        == BATCH_SIZE
    )

    assert (
        ack["completed_generations"]
        == 1
    )

    # Run is intentionally incomplete.
    summary = json.loads(
        (
            run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    assert summary["finalized"] is False

    assert (
        summary["evaluations_done"]
        == BATCH_SIZE
    )
