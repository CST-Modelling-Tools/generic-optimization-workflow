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
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


BATCH_SIZE = 4
MAX_EVALUATIONS = 12
RUN_ID = "acor-resume-determinism"


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
    Mathematical trajectory only.

    Runtime timestamps, filesystem paths and wall times are
    deliberately excluded.
    """

    return [
        {
            "generation_id": row[
                "generation_id"
            ],
            "candidate_index": row[
                "candidate_index"
            ],
            "candidate_id": row[
                "candidate_id"
            ],
            "candidate_local_id": row[
                "candidate_local_id"
            ],
            "attempt_id": row[
                "attempt_id"
            ],
            "params": row[
                "params"
            ],
            "fitness": row[
                "fitness"
            ],
        }
        for row in rows
    ]


def test_acor_pause_resume_matches_continuous_run_exactly(
    tmp_path: Path,
) -> None:

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # ============================================================
    # Deterministic evaluator
    # ============================================================

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

    # ============================================================
    # ACOR config
    # ============================================================

    config = {
        "id": "acor-local-resume-test",

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
    # A. Continuous run
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

    continuous_root = (
        continuous_out
        / "runs"
        / RUN_ID
    )

    continuous_rows = _read_jsonl(
        continuous_root
        / "results.jsonl"
    )

    assert (
        len(continuous_rows)
        == MAX_EVALUATIONS
    )

    continuous_checkpoint = (
        CheckpointStore(
            continuous_root
        ).load()
    )

    assert (
        continuous_checkpoint.manifest[
            "status"
        ]
        == "completed"
    )

    assert (
        continuous_checkpoint.optimizer_state[
            "optimizer"
        ]
        == "acor"
    )

    # ============================================================
    # B. Same run with PAUSE after first complete generation
    # ============================================================

    resumed_out = (
        tmp_path
        / "paused-resumed"
    )

    resumed_root = (
        resumed_out
        / "runs"
        / RUN_ID
    )

    request = pause_request_path(
        resumed_root
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
                "request_id": (
                    "acor-resume-pause-0001"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    paused_result = run_local_optimization(
        problem,
        outdir=resumed_out,
        run_id=RUN_ID,
    )

    assert paused_result == (
        resumed_root
        / "results.jsonl"
    )

    paused_rows = _read_jsonl(
        paused_result
    )

    assert (
        len(paused_rows)
        == BATCH_SIZE
    )

    paused_checkpoint = (
        CheckpointStore(
            resumed_root
        ).load()
    )

    assert (
        paused_checkpoint.manifest[
            "status"
        ]
        == "paused"
    )

    assert (
        paused_checkpoint.manifest[
            "optimizer"
        ]
        == "acor"
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

    assert not request.exists()

    ack = pause_ack_path(
        resumed_root
    )

    assert ack.is_file()

    ack_payload = json.loads(
        ack.read_text(
            encoding="utf-8",
        )
    )

    assert (
        ack_payload["request_id"]
        == "acor-resume-pause-0001"
    )

    assert (
        ack_payload["evaluations_done"]
        == BATCH_SIZE
    )

    # ============================================================
    # C. Resume from checkpoint
    # ============================================================

    resume_local_optimization(
        problem,
        outdir=resumed_out,
        run_id=RUN_ID,
    )

    resumed_rows = _read_jsonl(
        resumed_root
        / "results.jsonl"
    )

    assert (
        len(resumed_rows)
        == MAX_EVALUATIONS
    )

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
    # D. Exact mathematical trajectory
    # ============================================================

    assert (
        _trajectory(resumed_rows)
        == _trajectory(continuous_rows)
    )

    # ============================================================
    # E. Exact final optimizer state
    # ============================================================

    resumed_checkpoint = (
        CheckpointStore(
            resumed_root
        ).load()
    )

    assert (
        resumed_checkpoint.manifest[
            "status"
        ]
        == "completed"
    )

    assert (
        resumed_checkpoint.manifest[
            "evaluations_done"
        ]
        == MAX_EVALUATIONS
    )

    assert (
        resumed_checkpoint.manifest[
            "completed_generations"
        ]
        == 3
    )

    assert (
        resumed_checkpoint.manifest[
            "next_generation"
        ]
        == 3
    )

    assert (
        resumed_checkpoint.optimizer_state
        == continuous_checkpoint.optimizer_state
    )

    # ============================================================
    # F. Exact best-so-far
    # ============================================================

    continuous_summary = json.loads(
        (
            continuous_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    resumed_summary = json.loads(
        (
            resumed_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        resumed_summary["best"]
        == continuous_summary["best"]
    )
