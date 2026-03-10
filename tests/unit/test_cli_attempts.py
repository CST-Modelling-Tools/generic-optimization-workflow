from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from gow.cli import app


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_manual_evaluate_writes_requested_attempt_metadata(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    config = {
        "id": "toy-manual-attempt",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
            "y": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", str((Path.cwd() / "tests" / "toy_eval.py").resolve())],
            "timeout_s": 30,
        },
        "optimizer": {
            "name": "random_search",
            "seed": 123,
            "max_evaluations": 1,
            "batch_size": 1,
        },
    }

    config_yaml = project_dir / "problem.yaml"
    _write_yaml(config_yaml, config)

    outdir = tmp_path / "out"
    candidate_id = "r7c3f3a2a_g000002_c000014"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(config_yaml),
            "--outdir",
            str(outdir),
            "--run-id",
            "manual-run",
            "--candidate-id",
            candidate_id,
            "--attempt-index",
            "3",
            "--param",
            "x=0.5",
            "--param",
            "y=-0.25",
        ],
    )

    assert result.exit_code == 0, result.stdout

    workdir = outdir / "runs" / "manual-run" / candidate_id
    input_payload = json.loads((workdir / "input.json").read_text(encoding="utf-8"))
    output_payload = json.loads((workdir / "output.json").read_text(encoding="utf-8"))

    assert input_payload["candidate_id"] == candidate_id
    assert input_payload["candidate_local_id"] == "g000002_c000014"
    assert input_payload["attempt_id"] == f"{candidate_id}_a003"
    assert output_payload["status"] == "ok"
    assert output_payload["objective"] == 0.3125
