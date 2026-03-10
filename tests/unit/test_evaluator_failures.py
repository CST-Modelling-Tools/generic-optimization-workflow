from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from gow.config import load_problem_config
from gow.run import run_local_optimization


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_evaluator_missing_output_json_is_recorded_as_failure(tmp_path: Path) -> None:
    """
    If the evaluator exits without producing output.json, the candidate should be recorded
    as failed with an explanatory error.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Evaluator: intentionally DOES NOT write output.json.
    # It exits with code 0 to test "missing output.json" specifically.
    eval_py = project_dir / "bad_eval.py"
    _write_text(
        eval_py,
        """
# Intentionally do nothing: no output.json is produced.
# Exit code 0 to isolate the "missing output" failure mode.
raise SystemExit(0)
""".lstrip(),
    )

    eval_py_abs = str(eval_py.resolve())

    config = {
        "id": "toy-failure",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", eval_py_abs],
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

    cfg = load_problem_config(config_yaml)

    outdir = tmp_path / "out"
    run_id = "test-run-failure-0001"

    run_local_optimization(cfg, outdir=outdir, run_id=run_id)

    rows = _read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1

    r0 = rows[0]
    assert "fitness" in r0
    fitness = r0["fitness"]
    assert isinstance(fitness, dict)

    # These fields exist in your FitnessResult schema as seen in earlier runs.
    assert fitness.get("status") == "failed"
    assert fitness.get("error"), "Expected an explanatory error message for failure"
    assert "output.json" in str(fitness.get("error")), "Error should mention missing output.json"
    assert fitness.get("failure_kind") == "missing_output"
    assert r0.get("failure_kind") == "missing_output"
    assert isinstance(r0.get("evaluator"), dict)
    assert "resolved_command" in r0["evaluator"]
    assert _parse_timestamp(r0["finished_at"]) >= _parse_timestamp(r0["started_at"])


def test_evaluator_nonzero_exit_without_output_is_recorded_as_nonzero_exit(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    eval_py = project_dir / "bad_exit_eval.py"
    _write_text(
        eval_py,
        """
raise SystemExit(3)
""".lstrip(),
    )

    config = {
        "id": "toy-nonzero-exit",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", str(eval_py.resolve())],
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
    cfg = load_problem_config(config_yaml)

    outdir = tmp_path / "out"
    run_local_optimization(cfg, outdir=outdir, run_id="test-run-nonzero-exit")

    rows = _read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["fitness"]["status"] == "failed"
    assert r0["fitness"]["failure_kind"] == "nonzero_exit"
    assert r0["failure_kind"] == "nonzero_exit"
    assert r0["returncode"] == 3


def test_evaluator_invalid_output_is_recorded_as_invalid_output(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    eval_py = project_dir / "invalid_output_eval.py"
    _write_text(
        eval_py,
        """
from pathlib import Path
Path("output.json").write_text("{not valid json", encoding="utf-8")
raise SystemExit(0)
""".lstrip(),
    )

    config = {
        "id": "toy-invalid-output",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", str(eval_py.resolve())],
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
    cfg = load_problem_config(config_yaml)

    outdir = tmp_path / "out"
    run_local_optimization(cfg, outdir=outdir, run_id="test-run-invalid-output")

    rows = _read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["fitness"]["status"] == "failed"
    assert r0["fitness"]["failure_kind"] == "invalid_output"
    assert r0["failure_kind"] == "invalid_output"
    assert "invalid output.json" in r0["fitness"]["error"].lower()


def test_evaluator_timeout_is_recorded_as_timeout(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    eval_py = project_dir / "slow_eval.py"
    _write_text(
        eval_py,
        """
import time
time.sleep(2)
""".lstrip(),
    )

    config = {
        "id": "toy-timeout",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", str(eval_py.resolve())],
            "timeout_s": 1,
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
    cfg = load_problem_config(config_yaml)

    outdir = tmp_path / "out"
    run_local_optimization(cfg, outdir=outdir, run_id="test-run-timeout")

    rows = _read_jsonl(outdir / "results.jsonl")
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["fitness"]["status"] == "failed"
    assert r0["fitness"]["failure_kind"] == "timeout"
    assert r0["failure_kind"] == "timeout"
    assert r0["returncode"] == 124
