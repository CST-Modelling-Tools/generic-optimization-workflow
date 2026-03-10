from __future__ import annotations

import json
from pathlib import Path

import yaml

from gow.candidate_ids import format_candidate_id
from gow.config import load_problem_config
from gow.fw.tasks import EvaluateCandidateTask
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


def test_fireworks_evaluate_task_preserves_local_provenance_fields(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    eval_py = project_dir / "toy_eval.py"
    _write_text(
        eval_py,
        """
import json
from pathlib import Path

inp = json.loads(Path("input.json").read_text(encoding="utf-8"))
params = inp.get("params", {})
x = float(params.get("x", 0.0))
objective = x * x

out = {
    "status": "ok",
    "metrics": {"objective": objective},
    "objective": objective,
    "constraints": {},
    "artifacts": {},
    "error": None,
}

Path("output.json").write_text(json.dumps(out), encoding="utf-8")
""".lstrip(),
    )

    config = {
        "id": "toy-fw-parity",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.25, "bounds": [-1.0, 1.0]},
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

    local_outdir = tmp_path / "out-local"
    local_run_id = "parity-local-run"
    run_local_optimization(cfg, outdir=local_outdir, run_id=local_run_id)
    local_record = _read_jsonl(local_outdir / "results.jsonl")[0]

    fw_outdir = tmp_path / "out-fw"
    fw_run_id = "parity-fw-run"
    generation_id = 0
    candidate_index = 0
    attempt_index = 0
    candidate_id = format_candidate_id(generation_id, candidate_index, run_id=fw_run_id)

    task = EvaluateCandidateTask(
        {
            "problem_config": str(config_yaml.resolve()),
            "outdir": str(fw_outdir.resolve()),
            "run_id": fw_run_id,
            "candidate_id": candidate_id,
            "candidate_params": {"x": 0.25},
            "generation_id": generation_id,
            "candidate_index": candidate_index,
            "attempt_index": attempt_index,
        }
    )
    action = task.run_task({})
    fw_record = dict(action.stored_data)

    keys = [
        "candidate_id",
        "candidate_local_id",
        "attempt_id",
        "attempt_index",
        "generation_id",
        "candidate_index",
    ]
    for key in keys:
        assert key in fw_record
        assert key in local_record

    assert fw_record["candidate_id"] == candidate_id
    assert fw_record["candidate_local_id"] == "g000000_c000000"
    assert fw_record["attempt_id"] == f"{candidate_id}_a000"
    assert fw_record["attempt_index"] == 0
    assert fw_record["generation_id"] == 0
    assert fw_record["candidate_index"] == 0

    fw_result_path = fw_outdir / "runs" / fw_run_id / candidate_id / "result.json"
    assert fw_result_path.exists()
    fw_result_payload = json.loads(fw_result_path.read_text(encoding="utf-8"))
    for key in keys:
        assert fw_result_payload[key] == fw_record[key]
