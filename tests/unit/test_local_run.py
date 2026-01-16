from __future__ import annotations

import json
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


def test_local_run_smoke_creates_expected_outputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Evaluator must produce output.json compatible with FitnessResult (Pydantic).
    # We'll emit:
    # - status: "ok"
    # - metrics.objective: x^2
    # - error: null
    # - constraints/artifacts: empty
    eval_py = project_dir / "toy_eval.py"
    _write_text(
        eval_py,
        """
import json
from pathlib import Path

inp = json.loads(Path("input.json").read_text(encoding="utf-8"))
params = inp.get("parameters", {})
x = float(params.get("x", 0.0))
objective = x * x

out = {
    "status": "ok",
    "metrics": {"objective": objective},
    "constraints": {},
    "artifacts": {},
    "error": None,
}

Path("output.json").write_text(json.dumps(out), encoding="utf-8")
""".lstrip(),
    )

    eval_py_abs = str(eval_py.resolve())

    config = {
        "id": "toy-sphere",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.25, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["{python}", eval_py_abs],
            "timeout_s": 30,
        },
        "optimizer": {
            "name": "random_search",
            "seed": 123,
            "max_evaluations": 5,
            "batch_size": 2,
        },
    }

    config_yaml = project_dir / "problem.yaml"
    _write_yaml(config_yaml, config)

    cfg = load_problem_config(config_yaml)

    outdir = tmp_path / "out"
    run_id = "test-run-0001"

    run_local_optimization(cfg, outdir=outdir, run_id=run_id)

    assert (outdir / "summary.json").exists()
    assert (outdir / "results.jsonl").exists()

    run_root = outdir / "runs" / run_id
    assert run_root.exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "results.jsonl").exists()

    rows = _read_jsonl(outdir / "results.jsonl")
    assert len(rows) == cfg.optimizer.max_evaluations

    # Align with your actual JSONL structure:
    for r in rows:
        assert "candidate_id" in r
        assert "fitness" in r
        assert isinstance(r["fitness"], dict)
        assert r["fitness"].get("status") in ("ok", "failed")