from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from gow.cli import app


runner = CliRunner()


def test_run_cli_persists_resume_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = (
        tmp_path
        / "problem.yaml"
    )

    config.write_text(
        "id: context-test\n",
        encoding="utf-8",
    )

    outdir = (
        tmp_path
        / "results"
    )

    run_id = "context-run-001"

    run_root = (
        outdir
        / "runs"
        / run_id
    )

    results_path = (
        run_root
        / "results.jsonl"
    )

    fake_problem = object()

    def fake_load_problem_config(
        path: Path,
    ):
        assert path == config.resolve()
        return fake_problem

    def fake_run_local_optimization(
        problem,
        *,
        outdir,
        run_id,
        archive_generations,
        delete_archived_workdirs,
    ):
        assert problem is fake_problem
        assert Path(outdir) == (
            tmp_path
            / "results"
        ).resolve()

        assert run_id == "context-run-001"
        assert archive_generations is False
        assert delete_archived_workdirs is False

        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_path.write_text(
            "",
            encoding="utf-8",
        )

        return results_path

    monkeypatch.setattr(
        "gow.cli.load_problem_config",
        fake_load_problem_config,
    )

    monkeypatch.setattr(
        "gow.cli.run_local_optimization",
        fake_run_local_optimization,
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(config),
            "--outdir",
            str(outdir),
            "--run-id",
            run_id,
        ],
    )

    assert result.exit_code == 0, result.output

    context_path = (
        run_root
        / "run_context.json"
    )

    assert context_path.is_file()

    context = json.loads(
        context_path.read_text(
            encoding="utf-8"
        )
    )

    assert context["schema_version"] == 1
    assert context["run_id"] == run_id

    assert (
        Path(context["config_path"])
        == config.resolve()
    )

    assert context["config_sha256"] == (
        hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
    )

    python_executable = Path(
        context["python_executable"]
    )

    assert python_executable.is_absolute()
    assert python_executable.is_file()

    assert (
        Path(context["results_dir"])
        == outdir.resolve()
    )

    assert "Results:" in result.output
