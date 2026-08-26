from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gow.cli import app


runner = CliRunner()


def test_resume_cli_forwards_required_runtime_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "problem.yaml"
    config.write_text(
        "id: ignored-by-mock\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "results"

    fake_problem = object()
    calls: dict[str, object] = {}

    def fake_load_problem_config(path: Path):
        calls["config"] = path
        return fake_problem

    def fake_resume_local_optimization(
        problem,
        *,
        outdir,
        run_id,
        archive_generations,
        delete_archived_workdirs,
    ):
        calls["problem"] = problem
        calls["outdir"] = outdir
        calls["run_id"] = run_id
        calls["archive_generations"] = archive_generations
        calls["delete_archived_workdirs"] = (
            delete_archived_workdirs
        )

        return (
            Path(outdir)
            / "runs"
            / str(run_id)
            / "results.jsonl"
        )

    monkeypatch.setattr(
        "gow.cli.load_problem_config",
        fake_load_problem_config,
    )

    monkeypatch.setattr(
        "gow.cli.resume_local_optimization",
        fake_resume_local_optimization,
    )

    result = runner.invoke(
        app,
        [
            "resume",
            str(config),
            "--outdir",
            str(outdir),
            "--run-id",
            "paused-run-001",
            "--archive-generations",
            "--delete-archived-workdirs",
        ],
    )

    assert result.exit_code == 0, result.output

    assert (
        Path(calls["config"]).resolve()
        == config.resolve()
    )

    assert calls["problem"] is fake_problem

    assert (
        Path(calls["outdir"]).resolve()
        == outdir.resolve()
    )

    assert calls["run_id"] == "paused-run-001"

    assert (
        calls["archive_generations"]
        is True
    )

    assert (
        calls["delete_archived_workdirs"]
        is True
    )

    assert "Results:" in result.output
    assert "paused-run-001" in result.output


def test_resume_cli_requires_run_id(
    tmp_path: Path,
) -> None:
    config = tmp_path / "problem.yaml"
    config.write_text(
        "id: test\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "resume",
            str(config),
        ],
    )

    assert result.exit_code != 0
    assert "--run-id" in result.output


def test_root_help_exposes_resume_command() -> None:
    result = runner.invoke(
        app,
        ["--help"],
    )

    assert result.exit_code == 0
    assert "resume" in result.output.lower()
