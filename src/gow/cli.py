from __future__ import annotations

from pathlib import Path
import typer

from gow.config import load_problem_config
from gow.run import run_local_optimization

app = typer.Typer(help="Generic Optimization Workflow (gow)")
commands = typer.Typer(help="Commands")

app.add_typer(commands)


@commands.command("run")
def run_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to problem config (YAML/JSON)."),
    outdir: Path = typer.Option(Path("runs"), "--outdir", "-o", help="Output directory for run artifacts."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id (defaults to a UUID)."),
):
    """Run a local optimization loop (no FireWorks) using the provided problem configuration."""
    problem = load_problem_config(config)
    results_path = run_local_optimization(problem, outdir=outdir, run_id=run_id)
    typer.echo(f"Results: {results_path}")


@commands.command("info")
def info():
    """Print basic installation info."""
    typer.echo("gow is installed and commands are registered correctly.")


if __name__ == "__main__":
    app()