from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from gow.config import load_problem_config
from gow.run import run_local_optimization

app = typer.Typer(help="Generic Optimization Workflow (gow)")
commands = typer.Typer(help="Commands")

app.add_typer(commands)

def _parse_kv_params(items: List[str]) -> dict:
    """
    Parse ["x=1.2", "mode=a"] into {"x": 1.2, "mode": "a"} with light type casting.
    """
    out = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --param '{item}'. Use NAME=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        # basic casting: int -> float -> bool -> string
        if v.lower() in {"true", "false"}:
            out[k] = (v.lower() == "true")
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass

        out[k] = v
    return out

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

@commands.command("evaluate")
def evaluate_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to problem config (YAML/JSON)."),
    outdir: Path = typer.Option(Path("runs"), "--outdir", "-o", help="Output directory for run artifacts."),
    run_id: str = typer.Option("manual", "--run-id", help="Run id used to build the workdir path."),
    candidate_id: str = typer.Option("manual", "--candidate-id", help="Candidate id used to build the workdir path."),
    param: List[str] = typer.Option([], "--param", "-p", help="Override parameter as NAME=VALUE (repeatable)."),
    params_file: Optional[Path] = typer.Option(None, "--params-file", help="JSON file with parameter overrides."),
):
    """
    Evaluate a single candidate (useful for debugging external evaluators).
    """
    problem = load_problem_config(config)

    overrides = {}
    if params_file is not None:
        data = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter("--params-file must contain a JSON object (dict).")
        overrides.update(data)

    overrides.update(_parse_kv_params(param))

    # Workdir layout: <outdir>/<problem_id>/<run_id>/<candidate_id>/
    workdir = outdir / problem.id / run_id / candidate_id

    from gow.evaluation import evaluate_candidate  # local import to keep CLI lightweight

    res = evaluate_candidate(
        problem,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_params=overrides,
        workdir=workdir,
    )

    typer.echo(f"Workdir: {workdir}")
    typer.echo(f"Status:  {res.fitness.status}")
    typer.echo(f"Objective: {res.fitness.objective}")
    typer.echo(f"Metrics: {res.fitness.metrics}")
    if res.fitness.error:
        typer.echo(f"Error: {res.fitness.error}")
    typer.echo(f"Return code: {res.returncode}  Wall time (s): {res.wall_time_s:.3f}")

if __name__ == "__main__":
    app()