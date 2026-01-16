# src/gow/cli.py
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer

from gow.candidate_ids import format_candidate_id
from gow.config import load_problem_config
from gow.layout import candidate_workdir, launchers_dir as default_launchers_dir, run_root
from gow.run import run_local_optimization


app = typer.Typer(help="Generic Optimization Workflow (gow)")
commands = typer.Typer(help="Commands")
fw_app = typer.Typer(help="FireWorks backend (optional)")

app.add_typer(commands)
app.add_typer(fw_app, name="fw")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

ENV_OUTDIR = "GOW_OUTDIR"


def _default_run_id() -> str:
    return str(uuid.uuid4())


def _resolve_results_dir(config: Path, outdir: Optional[Path]) -> Path:
    """
    Resolution order:
      1) explicit --outdir
      2) env var GOW_OUTDIR
      3) <config_dir>/results

    IMPORTANT: this is the flattened results directory (problem root).
    Outputs go directly under:
      <outdir>/
        results.jsonl
        summary.json
        runs/...
        launchers/...
    """
    if outdir is not None:
        return outdir.expanduser().resolve()

    env = os.environ.get(ENV_OUTDIR)
    if env:
        return Path(env).expanduser().resolve()

    return config.expanduser().resolve().parent / "results"


def _parse_kv_params(items: List[str]) -> dict:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --param '{item}'. Use NAME=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick_best(
    records: Iterable[Dict[str, Any]],
    *,
    direction: str = "minimize",
) -> list[Dict[str, Any]]:
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be 'minimize' or 'maximize'")

    good: List[Dict[str, Any]] = []
    for r in records:
        fit = (r or {}).get("fitness", {}) or {}
        if fit.get("status") != "ok":
            continue
        obj = fit.get("objective", None)
        if obj is None:
            continue
        try:
            obj_val = float(obj)
        except (TypeError, ValueError):
            continue
        rr = dict(r)
        rr["_objective"] = obj_val
        good.append(rr)

    reverse = (direction == "maximize")
    good.sort(key=lambda x: x["_objective"], reverse=reverse)
    return good


def _optimizer_kwargs(opt_cfg: Any) -> Dict[str, Any]:
    """
    Extract optimizer-specific kwargs from opt_cfg.

    In GOW, optimizer hyperparameters are stored in opt_cfg.settings (dict).
    This function flattens that dict and returns it as kwargs for make_optimizer().

    Works with:
      - Pydantic v2: model_dump()
      - Pydantic v1: dict()
      - simple objects: __dict__
    """
    if hasattr(opt_cfg, "model_dump"):
        data = opt_cfg.model_dump()
    elif hasattr(opt_cfg, "dict"):
        data = opt_cfg.dict()
    else:
        data = dict(getattr(opt_cfg, "__dict__", {}) or {})

    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"optimizer.settings must be a dict, got {type(settings)}")

    for k in ("name", "seed", "max_evaluations", "batch_size", "settings"):
        data.pop(k, None)

    out = {k: v for k, v in data.items() if not str(k).startswith("_")}
    out.update(settings)
    out = {k: v for k, v in out.items() if not str(k).startswith("_")}
    return out


@contextmanager
def _pushd(path: Path):
    """
    Temporarily chdir. Ensures FireWorks launcher_* directories are created under `path`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


# -----------------------------------------------------------------------------
# Core (no FireWorks)
# -----------------------------------------------------------------------------

@commands.command("run")
def run_cmd(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to optimization specs (YAML/JSON).",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id (defaults to a UUID)."),
):
    """
    Run a local optimization loop (no FireWorks).

    Outputs:
      <outdir>/results.jsonl
      <outdir>/summary.json
      <outdir>/runs/<run_id>/c000000/...
    """
    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    problem = load_problem_config(config_abs)
    results_path = run_local_optimization(problem, outdir=results_dir, run_id=run_id)
    typer.echo(f"Results: {results_path}")


@commands.command("info")
def info():
    typer.echo("gow is installed and commands are registered correctly.")


@commands.command("evaluate")
def evaluate_cmd(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to optimization specs (YAML/JSON).",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str = typer.Option("manual", "--run-id", help="Run id used to build the workdir path."),
    candidate_id: str = typer.Option("manual", "--candidate-id", help="Candidate id used to build the workdir path."),
    param: List[str] = typer.Option([], "--param", "-p", help="Override parameter as NAME=VALUE (repeatable)."),
    params_file: Optional[Path] = typer.Option(None, "--params-file", help="JSON file with parameter overrides."),
):
    """
    Evaluate a single candidate (useful for debugging external evaluators).
    """
    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    problem = load_problem_config(config_abs)

    overrides: Dict[str, Any] = {}
    if params_file is not None:
        data = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter("--params-file must contain a JSON object (dict).")
        overrides.update(data)

    overrides.update(_parse_kv_params(param))

    workdir = candidate_workdir(results_dir, run_id, candidate_id)

    from gow.evaluation import evaluate_candidate  # local import

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


@commands.command("best")
def best_cmd(
    results_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Results dir (<outdir>)."),
    top: int = typer.Option(1, "--top", "-n", min=1, help="Show top N candidates."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optimization specs (YAML/JSON) to read objective direction."),
    direction: Optional[str] = typer.Option(None, "--direction", help="Override objective direction: minimize or maximize."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="If set, read runs/<run_id>/results.jsonl instead of <outdir>/results.jsonl."),
):
    """
    Show the best candidate(s).

    Reads:
      - <outdir>/results.jsonl (default), OR
      - <outdir>/runs/<run_id>/results.jsonl if --run-id is provided.
    """
    results_dir = results_dir.expanduser().resolve()
    if run_id:
        results_path = results_dir / "runs" / run_id / "results.jsonl"
    else:
        results_path = results_dir / "results.jsonl"

    if not results_path.exists():
        raise typer.BadParameter(f"Could not find results.jsonl at {results_path}")

    chosen_direction = "minimize"
    if config is not None:
        problem = load_problem_config(config.expanduser().resolve())
        chosen_direction = problem.objective.direction
    if direction is not None:
        direction = direction.strip().lower()
        if direction not in {"minimize", "maximize"}:
            raise typer.BadParameter("--direction must be 'minimize' or 'maximize'")
        chosen_direction = direction

    ranked = _pick_best(_iter_jsonl(results_path), direction=chosen_direction)
    if not ranked:
        typer.echo("No successful candidates with an objective found.")
        raise typer.Exit(code=1)

    show = ranked[:top]
    for i, r in enumerate(show, start=1):
        fit = r["fitness"]
        typer.echo(f"#{i}")
        typer.echo(f"  candidate_id: {r.get('candidate_id')}")
        typer.echo(f"  objective:    {fit.get('objective')}")
        typer.echo(f"  metrics:      {fit.get('metrics')}")
        typer.echo(f"  workdir:      {r.get('workdir')}")
        if i == 1:
            typer.echo(f"  direction:    {chosen_direction}")
        if top > 1 and i != top:
            typer.echo("")


# -----------------------------------------------------------------------------
# FireWorks backend
# -----------------------------------------------------------------------------

@fw_app.command("evaluate")
def fw_evaluate_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to optimization specs (YAML/JSON)."),
    launchpad: Optional[Path] = typer.Option(None, "--launchpad", help="Path to my_launchpad.yaml (FireWorks LaunchPad config)."),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str = typer.Option("fw-manual", "--run-id", help="Run id used to build the workdir path."),
    candidate_id: str = typer.Option("c000000", "--candidate-id", help="Candidate id used to build the workdir path."),
    param: List[str] = typer.Option([], "--param", "-p", help="Override parameter as NAME=VALUE (repeatable)."),
    params_file: Optional[Path] = typer.Option(None, "--params-file", help="JSON file with parameter overrides."),
    launch: bool = typer.Option(False, "--launch/--no-launch", help="Launch immediately (rapidfire) after submitting."),
    launch_dir: Optional[Path] = typer.Option(
        None,
        "--launch-dir",
        help="Directory to place FireWorks launcher_* dirs (default: <outdir>/launchers).",
    ),
    sleep: int = typer.Option(0, "--sleep", help="Seconds to sleep between rocket launches (rapidfire)."),
    nlaunches: int = typer.Option(0, "--nlaunches", help="Max launches for rapidfire (0 means until queue empty)."),
):
    """
    Submit a single-candidate evaluation workflow to FireWorks (optionally launch).
    """
    overrides: Dict[str, Any] = {}
    if params_file is not None:
        data = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter("--params-file must contain a JSON object (dict).")
        overrides.update(data)
    overrides.update(_parse_kv_params(param))

    try:
        from fireworks.core.rocket_launcher import rapidfire
        from gow.fw.launchpad import load_launchpad
        from gow.fw.workflow import SingleEvalSpec, build_single_evaluate_workflow
    except RuntimeError as e:
        raise typer.BadParameter(str(e)) from e
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    problem = load_problem_config(config_abs)
    launchers_dir = (launch_dir.expanduser().resolve() if launch_dir else default_launchers_dir(results_dir))

    lp = load_launchpad(launchpad)

    spec = SingleEvalSpec(
        problem_config=config_abs,
        outdir=results_dir,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_params=overrides,
    )
    wf = build_single_evaluate_workflow(spec)

    id_map = lp.add_wf(wf)
    fw_id = next(iter(id_map.values()), None) if isinstance(id_map, dict) else None

    typer.echo(f"Submitted workflow. id_map={id_map}  fw_id={fw_id}")
    typer.echo(f"Results dir: {results_dir}")
    typer.echo(f"Launchers dir: {launchers_dir}")

    if launch:
        with _pushd(launchers_dir):
            rapidfire(lp, nlaunches=nlaunches, sleep_time=sleep)
        typer.echo("Launch complete for current queue.")


@fw_app.command("run")
def fw_run_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to optimization specs (YAML/JSON)."),
    launchpad: Optional[Path] = typer.Option(None, "--launchpad", help="Path to my_launchpad.yaml (FireWorks LaunchPad config)."),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (defaults to UUID)."),
    launch: bool = typer.Option(True, "--launch/--no-launch", help="Launch immediately (rapidfire) after submitting."),
    launch_dir: Optional[Path] = typer.Option(
        None,
        "--launch-dir",
        help="Directory to place FireWorks launcher_* dirs (default: <outdir>/launchers).",
    ),
    sleep: int = typer.Option(0, "--sleep", help="Seconds to sleep between rocket launches (rapidfire)."),
    nlaunches: int = typer.Option(0, "--nlaunches", help="Max launches for rapidfire (0 means until queue empty)."),
):
    """
    Submit AND (optionally) launch a full optimization loop using FireWorks.

    NOTE: This is a simple synchronous loop (submit batch -> launch -> read results -> tell).
    """
    try:
        from fireworks.core.rocket_launcher import rapidfire
        from gow.fw.launchpad import load_launchpad
        from gow.fw.workflow import SingleEvalSpec, build_single_evaluate_workflow
        from gow.optimizer import make_optimizer
    except RuntimeError as e:
        raise typer.BadParameter(str(e)) from e
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    problem = load_problem_config(config_abs)
    lp = load_launchpad(launchpad)

    run_id_val = run_id or _default_run_id()
    launchers_dir = (launch_dir.expanduser().resolve() if launch_dir else default_launchers_dir(results_dir))

    opt_cfg = problem.optimizer
    opt_kwargs = _optimizer_kwargs(opt_cfg)

    name_norm = str(opt_cfg.name).lower().strip()
    if name_norm in {"differential_evolution", "de"}:
        opt_kwargs.setdefault("population_size", opt_cfg.batch_size)
        if opt_cfg.max_evaluations % opt_cfg.batch_size != 0:
            raise ValueError(
                "Differential Evolution requires max_evaluations to be a multiple of batch_size "
                "(one full population per generation). "
                f"Got max_evaluations={opt_cfg.max_evaluations}, batch_size={opt_cfg.batch_size}."
            )

    optimizer = make_optimizer(opt_cfg.name, seed=opt_cfg.seed, **opt_kwargs)

    typer.echo(f"Problem: {problem.id}")
    typer.echo(f"run_id:  {run_id_val}")
    typer.echo(f"results_dir: {results_dir}")
    typer.echo(f"launchers_dir: {launchers_dir}")
    typer.echo(f"max_evaluations={opt_cfg.max_evaluations}  batch_size={opt_cfg.batch_size}")

    def _read_candidate_fitness(workdir: Path) -> Dict[str, Any]:
        result_path = workdir / "result.json"
        if not result_path.exists():
            return {"status": "failed", "error": f"Missing result.json at {result_path}"}
        try:
            rec = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"status": "failed", "error": f"Failed to parse result.json: {e}"}
        fit = (rec.get("fitness") or {})
        if "status" not in fit:
            fit["status"] = "failed"
        return fit

    n_done = 0
    while n_done < opt_cfg.max_evaluations:
        n_batch = min(opt_cfg.batch_size, opt_cfg.max_evaluations - n_done)
        generation_id = n_done // opt_cfg.batch_size

        candidates = optimizer.ask(problem, n_batch)

        candidate_ids: List[str] = []
        for i, cand in enumerate(candidates):
            idx = n_done + i
            candidate_id = format_candidate_id(generation_id=generation_id, candidate_index=idx)
            candidate_ids.append(candidate_id)

            spec = SingleEvalSpec(
                problem_config=config_abs,
                outdir=results_dir,
                run_id=run_id_val,
                candidate_id=candidate_id,
                candidate_params=cand,
            )
            wf = build_single_evaluate_workflow(spec)
            lp.add_wf(wf)

        typer.echo(f"Submitted batch of {len(candidates)} candidate(s).")

        if launch:
            with _pushd(launchers_dir):
                rapidfire(lp, nlaunches=nlaunches, sleep_time=sleep)
            typer.echo("Launch complete for current queue.")

        fitness_dicts: List[Dict[str, Any]] = []
        for candidate_id in candidate_ids:
            workdir = candidate_workdir(results_dir, run_id_val, candidate_id)
            fitness_dicts.append(_read_candidate_fitness(workdir))

        try:
            optimizer.tell(candidates, fitness_dicts)
        except Exception as e:
            typer.echo(f"Warning: optimizer.tell failed: {e}")

        n_done += n_batch

    typer.echo("Done.")
    typer.echo(f"Results dir: {results_dir}")
    typer.echo(f"Results.jsonl: {results_dir / 'results.jsonl'}")
    typer.echo(f"Run results.jsonl: {run_root(results_dir, run_id_val) / 'results.jsonl'}")
    typer.echo(f"Launchers dir: {launchers_dir}")


if __name__ == "__main__":
    app()