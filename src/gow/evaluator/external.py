from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .io import read_output_json, write_input_json
from .models import FitnessResult


@dataclass(frozen=True)
class ExternalRunResult:
    fitness: FitnessResult
    returncode: int
    wall_time_s: float
    stdout_path: Path
    stderr_path: Path
    input_path: Path
    output_path: Path
    workdir: Path


class EvaluatorExecutionError(RuntimeError):
    pass


def run_external_evaluator(
    *,
    command: List[str],
    workdir: str | Path,
    run_id: str,
    candidate_id: str,
    params: Dict,
    context: Optional[Dict] = None,
    timeout_s: int = 600,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    input_filename: str = "input.json",
    output_filename: str = "output.json",
) -> ExternalRunResult:
    """
    Run an external evaluator command following the contract:
      <command...> --input input.json --output output.json

    Writes input.json, runs the command, captures stdout/stderr,
    reads output.json into FitnessResult (or produces a failed FitnessResult on errors).
    """
    if not command or not isinstance(command, list):
        raise ValueError("command must be a non-empty list of strings")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / input_filename
    output_path = workdir / output_filename
    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"

    write_input_json(
        input_path,
        run_id=run_id,
        candidate_id=candidate_id,
        params=params,
        context=context or {},
    )

    cmd = list(command)
    cmd.extend(["--input", input_filename, "--output", output_filename])
    if extra_args:
        cmd.extend(extra_args)

    proc_env = os.environ.copy()
    if env:
        proc_env.update({str(k): str(v) for k, v in env.items()})

    t0 = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open("w", encoding="utf-8") as f_err:
            completed = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=proc_env,
                stdout=f_out,
                stderr=f_err,
                timeout=timeout_s,
                check=False,
                text=True,
            )
        wall = time.time() - t0

        # Try to parse evaluator output.json if it exists
        if output_path.exists():
            fitness = read_output_json(output_path)
        else:
            fitness = FitnessResult(
                status="failed",
                metrics={},
                objective=None,
                error=f"Evaluator did not produce {output_filename}. Return code: {completed.returncode}",
            )

        return ExternalRunResult(
            fitness=fitness,
            returncode=completed.returncode,
            wall_time_s=wall,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            input_path=input_path,
            output_path=output_path,
            workdir=workdir,
        )

    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        # best effort: record a failure fitness
        fitness = FitnessResult(
            status="failed",
            metrics={},
            objective=None,
            error=f"Evaluator timed out after {timeout_s} seconds",
            constraints={"timeout_s": timeout_s, "wall_time_s": wall},
        )
        return ExternalRunResult(
            fitness=fitness,
            returncode=124,
            wall_time_s=wall,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            input_path=input_path,
            output_path=output_path,
            workdir=workdir,
        )