from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.run.control import (
    pause_ack_path,
    pause_request_path,
)


BATCH_SIZE = 4
MAX_EVALUATIONS = 12
RUN_ID = "acor-real-process-resume"


def _read_jsonl(
    path: Path,
) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def _trajectory(
    rows: list[dict],
) -> list[dict]:
    return [
        {
            "generation_id": row["generation_id"],
            "candidate_index": row["candidate_index"],
            "candidate_id": row["candidate_id"],
            "candidate_local_id": row["candidate_local_id"],
            "attempt_id": row["attempt_id"],
            "params": row["params"],
            "fitness": row["fitness"],
        }
        for row in rows
    ]


def _run_driver(
    *,
    driver: Path,
    mode: str,
    config_path: Path,
    outdir: Path,
    repo_root: Path,
) -> dict:

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            mode,
            str(config_path),
            str(outdir),
            RUN_ID,
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, (
        "\n"
        f"MODE: {mode}\n"
        f"RETURN CODE: {completed.returncode}\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}\n"
    )

    output_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    assert output_lines, (
        f"Subprocess {mode!r} produced no stdout"
    )

    payload = json.loads(
        output_lines[-1]
    )

    assert payload["mode"] == mode
    assert isinstance(payload["pid"], int)
    assert payload["pid"] > 0
    assert isinstance(
        payload["process_instance_id"],
        str,
    )
    assert payload["process_instance_id"]

    return payload


def test_acor_pause_resume_survives_real_process_restart(
    tmp_path: Path,
) -> None:

    repo_root = Path(__file__).resolve().parents[2]

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # ============================================================
    # Evaluator
    # ============================================================

    evaluator = project_dir / "eval.py"

    evaluator.write_text(
        '''
import json
from pathlib import Path

payload = json.loads(
    Path("input.json").read_text(
        encoding="utf-8"
    )
)

x = float(payload["params"]["x"])
y = float(payload["params"]["y"])

objective = (
    2.0 * x * x
    + 0.5 * y * y
    - 0.3 * x * y
    + 0.1 * x
)

Path("output.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "metrics": {
                "objective": objective,
            },
            "objective": objective,
            "constraints": {},
            "artifacts": {},
            "error": None,
        }
    ),
    encoding="utf-8",
)
'''.lstrip(),
        encoding="utf-8",
    )

    # ============================================================
    # Problem config
    # ============================================================

    config = {
        "id": "acor-real-process-resume-test",

        "objective": {
            "direction": "minimize",
        },

        "parameters": {
            "x": {
                "type": "real",
                "value": 0.0,
                "bounds": [-5.0, 5.0],
            },

            "y": {
                "type": "real",
                "value": 0.0,
                "bounds": [-4.0, 4.0],
            },
        },

        "evaluator": {
            "command": [
                "{python}",
                str(evaluator.resolve()),
            ],
            "timeout_s": 30,
        },

        "optimizer": {
            "name": "acor",
            "seed": 24680,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": BATCH_SIZE,

            "settings": {
                "q": 0.18,
                "xi": 0.82,
                "max_generations": 10,
                "min_sigma": 1e-10,
                "bound_strategy": "resample",
            },
        },
    }

    config_path = project_dir / "problem.yaml"

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Driver executed by genuinely separate Python processes.
    # ============================================================

    driver = project_dir / "driver.py"

    driver.write_text(
        '''
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from gow.config import load_problem_config
from gow.run import (
    resume_local_optimization,
    run_local_optimization,
)


def main() -> int:

    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: driver.py MODE CONFIG OUTDIR RUN_ID"
        )

    mode = sys.argv[1]
    config_path = Path(sys.argv[2])
    outdir = Path(sys.argv[3])
    run_id = sys.argv[4]

    problem = load_problem_config(
        config_path
    )

    if mode == "run":
        result_path = run_local_optimization(
            problem,
            outdir=outdir,
            run_id=run_id,
        )

    elif mode == "resume":
        result_path = resume_local_optimization(
            problem,
            outdir=outdir,
            run_id=run_id,
        )

    else:
        raise SystemExit(
            f"unsupported mode: {mode}"
        )

    print(
        json.dumps(
            {
                "mode": mode,
                "pid": os.getpid(),
                "process_instance_id": uuid.uuid4().hex,
                "result_path": str(result_path),
            },
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.lstrip(),
        encoding="utf-8",
    )

    # ============================================================
    # A. Independent continuous process
    # ============================================================

    continuous_out = (
        tmp_path
        / "continuous"
    )

    continuous_process = _run_driver(
        driver=driver,
        mode="run",
        config_path=config_path,
        outdir=continuous_out,
        repo_root=repo_root,
    )

    continuous_run_root = (
        continuous_out
        / "runs"
        / RUN_ID
    )

    continuous_results = (
        continuous_run_root
        / "results.jsonl"
    )

    assert continuous_results.is_file()

    continuous_rows = _read_jsonl(
        continuous_results
    )

    assert len(continuous_rows) == MAX_EVALUATIONS

    continuous_checkpoint = CheckpointStore(
        continuous_run_root
    ).load()

    assert (
        continuous_checkpoint.manifest["status"]
        == "completed"
    )

    # ============================================================
    # B. Separate process that must PAUSE after generation zero
    # ============================================================

    resumed_out = (
        tmp_path
        / "paused-resumed"
    )

    resumed_run_root = (
        resumed_out
        / "runs"
        / RUN_ID
    )

    request_path = pause_request_path(
        resumed_run_root
    )

    request_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "pause",
                "request_id": "acor-real-process-pause-0001",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pause_process = _run_driver(
        driver=driver,
        mode="run",
        config_path=config_path,
        outdir=resumed_out,
        repo_root=repo_root,
    )

    # The Python process above is now completely gone.
    assert pause_process["pid"] != os.getpid()

    paused_results = (
        resumed_run_root
        / "results.jsonl"
    )

    assert paused_results.is_file()

    paused_rows = _read_jsonl(
        paused_results
    )

    assert len(paused_rows) == BATCH_SIZE

    paused_checkpoint = CheckpointStore(
        resumed_run_root
    ).load()

    assert (
        paused_checkpoint.manifest["status"]
        == "paused"
    )

    assert (
        paused_checkpoint.manifest["evaluations_done"]
        == BATCH_SIZE
    )

    assert (
        paused_checkpoint.manifest["next_generation"]
        == 1
    )

    assert not request_path.exists()

    ack_path = pause_ack_path(
        resumed_run_root
    )

    assert ack_path.is_file()

    ack = json.loads(
        ack_path.read_text(
            encoding="utf-8",
        )
    )

    assert ack["status"] == "paused"

    assert (
        ack["request_id"]
        == "acor-real-process-pause-0001"
    )

    # ============================================================
    # C. SECOND independent process performs RESUME
    # ============================================================

    resume_process = _run_driver(
        driver=driver,
        mode="resume",
        config_path=config_path,
        outdir=resumed_out,
        repo_root=repo_root,
    )

    # Explicitly prove that PAUSE and RESUME came from
    # different driver invocations. PIDs cannot be used as a
    # permanent identity because the OS may reuse them after exit.
    assert (
        resume_process["process_instance_id"]
        != pause_process["process_instance_id"]
    )

    # The resume driver is also a real subprocess, not pytest.
    assert (
        resume_process["pid"]
        != os.getpid()
    )

    # ============================================================
    # D. Finished resumed run
    # ============================================================

    resumed_results = (
        resumed_run_root
        / "results.jsonl"
    )

    assert resumed_results.is_file()

    resumed_rows = _read_jsonl(
        resumed_results
    )

    assert len(resumed_rows) == MAX_EVALUATIONS

    assert [
        row["candidate_index"]
        for row in resumed_rows
    ] == list(
        range(MAX_EVALUATIONS)
    )

    assert [
        row["generation_id"]
        for row in resumed_rows
    ] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    # ============================================================
    # E. Exact mathematical trajectory
    # ============================================================

    assert (
        _trajectory(resumed_rows)
        == _trajectory(continuous_rows)
    )

    # ============================================================
    # F. Exact final optimizer state
    # ============================================================

    resumed_checkpoint = CheckpointStore(
        resumed_run_root
    ).load()

    assert (
        resumed_checkpoint.manifest["status"]
        == "completed"
    )

    assert (
        resumed_checkpoint.manifest["evaluations_done"]
        == MAX_EVALUATIONS
    )

    assert (
        resumed_checkpoint.manifest["completed_generations"]
        == 3
    )

    assert (
        resumed_checkpoint.optimizer_state
        == continuous_checkpoint.optimizer_state
    )

    # ============================================================
    # G. Final best identical
    # ============================================================

    continuous_summary = json.loads(
        (
            continuous_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    resumed_summary = json.loads(
        (
            resumed_run_root
            / "summary.json"
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        resumed_summary["best"]
        == continuous_summary["best"]
    )

    # Continuous, PAUSE and RESUME are three independent
    # driver invocations. Use process-instance identity rather
    # than PID because sequential subprocesses may reuse a PID.
    assert (
        continuous_process["process_instance_id"]
        != pause_process["process_instance_id"]
    )

    assert (
        continuous_process["process_instance_id"]
        != resume_process["process_instance_id"]
    )

    assert (
        pause_process["process_instance_id"]
        != resume_process["process_instance_id"]
    )
