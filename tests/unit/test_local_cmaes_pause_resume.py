from __future__ import annotations

import json
import pickle
from pathlib import Path

import yaml

from gow.checkpoint import CheckpointStore
from gow.config import load_problem_config
from gow.run import resume_local_optimization, run_local_optimization
from gow.run.control import pause_ack_path, pause_request_path


BATCH_SIZE = 6
MAX_EVALUATIONS = 18
RUN_ID = "cmaes-resume-determinism"
REQUEST_ID = "cmaes-resume-request-0001"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trajectory(rows: list[dict]) -> list[dict]:
    keys = (
        "generation_id",
        "candidate_index",
        "candidate_id",
        "candidate_local_id",
        "attempt_id",
        "params",
        "fitness",
    )

    return [
        {
            key: row[key]
            for key in keys
        }
        for row in rows
    ]


def _write_project(root: Path) -> Path:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator = root / "eval.py"

    evaluator.write_text(
        """
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
    1.7 * x * x
    + 0.55 * y * y
    - 0.21 * x * y
    + 0.04 * x
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
""".lstrip(),
        encoding="utf-8",
    )

    config = {
        "id": "cmaes-resume-determinism-test",
        "objective": {
            "direction": "minimize",
        },
        "parameters": {
            "x": {
                "type": "real",
                "value": 0.20,
                "bounds": [-4.0, 4.0],
            },
            "y": {
                "type": "real",
                "value": -0.15,
                "bounds": [-3.0, 3.0],
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
            "name": "cmaes",
            "seed": 24680,
            "max_evaluations": MAX_EVALUATIONS,
            "batch_size": BATCH_SIZE,
            "settings": {
                "sigma0": 0.18,
                "max_generations": 20,
            },
        },
    }

    config_path = root / "problem.yaml"

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def test_local_cmaes_pause_resume_is_exact(
    tmp_path: Path,
) -> None:

    config_path = _write_project(
        tmp_path / "project"
    )

    # ============================================================
    # A. Ejecución continua de referencia
    # ============================================================

    continuous_problem = load_problem_config(
        config_path
    )

    continuous_outdir = (
        tmp_path / "continuous"
    )

    continuous_result = run_local_optimization(
        continuous_problem,
        outdir=continuous_outdir,
        run_id=RUN_ID,
    )

    continuous_rows = _read_jsonl(
        continuous_result
    )

    assert (
        len(continuous_rows)
        == MAX_EVALUATIONS
    )

    continuous_root = (
        continuous_outdir
        / "runs"
        / RUN_ID
    )

    continuous_checkpoint = CheckpointStore(
        continuous_root
    ).load()

    assert (
        continuous_checkpoint.manifest["status"]
        == "completed"
    )

    assert (
        continuous_checkpoint.manifest[
            "evaluations_done"
        ]
        == MAX_EVALUATIONS
    )

    # ============================================================
    # B. Ejecución con PAUSE tras la primera generación
    # ============================================================

    resumed_problem = load_problem_config(
        config_path
    )

    resumed_outdir = (
        tmp_path / "resumed"
    )

    resumed_root = (
        resumed_outdir
        / "runs"
        / RUN_ID
    )

    request_path = pause_request_path(
        resumed_root
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
                "request_id": REQUEST_ID,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    paused_result = run_local_optimization(
        resumed_problem,
        outdir=resumed_outdir,
        run_id=RUN_ID,
    )

    paused_rows = _read_jsonl(
        paused_result
    )

    assert (
        len(paused_rows)
        == BATCH_SIZE
    )

    paused_checkpoint = CheckpointStore(
        resumed_root
    ).load()

    paused_manifest = (
        paused_checkpoint.manifest
    )

    assert (
        paused_manifest["status"]
        == "paused"
    )

    assert (
        paused_manifest["evaluations_done"]
        == BATCH_SIZE
    )

    assert (
        paused_manifest["completed_generations"]
        == 1
    )

    assert (
        paused_manifest["next_generation"]
        == 1
    )

    paused_state = (
        paused_checkpoint.optimizer_state
    )

    assert (
        paused_state["optimizer"]
        == "cmaes"
    )

    assert (
        paused_state["generation"]
        == 1
    )

    assert (
        paused_state["last_xs"]
        == []
    )

    assert (
        paused_state["es_countiter"]
        == 1
    )

    assert (
        paused_state["es_countevals"]
        == BATCH_SIZE
    )

    assert not request_path.exists()

    ack_path = pause_ack_path(
        resumed_root
    )

    assert ack_path.is_file()

    ack = json.loads(
        ack_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        ack["status"]
        == "paused"
    )

    assert (
        ack["request_id"]
        == REQUEST_ID
    )

    # ============================================================
    # C. RESUME
    # ============================================================

    final_result = resume_local_optimization(
        resumed_problem,
        outdir=resumed_outdir,
        run_id=RUN_ID,
    )

    resumed_rows = _read_jsonl(
        final_result
    )

    assert (
        len(resumed_rows)
        == MAX_EVALUATIONS
    )

    assert [
        row["candidate_index"]
        for row in resumed_rows
    ] == list(
        range(MAX_EVALUATIONS)
    )

    assert [
        row["generation_id"]
        for row in resumed_rows
    ] == (
        [0] * BATCH_SIZE
        + [1] * BATCH_SIZE
        + [2] * BATCH_SIZE
    )

    # ============================================================
    # D. Trayectoria completa exacta
    #
    # Ya compara:
    # - generation_id
    # - candidate_index
    # - candidate_id
    # - candidate_local_id
    # - attempt_id
    # - params
    # - fitness
    # ============================================================

    assert (
        _trajectory(resumed_rows)
        == _trajectory(continuous_rows)
    )

    # ============================================================
    # E. Checkpoint final
    # ============================================================

    resumed_checkpoint = CheckpointStore(
        resumed_root
    ).load()

    resumed_manifest = (
        resumed_checkpoint.manifest
    )

    assert (
        resumed_manifest["status"]
        == "completed"
    )

    assert (
        resumed_manifest["evaluations_done"]
        == MAX_EVALUATIONS
    )

    assert (
        resumed_manifest["completed_generations"]
        == 3
    )

    assert (
        resumed_manifest["next_generation"]
        == 3
    )

    # ============================================================
    # F. Estado GOW exacto
    #
    # es_pickle NO se compara byte a byte porque contiene estado
    # auxiliar/runtime interno de pycma.
    # ============================================================

    continuous_state = (
        continuous_checkpoint.optimizer_state
    )

    resumed_state = (
        resumed_checkpoint.optimizer_state
    )

    assert (
        set(resumed_state)
        == set(continuous_state)
    )

    continuous_without_pickle = dict(
        continuous_state
    )

    resumed_without_pickle = dict(
        resumed_state
    )

    continuous_pickle = (
        continuous_without_pickle.pop(
            "es_pickle"
        )
    )

    resumed_pickle = (
        resumed_without_pickle.pop(
            "es_pickle"
        )
    )

    assert (
        resumed_without_pickle
        == continuous_without_pickle
    )

    assert isinstance(
        continuous_pickle,
        bytes,
    )

    assert isinstance(
        resumed_pickle,
        bytes,
    )

    assert continuous_pickle
    assert resumed_pickle

    # ============================================================
    # G. Estado matemático pycma
    # ============================================================

    continuous_es = pickle.loads(
        continuous_pickle
    )

    resumed_es = pickle.loads(
        resumed_pickle
    )

    assert (
        int(resumed_es.countiter)
        == int(continuous_es.countiter)
    )

    assert (
        int(resumed_es.countevals)
        == int(continuous_es.countevals)
    )

    assert (
        float(resumed_es.sigma)
        == float(continuous_es.sigma)
    )

    assert [
        float(value)
        for value in resumed_es.mean
    ] == [
        float(value)
        for value in continuous_es.mean
    ]

    # ============================================================
    # H. Prueba fuerte de continuación
    #
    # Dos estados equivalentes deben generar exactamente la misma
    # siguiente población CMA-ES.
    # ============================================================

    continuous_next = [
        [
            float(value)
            for value in candidate
        ]
        for candidate in continuous_es.ask(
            number=BATCH_SIZE
        )
    ]

    resumed_next = [
        [
            float(value)
            for value in candidate
        ]
        for candidate in resumed_es.ask(
            number=BATCH_SIZE
        )
    ]

    assert (
        resumed_next
        == continuous_next
    )

    # best_score y best_candidate ya forman parte de los estados
    # GOW comparados exactamente arriba.