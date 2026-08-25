from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from gow.config import load_problem_config
from gow.optimizer.cmaes import CMAESOptimizer


BATCH_SIZE = 6
SEED = 24680
SIGMA0 = 0.18
MAX_GENERATIONS = 20


def _problem(
    tmp_path: Path,
):
    noop = tmp_path / "noop.py"

    noop.write_text(
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )

    config = {
        "id": "cmaes-checkpoint-test",

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
                str(noop.resolve()),
            ],
            "timeout_s": 30,
        },

        "optimizer": {
            "name": "cmaes",
            "seed": SEED,
            "max_evaluations": 24,
            "batch_size": BATCH_SIZE,

            "settings": {
                "sigma0": SIGMA0,
                "max_generations": MAX_GENERATIONS,
            },
        },
    }

    path = (
        tmp_path
        / "problem.yaml"
    )

    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return load_problem_config(
        path
    )


def _optimizer(
    *,
    batch_size=BATCH_SIZE,
    sigma0=SIGMA0,
    max_generations=MAX_GENERATIONS,
    seed=SEED,
) -> CMAESOptimizer:

    return CMAESOptimizer(
        batch_size=batch_size,
        sigma0=sigma0,
        max_generations=max_generations,
        seed=seed,
    )


def _fitness(
    candidates,
):
    rows = []

    for candidate in candidates:

        x = float(
            candidate["x"]
        )

        y = float(
            candidate["y"]
        )

        objective = (
            1.7 * x * x
            + 0.55 * y * y
            - 0.21 * x * y
            + 0.04 * x
        )

        rows.append(
            {
                "status": "ok",
                "objective": objective,
                "metrics": {
                    "objective": objective,
                },
            }
        )

    return rows


def _complete_generation(
    optimizer,
    problem,
):
    candidates = optimizer.ask(
        problem,
        BATCH_SIZE,
    )

    optimizer.tell(
        candidates,
        _fitness(candidates),
    )

    return candidates


def _without_pickle(
    state: dict,
) -> dict:
    result = dict(
        state
    )

    result.pop(
        "es_pickle",
        None,
    )

    return result


def test_cmaes_checkpoint_round_trip_and_exact_continuation(
    tmp_path: Path,
) -> None:

    problem = _problem(
        tmp_path
    )

    original = _optimizer()

    _complete_generation(
        original,
        problem,
    )

    state = original.state_dict()

    assert state["schema_version"] == 1
    assert state["optimizer"] == "cmaes"
    assert state["initialized"] is True
    assert state["generation"] == 1
    assert state["last_xs"] == []
    assert state["es_countiter"] == 1
    assert state["es_countevals"] == BATCH_SIZE
    assert isinstance(
        state["es_pickle"],
        bytes,
    )
    assert state["es_pickle"]

    # Simulate the object construction performed by the GOW
    # optimizer registry during a future resume. batch_size can
    # still be None before the first ask().
    restored = _optimizer(
        batch_size=None,
    )

    restored.load_state_dict(
        state
    )

    assert restored.batch_size == BATCH_SIZE
    assert restored.population_size == BATCH_SIZE
    assert restored.diagnostics() == original.diagnostics()

    # Continue several complete generations. Every candidate
    # must remain exactly identical.
    for _ in range(3):

        original_candidates = original.ask(
            problem,
            BATCH_SIZE,
        )

        restored_candidates = restored.ask(
            problem,
            BATCH_SIZE,
        )

        assert (
            restored_candidates
            == original_candidates
        )

        original_fitness = _fitness(
            original_candidates
        )

        restored_fitness = _fitness(
            restored_candidates
        )

        assert (
            restored_fitness
            == original_fitness
        )

        original.tell(
            original_candidates,
            original_fitness,
        )

        restored.tell(
            restored_candidates,
            restored_fitness,
        )

    original_state = (
        original.state_dict()
    )

    restored_state = (
        restored.state_dict()
    )

    assert (
        _without_pickle(
            restored_state
        )
        == _without_pickle(
            original_state
        )
    )


def test_cmaes_checkpoint_rejects_mid_generation(
    tmp_path: Path,
) -> None:

    problem = _problem(
        tmp_path
    )

    optimizer = _optimizer()

    optimizer.ask(
        problem,
        BATCH_SIZE,
    )

    with pytest.raises(
        RuntimeError,
        match="between generations",
    ):
        optimizer.state_dict()


def test_cmaes_checkpoint_rejects_incompatible_configuration(
    tmp_path: Path,
) -> None:

    problem = _problem(
        tmp_path
    )

    original = _optimizer()

    _complete_generation(
        original,
        problem,
    )

    state = original.state_dict()

    incompatible = _optimizer(
        batch_size=None,
        sigma0=0.25,
    )

    with pytest.raises(
        ValueError,
        match="sigma0 mismatch",
    ):
        incompatible.load_state_dict(
            state
        )


def test_cmaes_checkpoint_rejects_cma_version_mismatch(
    tmp_path: Path,
) -> None:

    problem = _problem(
        tmp_path
    )

    original = _optimizer()

    _complete_generation(
        original,
        problem,
    )

    state = copy.deepcopy(
        original.state_dict()
    )

    state["cma_version"] = (
        "definitely-not-installed"
    )

    restored = _optimizer(
        batch_size=None,
    )

    with pytest.raises(
        ValueError,
        match="version mismatch",
    ):
        restored.load_state_dict(
            state
        )


def test_cmaes_checkpoint_rejects_corrupt_internal_pickle(
    tmp_path: Path,
) -> None:

    problem = _problem(
        tmp_path
    )

    original = _optimizer()

    _complete_generation(
        original,
        problem,
    )

    state = dict(
        original.state_dict()
    )

    state["es_pickle"] = (
        b"not-a-valid-cma-pickle"
    )

    restored = _optimizer(
        batch_size=None,
    )

    with pytest.raises(
        ValueError,
        match="invalid internal strategy pickle",
    ):
        restored.load_state_dict(
            state
        )
