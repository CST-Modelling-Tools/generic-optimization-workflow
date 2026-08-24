from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gow.checkpoint import CheckpointStore
from gow.config.models import ProblemConfig
from gow.optimizer.acor import ACOROptimizer


BATCH_SIZE = 4


def _problem() -> ProblemConfig:
    return ProblemConfig.model_validate(
        {
            "id": "acor-checkpoint-test",

            "objective": {
                "direction": "minimize",
            },

            "parameters": {
                "x": {
                    "type": "real",
                    "value": 0.0,
                    "bounds": [-4.0, 4.0],
                },

                "y": {
                    "type": "real",
                    "value": 0.0,
                    "bounds": [-3.0, 3.0],
                },

                "count": {
                    "type": "int",
                    "value": 2,
                    "bounds": [0, 7],
                },
            },

            # The optimizer tests never execute this evaluator.
            "evaluator": {
                "command": [
                    "unused-evaluator",
                ],
            },

            "optimizer": {
                "name": "acor",
                "seed": 12345,
                "max_evaluations": 20,
                "batch_size": BATCH_SIZE,
            },
        }
    )


def _fitness(
    candidates: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    for candidate in candidates:

        x = float(
            candidate["x"]
        )
        y = float(
            candidate["y"]
        )
        count = int(
            candidate["count"]
        )

        objective = (
            x * x
            + 2.0 * y * y
            + 0.125 * x * y
            + 0.05 * count
        )

        rows.append(
            {
                "status": "ok",
                "objective": objective,
            }
        )

    return rows


def _optimizer(
    *,
    seed: int,
    q: float = 0.17,
    batch_size: int | None = BATCH_SIZE,
) -> ACOROptimizer:

    return ACOROptimizer(
        batch_size=batch_size,
        q=q,
        xi=0.79,
        max_generations=10,
        min_sigma=1e-10,
        bound_strategy="resample",
        seed=seed,
    )


def test_acor_checkpoint_round_trip_and_exact_continuation(
    tmp_path: Path,
) -> None:

    problem = _problem()

    original = _optimizer(
        seed=12345,
    )

    first_candidates = original.ask(
        problem,
        BATCH_SIZE,
    )

    original.tell(
        first_candidates,
        _fitness(first_candidates),
    )

    state = original.state_dict()

    assert state["schema_version"] == 1
    assert state["optimizer"] == "acor"
    assert state["generation"] == 1
    assert state["awaiting_tell"] is False

    assert len(
        state["archive"]
    ) == BATCH_SIZE

    # Persist through the real CheckpointStore so RNG tuples and the complete
    # archive make a genuine disk round-trip.
    run_root = (
        tmp_path
        / "runs"
        / "acor-run"
    )

    store = CheckpointStore(
        run_root
    )

    store.save(
        manifest={
            "schema_version": 1,
            "run_id": "acor-run",
            "problem_id": problem.id,
            "status": "paused",
            "optimizer": "acor",
            "evaluations_done": BATCH_SIZE,
            "completed_generations": 1,
            "next_generation": 1,
            "max_evaluations": 20,
        },
        optimizer_state=state,
    )

    loaded = store.load()

    restored = _optimizer(
        # Deliberately different seed:
        # continuation must come from persisted RNG state.
        seed=999999,
    )

    restored.load_state_dict(
        loaded.optimizer_state
    )

    assert (
        restored.state_dict()
        == state
    )

    next_original = original.ask(
        problem,
        BATCH_SIZE,
    )

    next_restored = restored.ask(
        problem,
        BATCH_SIZE,
    )

    assert (
        next_restored
        == next_original
    )

    next_fitness = _fitness(
        next_original
    )

    original.tell(
        next_original,
        next_fitness,
    )

    restored.tell(
        next_restored,
        next_fitness,
    )

    assert (
        restored.state_dict()
        == original.state_dict()
    )


def test_acor_checkpoint_rejects_mid_generation_state() -> None:

    problem = _problem()

    optimizer = _optimizer(
        seed=123,
    )

    first = optimizer.ask(
        problem,
        BATCH_SIZE,
    )

    optimizer.tell(
        first,
        _fitness(first),
    )

    # A new ask() advances the RNG but has not yet updated the archive.
    optimizer.ask(
        problem,
        BATCH_SIZE,
    )

    with pytest.raises(
        RuntimeError,
        match="between generations",
    ):
        optimizer.state_dict()


def test_acor_checkpoint_rejects_configuration_mismatch() -> None:

    problem = _problem()

    original = _optimizer(
        seed=321,
    )

    candidates = original.ask(
        problem,
        BATCH_SIZE,
    )

    original.tell(
        candidates,
        _fitness(candidates),
    )

    state = original.state_dict()

    different_q = _optimizer(
        seed=111,
        q=0.25,
    )

    with pytest.raises(
        ValueError,
        match="configuration mismatch",
    ):
        different_q.load_state_dict(
            state
        )


def test_acor_checkpoint_adopts_batch_size_when_constructor_has_none() -> None:

    problem = _problem()

    original = _optimizer(
        seed=555,
    )

    candidates = original.ask(
        problem,
        BATCH_SIZE,
    )

    original.tell(
        candidates,
        _fitness(candidates),
    )

    state = deepcopy(
        original.state_dict()
    )

    restored = _optimizer(
        seed=987654,
        batch_size=None,
    )

    assert restored.batch_size is None

    restored.load_state_dict(
        state
    )

    assert (
        restored.batch_size
        == BATCH_SIZE
    )

    assert (
        restored.archive_size
        == BATCH_SIZE
    )

    assert (
        restored.ants
        == BATCH_SIZE
    )

    assert (
        restored.state_dict()
        == state
    )
