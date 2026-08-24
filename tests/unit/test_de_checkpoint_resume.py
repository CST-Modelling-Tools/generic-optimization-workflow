from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gow.checkpoint import CheckpointStore
from gow.config.models import (
    ExternalEvaluatorConfig,
    ObjectiveConfig,
    OptimizerConfig,
    ProblemConfig,
    RealParam,
)
from gow.optimizer.differential_evolution import (
    DifferentialEvolutionOptimizer,
)


POPULATION_SIZE = 8


def _problem() -> ProblemConfig:
    """Create a small deterministic continuous problem for optimizer tests."""

    return ProblemConfig(
        id="de-checkpoint-resume-test",
        parameters={
            "x": RealParam(
                value=0.25,
                bounds=[-5.0, 5.0],
                optimizable=True,
            ),
            "y": RealParam(
                value=-0.50,
                bounds=[-3.0, 3.0],
                optimizable=True,
            ),
            "z": RealParam(
                value=1.00,
                bounds=[-2.0, 2.0],
                optimizable=True,
            ),
        },
        evaluator=ExternalEvaluatorConfig(
            command=["checkpoint-test-evaluator"],
        ),
        objective=ObjectiveConfig(
            direction="minimize",
        ),
        optimizer=OptimizerConfig(
            name="differential_evolution",
            seed=123456,
            max_evaluations=POPULATION_SIZE * 10,
            batch_size=POPULATION_SIZE,
        ),
    )


def _optimizer(seed: int) -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(
        population_size=POPULATION_SIZE,
        mutation_factor=0.73,
        crossover_rate=0.91,
        max_generations=10,
        seed=seed,
    )


def _fitness(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministic sphere-like objective used only by this unit test."""

    results: list[dict[str, Any]] = []

    for candidate in candidates:
        x = float(candidate["x"])
        y = float(candidate["y"])
        z = float(candidate["z"])

        objective = (
            x * x
            + 2.0 * y * y
            + 0.5 * z * z
        )

        results.append(
            {
                "status": "ok",
                "objective": objective,
            }
        )

    return results


def _complete_generation(
    optimizer: DifferentialEvolutionOptimizer,
    problem: ProblemConfig,
) -> list[dict[str, Any]]:
    candidates = optimizer.ask(
        problem,
        POPULATION_SIZE,
    )

    optimizer.tell(
        candidates,
        _fitness(candidates),
    )

    return candidates


def test_de_checkpoint_resume_preserves_exact_candidate_sequence(
    tmp_path: Path,
) -> None:
    problem = _problem()

    continuous = _optimizer(seed=123456)

    # Complete two generations normally.
    _complete_generation(continuous, problem)
    _complete_generation(continuous, problem)

    assert continuous.diagnostics() == {}

    state = continuous.state_dict()

    assert state["schema_version"] == 1
    assert state["optimizer"] == "differential_evolution"
    assert state["generation"] == 2
    assert state["last_targets"] == []

    # Exercise the real CheckpointStore, not only an in-memory dictionary.
    run_dir = tmp_path / "runs" / "de-resume-run"
    store = CheckpointStore(run_dir)

    store.save(
        manifest={
            "schema_version": 1,
            "run_id": "de-resume-run",
            "problem_id": problem.id,
            "status": "paused",
            "optimizer": "differential_evolution",
            "evaluations_done": POPULATION_SIZE * 2,
            "completed_generations": 2,
            "next_generation": 2,
            "max_evaluations": POPULATION_SIZE * 10,
        },
        optimizer_state=state,
    )

    loaded = store.load()

    # Use a deliberately different seed here.
    # load_state_dict() must replace its RNG state with the checkpoint RNG state.
    restored = _optimizer(seed=999999)
    restored.load_state_dict(loaded.optimizer_state)

    # First generation produced after restoration must be bit-for-bit equal
    # to what the uninterrupted optimizer would have produced.
    continuous_generation_2 = continuous.ask(
        problem,
        POPULATION_SIZE,
    )

    restored_generation_2 = restored.ask(
        problem,
        POPULATION_SIZE,
    )

    assert restored_generation_2 == continuous_generation_2

    continuous.tell(
        continuous_generation_2,
        _fitness(continuous_generation_2),
    )

    restored.tell(
        restored_generation_2,
        _fitness(restored_generation_2),
    )

    # Verify one further generation as well. This checks that tell() evolves
    # both restored and uninterrupted states identically after the checkpoint.
    continuous_generation_3 = continuous.ask(
        problem,
        POPULATION_SIZE,
    )

    restored_generation_3 = restored.ask(
        problem,
        POPULATION_SIZE,
    )

    assert restored_generation_3 == continuous_generation_3


def test_de_checkpoint_rejects_mid_generation_snapshot() -> None:
    problem = _problem()
    optimizer = _optimizer(seed=123456)

    optimizer.ask(
        problem,
        POPULATION_SIZE,
    )

    with pytest.raises(
        RuntimeError,
        match="between generations",
    ):
        optimizer.state_dict()


def test_de_checkpoint_rejects_incompatible_configuration() -> None:
    problem = _problem()

    original = _optimizer(seed=123456)
    _complete_generation(original, problem)

    state = original.state_dict()

    incompatible = DifferentialEvolutionOptimizer(
        population_size=POPULATION_SIZE,
        mutation_factor=0.50,
        crossover_rate=0.91,
        max_generations=10,
        seed=123456,
    )

    with pytest.raises(
        ValueError,
        match="configuration mismatch",
    ):
        incompatible.load_state_dict(state)
