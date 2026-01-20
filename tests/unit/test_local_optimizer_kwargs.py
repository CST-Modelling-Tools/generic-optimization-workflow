# tests/unit/test_local_optimizer_kwargs.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

from gow.run import local


# -----------------------------
# Minimal fakes for unit testing
# -----------------------------


@dataclass
class FakeOptimizerConfig:
    name: str
    seed: int | None
    batch_size: int
    settings: Dict[str, Any] = field(default_factory=dict)

    # Emulate "flattened" keys living directly on optimizer config
    # (legacy style): e.g. mutation_factor, crossover_rate, etc.
    flattened_extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Put flattened keys onto the instance so local._optimizer_kwargs can pick them up
        # via attribute / __dict__ inspection (as your implementation does).
        for k, v in self.flattened_extra.items():
            setattr(self, k, v)


@dataclass
class FakeProblemConfig:
    optimizer: FakeOptimizerConfig


# -----------------------------
# Tests
# -----------------------------


def test_de_defaults_population_size_to_batch_size_when_missing() -> None:
    opt = FakeOptimizerConfig(
        name="differential_evolution",
        seed=123,
        batch_size=50,
        settings={},           # no population_size
        flattened_extra={},    # no population_size
    )
    problem = FakeProblemConfig(optimizer=opt)

    opt_kwargs = local._optimizer_kwargs(problem)

    assert opt_kwargs["population_size"] == 50


def test_seed_passed_exactly_once_to_make_optimizer() -> None:
    opt = FakeOptimizerConfig(
        name="differential_evolution",
        seed=123,
        batch_size=50,
        settings={},           # seed not here
        flattened_extra={},    # seed not here
    )
    problem = FakeProblemConfig(optimizer=opt)

    opt_kwargs = local._optimizer_kwargs(problem)

    # "passed exactly once" here means: there's a single kwarg key "seed"
    # and its value comes from optimizer.seed.
    assert list(k for k in opt_kwargs.keys() if k == "seed") == ["seed"]
    assert opt_kwargs["seed"] == 123


def test_flattened_and_settings_are_merged_settings_override_flattened() -> None:
    opt = FakeOptimizerConfig(
        name="differential_evolution",
        seed=123,
        batch_size=50,
        flattened_extra={
            "mutation_factor": 0.5,
            "crossover_rate": 0.1,
        },
        settings={
            "mutation_factor": 0.8,  # overrides flattened 0.5
            "crossover_rate": 0.9,   # overrides flattened 0.1
            "max_generations": 10,   # new key from settings
        },
    )
    problem = FakeProblemConfig(optimizer=opt)

    opt_kwargs = local._optimizer_kwargs(problem)

    # merged + override behavior
    assert opt_kwargs["mutation_factor"] == 0.8
    assert opt_kwargs["crossover_rate"] == 0.9
    assert opt_kwargs["max_generations"] == 10

    # and seed is still injected
    assert opt_kwargs["seed"] == 123

    # and DE default still applies if population_size missing
    assert opt_kwargs["population_size"] == 50