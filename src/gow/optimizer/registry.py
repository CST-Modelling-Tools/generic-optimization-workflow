from __future__ import annotations

from .base import Optimizer
from .random_search import RandomSearchOptimizer
from .differential_evolution import DifferentialEvolutionOptimizer


def make_optimizer(name: str, *, seed: int | None = None, **kwargs) -> Optimizer:
    name = name.lower().strip()

    if name in {"random", "random_search", "rand"}:
        return RandomSearchOptimizer(seed=seed)

    if name in {"differential_evolution", "de"}:
        # kwargs may include: population_size, mutation_factor, crossover_rate, max_generations
        return DifferentialEvolutionOptimizer(seed=seed, **kwargs)

    raise ValueError(f"Unknown optimizer: {name}")