from __future__ import annotations

from .base import Optimizer
from .random_search import RandomSearchOptimizer
from .differential_evolution import DifferentialEvolutionOptimizer
from .cmaes import CMAESOptimizer
from .bayesian_optimization import BayesianOptimizationOptimizer
<<<<<<< HEAD
from .saasbo import SAASBOOptimizer
from .genetic_algorithm import GeneticAlgorithmOptimizer
=======

>>>>>>> origin/gow-optimizers-htc-unified

def make_optimizer(name: str, *, seed: int | None = None, **kwargs) -> Optimizer:
    name = name.lower().strip()

    if name in {"random", "random_search", "rand"}:
        return RandomSearchOptimizer(seed=seed)

    if name in {"differential_evolution", "de"}:
        # kwargs may include: population_size, mutation_factor, crossover_rate, max_generations
        return DifferentialEvolutionOptimizer(seed=seed, **kwargs)

    if name in {"cmaes", "cma-es", "covariance_matrix_adaptation"}:
        return CMAESOptimizer(seed=seed, **kwargs)

    if name in {"bayesian", "bayesian_optimization", "bo"}:
        return BayesianOptimizationOptimizer(seed=seed, **kwargs)

<<<<<<< HEAD
    if name in {"saasbo", "saas_bo"}:
        return SAASBOOptimizer(seed=seed, **kwargs)

    if name in {"ga", "genetic", "genetic_algorithm"}:
        return GeneticAlgorithmOptimizer(seed=seed, **kwargs)

    raise ValueError(f"Unknown optimizer: {name}")
=======
    raise ValueError(f"Unknown optimizer: {name}")
>>>>>>> origin/gow-optimizers-htc-unified
