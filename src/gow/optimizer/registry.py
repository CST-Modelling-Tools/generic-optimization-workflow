from __future__ import annotations

from .base import Optimizer
from .random_search import RandomSearchOptimizer
from .differential_evolution import DifferentialEvolutionOptimizer
from .cmaes import CMAESOptimizer
from .bayesian_optimization import BayesianOptimizationOptimizer
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .asa import ASAOptimizer
from .acor import ACOROptimizer
from .particle_swarm import ParticleSwarmOptimizer

def make_optimizer(name: str, *, seed: int | None = None, **kwargs) -> Optimizer:
    name = name.lower().strip()

    if name in {"random", "random_search", "rand"}:
        return RandomSearchOptimizer(seed=seed)

    if name in {"differential_evolution", "de"}:
        return DifferentialEvolutionOptimizer(seed=seed, **kwargs)

    if name in {"cmaes", "cma-es", "covariance_matrix_adaptation"}:
        return CMAESOptimizer(seed=seed, **kwargs)

    if name in {"bayesian", "bayesian_optimization", "bo"}:
        return BayesianOptimizationOptimizer(seed=seed, **kwargs)

    if name in {"saasbo", "saas_bo"}:
        from .saasbo import SAASBOOptimizer
        return SAASBOOptimizer(seed=seed, **kwargs)

    if name in {"ga", "genetic", "genetic_algorithm"}:
        return GeneticAlgorithmOptimizer(seed=seed, **kwargs)
    
    if name in {"asa", "adaptive_simulated_annealing", "adaptive_sa"}:
        return ASAOptimizer(seed=seed, **kwargs)

    if name in {"acor"}:
        return ACOROptimizer(seed=seed, **kwargs)
    
    if name in {"pso", "particle_swarm", "particle_swarm_optimization"}:
        return ParticleSwarmOptimizer(seed=seed, **kwargs)
    
    
    raise ValueError(f"Unknown optimizer: {name}")
