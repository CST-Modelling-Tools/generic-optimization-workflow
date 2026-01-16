from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from gow.config.models import ProblemConfig, RealParam, IntParam, CategoricalParam
from .base import Optimizer


class DifferentialEvolutionOptimizer(Optimizer):
    """
    Differential Evolution (DE/rand/1/bin) optimizer adapted to GOW.

    Notes / assumptions for GOW integration:
    - ask(problem, n) is called with a fixed n each time.
      For DE, n MUST equal population_size (one full generation at a time).
    - tell(candidates, fitness) is called once per ask, with same ordering/length.
    - Fitness dicts must contain a numeric value under one of:
        "fitness", "objective", "score", or "loss".
      If "loss" is used, it's treated as the objective value.
    - Objective direction is read from problem.objective.direction if present
      ("minimize" or "maximize"). Default: "maximize".
    - Supports RealParam and IntParam. Categorical params are rejected.
    """

    def __init__(
        self,
        *,
        population_size: int = 20,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.9,
        max_generations: int = 50,
        seed: int | None = None,
    ):
        if population_size < 4:
            raise ValueError("population_size must be >= 4 for DE (needs 3 distinct donors + target)")
        if not (0.0 < mutation_factor):
            raise ValueError("mutation_factor must be > 0")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")

        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self._rng = random.Random(seed)

        # DE state
        self._initialized: bool = False
        self._generation: int = 0

        # Ordered param names & metadata (built from ProblemConfig on first ask)
        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Any]] = {}  # name -> ("real"/"int", (lo,hi))
        self._direction: str = "maximize"

        # Population + fitness (comparison is always "higher is better" after normalization)
        self._population: List[Dict[str, Any]] = []
        self._fitness: List[float | None] = []

        # For generation > 0, ask() returns trial vectors; tell() compares them vs targets.
        self._last_targets: List[int] = []  # indices in population that trials correspond to

    # ------------------------
    # GOW Optimizer interface
    # ------------------------

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n != self.population_size:
            raise ValueError(
                f"DifferentialEvolutionOptimizer requires ask(..., n=population_size). "
                f"Got n={n}, population_size={self.population_size}."
            )

        if not self._initialized:
            self._initialize_from_problem(problem)

        if self._generation == 0:
            # Initial population
            return [dict(ind) for ind in self._population]

        # Generate one trial per target i
        trials: List[Dict[str, Any]] = []
        self._last_targets = list(range(self.population_size))

        for i in range(self.population_size):
            trial = self._make_trial(i)
            trials.append(trial)

        return trials

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); DE is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}")
        if len(candidates) != self.population_size:
            raise ValueError(
                f"DifferentialEvolutionOptimizer expects exactly population_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.population_size}"
            )

        # Normalize objective so "higher is better" for internal comparisons
        cand_scores = [self._normalize_score(fdict) for fdict in fitness]

        if self._generation == 0:
            # First evaluation assigns initial fitnesses
            self._fitness = list(cand_scores)
            # (population already set)
            self._generation += 1
            return

        # Generation > 0: candidates are trials for targets in _last_targets
        if not self._last_targets or len(self._last_targets) != self.population_size:
            raise RuntimeError("tell(): missing target mapping from previous ask().")

        for j, target_idx in enumerate(self._last_targets):
            trial = candidates[j]
            trial_score = cand_scores[j]
            target_score = self._fitness[target_idx]

            # If target hasn't been evaluated for some reason, accept trial.
            if target_score is None or trial_score > target_score:
                self._population[target_idx] = trial
                self._fitness[target_idx] = trial_score

        self._generation += 1

    # ------------------------
    # Internal helpers
    # ------------------------

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        # Objective direction
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for Differential Evolution.")

        # Validate and store bounds/specs
        self._param_names = []
        self._param_specs = {}

        for name, p in params.items():
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not (lo < hi):
                    raise ValueError(f"Real param '{name}' must have lo < hi (got {lo}, {hi})")
                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                lo, hi = int(p.bounds[0]), int(p.bounds[1])
                if lo > hi:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo}, {hi})")
                self._param_names.append(name)
                self._param_specs[name] = ("int", (lo, hi))

            elif isinstance(p, CategoricalParam):
                # Plain DE doesn't naturally support categorical variables.
                raise ValueError(
                    f"Differential Evolution does not support categorical param '{name}'. "
                    f"Use RandomSearch or encode categoricals into numeric space first."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        # Initialize population uniformly in bounds
        self._population = [self._random_individual() for _ in range(self.population_size)]
        self._fitness = [None] * self.population_size

        self._generation = 0
        self._initialized = True

    def _random_individual(self) -> Dict[str, Any]:
        ind: Dict[str, Any] = {}
        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            if kind == "real":
                ind[name] = self._rng.uniform(lo, hi)
            else:  # int
                ind[name] = self._rng.randint(lo, hi)
        return ind

    def _make_trial(self, target_idx: int) -> Dict[str, Any]:
        # Choose a,b,c distinct and distinct from target
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self._rng.sample(indices, 3)

        base = self._population[a]
        diff1 = self._population[b]
        diff2 = self._population[c]

        # Binomial crossover with j_rand to force at least one mutated dimension
        j_rand = self._rng.randrange(len(self._param_names))

        trial: Dict[str, Any] = {}
        for j, name in enumerate(self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            do_cross = (self._rng.random() < self.crossover_rate) or (j == j_rand)
            if do_cross:
                mutated = base[name] + self.mutation_factor * (diff1[name] - diff2[name])
                mutated = self._clip(mutated, lo, hi)

                if kind == "int":
                    mutated = int(round(mutated))
                    mutated = int(self._clip(mutated, lo, hi))

                trial[name] = mutated
            else:
                trial[name] = self._population[target_idx][name]

        return trial

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def _get_direction(self, problem: ProblemConfig) -> str:
        # Try to read problem.objective.direction; default to maximize
        direction = "maximize"
        obj = getattr(problem, "objective", None)
        if obj is not None:
            direction = getattr(obj, "direction", direction)
        if direction is None:
            direction = "maximize"
        direction = str(direction).lower().strip()
        if direction not in {"minimize", "maximize"}:
            # Be permissive, but explicit:
            raise ValueError(f"Unknown objective direction '{direction}' (expected 'minimize' or 'maximize').")
        return direction

    def _normalize_score(self, fitness_dict: Dict[str, Any]) -> float:
        # Extract numeric objective value
        val = None
        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                val = fitness_dict[k]
                break
        if val is None:
            raise KeyError(
                "Fitness dict must contain one of: 'fitness', 'objective', 'score', or 'loss'. "
                f"Got keys={list(fitness_dict.keys())}"
            )

        if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(float(val)):
            raise ValueError(f"Fitness value must be a finite number; got {val!r}")

        v = float(val)

        # Internally we compare "higher is better"
        return -v if self._direction == "minimize" else v

    # Optional convenience if a runner wants to check termination.
    def is_done(self) -> bool:
        return self._generation >= self.max_generations