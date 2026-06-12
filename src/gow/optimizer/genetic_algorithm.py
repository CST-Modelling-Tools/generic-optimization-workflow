from __future__ import annotations

import random
from typing import Any, Dict, List

from gow.config.models import IntParam, ProblemConfig, RealParam
from .base import Optimizer


class GeneticAlgorithmOptimizer(Optimizer):

    def __init__(
        self,
        *,
        population_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        seed: int | None = None,
        **kwargs,
    ):

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size

        self._rng = random.Random(seed)

        self._initialized = False
        self._generation = 0

        self._population = []
        self._fitness = []

        self._param_specs = {}
        self._param_names = []

    def ask(self, problem: ProblemConfig, n: int):

        if not self._initialized:
            self._initialize(problem)

        if self._generation == 0:
            return [dict(x) for x in self._population]

        new_population = []

        elite_count = max(1, int(self.population_size * self.elite_fraction))

        ranked = sorted(
            zip(self._population, self._fitness),
            key=lambda x: x[1],
            reverse=True,
        )

        elites = [dict(x[0]) for x in ranked[:elite_count]]

        new_population.extend(elites)

        while len(new_population) < self.population_size:

            p1 = self._tournament_select()
            p2 = self._tournament_select()

            child = self._crossover(p1, p2)
            child = self._mutate(child)

            new_population.append(child)

        self._population = new_population

        return [dict(x) for x in self._population]

    def tell(self, candidates, fitness):

        scores = []

        for f in fitness:

            if "fitness" in f:
                scores.append(float(f["fitness"]))

            elif "objective" in f:
                scores.append(float(f["objective"]))

            elif "score" in f:
                scores.append(float(f["score"]))

            else:
                scores.append(float("-inf"))

        self._fitness = scores
        self._generation += 1

    def is_done(self):

        return self._generation >= self.generations

    def diagnostics(self):

        if not self._fitness:
            return {}

        return {
            "generation": self._generation,
            "best_fitness": max(self._fitness),
        }

    def _initialize(self, problem):

        params = problem.optimizable_parameters()

        for name, p in params.items():

            if isinstance(p, RealParam):

                lo, hi = p.bounds

                self._param_specs[name] = ("real", lo, hi)

            elif isinstance(p, IntParam):

                lo, hi = p.bounds

                self._param_specs[name] = ("int", lo, hi)

        self._param_names = list(self._param_specs.keys())

        self._population = [
            self._random_individual()
            for _ in range(self.population_size)
        ]

        self._fitness = [float("-inf")] * self.population_size

        self._initialized = True

    def _random_individual(self):

        ind = {}

        for name in self._param_names:

            kind, lo, hi = self._param_specs[name]

            if kind == "real":
                ind[name] = self._rng.uniform(lo, hi)

            else:
                ind[name] = self._rng.randint(int(lo), int(hi))

        return ind

    def _tournament_select(self):

        idxs = self._rng.sample(
            range(len(self._population)),
            self.tournament_size,
        )

        best_idx = max(idxs, key=lambda i: self._fitness[i])

        return dict(self._population[best_idx])

    def _crossover(self, p1, p2):

        child = {}

        for name in self._param_names:

            if self._rng.random() < self.crossover_rate:
                child[name] = p1[name]
            else:
                child[name] = p2[name]

        return child

    def _mutate(self, child):

        for name in self._param_names:

            if self._rng.random() < self.mutation_rate:

                kind, lo, hi = self._param_specs[name]

                if kind == "real":
                    child[name] = self._rng.uniform(lo, hi)

                else:
                    child[name] = self._rng.randint(int(lo), int(hi))

        return child