from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import cma

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class CMAESOptimizer(Optimizer):
    """
    CMA-ES optimizer adapted to GOW.

    Internal search is done in normalized space [0, 1]^n.
    """

    def __init__(
        self,
        *,
        population_size: int = 16,
        sigma0: float = 0.05,
        max_generations: int = 100,
        seed: int | None = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2 for CMA-ES")
        if sigma0 <= 0.0:
            raise ValueError("sigma0 must be > 0")
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")

        self.population_size = int(population_size)
        self.sigma0 = float(sigma0)
        self.max_generations = int(max_generations)
        self.seed = seed

        self._initialized = False
        self._generation = 0

        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction = "maximize"

        self._es = None
        self._last_xs: List[List[float]] = []

        self._best_score = None
        self._best_candidate = None

        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n != self.population_size:
            raise ValueError(
                "CMAESOptimizer requires ask(..., n=population_size). "
                f"Got n={n}, population_size={self.population_size}."
            )

        if not self._initialized:
            self._initialize_from_problem(problem)

        xs = self._es.ask(number=self.population_size)
        self._last_xs = [list(x) for x in xs]

        return [self._normalized_to_candidate(x) for x in self._last_xs]

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); CMA-ES is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        if len(candidates) != self.population_size:
            raise ValueError(
                "CMAESOptimizer expects exactly population_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.population_size}"
            )

        if not self._last_xs or len(self._last_xs) != self.population_size:
            raise RuntimeError("tell(): missing CMA-ES vectors from previous ask().")

        scores = [self._normalize_score(fdict) for fdict in fitness]

        losses = []
        for score, cand in zip(scores, candidates):
            if score == float("-inf"):
                losses.append(1e100)
            else:
                losses.append(-score)

                if self._best_score is None or score > self._best_score:
                    self._best_score = score
                    self._best_candidate = dict(cand)

        self._es.tell(self._last_xs, losses)
        self._generation += 1
        self._last_xs = []

    def is_done(self) -> bool:
        if not self._initialized:
            return False
        return self._generation >= self.max_generations or bool(self._es.stop())

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "population_size": self.population_size,
            "sigma": float(getattr(self._es, "sigma", self.sigma0)) if self._es is not None else self.sigma0,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
        }

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for CMA-ES.")

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
                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])
                if lo_i > hi_i:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})")
                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"CMA-ES does not support categorical param '{name}'. "
                    "Use RandomSearch or encode categoricals into numeric space first."
                )
            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        dim = len(self._param_names)

        x0 = []
        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            p = problem.parameters[name]
            val = float(p.value)
            val = self._clip(val, lo, hi)
            x0.append((val - lo) / (hi - lo))

        opts = {
            "popsize": self.population_size,
            "bounds": [[0.0] * dim, [1.0] * dim],
            "verbose": -9,
            "tolx": 1e-12,
            "tolfun": 1e-12,
        }

        if self.seed is not None:
            opts["seed"] = int(self.seed)

        self._es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        self._generation = 0
        self._last_xs = []
        self._initialized = True

    def _normalized_to_candidate(self, x: List[float]) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}

        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            value = self._clip(float(value), 0.0, 1.0)
            real_value = lo + value * (hi - lo)

            if abs(real_value - lo) < 1e-15:
                real_value = lo
            if abs(real_value - hi) < 1e-15:
                real_value = hi

            if kind == "int":
                real_value = int(round(real_value))
                real_value = int(self._clip(float(real_value), lo, hi))

            cand[name] = real_value

        return cand

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        val: Any = None
        key: str | None = None

        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        if key is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("fitness", "objective", "score", "loss"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        if isinstance(val, str) and not val.strip():
            self._n_missing_score += 1
            return float("-inf")

        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        if key == "loss":
            x = -x

        if self._direction == "minimize":
            x = -x

        return x

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _get_direction(problem: ProblemConfig) -> str:
        direction = "maximize"
        obj = getattr(problem, "objective", None)
        if obj is not None:
            direction = getattr(obj, "direction", direction) or direction
        direction = str(direction).lower().strip()
        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"Unknown objective direction '{direction}' (expected 'minimize' or 'maximize')."
            )
        return direction