from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from skopt import Optimizer as SkoptOptimizer
from skopt.space import Integer, Real

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class BayesianOptimizationOptimizer(Optimizer):
    """
    Bayesian Optimization optimizer adapted to GOW using scikit-optimize.
    Uses skopt.Optimizer with ask/tell interface.
    """

    def __init__(
        self,
        *,
        n_initial_points: int = 20,
        base_estimator: str = "GP",
        acquisition_function: str = "EI",
        max_iterations: int = 100,
        acq_optimizer: str = "auto",
        batch_strategy: str = "cl_min",
        seed: int | None = None,
        **kwargs,
    ):
        if n_initial_points < 1:
            raise ValueError("n_initial_points must be >= 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        self.n_initial_points = int(n_initial_points)
        self.base_estimator = base_estimator
        self.acquisition_function = acquisition_function
        self.max_iterations = int(max_iterations)
        self.acq_optimizer = acq_optimizer
        self.batch_strategy = batch_strategy
        self.seed = seed

        self._initialized = False
        self._iteration = 0

        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction = "maximize"

        self._optimizer: SkoptOptimizer | None = None

        # IMPORTANT:
        # This stores the vectors actually delivered to the evaluator after repair.
        # Therefore skopt.tell() learns f(repaired_x), not f(raw_x).
        self._last_xs: List[List[Any]] = []

        self._best_score: float | None = None
        self._best_candidate: Dict[str, Any] | None = None

        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n < 1:
            raise ValueError("ask(..., n) requires n >= 1")

        if not self._initialized:
            self._initialize_from_problem(problem)

        if self._optimizer is None:
            raise RuntimeError("Bayesian optimizer was not initialized correctly.")

        raw_xs = self._optimizer.ask(n_points=n, strategy=self.batch_strategy)

        if n == 1 and raw_xs and not isinstance(raw_xs[0], list):
            raw_xs = [raw_xs]

        raw_xs = [list(x) for x in raw_xs]

        candidates = [self._vector_to_candidate(x) for x in raw_xs]
        repaired_candidates = [
            self._repair_candidate(problem, cand) for cand in candidates
        ]

        self._last_xs = [
            self._candidate_to_vector(cand) for cand in repaired_candidates
        ]

        return repaired_candidates

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        if not self._initialized or self._optimizer is None:
            raise RuntimeError(
                "tell() called before first ask(); Bayesian optimizer is not initialized."
            )

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        if len(self._last_xs) != len(candidates):
            raise RuntimeError(
                "tell(): number of candidates does not match the previous ask() call."
            )

        scores = [self._normalize_score(fdict) for fdict in fitness]

        losses: List[float] = []

        for score, cand in zip(scores, candidates):
            if score == float("-inf"):
                losses.append(1.0e100)
                continue

            losses.append(-score)

            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_candidate = dict(cand)

        self._optimizer.tell(self._last_xs, losses)

        self._iteration += 1
        self._last_xs = []

    def is_done(self) -> bool:
        if not self._initialized:
            return False
        return self._iteration >= self.max_iterations

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "iteration": self._iteration,
            "n_initial_points": self.n_initial_points,
            "base_estimator": self.base_estimator,
            "acquisition_function": self.acquisition_function,
            "acq_optimizer": self.acq_optimizer,
            "batch_strategy": self.batch_strategy,
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
            raise ValueError("No optimizable parameters found for Bayesian Optimization.")

        dimensions = []
        self._param_names = []
        self._param_specs = {}

        for name, p in params.items():
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(
                        f"Optimizable real param '{name}' missing bounds=[lo,hi]"
                    )

                lo, hi = float(p.bounds[0]), float(p.bounds[1])

                if not (lo < hi):
                    raise ValueError(
                        f"Real param '{name}' must have lo < hi (got {lo}, {hi})"
                    )

                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))
                dimensions.append(Real(lo, hi, name=name))

            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(
                        f"Optimizable int param '{name}' missing bounds=[lo,hi]"
                    )

                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])

                if lo_i > hi_i:
                    raise ValueError(
                        f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})"
                    )

                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))
                dimensions.append(Integer(lo_i, hi_i, name=name))

            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"BayesianOptimizationOptimizer does not support categorical param '{name}'. "
                    "Use numeric encoding or another optimizer."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        self._optimizer = SkoptOptimizer(
            dimensions=dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            acq_optimizer=self.acq_optimizer,
            random_state=self.seed,
        )

        self._iteration = 0
        self._last_xs = []
        self._initialized = True

    def _vector_to_candidate(self, x: List[Any]) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}

        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            if kind == "int":
                value = int(round(float(value)))
                value = int(self._clip(float(value), lo, hi))
            else:
                value = float(value)
                value = self._clip(value, lo, hi)

            cand[name] = value

        return cand

    def _candidate_to_vector(self, cand: Dict[str, Any]) -> List[Any]:
        return [cand[name] for name in self._param_names]

    def _repair_candidate(
        self,
        problem: ProblemConfig,
        cand: Dict[str, Any],
    ) -> Dict[str, Any]:
        repaired = dict(cand)

        if "r_min" in repaired:
            receiver_radius = self._get_param_value(
                problem, repaired, "flat_receiver_radius", 0.0
            )
            min_clearance = self._get_param_value(
                problem, repaired, "min_tower_clearance", 0.0
            )
            mh = self._get_param_value(problem, repaired, "mirror_height", 4.06)
            mw = self._get_param_value(problem, repaired, "mirror_width", 4.06)

            diag = math.sqrt(mh * mh + mw * mw)
            r_inner = receiver_radius + min_clearance + 0.5 * diag
            repaired["r_min"] = max(float(repaired["r_min"]), float(r_inner))

        if "chord_diameter_factor" in repaired and "rowrow_diameter_factor" in repaired:
            c = float(repaired["chord_diameter_factor"])
            rr = float(repaired["rowrow_diameter_factor"])
            if rr < c:
                repaired["rowrow_diameter_factor"] = c

        return repaired

    def _get_param_value(
        self,
        problem: ProblemConfig,
        cand: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        if key in cand:
            return float(cand[key])

        p = problem.parameters.get(key)
        if p is None:
            return float(default)

        return float(p.value)

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
                f"Unknown objective direction '{direction}' "
                "(expected 'minimize' or 'maximize')."
            )

        return direction