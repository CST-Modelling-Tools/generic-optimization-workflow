from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class ACOROptimizer(Optimizer):
    """
    ACOR optimizer adapted to GOW.

    ACOR = Ant Colony Optimization for Continuous Domains.

    This implementation is dimension-agnostic:
    - it reads all optimizable numeric parameters from the GOW problem config;
    - it works internally in normalized space [0, 1]^n;
    - it uses a solution archive as the continuous pheromone model;
    - it samples new candidates from Gaussian kernels centered on archive solutions.

    Notes:
    - Categorical parameters are not supported.
    - Integer parameters are supported by sampling in normalized continuous space
      and rounding back to the integer domain.
    """

    def __init__(
        self,
        *,
        archive_size: int = 50,
        q: float = 0.1,
        xi: float = 0.85,
        ants: int | None = None,
        max_generations: int | None = None,
        include_initial_candidate: bool = True,
        min_sigma: float = 1e-12,
        bound_strategy: str = "clip",
        seed: int | None = None,
    ):
        if archive_size < 2:
            raise ValueError("archive_size must be >= 2 for ACOR")
        if q <= 0.0:
            raise ValueError("q must be > 0")
        if xi <= 0.0:
            raise ValueError("xi must be > 0")
        if ants is not None and ants < 1:
            raise ValueError("ants must be >= 1 when provided")
        if max_generations is not None and max_generations < 1:
            raise ValueError("max_generations must be >= 1 when provided")
        if min_sigma <= 0.0:
            raise ValueError("min_sigma must be > 0")

        bound_strategy = str(bound_strategy).lower().strip()
        if bound_strategy not in {"clip", "resample"}:
            raise ValueError("bound_strategy must be either 'clip' or 'resample'")

        self.archive_size = int(archive_size)
        self.q = float(q)
        self.xi = float(xi)
        self.ants = int(ants) if ants is not None else None
        self.max_generations = int(max_generations) if max_generations is not None else None
        self.include_initial_candidate = bool(include_initial_candidate)
        self.min_sigma = float(min_sigma)
        self.bound_strategy = bound_strategy
        self.seed = seed

        self._rng = random.Random(seed)

        self._initialized = False
        self._generation = 0

        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction = "maximize"

        # Archive entries:
        # {"x": normalized_vector, "candidate": candidate_dict, "score": internal_score}
        # Internal score is always maximized.
        self._archive: List[Dict[str, Any]] = []

        self._initial_candidate_used = False

        self._best_score = None
        self._best_candidate = None

        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0
        self._n_invalid_candidates = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n < 1:
            raise ValueError("ACOROptimizer requires ask(..., n>=1)")

        if not self._initialized:
            self._initialize_from_problem(problem)

        # First generation: initialize the archive with random candidates,
        # optionally including the baseline candidate from the YAML values.
        if not self._archive:
            return self._initial_candidates(n)

        return [self._sample_candidate_from_archive() for _ in range(n)]

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0
        self._n_invalid_candidates = 0

        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); ACOR is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        new_entries: List[Dict[str, Any]] = []

        for cand, fdict in zip(candidates, fitness):
            score = self._normalize_score(fdict)

            if score == float("-inf"):
                continue

            try:
                x = self._candidate_to_normalized(cand)
            except Exception:
                self._n_invalid_candidates += 1
                continue

            new_entries.append(
                {
                    "x": x,
                    "candidate": dict(cand),
                    "score": score,
                }
            )

            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_candidate = dict(cand)

        if new_entries:
            self._archive.extend(new_entries)
            self._archive.sort(key=lambda row: row["score"], reverse=True)
            self._archive = self._archive[: self.archive_size]

        self._generation += 1

    def is_done(self) -> bool:
        if self.max_generations is None:
            return False
        return self._generation >= self.max_generations

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "archive_size": self.archive_size,
            "archive_size_current": len(self._archive),
            "q": self.q,
            "xi": self.xi,
            "ants": self.ants,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
            "n_invalid_candidates": self._n_invalid_candidates,
        }

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for ACOR.")

        self._param_names = []
        self._param_specs = {}

        initial_x0: List[float] = []

        for name, p in params.items():
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not (lo < hi):
                    raise ValueError(f"Real param '{name}' must have lo < hi (got {lo}, {hi})")
                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

                val = float(p.value) if p.value is not None else 0.5 * (lo + hi)
                val = self._clip(val, lo, hi)
                initial_x0.append((val - lo) / (hi - lo))

            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])
                if lo_i > hi_i:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})")
                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

                lo, hi = float(lo_i), float(hi_i)
                val = float(p.value) if p.value is not None else 0.5 * (lo + hi)
                val = self._clip(val, lo, hi)
                initial_x0.append((val - lo) / (hi - lo))

            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"ACOR does not support categorical param '{name}'. "
                    "Use numeric encoding or another optimizer for categorical variables."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        self._initial_x0 = initial_x0
        self._initialized = True

    def _initial_candidates(self, n: int) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []

        if self.include_initial_candidate and not self._initial_candidate_used:
            cands.append(self._normalized_to_candidate(self._problem_default_x()))
            self._initial_candidate_used = True

        while len(cands) < n:
            cands.append(self._normalized_to_candidate(self._random_x()))

        return cands

    def _problem_default_x(self) -> List[float]:
        # Values were already read from problem during initialization via param specs.
        # We do not keep the whole problem object here, so this method creates
        # a neutral point at the center when called outside direct problem access.
        #
        # The real baseline values are injected in _initialize_from_problem through
        # self._initial_x0 if available.
        if hasattr(self, "_initial_x0"):
            return list(self._initial_x0)
        return [0.5 for _ in self._param_names]

    def _random_x(self) -> List[float]:
        return [self._rng.random() for _ in self._param_names]

    def _archive_weights(self) -> List[float]:
        k = len(self._archive)
        if k <= 0:
            raise RuntimeError("Cannot compute ACOR weights with an empty archive.")

        denom = self.q * k
        weights = [
            math.exp(-((rank ** 2) / (2.0 * denom * denom)))
            / (denom * math.sqrt(2.0 * math.pi))
            for rank in range(k)
        ]

        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return [1.0 / k for _ in range(k)]

        return [w / total for w in weights]

    def _choose_archive_index(self) -> int:
        weights = self._archive_weights()
        r = self._rng.random()
        acc = 0.0

        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i

        return len(weights) - 1

    def _sample_candidate_from_archive(self) -> Dict[str, Any]:
        idx = self._choose_archive_index()
        center = self._archive[idx]["x"]
        k = len(self._archive)

        x_new: List[float] = []

        for dim, mu in enumerate(center):
            if k > 1:
                avg_distance = sum(
                    abs(float(row["x"][dim]) - float(mu))
                    for row in self._archive
                ) / float(k - 1)
                sigma = self.xi * avg_distance
            else:
                sigma = 0.5 * self.xi

            sigma = max(float(sigma), self.min_sigma)
            x_new.append(self._sample_bounded_gaussian(float(mu), sigma))

        return self._normalized_to_candidate(x_new)

    def _sample_bounded_gaussian(self, mu: float, sigma: float) -> float:
        if self.bound_strategy == "resample":
            for _ in range(32):
                x = self._rng.gauss(mu, sigma)
                if 0.0 <= x <= 1.0:
                    return x
            return self._clip(self._rng.gauss(mu, sigma), 0.0, 1.0)

        return self._clip(self._rng.gauss(mu, sigma), 0.0, 1.0)

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

    def _candidate_to_normalized(self, candidate: Mapping[str, Any]) -> List[float]:
        x: List[float] = []

        for name in self._param_names:
            if name not in candidate:
                raise KeyError(f"Candidate missing parameter '{name}'")

            kind, (lo, hi) = self._param_specs[name]
            val = float(candidate[name])

            if kind == "int":
                val = float(int(round(val)))

            val = self._clip(val, lo, hi)
            x.append((val - lo) / (hi - lo))

        return x

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        key: str | None = None
        val: Any = None

        for k in ("objective", "score", "loss", "fitness"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        if isinstance(val, Mapping):
            nested = val
            key = None
            val = None
            for k in ("objective", "score", "loss", "fitness"):
                if k in nested:
                    key = k
                    val = nested[k]
                    break

        if val is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("objective", "score", "loss", "fitness"):
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
