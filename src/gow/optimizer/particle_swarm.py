from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class ParticleSwarmOptimizer(Optimizer):
    """
    Particle Swarm Optimization with inertia weight.

    Update rule:
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v

    GOW integration:
      - ask(problem, n) requires n == swarm_size.
      - tell(candidates, fitness) updates pbest and gbest.
      - Internal score is higher-is-better.
      - If objective.direction is minimize, the objective is internally negated.
      - Supports RealParam and IntParam.
      - Rejects CategoricalParam.
    """

    def __init__(
        self,
        *,
        swarm_size: int = 40,
        max_generations: int = 50,
        inertia_weight: float = 0.729,
        cognitive_coefficient: float = 1.49445,
        social_coefficient: float = 1.49445,
        velocity_clamp_fraction: float = 0.2,
        boundary_handling: str = "clamp",
        include_initial_candidate: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        if swarm_size < 1:
            raise ValueError("swarm_size must be >= 1")
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if inertia_weight < 0.0:
            raise ValueError("inertia_weight must be >= 0")
        if cognitive_coefficient < 0.0:
            raise ValueError("cognitive_coefficient must be >= 0")
        if social_coefficient < 0.0:
            raise ValueError("social_coefficient must be >= 0")
        if velocity_clamp_fraction <= 0.0:
            raise ValueError("velocity_clamp_fraction must be > 0")
        if boundary_handling not in {"clamp"}:
            raise ValueError("Only boundary_handling='clamp' is currently supported")

        self.swarm_size = int(swarm_size)
        self.max_generations = int(max_generations)
        self.inertia_weight = float(inertia_weight)
        self.cognitive_coefficient = float(cognitive_coefficient)
        self.social_coefficient = float(social_coefficient)
        self.velocity_clamp_fraction = float(velocity_clamp_fraction)
        self.boundary_handling = boundary_handling
        self.include_initial_candidate = bool(include_initial_candidate)

        self._rng = random.Random(seed)

        self._initialized = False
        self._generation = 0
        self._direction = "minimize"

        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, float, float]] = {}

        self._positions: List[Dict[str, float]] = []
        self._velocities: List[Dict[str, float]] = []

        self._pbest_positions: List[Dict[str, float]] = []
        self._pbest_scores: List[float | None] = []

        self._gbest_position: Dict[str, float] | None = None
        self._gbest_score: float | None = None

        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n != self.swarm_size:
            raise ValueError(
                "ParticleSwarmOptimizer requires ask(..., n=swarm_size). "
                f"Got n={n}, swarm_size={self.swarm_size}."
            )

        if not self._initialized:
            self._initialize(problem)

        if self._generation > 0:
            self._move_swarm()

        return [self._candidate_from_position(pos) for pos in self._positions]

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); PSO is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        if len(candidates) != self.swarm_size:
            raise ValueError(
                "ParticleSwarmOptimizer expects exactly swarm_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.swarm_size}"
            )

        scores = [self._normalize_score(fdict) for fdict in fitness]

        for i, score in enumerate(scores):
            if score == float("-inf"):
                continue

            pos = self._position_from_candidate(candidates[i])

            old_score = self._pbest_scores[i]
            if old_score is None or score > old_score:
                self._pbest_positions[i] = dict(pos)
                self._pbest_scores[i] = score

            if self._gbest_score is None or score > self._gbest_score:
                self._gbest_position = dict(pos)
                self._gbest_score = score

        self._generation += 1

    def is_done(self) -> bool:
        return self._generation >= self.max_generations

    def diagnostics(self) -> Dict[str, Any]:
        if self._gbest_score is None:
            return {
                "generation": self._generation,
                "best_objective": None,
                "status_failed": self._n_status_failed,
                "missing_score": self._n_missing_score,
                "non_numeric": self._n_non_numeric,
                "non_finite": self._n_non_finite,
            }

        best_objective = -self._gbest_score if self._direction == "minimize" else self._gbest_score

        return {
            "generation": self._generation,
            "best_objective": best_objective,
            "best_internal_score": self._gbest_score,
            "status_failed": self._n_status_failed,
            "missing_score": self._n_missing_score,
            "non_numeric": self._n_non_numeric,
            "non_finite": self._n_non_finite,
        }

    def _initialize(self, problem: ProblemConfig) -> None:
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for Particle Swarm Optimization.")

        self._param_names = []
        self._param_specs = {}

        for name, p in params.items():
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                kind = "real"

            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                kind = "int"

            elif isinstance(p, CategoricalParam):
                raise TypeError(
                    f"ParticleSwarmOptimizer does not support categorical optimizable param '{name}'."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

            if not lo < hi:
                raise ValueError(f"Invalid bounds for '{name}': [{lo}, {hi}]")

            self._param_names.append(name)
            self._param_specs[name] = (kind, lo, hi)

        self._positions = []
        self._velocities = []

        for i in range(self.swarm_size):
            if i == 0 and self.include_initial_candidate:
                pos = self._initial_position_from_problem(problem)
            else:
                pos = self._random_position()

            vel = self._random_velocity()

            self._positions.append(pos)
            self._velocities.append(vel)

        self._pbest_positions = [dict(pos) for pos in self._positions]
        self._pbest_scores = [None] * self.swarm_size

        self._gbest_position = None
        self._gbest_score = None
        self._initialized = True

    def _move_swarm(self) -> None:
        if self._gbest_position is None:
            self._positions = [self._random_position() for _ in range(self.swarm_size)]
            self._velocities = [self._random_velocity() for _ in range(self.swarm_size)]
            return

        new_positions: List[Dict[str, float]] = []
        new_velocities: List[Dict[str, float]] = []

        for i in range(self.swarm_size):
            pos = self._positions[i]
            vel = self._velocities[i]
            pbest = self._pbest_positions[i]

            new_pos: Dict[str, float] = {}
            new_vel: Dict[str, float] = {}

            for name in self._param_names:
                kind, lo, hi = self._param_specs[name]
                span = hi - lo
                vmax = self.velocity_clamp_fraction * span

                r1 = self._rng.random()
                r2 = self._rng.random()

                v = (
                    self.inertia_weight * vel[name]
                    + self.cognitive_coefficient * r1 * (pbest[name] - pos[name])
                    + self.social_coefficient * r2 * (self._gbest_position[name] - pos[name])
                )

                v = min(max(v, -vmax), vmax)
                x = pos[name] + v

                if x < lo:
                    x = lo
                    v = 0.0
                elif x > hi:
                    x = hi
                    v = 0.0

                if kind == "int":
                    x = float(round(x))
                    x = min(max(x, lo), hi)

                new_pos[name] = float(x)
                new_vel[name] = float(v)

            new_positions.append(new_pos)
            new_velocities.append(new_vel)

        self._positions = new_positions
        self._velocities = new_velocities

    def _random_position(self) -> Dict[str, float]:
        pos: Dict[str, float] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]

            if kind == "real":
                pos[name] = self._rng.uniform(lo, hi)
            else:
                pos[name] = float(self._rng.randint(int(lo), int(hi)))

        return pos

    def _random_velocity(self) -> Dict[str, float]:
        vel: Dict[str, float] = {}

        for name in self._param_names:
            _kind, lo, hi = self._param_specs[name]
            vmax = self.velocity_clamp_fraction * (hi - lo)
            vel[name] = self._rng.uniform(-vmax, vmax)

        return vel

    def _initial_position_from_problem(self, problem: ProblemConfig) -> Dict[str, float]:
        pos: Dict[str, float] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]
            p = problem.parameters.get(name)
            value = getattr(p, "value", None)

            if value is None:
                if kind == "real":
                    x = self._rng.uniform(lo, hi)
                else:
                    x = float(self._rng.randint(int(lo), int(hi)))
            else:
                x = float(value)

            x = min(max(x, lo), hi)

            if kind == "int":
                x = float(round(x))
                x = min(max(x, lo), hi)

            pos[name] = float(x)

        return pos

    def _candidate_from_position(self, pos: Dict[str, float]) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]
            x = min(max(pos[name], lo), hi)

            if kind == "int":
                cand[name] = int(round(x))
            else:
                cand[name] = float(x)

        return cand

    def _position_from_candidate(self, cand: Dict[str, Any]) -> Dict[str, float]:
        pos: Dict[str, float] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]
            x = float(cand[name])
            x = min(max(x, lo), hi)

            if kind == "int":
                x = float(round(x))
                x = min(max(x, lo), hi)

            pos[name] = float(x)

        return pos

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

    def _get_direction(self, problem: ProblemConfig) -> str:
        objective = getattr(problem, "objective", None)
        direction = getattr(objective, "direction", None)

        if direction is None:
            return "minimize"

        direction = str(direction).lower().strip()

        if direction in {"minimize", "min"}:
            return "minimize"

        if direction in {"maximize", "max"}:
            return "maximize"

        return "minimize"
