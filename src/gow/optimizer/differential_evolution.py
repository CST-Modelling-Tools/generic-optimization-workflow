from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class DifferentialEvolutionOptimizer(Optimizer):
    """
    Differential Evolution (DE/rand/1/bin) optimizer adapted to GOW.

    GOW integration assumptions:
      - ask(problem, n) is called repeatedly with a fixed n.
        For DE, n MUST equal population_size (one full generation at a time).
      - tell(candidates, fitness) is called once per ask(), with matching ordering/length.
      - Fitness dicts should contain a numeric value under one of:
          "fitness", "objective", "score", or "loss".
        If "loss" is used, smaller is better (it is inverted internally).
      - Objective direction is read from problem.objective.direction if present
        ("minimize" or "maximize"). Default: "maximize".
      - Supports RealParam and IntParam as optimizable variables.
        Categorical optimizables are rejected (plain DE is not categorical-native).

    Robustness (Option A):
      - Any failed evaluation or missing/non-numeric score is treated as the worst candidate
        by returning -inf after normalization (higher-is-better internal convention).
        This is valid for BOTH maximize and minimize objective directions.
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
            raise ValueError(
                "population_size must be >= 4 for DE (needs 3 distinct donors + target)"
            )
        if mutation_factor <= 0.0:
            raise ValueError("mutation_factor must be > 0")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")

        self.population_size = int(population_size)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.max_generations = int(max_generations)

        self._rng = random.Random(seed)

        # DE state
        self._initialized: bool = False
        self._generation: int = 0

        # Ordered param names & metadata (built from ProblemConfig on first ask)
        self._param_names: List[str] = []
        # name -> ("real"/"int", (lo, hi))
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction: str = "maximize"  # "maximize" | "minimize"

        # Population + fitness (after normalization, higher is better)
        self._population: List[Dict[str, Any]] = []
        self._fitness: List[float | None] = []

        # For generation > 0, ask() returns trial vectors; tell() compares them vs targets.
        self._last_targets: List[int] = []

        # Simple diagnostics (useful when many candidates fail)
        self._n_status_failed: int = 0
        self._n_missing_score: int = 0
        self._n_non_numeric: int = 0
        self._n_non_finite: int = 0

    # ------------------------
    # GOW Optimizer interface
    # ------------------------

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n != self.population_size:
            raise ValueError(
                "DifferentialEvolutionOptimizer requires ask(..., n=population_size). "
                f"Got n={n}, population_size={self.population_size}."
            )

        if not self._initialized:
            self._initialize_from_problem(problem)

        if self._generation == 0:
            # Initial population (first "generation" evaluation)
            self._last_targets = list(range(self.population_size))
            return [dict(ind) for ind in self._population]

        # Subsequent generations: return one trial per target
        trials: List[Dict[str, Any]] = []
        self._last_targets = list(range(self.population_size))

        for target_idx in self._last_targets:
            trial = self._make_trial(target_idx)
            trial = self._repair_candidate(problem, trial)
            trials.append(trial)

        return trials

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:

        # Reset per-generation diagnostics
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); DE is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )
        if len(candidates) != self.population_size:
            raise ValueError(
                "DifferentialEvolutionOptimizer expects exactly population_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.population_size}"
            )

        if not self._last_targets or len(self._last_targets) != self.population_size:
            raise RuntimeError("tell(): missing target mapping from previous ask().")

        cand_scores = [self._normalize_score(fdict) for fdict in fitness]

        if self._generation == 0:
            # First evaluation assigns initial fitnesses
            self._fitness = list(cand_scores)
            self._generation += 1
            self._last_targets = []
            return

        # Generation > 0: candidates are trials for targets in _last_targets
        for j, target_idx in enumerate(self._last_targets):
            trial = candidates[j]
            trial_score = cand_scores[j]
            target_score = self._fitness[target_idx]

            # If target is unknown, accept a non-worst trial.
            if target_score is None:
                if trial_score != float("-inf"):
                    self._population[target_idx] = trial
                    self._fitness[target_idx] = trial_score
                continue

            # Normal DE selection (higher is better after normalization)
            if trial_score > target_score:
                self._population[target_idx] = trial
                self._fitness[target_idx] = trial_score

        self._generation += 1
        self._last_targets = []

    def is_done(self) -> bool:
        """True when the optimizer has reached its internal generation limit."""
        return self._generation >= self.max_generations


    # ------------------------
    # Checkpoint / resume
    # ------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return all state required to continue DE exactly.

        Checkpoint v1 is deliberately restricted to generation boundaries.
        A checkpoint may therefore be created only after tell() has completed
        and cleared the target mapping from the preceding ask() call.
        """

        if self._last_targets:
            raise RuntimeError(
                "Differential Evolution checkpoint can only be created "
                "between generations, after tell() has completed."
            )

        return {
            "schema_version": 1,
            "optimizer": "differential_evolution",

            "configuration": {
                "population_size": self.population_size,
                "mutation_factor": self.mutation_factor,
                "crossover_rate": self.crossover_rate,
                "max_generations": self.max_generations,
            },

            "initialized": self._initialized,
            "generation": self._generation,

            "param_names": list(self._param_names),
            "param_specs": dict(self._param_specs),
            "direction": self._direction,

            "population": [
                dict(individual)
                for individual in self._population
            ],
            "fitness": list(self._fitness),

            # Checkpoint v1 never persists a partially evaluated generation.
            "last_targets": [],

            # Critical for exact deterministic continuation.
            "rng_state": self._rng.getstate(),

            "diagnostics": {
                "n_status_failed": self._n_status_failed,
                "n_missing_score": self._n_missing_score,
                "n_non_numeric": self._n_non_numeric,
                "n_non_finite": self._n_non_finite,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a state previously returned by state_dict()."""

        if not isinstance(state, dict):
            raise TypeError(
                "Differential Evolution checkpoint state must be a dictionary"
            )

        if state.get("schema_version") != 1:
            raise ValueError(
                "Unsupported Differential Evolution checkpoint schema_version: "
                f"{state.get('schema_version')!r}"
            )

        if state.get("optimizer") != "differential_evolution":
            raise ValueError(
                "Checkpoint optimizer mismatch: expected "
                "'differential_evolution', got "
                f"{state.get('optimizer')!r}"
            )

        # ------------------------------------------------------------
        # Configuration compatibility
        # ------------------------------------------------------------

        configuration = state.get("configuration")

        if not isinstance(configuration, dict):
            raise ValueError(
                "Differential Evolution checkpoint is missing configuration"
            )

        expected_configuration = {
            "population_size": self.population_size,
            "mutation_factor": self.mutation_factor,
            "crossover_rate": self.crossover_rate,
            "max_generations": self.max_generations,
        }

        for key, current_value in expected_configuration.items():
            if key not in configuration:
                raise ValueError(
                    "Differential Evolution checkpoint configuration "
                    f"is missing {key!r}"
                )

            checkpoint_value = configuration[key]

            if checkpoint_value != current_value:
                raise ValueError(
                    "Differential Evolution checkpoint configuration mismatch "
                    f"for {key}: checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"
                )

        # ------------------------------------------------------------
        # General state
        # ------------------------------------------------------------

        initialized = state.get("initialized")

        if not isinstance(initialized, bool):
            raise ValueError(
                "Checkpoint field 'initialized' must be boolean"
            )

        generation = state.get("generation")

        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
        ):
            raise ValueError(
                "Checkpoint field 'generation' must be a non-negative integer"
            )

        direction = state.get("direction")

        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                "Checkpoint direction must be 'minimize' or 'maximize'"
            )

        # ------------------------------------------------------------
        # Parameter metadata
        # ------------------------------------------------------------

        param_names = state.get("param_names")

        if not isinstance(param_names, list):
            raise ValueError(
                "Checkpoint field 'param_names' must be a list"
            )

        if not all(
            isinstance(name, str) and name
            for name in param_names
        ):
            raise ValueError(
                "Checkpoint contains invalid parameter names"
            )

        param_specs_raw = state.get("param_specs")

        if not isinstance(param_specs_raw, dict):
            raise ValueError(
                "Checkpoint field 'param_specs' must be a dictionary"
            )

        param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}

        for name in param_names:
            if name not in param_specs_raw:
                raise ValueError(
                    f"Checkpoint is missing parameter specification for {name!r}"
                )

            spec = param_specs_raw[name]

            if (
                not isinstance(spec, (tuple, list))
                or len(spec) != 2
                or spec[0] not in {"real", "int"}
                or not isinstance(spec[1], (tuple, list))
                or len(spec[1]) != 2
            ):
                raise ValueError(
                    f"Invalid parameter specification for {name!r}: {spec!r}"
                )

            kind = str(spec[0])
            lo = float(spec[1][0])
            hi = float(spec[1][1])

            if kind == "real" and not lo < hi:
                raise ValueError(
                    f"Invalid real bounds for {name!r}: {lo}, {hi}"
                )

            if kind == "int" and lo > hi:
                raise ValueError(
                    f"Invalid integer bounds for {name!r}: {lo}, {hi}"
                )

            param_specs[name] = (
                kind,
                (lo, hi),
            )

        # ------------------------------------------------------------
        # Population
        # ------------------------------------------------------------

        population_raw = state.get("population")

        if not isinstance(population_raw, list):
            raise ValueError(
                "Checkpoint field 'population' must be a list"
            )

        population: List[Dict[str, Any]] = []

        for index, individual in enumerate(population_raw):
            if not isinstance(individual, dict):
                raise ValueError(
                    f"Checkpoint population entry {index} is not a dictionary"
                )

            missing = [
                name
                for name in param_names
                if name not in individual
            ]

            if missing:
                raise ValueError(
                    f"Checkpoint population entry {index} "
                    f"is missing parameters: {missing}"
                )

            population.append(dict(individual))

        # ------------------------------------------------------------
        # Fitness
        # ------------------------------------------------------------

        fitness_raw = state.get("fitness")

        if not isinstance(fitness_raw, list):
            raise ValueError(
                "Checkpoint field 'fitness' must be a list"
            )

        fitness: List[float | None] = []

        for value in fitness_raw:
            if value is None:
                fitness.append(None)
            else:
                try:
                    fitness.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid checkpoint fitness value: {value!r}"
                    ) from exc

        # ------------------------------------------------------------
        # Generation boundary
        # ------------------------------------------------------------

        last_targets = state.get("last_targets", [])

        if not isinstance(last_targets, list):
            raise ValueError(
                "Checkpoint field 'last_targets' must be a list"
            )

        if last_targets:
            raise ValueError(
                "Differential Evolution checkpoint contains a partial "
                "generation. Checkpoint v1 only supports generation boundaries."
            )

        if initialized:
            if not param_names:
                raise ValueError(
                    "Initialized DE checkpoint contains no parameters"
                )

            if len(population) != self.population_size:
                raise ValueError(
                    "Differential Evolution checkpoint population size mismatch: "
                    f"checkpoint={len(population)}, "
                    f"expected={self.population_size}"
                )

            if len(fitness) != self.population_size:
                raise ValueError(
                    "Differential Evolution checkpoint fitness size mismatch: "
                    f"checkpoint={len(fitness)}, "
                    f"expected={self.population_size}"
                )

        # ------------------------------------------------------------
        # RNG
        # ------------------------------------------------------------

        rng_state = state.get("rng_state")

        if rng_state is None:
            raise ValueError(
                "Differential Evolution checkpoint is missing RNG state"
            )

        rng_probe = random.Random()

        try:
            rng_probe.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Differential Evolution checkpoint contains invalid RNG state"
            ) from exc

        # ------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------

        diagnostics = state.get("diagnostics", {})

        if not isinstance(diagnostics, dict):
            raise ValueError(
                "Checkpoint diagnostics must be a dictionary"
            )

        diagnostic_values: Dict[str, int] = {}

        for key in (
            "n_status_failed",
            "n_missing_score",
            "n_non_numeric",
            "n_non_finite",
        ):
            value = diagnostics.get(key, 0)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"Checkpoint diagnostic {key!r} "
                    "must be a non-negative integer"
                )

            diagnostic_values[key] = value

        # ------------------------------------------------------------
        # Commit validated state to this optimizer
        # ------------------------------------------------------------

        self._initialized = initialized
        self._generation = generation

        self._param_names = list(param_names)
        self._param_specs = param_specs
        self._direction = str(direction)

        self._population = population
        self._fitness = fitness

        self._last_targets = []

        self._rng.setstate(rng_state)

        self._n_status_failed = diagnostic_values["n_status_failed"]
        self._n_missing_score = diagnostic_values["n_missing_score"]
        self._n_non_numeric = diagnostic_values["n_non_numeric"]
        self._n_non_finite = diagnostic_values["n_non_finite"]


    # ------------------------
    # Internal helpers
    # ------------------------

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

        # Optional: fallback to metrics dict
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

    def _get_param_value(self, problem: ProblemConfig, cand: Dict[str, Any], key: str, default: float) -> float:
        # Candidate overrides problem defaults if present
        if key in cand:
            return float(cand[key])
        p = problem.parameters.get(key)
        if p is None:
            return float(default)
        return float(p.value)

    def _repair_candidate(self, problem: ProblemConfig, cand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort repair for known problematic layout knobs (HFE radial-staggered).

        This is intentionally conservative:
          - only touches keys present in cand (or relevant to cand keys)
          - uses problem defaults for "fixed" values if needed
        """
        repaired = dict(cand)

        # If r_min is being optimized, enforce geometric lower bound:
        # r_min >= receiver_radius + min_clearance + 0.5*diag
        if "r_min" in repaired:
            receiver_radius = self._get_param_value(problem, repaired, "flat_receiver_radius", 0.0)
            min_clearance = self._get_param_value(problem, repaired, "min_tower_clearance", 0.0)
            mh = self._get_param_value(problem, repaired, "mirror_height", 4.06)
            mw = self._get_param_value(problem, repaired, "mirror_width", 4.06)

            diag = math.sqrt(mh * mh + mw * mw)
            r_inner = receiver_radius + min_clearance + 0.5 * diag
            repaired["r_min"] = max(float(repaired["r_min"]), float(r_inner))

        # Keep row-to-row spacing >= within-row spacing if both are present.
        if "chord_diameter_factor" in repaired and "rowrow_diameter_factor" in repaired:
            c = float(repaired["chord_diameter_factor"])
            rr = float(repaired["rowrow_diameter_factor"])
            if rr < c:
                repaired["rowrow_diameter_factor"] = c

        return repaired

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for Differential Evolution.")

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
                # store as floats, but we'll cast back to int when mutating
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"Differential Evolution does not support categorical param '{name}'. "
                    "Use RandomSearch or encode categoricals into numeric space first."
                )
            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        if not self._param_names:
            raise ValueError("No supported optimizable parameters found for Differential Evolution.")

        self._population = [self._random_individual() for _ in range(self.population_size)]
        self._fitness = [None] * self.population_size

        self._generation = 0
        self._last_targets = []
        self._initialized = True

        # Reset diagnostics
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def _random_individual(self) -> Dict[str, Any]:
        ind: Dict[str, Any] = {}
        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            if kind == "real":
                ind[name] = self._rng.uniform(lo, hi)
            else:  # int
                ind[name] = int(self._rng.randint(int(lo), int(hi)))
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
                    mutated = int(self._clip(float(mutated), lo, hi))

                trial[name] = mutated
            else:
                trial[name] = self._population[target_idx][name]

        return trial

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