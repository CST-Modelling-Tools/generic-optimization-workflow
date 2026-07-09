from __future__ import annotations

"""
Adaptive Simulated Annealing optimizer for GOW.

This module implements a GOW-compatible Adaptive Simulated Annealing optimizer.

Scientific basis
----------------
The implementation follows the main ideas of Ingber-style Adaptive Simulated
Annealing:

    - heavy-tailed ASA generating distribution,
    - per-parameter temperatures,
    - separate cost temperature,
    - dimension-aware temperature schedule,

        T_i(k_i) = T0_i * exp(-c_i * k_i ** (q_i / D))

    - optional quenching through q_i,
    - reannealing through approximate parameter sensitivities.

Important implementation note
-----------------------------
This is not a verbatim port of Lester Ingber's original C ASA code. It is a
GOW-native optimizer that keeps the ask/tell workflow while preserving the
central ASA mechanisms.

GOW integration
---------------
The class follows the same integration style as other GOW optimizers:

    - class ASAOptimizer(Optimizer)
    - ask(problem, n)
    - tell(candidates, fitness)
    - is_done()
    - diagnostics()

It supports bounded RealParam and IntParam parameters. Categorical parameters
are rejected unless they are encoded numerically before reaching this optimizer.
"""

# -----------------------------------------------------------------------------
# Beginner reading guide
# -----------------------------------------------------------------------------
# Lines that start with '#' are comments. Python ignores them when executing
# the file. They are included only to help a reader follow the optimizer.
#
# Key Python ideas used here:
#   - A class groups related data and behavior.
#   - A method is a function that belongs to a class.
#   - 'self' means the current ASA optimizer object.
#   - A dictionary stores values by name, for example {'p0': 1.2}.
#   - A list stores several values in order.
#   - None means that a value does not exist yet.
#   - Optional[...] means a value can exist or can be None.
#   - GOW calls ask() to request candidates and tell() to return results.
#
# Maintenance rule for this commented version:
#   The executable ASA code is kept intact. Only '#' comments were added.

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam

from .base import Optimizer


# @dataclass asks Python to automatically create simple storage behavior for
# the _State container below.
@dataclass
# _State is an internal container for one ASA state or candidate.
# It keeps the real values, normalized values, score, cost, and metadata
# together so they can be passed around as one object.
class _State:
    """
    Internal ASA state.

    values:
        Candidate dictionary in real parameter space.

    normalized:
        Candidate dictionary in normalized [0, 1] space.

    score:
        Normalized GOW score. Higher is better internally.

    cost:
        ASA cost. Lower is better internally. Defined as cost = -score.

    metadata:
        Optional proposal diagnostics.
    """

    values: Dict[str, Any]
    normalized: Dict[str, float]
    score: Optional[float] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# ASAOptimizer is the main optimizer class used by GOW when the YAML selects
# the ASA optimizer. It inherits from the common Optimizer base class.
class ASAOptimizer(Optimizer):
    """
    Adaptive Simulated Annealing optimizer adapted to GOW.

    Expected YAML usage example:

        optimizer:
          name: asa
          seed: 99
          max_evaluations: 800
          batch_size: 32

          settings:
            max_iterations: 800
            batch_size: 32

            generating_distribution: ingber_asa
            temperature_schedule: ingber_asa
            acceptance_rule: metropolis_exp
            reannealing_method: hybrid

            initial_temperature: 1.0
            final_temperature: 0.001

            parameter_quench_factor: 1.0
            cost_quench_factor: 1.0
            asa_temperature_scale: 1.0

            mutation_parameter_fraction_start: 0.55
            mutation_parameter_fraction_end: 0.18

            reanneal_interval: 64
            target_acceptance_rate: 0.25

            restart_interval: 240
            restart_sigma: 0.012
            include_initial_candidate: true

    Notes:
        - Internal score convention follows DifferentialEvolutionOptimizer:
          higher score is better.
        - ASA cost is cost = -score.
        - The optimizer can run in batch mode because GOW evaluates candidates
          externally and then returns all results through tell().
        - Batch mode is a GOW/HPC adaptation: candidates in one ask() call are
          generated from the current state available at ask() time.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_iterations: int = 800,
        max_evaluations: int | None = None,
        batch_size: int = 32,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        cooling_rate: float = 0.985,
        sigma_start: float = 0.045,
        sigma_min: float = 0.004,
        sigma_max: float = 0.20,
        mutation_parameter_fraction_start: float = 0.55,
        mutation_parameter_fraction_end: float = 0.18,
        reanneal_interval: int = 64,
        target_acceptance_rate: float = 0.25,
        restart_interval: int = 240,
        restart_sigma: float = 0.012,
        include_initial_candidate: bool = True,
        generating_distribution: str = "ingber_asa",
        temperature_schedule: str = "ingber_asa",
        acceptance_rule: str = "metropolis_exp",
        reannealing_method: str = "hybrid",
        parameter_quench_factor: float = 1.0,
        cost_quench_factor: float = 1.0,
        asa_temperature_scale: float = 1.0,
        reanneal_temperature_min_factor: float = 0.25,
        reanneal_temperature_max_factor: float = 4.0,
        sensitivity_floor: float = 1.0e-18,
        **kwargs: Any,
    ) -> None:
        # __init__ runs once when the optimizer object is created.
        # It stores YAML settings such as temperatures, batch size, and restart options.
        # It also creates empty internal variables that will be filled later by ask() and tell().
        # No candidate is evaluated in this method.
        if max_evaluations is not None:
            max_iterations = int(max_evaluations)

        self.seed = seed
        self._rng = random.Random(seed)

        self.max_iterations = int(max_iterations)
        self.batch_size = int(batch_size)

        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.cooling_rate = float(cooling_rate)

        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.mutation_parameter_fraction_start = float(mutation_parameter_fraction_start)
        self.mutation_parameter_fraction_end = float(mutation_parameter_fraction_end)

        self.reanneal_interval = int(reanneal_interval)
        self.target_acceptance_rate = float(target_acceptance_rate)

        self.restart_interval = int(restart_interval)
        self.restart_sigma = float(restart_sigma)
        self.include_initial_candidate = bool(include_initial_candidate)

        self.generating_distribution = str(generating_distribution).strip().lower()
        self.temperature_schedule = str(temperature_schedule).strip().lower()
        self.acceptance_rule = str(acceptance_rule).strip().lower()
        self.reannealing_method = str(reannealing_method).strip().lower()

        self.parameter_quench_factor = float(parameter_quench_factor)
        self.cost_quench_factor = float(cost_quench_factor)
        self.asa_temperature_scale = float(asa_temperature_scale)

        self.reanneal_temperature_min_factor = float(reanneal_temperature_min_factor)
        self.reanneal_temperature_max_factor = float(reanneal_temperature_max_factor)
        self.sensitivity_floor = float(sensitivity_floor)

        self.extra_config = dict(kwargs)

        # GOW / optimizer state
        self._initialized: bool = False
        self._done: bool = False
        self._generation: int = 0
        self._evaluations_seen: int = 0

        # Parameter metadata
        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction: str = "maximize"

        # Initial candidate
        self._initial_values: Dict[str, Any] = {}
        self._initial_normalized: Dict[str, float] = {}
        self._initial_candidate_emitted: bool = False

        # ASA state
        self._current_state: Optional[_State] = None
        self._best_state: Optional[_State] = None

        # Temperatures
        self.cost_temperature: float = self.initial_temperature
        self.temperature: float = self.initial_temperature  # backward-compatible alias
        self.cost_annealing_index: int = 0

        self.param_initial_temperatures: Dict[str, float] = {}
        self.param_final_temperatures: Dict[str, float] = {}
        self.param_temperatures: Dict[str, float] = {}
        self.param_temperature_multipliers: Dict[str, float] = {}
        self.param_annealing_indices: Dict[str, int] = {}
        self.param_annealing_coefficients: Dict[str, float] = {}

        # Gaussian fallback / acceptance-rate adaptation
        self.per_param_sigma: Dict[str, float] = {}
        self.per_param_attempts: Dict[str, int] = {}
        self.per_param_accepts: Dict[str, int] = {}

        # Historical sensitivity reannealing
        self.param_sensitivity_sum: Dict[str, float] = {}
        self.param_sensitivity_count: Dict[str, int] = {}
        self.param_last_sensitivity: Dict[str, Optional[float]] = {}

        # Metadata bridge between ask() and tell()
        self._pending_metadata_by_key: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}

        # Diagnostics
        self.accepted_count: int = 0
        self.rejected_count: int = 0
        self.worse_accepted_count: int = 0
        self.improvement_count: int = 0
        self.last_improvement_evaluation: int = 0

        self.window_accepted: int = 0
        self.window_total: int = 0

        self._n_status_failed: int = 0
        self._n_missing_score: int = 0
        self._n_non_numeric: int = 0
        self._n_non_finite: int = 0

        # Validate configuration values before the run starts. This prevents
        # invalid YAML settings from reaching the optimization loop.
        self._validate_config()

    # ------------------------------------------------------------------
    # GOW Optimizer interface
    # ------------------------------------------------------------------

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        """
        Return n candidate parameter dictionaries to GOW.
        """
        # ask() is called by GOW when it needs new candidates to evaluate.
        # The first call initializes ASA from the GOW problem: parameter names, bounds, and initial values.
        # Then ASA generates n candidate states and returns only their real parameter values to GOW.
        # The evaluator is outside this file; ask() only proposes candidate parameter dictionaries.
        if self._done:
            return []

        if not self._initialized:
            self._initialize_from_problem(problem)

        requested = int(n)
        if requested <= 0:
            raise ValueError(f"ASAOptimizer.ask() requires n > 0, got n={n}")

        states = self._generate_candidate_states(requested)

        self._pending_metadata_by_key = {
            self._candidate_key(state.values): dict(state.metadata or {})
            for state in states
        }

        return [dict(state.values) for state in states]

    def tell(self, candidates: Sequence[Dict[str, Any]], fitness: Sequence[Any]) -> None:
        """
        Update optimizer state from evaluated candidates and fitness dicts.
        """
        # tell() is called by GOW after the evaluator has computed fitness results.
        # Each candidate is matched with one fitness result in the same list position.
        # ASA converts fitness into an internal score, decides whether to accept each candidate,
        # updates counters, cools temperatures, and may trigger reannealing or a soft restart.
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); ASA is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        # Reset per-batch diagnostics
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        for candidate, raw_fitness in zip(candidates, fitness):
            if self._done:
                break

            score = self._normalize_score(raw_fitness)

            normalized = self._normalize_candidate(candidate)
            metadata = self._lookup_pending_metadata(candidate)
            metadata.update(
                {
                    "generation": self._generation,
                    "evaluation": self._evaluations_seen,
                    "raw_fitness": raw_fitness,
                    "cost_temperature": self.cost_temperature,
                }
            )

            if score == float("-inf"):
                proposed = _State(
                    values=dict(candidate),
                    normalized=normalized,
                    score=score,
                    cost=None,
                    metadata=metadata,
                )
            else:
                proposed = _State(
                    values=dict(candidate),
                    normalized=normalized,
            # Each processed evaluation cools the cost temperature. This makes
            # the search gradually more selective.
                    score=score,
                    cost=-score,
                    metadata=metadata,
            # Reannealing is an adaptation step. It adjusts temperatures or
            # sigmas using information collected during recent evaluations.
                )

            accepted = self._process_proposed_state(proposed)
            # A soft restart is only considered when restart_interval is active.
            # It helps the optimizer continue after a long period without a new
            # best solution.

            self._evaluations_seen += 1
            self._cool_cost_temperature(accepted=accepted)

            if self.reanneal_interval > 0 and self._evaluations_seen % self.reanneal_interval == 0:
                self._reanneal()

            if self.restart_interval > 0:
                stale_for = self._evaluations_seen - self.last_improvement_evaluation
                if stale_for >= self.restart_interval and self._best_state is not None:
                    self._soft_restart_around_best()
                    self.last_improvement_evaluation = self._evaluations_seen

            if self._evaluations_seen >= self.max_iterations:
                self._done = True

        self._generation += 1
        self._pending_metadata_by_key = {}

    def is_done(self) -> bool:
        """Return true when ASA has reached max_iterations."""
        # GOW uses this method to know whether ASA has reached its stopping condition.
        return self._done or self._evaluations_seen >= self.max_iterations

    def diagnostics(self) -> Dict[str, Any]:
        """Return small JSON-serializable diagnostics."""
        # This method reports the current state of the optimizer without changing it.
        # The returned dictionary can be serialized to JSON and inspected in logs.
        total_decisions = self.accepted_count + self.rejected_count
        acceptance_rate = None if total_decisions == 0 else self.accepted_count / total_decisions
        worse_acceptance_rate = (
            None if total_decisions == 0 else self.worse_accepted_count / total_decisions
        )

        return {
            "optimizer": "asa",
            "algorithm_family": "adaptive_simulated_annealing",
            "algorithm_variant": "gow_ingber_asa_like",
            "is_verbatim_ingber_c_asa": False,
            "seed": self.seed,
            "generation": self._generation,
            "max_iterations": self.max_iterations,
            "evaluations_seen": self._evaluations_seen,
            "done": self.is_done(),
            "direction": self._direction,
            "generating_distribution": self.generating_distribution,
            "temperature_schedule": self.temperature_schedule,
            "acceptance_rule": self.acceptance_rule,
            "reannealing_method": self.reannealing_method,
            "cost_temperature": self.cost_temperature,
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "cooling_rate": self.cooling_rate,
            "parameter_quench_factor": self.parameter_quench_factor,
            "cost_quench_factor": self.cost_quench_factor,
            "asa_temperature_scale": self.asa_temperature_scale,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "worse_accepted_count": self.worse_accepted_count,
            "improvement_count": self.improvement_count,
            "acceptance_rate": acceptance_rate,
            "worse_acceptance_rate": worse_acceptance_rate,
            "current_score": None if self._current_state is None else self._current_state.score,
            "current_cost": None if self._current_state is None else self._current_state.cost,
            "best_score": None if self._best_state is None else self._best_state.score,
            "best_cost": None if self._best_state is None else self._best_state.cost,
            "best_values": None if self._best_state is None else self._best_state.values,
            "mutation_parameter_fraction": self._current_mutation_parameter_fraction(),
            "sigma_min_current": None if not self.per_param_sigma else min(self.per_param_sigma.values()),
            "sigma_max_current": None if not self.per_param_sigma else max(self.per_param_sigma.values()),
            "sigma_mean_current": None if not self.per_param_sigma else (
                sum(self.per_param_sigma.values()) / len(self.per_param_sigma)
            ),
            "param_temperature_min_current": None if not self.param_temperatures else min(self.param_temperatures.values()),
            "param_temperature_max_current": None if not self.param_temperatures else max(self.param_temperatures.values()),
            "param_temperature_mean_current": None if not self.param_temperatures else (
                sum(self.param_temperatures.values()) / len(self.param_temperatures)
            ),
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
        }

    # ------------------------------------------------------------------
    # Core ASA logic
    # ------------------------------------------------------------------

    def _generate_candidate_states(self, count: int) -> List[_State]:
        """
        Generate candidate states.

        If include_initial_candidate is enabled, the first emitted candidate is
        the YAML value candidate.
        """
        # This internal method creates the candidate states that ask() will return.
        # ASA works in normalized [0, 1] space so all parameters share the same scale.
        # After mutation, candidates are converted back to real parameter values for GOW.
        out: List[_State] = []

        if self.include_initial_candidate and not self._initial_candidate_emitted:
            initial_state = _State(
                values=dict(self._initial_values),
                normalized=dict(self._initial_normalized),
                metadata={
                    "kind": "initial_yaml_value",
                    "mutated_parameters": [],
                    "normalized_step_deltas": {},
                },
            )
            out.append(initial_state)
            self._initial_candidate_emitted = True

        while len(out) < count:
            if self._current_state is None:
                base_normalized = dict(self._initial_normalized)
            else:
                base_normalized = dict(self._current_state.normalized)

            candidate_normalized, mutated_names, step_deltas = self._mutate_normalized(base_normalized)
            candidate_values = self._denormalize_candidate(candidate_normalized)

            out.append(
                _State(
                    values=candidate_values,
                    normalized=candidate_normalized,
                    metadata={
                        "kind": "proposal",
                        "mutated_parameters": mutated_names,
                        "normalized_step_deltas": step_deltas,
                        "cost_temperature": self.cost_temperature,
                        "param_temperatures": {
                            name: self.param_temperatures.get(name)
                            for name in mutated_names
                        },
                        "generating_distribution": self.generating_distribution,
                    },
                )
            )

        return out

    def _process_proposed_state(self, proposed: _State) -> bool:
        """
        Decide whether to accept a proposed state.

        Returns:
            True if accepted, False otherwise.
        """
        # This method contains the ASA accept/reject decision.
        # Better candidates are accepted directly. Worse candidates can still be accepted
        # with a probability controlled by the current cost temperature.
        # That is the mechanism that helps simulated annealing escape local minima.
        if proposed.score == float("-inf") or proposed.cost is None:
            self.rejected_count += 1
            self.window_total += 1
            return False

        if self._current_state is None:
            self._current_state = self._copy_state(proposed)
            self._best_state = self._copy_state(proposed)

            self.accepted_count += 1
            self.window_accepted += 1
            self.window_total += 1
            self.improvement_count += 1
            self.last_improvement_evaluation = self._evaluations_seen
            return True

        assert self._current_state.cost is not None
        assert proposed.cost is not None

        previous_state = self._current_state
        delta_cost = proposed.cost - previous_state.cost

        if delta_cost <= 0.0:
            accept = True
            accepted_worse = False
        else:
            probability = self._acceptance_probability(delta_cost)
            accept = self._rng.random() < probability
            accepted_worse = accept

        self._record_param_attempts(proposed)

        if accept:
            self._current_state = self._copy_state(proposed)
            self.accepted_count += 1
            self.window_accepted += 1

            if accepted_worse:
                self.worse_accepted_count += 1

            self._record_param_accepts(proposed)
            self._record_historical_sensitivity(previous_state, proposed, delta_cost)

            assert proposed.score is not None
            if self._best_state is None or proposed.score > self._best_state.score:
                self._best_state = self._copy_state(proposed)
                self.improvement_count += 1
                self.last_improvement_evaluation = self._evaluations_seen
        else:
            self.rejected_count += 1

        self.window_total += 1
        return accept

    def _acceptance_probability(self, delta_cost: float) -> float:
        """
        Probability of accepting a worse move.

        metropolis_exp:
            p = exp(-delta_cost / T_cost)

        asa_logistic:
            p = 1 / (1 + exp(delta_cost / T_cost))
        """
        # This method computes the probability of accepting a worse move.
        # A high temperature gives worse moves a better chance of being accepted.
        # A low temperature makes ASA behave more strictly.
        temperature = max(self.cost_temperature, 1.0e-300)
        ratio = delta_cost / temperature

        if ratio > 700.0:
            return 0.0

        if self.acceptance_rule == "metropolis_exp":
            return math.exp(-ratio)

        if self.acceptance_rule == "asa_logistic":
            return 1.0 / (1.0 + math.exp(ratio))

        raise ValueError(f"Unsupported acceptance_rule: {self.acceptance_rule!r}")

    def _mutate_normalized(
        self,
        base: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
        """
        Mutate a subset of parameters in normalized space.
        """
        # This method changes a subset of parameters in normalized [0, 1] space.
        # Only some parameters are mutated in each proposal, controlled by the mutation fraction.
        # Each changed parameter also advances its own temperature schedule.
        candidate = dict(base)

        fraction = self._current_mutation_parameter_fraction()
        param_count = len(self._param_names)

        mutation_count = max(1, int(round(param_count * fraction)))
        mutation_count = min(param_count, mutation_count)

        mutated_names = self._rng.sample(self._param_names, mutation_count)
        step_deltas: Dict[str, float] = {}

        for name in mutated_names:
            old_value = candidate[name]

            if self.generating_distribution == "ingber_asa":
                temperature = self.param_temperatures[name]
                step = self._generate_ingber_asa_step(temperature)

            elif self.generating_distribution == "gaussian":
                sigma = self.per_param_sigma.get(name, self.sigma_start)
                step = self._rng.gauss(0.0, sigma)

            else:
                raise ValueError(
                    f"Unsupported generating_distribution: {self.generating_distribution!r}"
                )

            new_value = self._clip01(old_value + step)
            candidate[name] = new_value
            step_deltas[name] = new_value - old_value

            self.param_annealing_indices[name] = self.param_annealing_indices.get(name, 0) + 1
            self._refresh_parameter_temperature(name)

        return candidate, mutated_names, step_deltas

    def _generate_ingber_asa_step(self, temperature: float) -> float:
        """
        Generate one ASA normalized step y in [-1, 1].

        Inverse-CDF form:

            y = sign(u - 1/2) * T * [ (1 + 1/T)^|2u - 1| - 1 ]

        where u ~ U(0, 1).
        """
        # This method generates one ASA step using the Ingber-style heavy-tailed distribution.
        # Heavy-tailed means ASA usually makes small moves but still has a chance to make larger jumps.
        temperature = max(float(temperature), 1.0e-300)

        u = self._rng.random()
        sign = -1.0 if u < 0.5 else 1.0
        exponent = abs(2.0 * u - 1.0)

        log_base = math.log1p(1.0 / temperature)
        value = temperature * (math.exp(exponent * log_base) - 1.0)

        value = self._clip_value(value, 0.0, 1.0)
        return sign * value

    def _current_mutation_parameter_fraction(self) -> float:
        """
        Linearly decrease mutated parameter fraction during the run.
        """
        # This method computes how many parameters should be mutated as the run progresses.
        # At the beginning, more parameters may change together. Later, fewer parameters may change.
        if self.max_iterations <= 1:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, self._evaluations_seen / self.max_iterations))

        start = self.mutation_parameter_fraction_start
        end = self.mutation_parameter_fraction_end

        return start + (end - start) * progress

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------

    def _cool_cost_temperature(self, *, accepted: bool) -> None:
        """
        Update the cost temperature.

        For the Ingber-style schedule, the cost temperature index is advanced
        by accepted states. For geometric mode, it is advanced every evaluation.
        """
        # This method updates the cost temperature after an evaluation is processed.
        # The cost temperature controls acceptance of worse candidates.
        if self.temperature_schedule == "geometric":
            self.cost_temperature = max(
                self.final_temperature,
                self.cost_temperature * self.cooling_rate,
            )

        elif self.temperature_schedule == "ingber_asa":
            if accepted:
                self.cost_annealing_index += 1
                self.cost_temperature = self._asa_temperature(
                    initial=self.initial_temperature,
                    final=self.final_temperature,
                    index=self.cost_annealing_index,
                    quench_factor=self.cost_quench_factor,
                )

        else:
            raise ValueError(f"Unsupported temperature_schedule: {self.temperature_schedule!r}")

        self.temperature = self.cost_temperature

    def _refresh_parameter_temperature(self, name: str) -> None:
        """
        Refresh one parameter temperature from its annealing index and multiplier.
        """
        # This method updates the temperature of one parameter after that parameter is mutated.
        # Parameter temperature controls the scale of future steps for that parameter.
        if self.temperature_schedule == "geometric":
            current = self.param_temperatures[name] * self.cooling_rate
            self.param_temperatures[name] = max(self.param_final_temperatures[name], current)
            return

        if self.temperature_schedule == "ingber_asa":
            base = self._asa_temperature(
                initial=self.param_initial_temperatures[name],
                final=self.param_final_temperatures[name],
                index=self.param_annealing_indices[name],
                quench_factor=self.parameter_quench_factor,
            )
            multiplier = self.param_temperature_multipliers.get(name, 1.0)
            value = base * multiplier
            self.param_temperatures[name] = self._clip_value(
                value,
                self.param_final_temperatures[name],
                self.param_initial_temperatures[name],
            )
            return

        raise ValueError(f"Unsupported temperature_schedule: {self.temperature_schedule!r}")

    def _asa_temperature(
        self,
        *,
        initial: float,
        final: float,
        index: int,
        quench_factor: float,
    ) -> float:
        """
        Compute ASA-like dimension-aware annealing temperature.

            T(k) = T0 * exp(-c * k^(q/D))

        c is chosen so that T(max_iterations) is approximately final.
        """
        # This method computes the ASA temperature schedule.
        # The schedule depends on the number of parameters, so the cooling is dimension-aware.
        initial = max(float(initial), 1.0e-300)
        final = max(float(final), 1.0e-300)
        index = max(0, int(index))

        if index <= 0:
            return initial

        dimension = max(1, len(self._param_names))
        q_over_d = max(float(quench_factor), 1.0e-12) / float(dimension)

        horizon = max(1, self.max_iterations)
        denominator = float(horizon) ** q_over_d

        coefficient = -math.log(final / initial) / denominator
        coefficient *= self.asa_temperature_scale

        temperature = initial * math.exp(-coefficient * (float(index) ** q_over_d))
        return max(final, temperature)

    def _computed_asa_coefficient(self, initial: float, final: float, quench_factor: float) -> float:
        """
        Return ASA cooling coefficient for diagnostics.
        """
        # This helper computes the cooling coefficient used by the ASA schedule.
        # It is mainly useful for diagnostics.
        initial = max(float(initial), 1.0e-300)
        final = max(float(final), 1.0e-300)

        dimension = max(1, len(self._param_names))
        q_over_d = max(float(quench_factor), 1.0e-12) / float(dimension)

        horizon = max(1, self.max_iterations)
        denominator = float(horizon) ** q_over_d

        return -math.log(final / initial) / denominator * self.asa_temperature_scale

    # ------------------------------------------------------------------
    # Reannealing
    # ------------------------------------------------------------------

    def _reanneal(self) -> None:
        """
        Reanneal according to selected method.
        """
        # This method chooses and applies the configured reannealing strategy.
        # Reannealing adapts the search using acceptance-rate and/or sensitivity information.
        if self.reannealing_method == "none":
            self._reset_reanneal_windows()
            return

        if self.reannealing_method == "acceptance_rate":
            self._reanneal_acceptance_rate()

        elif self.reannealing_method == "historical_sensitivity":
            self._reanneal_historical_sensitivity()

        elif self.reannealing_method == "hybrid":
            self._reanneal_acceptance_rate()
            self._reanneal_historical_sensitivity()

        else:
            raise ValueError(f"Unsupported reannealing_method: {self.reannealing_method!r}")

        self._reset_reanneal_windows()

    def _reanneal_acceptance_rate(self) -> None:
        """
        Adapt Gaussian sigma and lightly rescale parameter temperatures using
        recent acceptance behavior.
        """
        # This reannealing method uses recent acceptance rate to adapt step sizes.
        # If too few moves are accepted, steps are reduced. If many are accepted, steps can expand.
        if self.window_total <= 0:
            return

        acceptance_rate = self.window_accepted / self.window_total

        if acceptance_rate < self.target_acceptance_rate * 0.5:
            global_factor = 0.70
            temp_factor = 0.85
        elif acceptance_rate > self.target_acceptance_rate * 1.5:
            global_factor = 1.15
            temp_factor = 1.10
        else:
            global_factor = 1.00
            temp_factor = 1.00

        for name in self._param_names:
            attempts = self.per_param_attempts.get(name, 0)
            accepts = self.per_param_accepts.get(name, 0)

            if attempts <= 0:
                local_factor = 1.00
            else:
                local_rate = accepts / attempts

                if local_rate < self.target_acceptance_rate * 0.5:
                    local_factor = 0.80
                elif local_rate > self.target_acceptance_rate * 1.5:
                    local_factor = 1.10
                else:
                    local_factor = 1.00

            new_sigma = self.per_param_sigma[name] * global_factor * local_factor
            self.per_param_sigma[name] = self._clip_value(
                new_sigma,
                self.sigma_min,
                self.sigma_max,
            )

            self._rescale_parameter_temperature(name, temp_factor * local_factor)

    def _reanneal_historical_sensitivity(self) -> None:
        """
        Reanneal parameter temperatures using historical sensitivity.

        Approximation:

            sensitivity_i ≈ |Δcost| / |Δx_i|

        More sensitive parameters get lower temperatures. Less sensitive
        parameters are allowed broader moves.
        """
        # This reannealing method estimates how sensitive the cost is to each parameter.
        # More sensitive parameters receive lower temperatures; less sensitive parameters can move more broadly.
        sensitivities: Dict[str, float] = {}

        for name in self._param_names:
            count = self.param_sensitivity_count.get(name, 0)
            if count <= 0:
                continue

            value = self.param_sensitivity_sum[name] / count
            if math.isfinite(value):
                sensitivities[name] = max(value, self.sensitivity_floor)
                self.param_last_sensitivity[name] = sensitivities[name]

        if not sensitivities:
            return

        mean_sensitivity = sum(sensitivities.values()) / len(sensitivities)
        mean_sensitivity = max(mean_sensitivity, self.sensitivity_floor)

        for name, sensitivity in sensitivities.items():
            raw_factor = mean_sensitivity / sensitivity
            factor = math.sqrt(raw_factor)

            factor = self._clip_value(
                factor,
                self.reanneal_temperature_min_factor,
                self.reanneal_temperature_max_factor,
            )

            self._rescale_parameter_temperature(name, factor)

    def _rescale_parameter_temperature(self, name: str, factor: float) -> None:
        """
        Rescale parameter temperature while preserving ASA schedule through a
        multiplier. This avoids losing reannealing effects at the next refresh.
        """
        # This method changes one parameter temperature while preserving the scheduled ASA cooling.
        # The change is stored as a multiplier so the effect survives later temperature refreshes.
        current = self.param_temperatures[name]
        target = current * factor

        target = self._clip_value(
            target,
            self.param_final_temperatures[name],
            self.param_initial_temperatures[name],
        )

        if self.temperature_schedule == "ingber_asa":
            base = self._asa_temperature(
                initial=self.param_initial_temperatures[name],
                final=self.param_final_temperatures[name],
                index=self.param_annealing_indices[name],
                quench_factor=self.parameter_quench_factor,
            )
            base = max(base, 1.0e-300)
            self.param_temperature_multipliers[name] = self._clip_value(
                target / base,
                1.0e-12,
                1.0e12,
            )

        self.param_temperatures[name] = target

    def _reset_reanneal_windows(self) -> None:
        """
        Reset window counters after reannealing.
        """
        # This method clears the temporary counters used during one reannealing window.
        self.window_accepted = 0
        self.window_total = 0

        self.per_param_attempts = {name: 0 for name in self._param_names}
        self.per_param_accepts = {name: 0 for name in self._param_names}

        self.param_sensitivity_sum = {name: 0.0 for name in self._param_names}
        self.param_sensitivity_count = {name: 0 for name in self._param_names}

    def _soft_restart_around_best(self) -> None:
        """
        Soft restart around the best known candidate.

        This is a GOW-practical option to recover from frozen chains.
        """
        # This method restarts the current state near the best solution found so far.
        # It is a soft restart because it does not erase the best solution; it only moves the current search point.
        if self._best_state is None:
            return

        base = dict(self._best_state.normalized)
        restarted: Dict[str, float] = {}

        for name in self._param_names:
            value = base[name] + self._rng.gauss(0.0, self.restart_sigma)
            restarted[name] = self._clip01(value)

        self._current_state = _State(
            values=self._denormalize_candidate(restarted),
            normalized=restarted,
            score=self._best_state.score,
            cost=self._best_state.cost,
            metadata={
                "kind": "soft_restart",
                "source_best_score": self._best_state.score,
                "source_best_cost": self._best_state.cost,
                "evaluation": self._evaluations_seen,
            },
        )

        reheated = math.sqrt(self.initial_temperature * self.final_temperature)

        self.cost_temperature = max(self.cost_temperature, reheated)
        self.temperature = self.cost_temperature

        for name in self._param_names:
            current = self.param_temperatures[name]
            if current < reheated:
                self._rescale_parameter_temperature(name, reheated / max(current, 1.0e-300))

    # ------------------------------------------------------------------
    # Initialization from GOW ProblemConfig
    # ------------------------------------------------------------------

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        """
        Extract parameter names, bounds and YAML initial values from GOW.
        """
        # This method reads parameter information from the GOW ProblemConfig.
        # It stores parameter names, parameter types, bounds, and YAML initial values.
        # It rejects categorical parameters because ASA needs numeric distances and numeric steps.
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for ASA.")

        self._param_names = []
        self._param_specs = {}
        self._initial_values = {}
        self._initial_normalized = {}

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
                    f"ASA does not support categorical param '{name}'. "
                    "Use RandomSearch or encode categoricals into numeric space first."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

            kind, (lo, hi) = self._param_specs[name]
            initial_value = getattr(p, "value", None)

            if initial_value is None:
                initial_value = 0.5 * (lo + hi)

            initial_value = self._clip_value(float(initial_value), lo, hi)

            if kind == "int":
                initial_value = int(round(initial_value))
                initial_value = int(self._clip_value(float(initial_value), lo, hi))

            self._initial_values[name] = initial_value
            self._initial_normalized[name] = self._normalize_value(name, float(initial_value))

        if not self._param_names:
            raise ValueError("No supported optimizable parameters found for ASA.")

        self._initialize_asa_state()
        self._initialized = True

    def _initialize_asa_state(self) -> None:
        """
        Initialize ASA temperatures, counters and diagnostics.
        """
        # This method initializes temperatures, counters, sigmas, and sensitivity accumulators.
        # It runs after parameter names are known because many dictionaries are keyed by parameter name.
        self.cost_temperature = self.initial_temperature
        self.temperature = self.cost_temperature
        self.cost_annealing_index = 0

        self.param_initial_temperatures = {
            name: self.initial_temperature for name in self._param_names
        }
        self.param_final_temperatures = {
            name: self.final_temperature for name in self._param_names
        }
        self.param_temperatures = {
            name: self.initial_temperature for name in self._param_names
        }
        self.param_temperature_multipliers = {
            name: 1.0 for name in self._param_names
        }
        self.param_annealing_indices = {
            name: 0 for name in self._param_names
        }
        self.param_annealing_coefficients = {
            name: self._computed_asa_coefficient(
                initial=self.initial_temperature,
                final=self.final_temperature,
                quench_factor=self.parameter_quench_factor,
            )
            for name in self._param_names
        }

        self.per_param_sigma = {
            name: self.sigma_start for name in self._param_names
        }
        self.per_param_attempts = {
            name: 0 for name in self._param_names
        }
        self.per_param_accepts = {
            name: 0 for name in self._param_names
        }

        self.param_sensitivity_sum = {
            name: 0.0 for name in self._param_names
        }
        self.param_sensitivity_count = {
            name: 0 for name in self._param_names
        }
        self.param_last_sensitivity = {
            name: None for name in self._param_names
        }

    # ------------------------------------------------------------------
    # Fitness handling
    # ------------------------------------------------------------------

    def _normalize_score(self, fitness_value: Any) -> float:
        """
        Convert GOW fitness dict into internal higher-is-better score.

        This follows the same convention used by DifferentialEvolutionOptimizer.
        """
        # This method converts evaluator output into ASA's internal score convention.
        # Inside ASA, higher score is always better. For minimization problems, the sign is inverted.
        # Invalid or missing evaluator results become -infinity, which ASA rejects.
        if isinstance(fitness_value, (int, float)):
            x = float(fitness_value)
            if not math.isfinite(x):
                self._n_non_finite += 1
                return float("-inf")
            if self._direction == "minimize":
                x = -x
            return x

        if not isinstance(fitness_value, Mapping):
            self._n_non_numeric += 1
            return float("-inf")

        status = fitness_value.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        val: Any = None
        key: str | None = None

        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_value:
                key = k
                val = fitness_value[k]
                break

        if key is None:
            metrics = fitness_value.get("metrics")
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

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    def _normalize_candidate(self, candidate: Mapping[str, Any]) -> Dict[str, float]:
        """
        Convert real parameter candidate to normalized [0, 1].
        """
        # This helper converts a candidate from real units into normalized [0, 1] values.
        return {
            name: self._normalize_value(name, float(candidate[name]))
            for name in self._param_names
        }

    def _denormalize_candidate(self, normalized: Mapping[str, float]) -> Dict[str, Any]:
        """
        Convert normalized [0, 1] candidate to real parameter space.
        """
        # This helper converts normalized [0, 1] values back to real parameter values.
        # Integer parameters are rounded and clipped before being returned.
        candidate: Dict[str, Any] = {}

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            value = self._denormalize_value(name, float(normalized[name]))

            if kind == "int":
                value = int(round(value))
                value = int(self._clip_value(float(value), lo, hi))

            candidate[name] = value

        return candidate

    def _normalize_value(self, name: str, value: float) -> float:
        """
        Normalize one real value to [0, 1].
        """
        # Normalize one value using that parameter's lower and upper bounds.
        _, (lo, hi) = self._param_specs[name]
        return self._clip01((value - lo) / (hi - lo))

    def _denormalize_value(self, name: str, value: float) -> float:
        """
        Denormalize one [0, 1] value to real bounds.
        """
        # Denormalize one value from [0, 1] back into the parameter's real bounds.
        _, (lo, hi) = self._param_specs[name]
        value = self._clip01(value)
        return lo + value * (hi - lo)

    @staticmethod
    def _clip_value(value: float, low: float, high: float) -> float:
        """
        Clip value to [low, high].
        """
        # Clip means force a number to remain inside a valid interval.
        return min(high, max(low, value))

    def _clip01(self, value: float) -> float:
        """
        Clip value to [0, 1].
        """
        # Special clipping helper for normalized values, where the valid interval is [0, 1].
        return self._clip_value(value, 0.0, 1.0)

    def _copy_state(self, state: _State) -> _State:
        """
        Deep-ish copy of a state.
        """
        # This helper copies a state so later edits do not accidentally change stored history.
        return _State(
            values=dict(state.values),
            normalized=dict(state.normalized),
            score=state.score,
            cost=state.cost,
            metadata=None if state.metadata is None else dict(state.metadata),
        )

    def _candidate_key(self, candidate: Mapping[str, Any]) -> Tuple[Tuple[str, float], ...]:
        """
        Stable key used to recover proposal metadata in tell().
        """
        # This helper creates a stable key used to recover metadata for a candidate.
        return tuple(
            (name, round(float(candidate[name]), 15))
            for name in sorted(self._param_names)
        )

    def _lookup_pending_metadata(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Recover metadata generated in ask().
        """
        # This helper retrieves metadata saved during ask() for a candidate returned through tell().
        return dict(self._pending_metadata_by_key.get(self._candidate_key(candidate), {}))

    def _record_param_attempts(self, proposed: _State) -> None:
        """
        Record which parameters were changed in a proposal.
        """
        # This method counts which parameters were attempted in a proposed mutation.
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []

        for name in mutated:
            if name in self.per_param_attempts:
                self.per_param_attempts[name] += 1

    def _record_param_accepts(self, proposed: _State) -> None:
        """
        Record accepted mutations per parameter.
        """
        # This method counts which mutated parameters were accepted.
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []

        for name in mutated:
            if name in self.per_param_accepts:
                self.per_param_accepts[name] += 1

    def _record_historical_sensitivity(
        self,
        previous: _State,
        proposed: _State,
        delta_cost: float,
    ) -> None:
        """
        Record approximate per-parameter sensitivity from accepted moves.
        """
        # This method estimates sensitivity from accepted moves.
        # The approximation is: sensitivity = absolute cost change / absolute normalized step.
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []
        step_deltas = proposed.metadata.get("normalized_step_deltas") or {}

        if not mutated:
            return

        abs_delta = abs(float(delta_cost))

        for name in mutated:
            step = abs(float(step_deltas.get(name, 0.0)))

            if step <= 0.0 and name in previous.normalized and name in proposed.normalized:
                step = abs(proposed.normalized[name] - previous.normalized[name])

            if step <= 0.0:
                continue

            sensitivity = abs_delta / max(step, 1.0e-300)

            if math.isfinite(sensitivity):
                self.param_sensitivity_sum[name] += sensitivity
                self.param_sensitivity_count[name] += 1

    # ------------------------------------------------------------------
    # Objective direction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_direction(problem: ProblemConfig) -> str:
        # This helper reads whether the objective should be minimized or maximized.
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """
        Validate configuration values early.
        """
        # This method checks all user-facing configuration values.
        # It raises clear errors for impossible or unsafe settings before optimization begins.
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be > 0")

        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be > 0")

        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must be <= initial_temperature")

        if not (0.0 < self.cooling_rate <= 1.0):
            raise ValueError("cooling_rate must be in (0, 1]")

        if self.sigma_start <= 0:
            raise ValueError("sigma_start must be > 0")

        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be > 0")

        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")

        if not (0.0 < self.mutation_parameter_fraction_start <= 1.0):
            raise ValueError("mutation_parameter_fraction_start must be in (0, 1]")

        if not (0.0 < self.mutation_parameter_fraction_end <= 1.0):
            raise ValueError("mutation_parameter_fraction_end must be in (0, 1]")

        if not (0.0 < self.target_acceptance_rate < 1.0):
            raise ValueError("target_acceptance_rate must be in (0, 1)")

        if self.restart_interval < 0:
            raise ValueError("restart_interval must be >= 0")

        if self.restart_sigma <= 0:
            raise ValueError("restart_sigma must be > 0")

        if self.generating_distribution not in {"ingber_asa", "gaussian"}:
            raise ValueError(
                "generating_distribution must be 'ingber_asa' or 'gaussian'"
            )

        if self.temperature_schedule not in {"ingber_asa", "geometric"}:
            raise ValueError(
                "temperature_schedule must be 'ingber_asa' or 'geometric'"
            )

        if self.acceptance_rule not in {"metropolis_exp", "asa_logistic"}:
            raise ValueError(
                "acceptance_rule must be 'metropolis_exp' or 'asa_logistic'"
            )

        if self.reannealing_method not in {
            "hybrid",
            "historical_sensitivity",
            "acceptance_rate",
            "none",
        }:
            raise ValueError(
                "reannealing_method must be 'hybrid', 'historical_sensitivity', "
                "'acceptance_rate', or 'none'"
            )

        if self.parameter_quench_factor <= 0:
            raise ValueError("parameter_quench_factor must be > 0")

        if self.cost_quench_factor <= 0:
            raise ValueError("cost_quench_factor must be > 0")

        if self.asa_temperature_scale <= 0:
            raise ValueError("asa_temperature_scale must be > 0")

        if self.reanneal_temperature_min_factor <= 0:
            raise ValueError("reanneal_temperature_min_factor must be > 0")

        if self.reanneal_temperature_max_factor < self.reanneal_temperature_min_factor:
            raise ValueError(
                "reanneal_temperature_max_factor must be >= "
                "reanneal_temperature_min_factor"
            )

        if self.sensitivity_floor <= 0:
            raise ValueError("sensitivity_floor must be > 0")


# Keep the earlier experimental class name available for old imports.
# Backward compatibility with earlier experimental file/class name.
AdaptiveSimulatedAnnealingOptimizer = ASAOptimizer