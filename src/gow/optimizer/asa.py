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
# How to read this file
# -----------------------------------------------------------------------------
# This file implements the ASA optimizer used by GOW.
#
# The important execution path is:
#   1. __init__() stores configuration and prepares empty state.
#   2. ask(problem, n) creates a batch of candidate parameter sets.
#   3. GOW evaluates those candidates outside this file.
#   4. tell(candidates, fitness) receives the evaluation results.
#   5. ASA accepts or rejects candidates, updates temperatures, and repeats.
#
# Key reading conventions:
#   - self means "this ASA optimizer object".
#   - values are real parameter values sent to the evaluator.
#   - normalized values are the same parameters scaled to [0, 1].
#   - score follows the GOW convention: higher is better internally.
#   - cost follows the ASA convention: lower is better internally.
#   - a batch is the group of candidates returned by one ask() call.
#
# The comments are intentionally explanatory, but the executable code is kept
# unchanged from the adjusted ASA implementation.
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam

from .base import Optimizer



# _State is a compact record used inside ASA.
# Instead of passing values, normalized values, score, cost, and metadata as
# separate variables, the optimizer stores them together in one object.
@dataclass
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



# Main optimizer class.
# GOW creates an instance of this class when the YAML selects name: asa.
# The Optimizer base class defines the interface expected by GOW.
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
            initial_temperature: 1.0
            final_temperature: 0.001
            reanneal_interval: 64
            restart_interval: 240
            restart_sigma: 0.012

    Advanced ASA options such as generating_distribution, temperature_schedule,
    acceptance_rule, reannealing_method, quenching factors, mutation fractions,
    and target_acceptance_rate already have defaults in the constructor. They can
    be exposed later only if the user really needs advanced tuning.

    Notes:
        - Internal score convention follows DifferentialEvolutionOptimizer:
          higher score is better.
        - ASA cost is cost = -score.
        - The optimizer can run in batch mode because GOW evaluates candidates
          externally and then returns all results through tell().
        - Batch mode is a GOW/HPC adaptation: candidates in one ask() call are
          generated from the current state available at ask() time.
    """

    # ------------------------------------------------------------------
    # Construction and configuration
    # ------------------------------------------------------------------
    # __init__ runs once, before the optimization loop starts.
    # It stores configuration values and creates empty variables for the
    # state that will be filled later by ask() and tell().
    def __init__(
        self,
        *,
        seed: int | None = None,
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
        # max_iterations is intentionally not exposed as a user setting.
        # In this implementation, the internal stopping limit must follow
        # GOW optimizer.max_evaluations. If an old YAML still contains
        # settings.max_iterations, it is ignored here to avoid conflicts.
        kwargs.pop("max_iterations", None)

        # The exact YAML value candidate is no longer emitted as a special
        # first candidate. If an old YAML still contains this option, it is
        # ignored so old files do not break the constructor.
        kwargs.pop("include_initial_candidate", None)

        # Temporary fallback. The authoritative value is synchronized from
        # the GOW ProblemConfig in ask(), when that object is available.
        if max_evaluations is None:
            max_evaluations = 800

        # Store reproducibility and budget information.
        # self._rng is a private random generator, so a seed gives repeatable
        # ASA proposals without changing Python's global random state.
        self.seed = seed
        self._rng = random.Random(seed)

        self.max_iterations = int(max_evaluations)
        self.batch_size = int(batch_size)

        # Temperature settings.
        # initial_temperature controls early exploration.
        # final_temperature is the lower limit reached after cooling.
        # cooling_rate is used only by the geometric schedule option.
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.cooling_rate = float(cooling_rate)

        # Gaussian proposal scales.
        # These are mainly used when generating_distribution="gaussian" or
        # when acceptance-rate reannealing adapts per-parameter step sizes.
        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # Fraction of parameters mutated per proposal.
        # ASA can start by changing many parameters together and later move
        # toward smaller, more local changes.
        self.mutation_parameter_fraction_start = float(mutation_parameter_fraction_start)
        self.mutation_parameter_fraction_end = float(mutation_parameter_fraction_end)

        # Reannealing settings.
        # Reannealing periodically adjusts search scales using recent
        # acceptance behavior and/or parameter sensitivity estimates.
        self.reanneal_interval = int(reanneal_interval)
        self.target_acceptance_rate = float(target_acceptance_rate)

        # Soft restart settings.
        # A soft restart moves the current state near the best known state
        # after a long period without improvement.
        self.restart_interval = int(restart_interval)
        self.restart_sigma = float(restart_sigma)
        self.include_initial_candidate = False

        # Advanced ASA mode selectors.
        # These strings choose the proposal distribution, temperature
        # schedule, acceptance rule, and reannealing strategy.
        self.generating_distribution = str(generating_distribution).strip().lower()
        self.temperature_schedule = str(temperature_schedule).strip().lower()
        self.acceptance_rule = str(acceptance_rule).strip().lower()
        self.reannealing_method = str(reannealing_method).strip().lower()

        # Quenching and scaling parameters for the ASA temperature schedule.
        # Values near 1 keep the behavior close to the default ASA mode.
        self.parameter_quench_factor = float(parameter_quench_factor)
        self.cost_quench_factor = float(cost_quench_factor)
        self.asa_temperature_scale = float(asa_temperature_scale)

        self.reanneal_temperature_min_factor = float(reanneal_temperature_min_factor)
        self.reanneal_temperature_max_factor = float(reanneal_temperature_max_factor)
        self.sensitivity_floor = float(sensitivity_floor)

        # Any extra keyword settings are stored for diagnostics/compatibility.
        # They are not used directly by the optimization loop.
        self.extra_config = dict(kwargs)

        # General optimizer state.
        # _initialized becomes True after ASA reads the problem definition.
        # _done becomes True when the evaluation budget is exhausted.
        self._initialized: bool = False
        self._done: bool = False
        self._generation: int = 0
        self._evaluations_seen: int = 0
        self._max_iterations_synced_from_problem: bool = False

        # Parameter metadata.
        # These fields are filled from the GOW ProblemConfig in
        # _initialize_from_problem().
        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction: str = "maximize"

        # Initial YAML reference values.
        # They define the starting reference in normalized space, but they
        # are not returned as a special first candidate.
        self._initial_values: Dict[str, Any] = {}
        self._initial_normalized: Dict[str, float] = {}

        # ASA search state.
        # _current_state is the state from which new proposals are generated.
        # _best_state is the best evaluated state found so far.
        self._current_state: Optional[_State] = None
        self._best_state: Optional[_State] = None

        # Temperature state.
        # cost_temperature controls acceptance of worse candidates.
        # param_temperatures control proposal sizes per parameter.
        self.cost_temperature: float = self.initial_temperature
        self.temperature: float = self.initial_temperature  # backward-compatible alias
        self.cost_annealing_index: int = 0

        self.param_initial_temperatures: Dict[str, float] = {}
        self.param_final_temperatures: Dict[str, float] = {}
        self.param_temperatures: Dict[str, float] = {}
        self.param_temperature_multipliers: Dict[str, float] = {}
        self.param_annealing_indices: Dict[str, int] = {}
        self.param_annealing_coefficients: Dict[str, float] = {}

        # Per-parameter Gaussian scales and recent acceptance counters.
        # These support acceptance-rate reannealing.
        self.per_param_sigma: Dict[str, float] = {}
        self.per_param_attempts: Dict[str, int] = {}
        self.per_param_accepts: Dict[str, int] = {}

        # Sensitivity accumulators.
        # They estimate which parameters strongly affect the objective.
        self.param_sensitivity_sum: Dict[str, float] = {}
        self.param_sensitivity_count: Dict[str, int] = {}
        self.param_last_sensitivity: Dict[str, Optional[float]] = {}

        # Metadata bridge between ask() and tell().
        # ask() stores how each candidate was generated; tell() recovers that
        # metadata after GOW returns the corresponding fitness.
        self._pending_metadata_by_key: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}

        # Diagnostic counters.
        # These do not drive the objective directly, but they make the run
        # easier to inspect and debug.
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

        # Validate configuration before the run begins. This catches invalid
        # YAML values early, before candidates are generated.
        self._validate_config()


    # ------------------------------------------------------------------
    # GOW optimizer interface
    # ------------------------------------------------------------------
    # ask() is the candidate-generation side of the ask/tell loop.
    # GOW calls it with n = optimizer.batch_size.
    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        """
        Return n candidate parameter dictionaries to GOW.
        """
        # If the optimizer already reached its evaluation budget, there is
        # nothing else to propose.
        if self._done:
            return []

        # Synchronize the internal budget with GOW max_evaluations once.
        # This prevents settings.max_iterations from acting as a separate
        # and conflicting stopping criterion.
        if not self._max_iterations_synced_from_problem:
            self._sync_max_iterations_from_problem(problem)

        # First ask() call: read parameter names, bounds, direction, and
        # starting reference values from the GOW problem.
        if not self._initialized:
            self._initialize_from_problem(problem)

        # n is the number of candidates GOW wants for this batch.
        # For ASA, this is the batch size used by one ask()/tell() cycle.
        requested = int(n)
        if requested <= 0:
            raise ValueError(f"ASAOptimizer.ask() requires n > 0, got n={n}")

        # Generate internal _State objects, then keep metadata so tell() can
        # understand how each returned candidate was produced.
        states = self._generate_candidate_states(requested)

        self._pending_metadata_by_key = {
            self._candidate_key(state.values): dict(state.metadata or {})
            for state in states
        }

        # GOW only receives real parameter dictionaries, not internal ASA
        # normalized values, costs, or metadata.
        return [dict(state.values) for state in states]

    # tell() is the result-processing side of the ask/tell loop.
    # GOW calls it after evaluating the candidates returned by ask().
    def tell(self, candidates: Sequence[Dict[str, Any]], fitness: Sequence[Any]) -> None:
        """
        Update optimizer state from evaluated candidates and fitness dicts.
        """
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); ASA is not initialized.")

        # The two lists must match position by position:
        # candidates[i] was evaluated and produced fitness[i].
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        # Reset diagnostics for this evaluated batch.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        # Process each evaluated candidate in order.
        for candidate, raw_fitness in zip(candidates, fitness):
            if self._done:
                break

            # Convert the evaluator output to ASA/GOW internal score.
            # The internal convention is always: higher score is better.
            score = self._normalize_score(raw_fitness)

            # Convert the real candidate values back to normalized [0, 1]
            # space and recover the metadata saved during ask().
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

            # A score of -inf means the evaluation was invalid or unusable.
            # Such proposals are represented but cannot be accepted.
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
                    score=score,
                    cost=-score,
                    metadata=metadata,
                )

            # Apply the ASA acceptance rule and update current/best states.
            accepted = self._process_proposed_state(proposed)

            # From this point on, one more evaluated candidate has been
            # processed by ASA.
            self._evaluations_seen += 1
            # Cooling gradually reduces the probability of accepting worse
            # candidates as the search progresses.
            self._cool_cost_temperature(accepted=accepted)

            # Periodically adapt temperatures and/or step sizes using recent
            # behavior. This is the "adaptive" part of ASA.
            if self.reanneal_interval > 0 and self._evaluations_seen % self.reanneal_interval == 0:
                self._reanneal()

            # If the run has gone too long without improvement, ASA can move
            # the current state near the best known state and continue.
            if self.restart_interval > 0:
                stale_for = self._evaluations_seen - self.last_improvement_evaluation
                if stale_for >= self.restart_interval and self._best_state is not None:
                    self._soft_restart_around_best()
                    self.last_improvement_evaluation = self._evaluations_seen

            # Stop when the internal evaluation budget is reached.
            # This budget is synchronized with GOW max_evaluations.
            if self._evaluations_seen >= self.max_iterations:
                self._done = True

        # One full tell() call corresponds to one completed GOW batch/cycle.
        self._generation += 1
        self._pending_metadata_by_key = {}

    # GOW uses is_done() to know whether it should stop asking for candidates.
    def is_done(self) -> bool:
        """Return true when ASA has reached the GOW evaluation budget."""
        return self._done or self._evaluations_seen >= self.max_iterations

    # diagnostics() reports the optimizer state without changing the search.
    def diagnostics(self) -> Dict[str, Any]:
        """Return small JSON-serializable diagnostics."""
        # Acceptance rates are useful indicators of whether ASA is exploring
        # too broadly, too narrowly, or at a reasonable scale.
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
    # Build the candidate states that ask() will return to GOW.
    def _generate_candidate_states(self, count: int) -> List[_State]:
        """
        Generate candidate states.

        Every emitted state is generated as an ASA proposal. The exact YAML
        value candidate is not emitted automatically as a special first point.
        """
        # Candidates are first built as _State objects because ASA needs
        # normalized values and metadata in addition to real values.
        out: List[_State] = []

        while len(out) < count:
            # Before the first accepted state exists, proposals are generated
            # around the normalized YAML reference values.
            if self._current_state is None:
                base_normalized = dict(self._initial_normalized)
            else:
                base_normalized = dict(self._current_state.normalized)

            # Mutate in normalized space, then convert back to real parameter
            # values so the external evaluator can understand the candidate.
            candidate_normalized, mutated_names, step_deltas = self._mutate_normalized(base_normalized)
            candidate_values = self._denormalize_candidate(candidate_normalized)

            # Store both the candidate values and how they were generated.
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

    # Decide whether a proposed candidate becomes the new current state.
    # This is where simulated annealing differs from greedy search.
    def _process_proposed_state(self, proposed: _State) -> bool:
        """
        Decide whether to accept a proposed state.

        Returns:
            True if accepted, False otherwise.
        """
        # Invalid candidates are rejected immediately.
        if proposed.score == float("-inf") or proposed.cost is None:
            self.rejected_count += 1
            self.window_total += 1
            return False

        # The first valid candidate initializes both current and best state.
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

        # delta_cost compares the proposed state against the current state.
        # Negative or zero means the proposal is better or equal for ASA.
        previous_state = self._current_state
        delta_cost = proposed.cost - previous_state.cost

        # Better moves are accepted directly.
        if delta_cost <= 0.0:
            accept = True
            accepted_worse = False
        else:
            # Worse moves are accepted only with a temperature-controlled
            # probability. This can help ASA escape local minima.
            probability = self._acceptance_probability(delta_cost)
            accept = self._rng.random() < probability
            accepted_worse = accept

        # Record which parameters were attempted. Reannealing uses this later.
        self._record_param_attempts(proposed)

        # If accepted, the proposed state becomes the new current state.
        if accept:
            self._current_state = self._copy_state(proposed)
            self.accepted_count += 1
            self.window_accepted += 1

            if accepted_worse:
                self.worse_accepted_count += 1

            # Accepted moves also contribute to sensitivity estimates.
            self._record_param_accepts(proposed)
            self._record_historical_sensitivity(previous_state, proposed, delta_cost)

            assert proposed.score is not None
            # Best state is tracked by score because internally higher score
            # is always better, even for minimization problems.
            if self._best_state is None or proposed.score > self._best_state.score:
                self._best_state = self._copy_state(proposed)
                self.improvement_count += 1
                self.last_improvement_evaluation = self._evaluations_seen
        else:
            self.rejected_count += 1

        self.window_total += 1
        return accept

    # Compute the probability of accepting a worse move.
    def _acceptance_probability(self, delta_cost: float) -> float:
        """
        Probability of accepting a worse move.

        metropolis_exp:
            p = exp(-delta_cost / T_cost)

        asa_logistic:
            p = 1 / (1 + exp(delta_cost / T_cost))
        """
        # Protect against division by zero while preserving the meaning of a
        # very small temperature.
        temperature = max(self.cost_temperature, 1.0e-300)
        ratio = delta_cost / temperature

        # exp(-ratio) would underflow for very large ratio, so return zero.
        if ratio > 700.0:
            return 0.0

        if self.acceptance_rule == "metropolis_exp":
            return math.exp(-ratio)

        if self.acceptance_rule == "asa_logistic":
            return 1.0 / (1.0 + math.exp(ratio))

        raise ValueError(f"Unsupported acceptance_rule: {self.acceptance_rule!r}")

    # Mutate a subset of parameters in normalized [0, 1] space.
    def _mutate_normalized(
        self,
        base: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
        """
        Mutate a subset of parameters in normalized space.
        """
        # Start from a copy of the base point so the original state is not
        # modified while building a proposal.
        candidate = dict(base)

        # Decide how many dimensions to modify in this proposal.
        fraction = self._current_mutation_parameter_fraction()
        param_count = len(self._param_names)

        mutation_count = max(1, int(round(param_count * fraction)))
        mutation_count = min(param_count, mutation_count)

        # Randomly choose which parameter names will be changed.
        mutated_names = self._rng.sample(self._param_names, mutation_count)
        step_deltas: Dict[str, float] = {}

        # Apply one random step to each selected parameter.
        for name in mutated_names:
            old_value = candidate[name]

            # Main ASA mode: heavy-tailed steps controlled by parameter
            # temperature.
            if self.generating_distribution == "ingber_asa":
                temperature = self.param_temperatures[name]
                step = self._generate_ingber_asa_step(temperature)

            # Alternative mode: local Gaussian steps controlled by sigma.
            elif self.generating_distribution == "gaussian":
                sigma = self.per_param_sigma.get(name, self.sigma_start)
                step = self._rng.gauss(0.0, sigma)

            else:
                raise ValueError(
                    f"Unsupported generating_distribution: {self.generating_distribution!r}"
                )

            # Keep normalized values inside [0, 1], which corresponds to
            # staying inside the original parameter bounds.
            new_value = self._clip01(old_value + step)
            candidate[name] = new_value
            step_deltas[name] = new_value - old_value

            # Each mutated parameter advances its own annealing counter and
            # receives an updated temperature for future proposals.
            self.param_annealing_indices[name] = self.param_annealing_indices.get(name, 0) + 1
            self._refresh_parameter_temperature(name)

        return candidate, mutated_names, step_deltas

    # Generate one Ingber-style ASA step in normalized space.
    # The heavy tail means small moves are common, but larger jumps remain
    # possible, especially at higher temperatures.
    def _generate_ingber_asa_step(self, temperature: float) -> float:
        """
        Generate one ASA normalized step y in [-1, 1].

        Inverse-CDF form:

            y = sign(u - 1/2) * T * [ (1 + 1/T)^|2u - 1| - 1 ]

        where u ~ U(0, 1).
        """
        temperature = max(float(temperature), 1.0e-300)

        # Draw one uniform random number and transform it into an ASA step.
        u = self._rng.random()
        sign = -1.0 if u < 0.5 else 1.0
        exponent = abs(2.0 * u - 1.0)

        log_base = math.log1p(1.0 / temperature)
        value = temperature * (math.exp(exponent * log_base) - 1.0)

        value = self._clip_value(value, 0.0, 1.0)
        return sign * value

    # Compute the current fraction of parameters to mutate.
    def _current_mutation_parameter_fraction(self) -> float:
        """
        Linearly decrease mutated parameter fraction during the run.
        """
        if self.max_iterations <= 1:
            progress = 1.0
        else:
            # progress moves from 0 at the start to 1 near the evaluation
            # budget limit.
            progress = min(1.0, max(0.0, self._evaluations_seen / self.max_iterations))

        start = self.mutation_parameter_fraction_start
        end = self.mutation_parameter_fraction_end

        return start + (end - start) * progress


    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------
    # Update the cost temperature after each processed evaluation.
    def _cool_cost_temperature(self, *, accepted: bool) -> None:
        """
        Update the cost temperature.

        For the Ingber-style schedule, the cost temperature index is advanced
        by accepted states. For geometric mode, it is advanced every evaluation.
        """
        # Geometric schedule: multiply by a fixed cooling rate each time.
        if self.temperature_schedule == "geometric":
            self.cost_temperature = max(
                self.final_temperature,
                self.cost_temperature * self.cooling_rate,
            )

        # ASA schedule: advance the cost temperature when a state is accepted.
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

        # Keep the older temperature alias synchronized for compatibility.
        self.temperature = self.cost_temperature

    # Refresh the temperature for one parameter after that parameter mutates.
    def _refresh_parameter_temperature(self, name: str) -> None:
        """
        Refresh one parameter temperature from its annealing index and multiplier.
        """
        if self.temperature_schedule == "geometric":
            current = self.param_temperatures[name] * self.cooling_rate
            self.param_temperatures[name] = max(self.param_final_temperatures[name], current)
            return

        if self.temperature_schedule == "ingber_asa":
            # Compute the schedule value, then apply any reannealing
            # multiplier that has been learned for this parameter.
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

    # Compute the ASA temperature curve used by cost and parameter schedules.
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
        initial = max(float(initial), 1.0e-300)
        final = max(float(final), 1.0e-300)
        index = max(0, int(index))

        if index <= 0:
            return initial

        # The schedule depends on the number of optimizable parameters.
        dimension = max(1, len(self._param_names))
        q_over_d = max(float(quench_factor), 1.0e-12) / float(dimension)

        horizon = max(1, self.max_iterations)
        denominator = float(horizon) ** q_over_d

        coefficient = -math.log(final / initial) / denominator
        coefficient *= self.asa_temperature_scale

        temperature = initial * math.exp(-coefficient * (float(index) ** q_over_d))
        return max(final, temperature)

    # Same coefficient as _asa_temperature(), returned for diagnostics.
    def _computed_asa_coefficient(self, initial: float, final: float, quench_factor: float) -> float:
        """
        Return ASA cooling coefficient for diagnostics.
        """
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
    # Reannealing adjusts search scales during the run.
    def _reanneal(self) -> None:
        """
        Reanneal according to selected method.
        """
        # "none" disables adaptation but still clears the recent windows.
        if self.reannealing_method == "none":
            self._reset_reanneal_windows()
            return

        # acceptance_rate uses recent accept/reject behavior.
        if self.reannealing_method == "acceptance_rate":
            self._reanneal_acceptance_rate()

        # historical_sensitivity uses how much each parameter changed cost.
        elif self.reannealing_method == "historical_sensitivity":
            self._reanneal_historical_sensitivity()

        # hybrid applies both adaptation mechanisms.
        elif self.reannealing_method == "hybrid":
            self._reanneal_acceptance_rate()
            self._reanneal_historical_sensitivity()

        else:
            raise ValueError(f"Unsupported reannealing_method: {self.reannealing_method!r}")

        self._reset_reanneal_windows()

    # Adapt search scale using the recent acceptance rate.
    def _reanneal_acceptance_rate(self) -> None:
        """
        Adapt Gaussian sigma and lightly rescale parameter temperatures using
        recent acceptance behavior.
        """
        if self.window_total <= 0:
            return

        # Compare observed acceptance against target_acceptance_rate.
        acceptance_rate = self.window_accepted / self.window_total

        # Too few acceptances: moves may be too large, so reduce scale.
        if acceptance_rate < self.target_acceptance_rate * 0.5:
            global_factor = 0.70
            temp_factor = 0.85
        # Many acceptances: moves may be too small, so expand scale.
        elif acceptance_rate > self.target_acceptance_rate * 1.5:
            global_factor = 1.15
            temp_factor = 1.10
        else:
            global_factor = 1.00
            temp_factor = 1.00

        # Apply the adjustment to each parameter separately.
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

    # Adapt temperatures using historical sensitivity estimates.
    def _reanneal_historical_sensitivity(self) -> None:
        """
        Reanneal parameter temperatures using historical sensitivity.

        Approximation:

            sensitivity_i ≈ |Δcost| / |Δx_i|

        More sensitive parameters get lower temperatures. Less sensitive
        parameters are allowed broader moves.
        """
        # Build a dictionary of average sensitivities for parameters that
        # have enough accepted-move information.
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

        # A parameter more sensitive than the mean gets a smaller temperature.
        # A less sensitive parameter gets a larger temperature.
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

    # Rescale one parameter temperature while preserving the ASA schedule.
    def _rescale_parameter_temperature(self, name: str, factor: float) -> None:
        """
        Rescale parameter temperature while preserving ASA schedule through a
        multiplier. This avoids losing reannealing effects at the next refresh.
        """
        current = self.param_temperatures[name]
        # factor > 1 expands future moves; factor < 1 narrows them.
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

    # Clear short-term counters after a reannealing step.
    def _reset_reanneal_windows(self) -> None:
        """
        Reset window counters after reannealing.
        """
        self.window_accepted = 0
        self.window_total = 0

        self.per_param_attempts = {name: 0 for name in self._param_names}
        self.per_param_accepts = {name: 0 for name in self._param_names}

        self.param_sensitivity_sum = {name: 0.0 for name in self._param_names}
        self.param_sensitivity_count = {name: 0 for name in self._param_names}

    # Soft restart moves the current state near the best known state.
    # It is not a full random restart; it keeps the search centered around
    # the best region found so far.
    def _soft_restart_around_best(self) -> None:
        """
        Soft restart around the best known candidate.

        This is a GOW-practical option to recover from frozen chains.
        """
        if self._best_state is None:
            return

        # Work around the best state in normalized space so restart_sigma has
        # the same interpretation for every parameter.
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

        # Reheat moderately so the restart is not immediately frozen.
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
    # Read objective direction, optimizable parameters, bounds, and YAML
    # starting values from the GOW problem definition.
    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        """
        Extract parameter names, bounds and YAML initial values from GOW.
        """
        # Store whether the original objective is minimize or maximize.
        self._direction = self._get_direction(problem)

        # Only parameters marked as optimizable are handled by ASA.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for ASA.")

        self._param_names = []
        self._param_specs = {}
        self._initial_values = {}
        self._initial_normalized = {}

        # Register every supported optimizable parameter.
        for name, p in params.items():
            # Real-valued parameter with continuous bounds.
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")

                lo, hi = float(p.bounds[0]), float(p.bounds[1])

                if not (lo < hi):
                    raise ValueError(f"Real param '{name}' must have lo < hi (got {lo}, {hi})")

                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            # Integer parameter. ASA still works internally with normalized
            # floats, but values are rounded before evaluation.
            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")

                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])

                if lo_i > hi_i:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})")

                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

            # ASA needs numeric distances and temperatures, so raw
            # categorical choices are rejected.
            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"ASA does not support categorical param '{name}'. "
                    "Use RandomSearch or encode categoricals into numeric space first."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

            kind, (lo, hi) = self._param_specs[name]
            # Use the YAML value as the reference point when available.
            initial_value = getattr(p, "value", None)

            # If no YAML value exists, use the middle of the bounds.
            if initial_value is None:
                initial_value = 0.5 * (lo + hi)

            initial_value = self._clip_value(float(initial_value), lo, hi)

            if kind == "int":
                initial_value = int(round(initial_value))
                initial_value = int(self._clip_value(float(initial_value), lo, hi))

            # Store both the real reference value and its normalized version.
            self._initial_values[name] = initial_value
            self._initial_normalized[name] = self._normalize_value(name, float(initial_value))

        if not self._param_names:
            raise ValueError("No supported optimizable parameters found for ASA.")

        # Once parameters are known, initialize ASA-specific temperatures
        # and counters.
        self._initialize_asa_state()
        self._initialized = True

    # Initialize temperatures, sigma values, counters, and sensitivity
    # accumulators for the current problem.
    def _initialize_asa_state(self) -> None:
        """
        Initialize ASA temperatures, counters and diagnostics.
        """
        self.cost_temperature = self.initial_temperature
        self.temperature = self.cost_temperature
        self.cost_annealing_index = 0

        # Each parameter starts with the same initial/final temperature range.
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
        # Precompute schedule coefficients for diagnostics.
        self.param_annealing_coefficients = {
            name: self._computed_asa_coefficient(
                initial=self.initial_temperature,
                final=self.final_temperature,
                quench_factor=self.parameter_quench_factor,
            )
            for name in self._param_names
        }

        # Start every Gaussian sigma at sigma_start.
        self.per_param_sigma = {
            name: self.sigma_start for name in self._param_names
        }
        self.per_param_attempts = {
            name: 0 for name in self._param_names
        }
        self.per_param_accepts = {
            name: 0 for name in self._param_names
        }

        # No accepted move sensitivities exist yet.
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
    # Convert evaluator output into ASA internal score.
    def _normalize_score(self, fitness_value: Any) -> float:
        """
        Convert GOW fitness dict into internal higher-is-better score.

        This follows the same convention used by DifferentialEvolutionOptimizer.
        """
        # Some evaluators may return a plain number directly.
        if isinstance(fitness_value, (int, float)):
            x = float(fitness_value)
            if not math.isfinite(x):
                self._n_non_finite += 1
                return float("-inf")
            if self._direction == "minimize":
                x = -x
            return x

        # Otherwise ASA expects a dictionary-like fitness object.
        if not isinstance(fitness_value, Mapping):
            self._n_non_numeric += 1
            return float("-inf")

        # Non-ok status means the external evaluation failed.
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

        # Fall back to common objective/fitness field names.
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
    # Convert a real candidate dictionary into normalized [0, 1] values.
    def _normalize_candidate(self, candidate: Mapping[str, Any]) -> Dict[str, float]:
        """
        Convert real parameter candidate to normalized [0, 1].
        """
        return {
            name: self._normalize_value(name, float(candidate[name]))
            for name in self._param_names
        }

    # Convert normalized values back into real parameter values for GOW.
    def _denormalize_candidate(self, normalized: Mapping[str, float]) -> Dict[str, Any]:
        """
        Convert normalized [0, 1] candidate to real parameter space.
        """
        candidate: Dict[str, Any] = {}

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            value = self._denormalize_value(name, float(normalized[name]))

            if kind == "int":
                value = int(round(value))
                value = int(self._clip_value(float(value), lo, hi))

            candidate[name] = value

        return candidate

    # Normalize one value using that parameter's bounds.
    def _normalize_value(self, name: str, value: float) -> float:
        """
        Normalize one real value to [0, 1].
        """
        _, (lo, hi) = self._param_specs[name]
        return self._clip01((value - lo) / (hi - lo))

    # Denormalize one value from [0, 1] to its real bounds.
    def _denormalize_value(self, name: str, value: float) -> float:
        """
        Denormalize one [0, 1] value to real bounds.
        """
        _, (lo, hi) = self._param_specs[name]
        value = self._clip01(value)
        return lo + value * (hi - lo)

    # Force a value to remain inside a closed interval.
    @staticmethod
    def _clip_value(value: float, low: float, high: float) -> float:
        """
        Clip value to [low, high].
        """
        return min(high, max(low, value))

    # Specialized clipping helper for normalized values.
    def _clip01(self, value: float) -> float:
        """
        Clip value to [0, 1].
        """
        return self._clip_value(value, 0.0, 1.0)

    # Copy a state so stored current/best states are not accidentally
    # modified later.
    def _copy_state(self, state: _State) -> _State:
        """
        Deep-ish copy of a state.
        """
        return _State(
            values=dict(state.values),
            normalized=dict(state.normalized),
            score=state.score,
            cost=state.cost,
            metadata=None if state.metadata is None else dict(state.metadata),
        )

    # Build a stable key from candidate values to match ask() metadata with
    # tell() results.
    def _candidate_key(self, candidate: Mapping[str, Any]) -> Tuple[Tuple[str, float], ...]:
        """
        Stable key used to recover proposal metadata in tell().
        """
        return tuple(
            (name, round(float(candidate[name]), 15))
            for name in sorted(self._param_names)
        )

    # Recover the metadata that was stored when this candidate was proposed.
    def _lookup_pending_metadata(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Recover metadata generated in ask().
        """
        return dict(self._pending_metadata_by_key.get(self._candidate_key(candidate), {}))

    # Count which parameters were attempted in a proposal.
    def _record_param_attempts(self, proposed: _State) -> None:
        """
        Record which parameters were changed in a proposal.
        """
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []

        for name in mutated:
            if name in self.per_param_attempts:
                self.per_param_attempts[name] += 1

    # Count which mutated parameters were part of an accepted proposal.
    def _record_param_accepts(self, proposed: _State) -> None:
        """
        Record accepted mutations per parameter.
        """
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []

        for name in mutated:
            if name in self.per_param_accepts:
                self.per_param_accepts[name] += 1

    # Estimate parameter sensitivity from accepted moves.
    def _record_historical_sensitivity(
        self,
        previous: _State,
        proposed: _State,
        delta_cost: float,
    ) -> None:
        """
        Record approximate per-parameter sensitivity from accepted moves.
        """
        if proposed.metadata is None:
            return

        mutated = proposed.metadata.get("mutated_parameters") or []
        step_deltas = proposed.metadata.get("normalized_step_deltas") or {}

        if not mutated:
            return

        # Sensitivity uses absolute cost change divided by absolute step size.
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
    # GOW evaluation budget synchronization
    # ------------------------------------------------------------------
    # Align ASA internal max_iterations with GOW optimizer.max_evaluations.
    def _sync_max_iterations_from_problem(self, problem: ProblemConfig) -> None:
        """
        Align ASA's internal stopping limit with GOW max_evaluations.
        """
        # Try the known places where GOW may store max_evaluations.
        optimizer_cfg = getattr(problem, "optimizer", None)
        max_evaluations = getattr(optimizer_cfg, "max_evaluations", None)

        if max_evaluations is None:
            max_evaluations = getattr(problem, "max_evaluations", None)

        if max_evaluations is None:
            optimizer_config = getattr(problem, "optimizer_config", None)
            max_evaluations = getattr(optimizer_config, "max_evaluations", None)

        # If GOW provides a budget, make it the internal ASA budget.
        if max_evaluations is not None:
            max_evaluations_int = int(max_evaluations)
            if max_evaluations_int <= 0:
                raise ValueError("optimizer.max_evaluations must be > 0")
            self.max_iterations = max_evaluations_int

        self._max_iterations_synced_from_problem = True


    # ------------------------------------------------------------------
    # Objective direction
    # ------------------------------------------------------------------
    # Read whether the original objective is a minimization or maximization.
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


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    # Validate user-facing and internal configuration values early.
    def _validate_config(self) -> None:
        """
        Validate configuration values early.
        """
        if self.max_iterations <= 0:
            raise ValueError("max_iterations/internal max_evaluations must be > 0")

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


AdaptiveSimulatedAnnealingOptimizer = ASAOptimizer
