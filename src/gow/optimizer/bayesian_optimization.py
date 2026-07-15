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
    Uses skopt.Optimizer with the ask/tell interface.

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    Bayesian Optimization (BO) is an optimization method designed for problems
    where evaluating one candidate can be expensive.

    In this project, one candidate is one complete set of parameter values.
    For example:

        {"p0": 2.26, "p1": -0.00093, "p2": 4.895, ...}

    GOW sends that candidate to an external evaluator. The evaluator computes
    the real objective value. In the Sun Position case, this can be an error
    metric such as:

        metrics.errors_arcsec.sun_vector.average

    The optimizer does not calculate that objective by itself. It only decides
    which candidates should be evaluated next.

    BO works differently from population-based optimizers such as GA or PSO.
    It does not move a swarm and it does not evolve a population. Instead, it
    builds an internal model of the objective function using the evaluations
    that have already been observed.

    That internal model is usually called a surrogate model.

    -------------------------------------------------------------------------
    MAIN BO CONCEPTS
    -------------------------------------------------------------------------

    Surrogate model:
        A cheaper model that tries to approximate the expensive objective.
        Instead of evaluating the real objective everywhere, BO learns from the
        points already evaluated and uses that learned model to choose better
        future points.

    base_estimator:
        The type of surrogate model used by scikit-optimize.

        Common examples in skopt are:
          - "GP"   : Gaussian Process
          - "RF"   : Random Forest
          - "ET"   : Extra Trees
          - "GBRT" : Gradient Boosted Regression Trees

    acquisition function:
        A function that decides which point looks promising according to the
        surrogate model.

        It balances two ideas:
          - exploitation: try near regions that already look good;
          - exploration : try uncertain regions that may hide a better result.

        Common examples in skopt are:
          - "EI"       : Expected Improvement
          - "PI"       : Probability of Improvement
          - "LCB"      : Lower Confidence Bound
          - "gp_hedge" : automatic mixture of acquisition functions

    acq_optimizer:
        The method used internally by skopt to optimize the acquisition
        function. In simple terms, it decides how skopt searches for the next
        candidate suggested by the acquisition function.

    batch_strategy:
        Strategy used when asking for more than one candidate at the same time.
        In GOW, this happens when batch_size is greater than 1.

        skopt uses strategies such as:
          - "cl_min"
          - "cl_mean"
          - "cl_max"

        These are constant-liar strategies. They allow skopt to propose a batch
        of several candidates before their real objective values are known.

    n_initial_points:
        Number of initial evaluations used before the surrogate model has
        enough information to guide the search properly.

        These points are not the same thing as batch_size. They are the initial
        observations used to start learning the objective behavior.

    max_iterations:
        Maximum number of completed BO rounds.

        In this implementation, one iteration is completed after one ask/tell
        cycle. If GOW requests a batch of 100 candidates, then one iteration
        corresponds to one completed batch of 100 evaluated candidates.

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores BO hyperparameters and prepares internal variables.
           It does not create the skopt optimizer yet because the problem
           parameters are not known until ask() receives the ProblemConfig.

      2. ask(problem, n)
           GOW calls this function to request new candidates.
           On the first call, the optimizer reads the problem bounds and creates
           the internal skopt optimizer.
           Then it asks skopt for n candidates.

      3. GOW evaluates the candidates outside this file.
           The real objective function is computed by the external evaluator.

      4. tell(candidates, fitness)
           GOW calls this function to return the evaluation results.
           BO gives those results back to skopt, so the surrogate model can be
           updated.

      5. ask(problem, n) is called again.
           Now the surrogate model has more information, so the new candidates
           are chosen using the acquisition function.

      6. The ask() / tell() cycle repeats until max_iterations is reached.

    -------------------------------------------------------------------------
    GOW INTEGRATION
    -------------------------------------------------------------------------

    GOW uses an ask/tell interface:

      - ask() produces candidates.
      - tell() receives results.

    In this implementation:

      - GOW's batch_size reaches this optimizer as n in ask(problem, n).
      - n controls how many candidates are requested in one BO round.
      - One completed tell() call counts as one BO iteration.
      - max_iterations limits the number of completed BO rounds.
      - max_iterations is therefore similar to the role of generations in GA or
        PSO, but BO is not an evolutionary algorithm.
      - Real and integer optimizable parameters are supported.
      - Optimizable categorical parameters are not supported.
      - Internally, skopt minimizes losses. Therefore this wrapper converts GOW
        scores so that the correct objective direction is respected.
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
        """
        Store the initial Bayesian Optimization configuration.

        This function runs only once, when the optimizer object is created.
        It does not start the optimization yet. It only stores the values that
        control BO and prepares empty variables where the optimizer state will
        be stored later.

        BO-specific hyperparameters:

        n_initial_points:
            Number of initial points used by skopt before relying mainly on the
            surrogate model and acquisition function.

            These points give BO its first information about the objective.
            Without initial observations, the surrogate model has no data from
            which to learn.

        base_estimator:
            Surrogate model used by skopt.

            Example:
                base_estimator = "GP"

            means:
                use a Gaussian Process as the model that approximates the
                objective function.

        acquisition_function:
            Rule used to decide which candidate looks promising according to
            the surrogate model.

            Example:
                acquisition_function = "EI"

            means:
                use Expected Improvement.

        max_iterations:
            Maximum number of BO rounds.

            In this implementation, one round is completed when tell() receives
            the results for the batch previously produced by ask().

        acq_optimizer:
            Internal method used by skopt to optimize the acquisition function.
            "auto" lets skopt choose a suitable method.

        batch_strategy:
            Strategy used by skopt when several candidates are requested in the
            same ask() call.

            This is important when GOW batch_size is greater than 1.

        seed:
            Optional random seed. It helps make the sequence of proposed
            candidates reproducible when the rest of the workflow is also
            deterministic.
        """

        # ------------------------------------------------------------------
        # Basic hyperparameter validation
        # ------------------------------------------------------------------
        # These checks prevent impossible configurations before the algorithm
        # starts running.
        #
        # For example, BO cannot start with zero initial points and cannot have
        # zero iterations.
        if n_initial_points < 1:
            raise ValueError("n_initial_points must be >= 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        # ------------------------------------------------------------------
        # Store BO hyperparameters
        # ------------------------------------------------------------------
        # These values usually come from the optimizer.settings block in the
        # YAML configuration.
        self.n_initial_points = int(n_initial_points)
        self.base_estimator = base_estimator
        self.acquisition_function = acquisition_function
        self.max_iterations = int(max_iterations)
        self.acq_optimizer = acq_optimizer
        self.batch_strategy = batch_strategy
        self.seed = seed

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the internal skopt optimizer has already been created.
        # At the beginning this is False because ask() has not been called yet.
        self._initialized = False

        # Number of completed BO iterations.
        # This is incremented in tell(), after one full batch has been evaluated.
        self._iteration = 0

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Names of the parameters that BO will optimize.
        #
        # Example:
        #   ["p0", "p1", "p2"]
        self._param_names: List[str] = []

        # Specifications of each optimizable parameter.
        #
        # The key is the parameter name.
        # The value is a tuple with:
        #   (kind, (lower_bound, upper_bound))
        #
        # Example:
        #   self._param_specs["p0"] = ("real", (0.0, 10.0))
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}

        # Objective direction: "minimize" or "maximize".
        # It is read from problem in _initialize_from_problem().
        self._direction = "maximize"

        # The real scikit-optimize Optimizer object.
        # It starts as None because it cannot be created until the parameter
        # bounds are known.
        self._optimizer: SkoptOptimizer | None = None

        # ------------------------------------------------------------------
        # Last candidates sent to the evaluator
        # ------------------------------------------------------------------
        # IMPORTANT:
        # This stores the vectors actually delivered to the evaluator after
        # repair. Therefore skopt.tell() learns f(repaired_x), not f(raw_x).
        #
        # In simple terms:
        #   - skopt may propose a raw candidate;
        #   - this implementation may repair it to satisfy extra constraints;
        #   - the repaired candidate is the one evaluated;
        #   - therefore the repaired candidate is the one sent back to skopt.
        self._last_xs: List[List[Any]] = []

        # ------------------------------------------------------------------
        # Best solution found so far
        # ------------------------------------------------------------------

        # Best internal score found by this wrapper.
        # Internally, higher score is better.
        self._best_score: float | None = None

        # Candidate associated with the best internal score.
        self._best_candidate: Dict[str, Any] | None = None

        # ------------------------------------------------------------------
        # Diagnostic counters
        # ------------------------------------------------------------------
        # These counters help identify whether the evaluator returned invalid,
        # incomplete, or non-numeric results.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        """
        Generate candidates for GOW to evaluate.

        This function answers the question:

            "Which points in the search space should be evaluated now?"

        GOW calls ask() and expects a list of candidates.
        Each candidate is a dictionary with parameter values.

        Example candidate:
            {"p0": 2.3, "p1": -0.001, "p2": 5}

        ask() flow:

          1. Read n, the number of candidates requested by GOW.
          2. If this is the first call, initialize the internal skopt optimizer.
          3. Ask skopt for n new candidate vectors.
          4. Convert those vectors into GOW candidate dictionaries.
          5. Repair candidates if special problem constraints require it.
          6. Store the repaired vectors so tell() can report the correct points
             back to skopt.
          7. Return the repaired candidate list to GOW.
        """

        # GOW passes the number of candidates requested for this BO round as n.
        # In normal GOW execution, this corresponds to optimizer.batch_size.
        if n < 1:
            raise ValueError("ask(..., n) requires n >= 1")

        # First call to ask(): the skopt optimizer does not exist yet.
        # It is created here because now we have access to the full problem
        # configuration, including parameter types and bounds.
        if not self._initialized:
            self._initialize_from_problem(problem)

        # Safety check. If initialization failed, the optimizer cannot continue.
        if self._optimizer is None:
            raise RuntimeError("Bayesian optimizer was not initialized correctly.")

        # Ask skopt for n candidate points.
        #
        # n_points=n:
        #   number of candidates requested in this batch.
        #
        # strategy=self.batch_strategy:
        #   strategy used when n > 1. This controls how a batch of candidates is
        #   proposed before their real objective values are known.
        raw_xs = self._optimizer.ask(n_points=n, strategy=self.batch_strategy)

        # skopt may return a slightly different shape when only one point is
        # requested. This block ensures raw_xs is always a list of points.
        if n == 1 and raw_xs and not isinstance(raw_xs[0], list):
            raw_xs = [raw_xs]

        # Convert every returned point to a standard Python list.
        # This makes the following conversion steps predictable.
        raw_xs = [list(x) for x in raw_xs]

        # Convert skopt vectors into GOW candidate dictionaries.
        candidates = [self._vector_to_candidate(x) for x in raw_xs]

        # Apply extra repair rules required by some heliostat-style problems.
        # For the Sun Position PSA parameter problem, these repair rules usually
        # do not modify p0, p1, ..., p14 because those special keys are absent.
        repaired_candidates = [
            self._repair_candidate(problem, cand) for cand in candidates
        ]

        # Store the exact repaired points that GOW will evaluate.
        # Later, tell() will pass these same points to skopt together with their
        # objective values.
        self._last_xs = [
            self._candidate_to_vector(cand) for cand in repaired_candidates
        ]

        # Return the list of candidates to GOW.
        return repaired_candidates

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """
        Receive evaluation results and update the BO model.

        This function answers the question:

            "How well did each candidate proposed by ask() perform?"

        GOW evaluates candidates outside the optimizer and then calls tell()
        with two lists:

          - candidates:
              the candidates that were evaluated;

          - fitness:
              the results obtained for those candidates.

        Order matters:

          - candidates[0] corresponds to fitness[0]
          - candidates[1] corresponds to fitness[1]
          - etc.

        In BO, tell() is where the observations are sent back to skopt.
        skopt then uses this information to update its surrogate model and to
        choose better candidates in the next ask() call.
        """

        # Reset diagnostic counters for this iteration.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        # tell() only makes sense after ask(), because ask() initializes the
        # optimizer and stores the candidate vectors in self._last_xs.
        if not self._initialized or self._optimizer is None:
            raise RuntimeError(
                "tell() called before first ask(); Bayesian optimizer is not initialized."
            )

        # Every candidate must have exactly one fitness result.
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        # The number of candidates received by tell() must match the number of
        # candidates produced by the previous ask() call.
        if len(self._last_xs) != len(candidates):
            raise RuntimeError(
                "tell(): number of candidates does not match the previous ask() call."
            )

        # Convert evaluator results into internal scores.
        # The internal rule used by this wrapper is always:
        #
        #     higher score = better candidate
        scores = [self._normalize_score(fdict) for fdict in fitness]

        # skopt minimizes losses, not scores.
        # This list will store one loss for each evaluated candidate.
        losses: List[float] = []

        # Loop over each score and its corresponding candidate.
        for score, cand in zip(scores, candidates):
            # A score of -inf means the evaluation was invalid or unusable.
            # skopt still needs a numeric loss, so invalid points receive a very
            # large loss. This makes them look extremely bad.
            if score == float("-inf"):
                losses.append(1.0e100)
                continue

            # Convert internal score to skopt loss.
            # Since skopt minimizes, loss = -score.
            losses.append(-score)

            # Store the best candidate found by this wrapper.
            # Because the internal rule is "higher score is better", the best
            # score is the largest score.
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_candidate = dict(cand)

        # Send evaluated points and losses back to skopt.
        # This is the key BO learning step: skopt updates its internal model
        # with the new observations.
        self._optimizer.tell(self._last_xs, losses)

        # One full BO iteration is complete after results for the previous ask()
        # have been processed.
        self._iteration += 1

        # Clear the last candidate vectors.
        # The next ask() call will fill this list again.
        self._last_xs = []

    def is_done(self) -> bool:
        """
        Return whether the optimizer should stop.

        BO stops when the number of completed iterations reaches
        max_iterations.

        In this implementation:

            one iteration = one completed ask/tell cycle

        If GOW uses batch_size = 100, then one iteration corresponds to one
        completed batch of 100 evaluated candidates.
        """

        # If the optimizer has not even started, it is not done.
        if not self._initialized:
            return False

        # Stop when the number of completed BO iterations reaches the configured
        # maximum.
        return self._iteration >= self.max_iterations

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return useful information about the current optimizer state.

        This function does not ask for new candidates and does not update the
        model. It only reports information.

        Main fields:

          - iteration:
              current BO iteration.

          - n_initial_points:
              number of initial observations requested by skopt.

          - base_estimator:
              surrogate model type.

          - acquisition_function:
              acquisition function used to select promising candidates.

          - acq_optimizer:
              method used to optimize the acquisition function.

          - batch_strategy:
              strategy used when requesting several candidates at once.

          - best_score_internal:
              best internal score found by this wrapper.

          - best_candidate:
              candidate associated with the best internal score.

          - n_status_failed, n_missing_score, n_non_numeric, n_non_finite:
              counters for issues detected in evaluator results.
        """

        # Return the main configuration and current state in a dictionary.
        # GOW can store or print this information for debugging and reporting.
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
        """
        Create the internal skopt optimizer from the GOW problem definition.

        This function is called automatically from ask() the first time GOW
        requests candidates.

        Initialization steps:

          1. Read the objective direction: minimize or maximize.
          2. Read the optimizable parameters of the problem.
          3. Validate that each parameter has valid bounds.
          4. Convert GOW parameter definitions into skopt dimensions.
          5. Create the skopt.Optimizer object.
          6. Mark this wrapper as initialized.

        Initialization uses only the problem definition and the BO
        hyperparameters stored in __init__().
        """

        # Store whether the real problem objective is minimize or maximize.
        self._direction = self._get_direction(problem)

        # GOW provides the parameters marked as optimizable.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for Bayesian Optimization.")

        # skopt dimensions will be stored here.
        # Each dimension represents one optimizable parameter.
        dimensions = []

        # Clear these structures before filling them for the current problem.
        self._param_names = []
        self._param_specs = {}

        # Loop over all optimizable parameters.
        # name is the parameter name.
        # p is the parameter configuration object.
        for name, p in params.items():
            # --------------------------------------------------------------
            # Real-valued parameter
            # --------------------------------------------------------------
            if isinstance(p, RealParam):
                # Real parameters must have bounds such as [lo, hi].
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(
                        f"Optimizable real param '{name}' missing bounds=[lo,hi]"
                    )

                # Convert bounds to float.
                lo, hi = float(p.bounds[0]), float(p.bounds[1])

                # Bounds must make sense: lower bound must be smaller than upper
                # bound.
                if not (lo < hi):
                    raise ValueError(
                        f"Real param '{name}' must have lo < hi (got {lo}, {hi})"
                    )

                # Store the parameter name so vectors can be converted back to
                # dictionaries in a stable order.
                self._param_names.append(name)

                # Store type and bounds for later conversions and repairs.
                self._param_specs[name] = ("real", (lo, hi))

                # Add a real-valued skopt dimension.
                dimensions.append(Real(lo, hi, name=name))

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            elif isinstance(p, IntParam):
                # Integer parameters must also have bounds such as [lo, hi].
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(
                        f"Optimizable int param '{name}' missing bounds=[lo,hi]"
                    )

                # Convert bounds to integers.
                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])

                # For integers, lo can be equal to hi only if the parameter has
                # one possible value. This implementation accepts lo <= hi.
                if lo_i > hi_i:
                    raise ValueError(
                        f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})"
                    )

                # Store the parameter name and its specification.
                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

                # Add an integer skopt dimension.
                dimensions.append(Integer(lo_i, hi_i, name=name))

            # --------------------------------------------------------------
            # Optimizable categorical parameter
            # --------------------------------------------------------------
            # This BO implementation does not support categorical parameters.
            #
            # Examples of categorical values would be:
            #   "small", "medium", "large"
            #   "red", "blue", "green"
            #
            # skopt can support categories in other contexts, but this GOW
            # wrapper intentionally rejects them here.
            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"BayesianOptimizationOptimizer does not support categorical param '{name}'. "
                    "Use numeric encoding or another optimizer."
                )

            # --------------------------------------------------------------
            # Unknown parameter type
            # --------------------------------------------------------------
            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        # Create the real scikit-optimize optimizer.
        #
        # This object performs the core Bayesian Optimization logic:
        #   - initial sampling;
        #   - surrogate model fitting;
        #   - acquisition function use;
        #   - candidate proposal through ask();
        #   - model update through tell().
        self._optimizer = SkoptOptimizer(
            dimensions=dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            acq_optimizer=self.acq_optimizer,
            random_state=self.seed,
        )

        # Reset iteration and last-candidate memory at the start of a run.
        self._iteration = 0
        self._last_xs = []

        # Mark the optimizer as ready to use.
        self._initialized = True

    def _vector_to_candidate(self, x: List[Any]) -> Dict[str, Any]:
        """
        Convert a skopt vector into a GOW candidate dictionary.

        skopt represents a point as a list because it only cares about the
        order of dimensions.

        Example skopt vector:
            [2.3, -0.001, 5]

        GOW expects a dictionary with parameter names.

        Example GOW candidate:
            {"p0": 2.3, "p1": -0.001, "p2": 5}

        This function also ensures that values remain inside bounds and that
        integer parameters are returned as integers.
        """

        # Candidate dictionary that will be returned to GOW.
        cand: Dict[str, Any] = {}

        # zip(x, self._param_names) pairs each value with its parameter name.
        # Example:
        #   value = 2.3, name = "p0"
        for value, name in zip(x, self._param_names):
            # Get parameter type and bounds.
            kind, (lo, hi) = self._param_specs[name]

            # Integer parameter handling.
            # skopt may provide a numeric value. This wrapper rounds it and
            # clamps it inside bounds.
            if kind == "int":
                value = int(round(float(value)))
                value = int(self._clip(float(value), lo, hi))
            else:
                # Real parameter handling.
                # Convert to float and clamp inside bounds.
                value = float(value)
                value = self._clip(value, lo, hi)

            # Store the value using the parameter name expected by GOW.
            cand[name] = value

        return cand

    def _candidate_to_vector(self, cand: Dict[str, Any]) -> List[Any]:
        """
        Convert a GOW candidate dictionary back into a skopt vector.

        This is the inverse of _vector_to_candidate().

        It is used before tell() sends evaluated points back to skopt.
        skopt needs vectors in the same dimension order used during
        initialization.
        """

        # Read candidate values in the exact same order as self._param_names.
        return [cand[name] for name in self._param_names]

    def _repair_candidate(
        self,
        problem: ProblemConfig,
        cand: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply extra safety corrections to a candidate.

        This function exists because some optimization problems may have
        constraints that are not simple lower/upper bounds.

        Example:
            A heliostat layout may require one distance to be at least as large
            as another distance.

        The function returns a repaired copy of the candidate.

        Important:
            For the Sun Position p0-p14 problem, these repair rules normally do
            not change anything because the special heliostat keys used below
            are not present.
        """

        # Work on a copy so the original candidate is not modified directly.
        repaired = dict(cand)

        # ------------------------------------------------------------------
        # Repair rule for r_min
        # ------------------------------------------------------------------
        # If the candidate contains r_min, make sure it is not smaller than a
        # physically reasonable inner radius.
        if "r_min" in repaired:
            # Read auxiliary values from the candidate or from fixed problem
            # parameters. If they do not exist, use safe defaults.
            receiver_radius = self._get_param_value(
                problem, repaired, "flat_receiver_radius", 0.0
            )
            min_clearance = self._get_param_value(
                problem, repaired, "min_tower_clearance", 0.0
            )
            mh = self._get_param_value(problem, repaired, "mirror_height", 4.06)
            mw = self._get_param_value(problem, repaired, "mirror_width", 4.06)

            # Mirror diagonal computed from height and width.
            diag = math.sqrt(mh * mh + mw * mw)

            # Minimum allowed inner radius.
            r_inner = receiver_radius + min_clearance + 0.5 * diag

            # Enforce r_min >= r_inner.
            repaired["r_min"] = max(float(repaired["r_min"]), float(r_inner))

        # ------------------------------------------------------------------
        # Repair rule for row/diameter relationship
        # ------------------------------------------------------------------
        # If both factors exist, rowrow_diameter_factor must not be smaller than
        # chord_diameter_factor.
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
        """
        Read a parameter value from the candidate or from the problem.

        Priority order:

          1. If the key is present in the candidate, use the candidate value.
          2. Otherwise, look for a fixed parameter in problem.parameters.
          3. If it is not found, use the provided default value.

        This helper is mainly used by _repair_candidate().
        """

        # Candidate values have priority because they are the values currently
        # being proposed for evaluation.
        if key in cand:
            return float(cand[key])

        # If the value is not in the candidate, it may be a fixed parameter in
        # the GOW problem configuration.
        p = problem.parameters.get(key)
        if p is None:
            return float(default)

        # GOW parameter objects store their value in p.value.
        return float(p.value)

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        """
        Convert the evaluator result into an internal score.

        Evaluators may return results using different keys, for example:

          - fitness
          - objective
          - score
          - loss

        This wrapper needs to compare all of them with one internal rule:

            higher internal score = better candidate

        This function normalizes the result.

        Important cases:

          - If the evaluator failed, return -inf.
          - If the objective value is missing, return -inf.
          - If the value is not numeric, return -inf.
          - If the real objective is minimization, invert the sign.

        -inf means "worse than any valid result".
        """

        # Some evaluators return a status field.
        # If status exists and is not "ok", treat the result as invalid.
        status = fitness_dict.get("status")

        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        # val will store the numeric value found.
        val: Any = None

        # key will store what kind of value was found: fitness, objective, etc.
        key: str | None = None

        # First, look for the value directly in the main dictionary.
        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        # If it is not directly present, it may be inside a sub-dictionary
        # called metrics.
        if key is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("fitness", "objective", "score", "loss"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        # If no value was found, the result cannot be used for comparison.
        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        # An empty string is also not a valid numeric value.
        if isinstance(val, str) and not val.strip():
            self._n_missing_score += 1
            return float("-inf")

        # Try to convert the value to a real number.
        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        # Reject NaN, +inf, and -inf as valid evaluator results.
        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        # If the evaluator returns loss, lower loss is normally better.
        # To keep the internal rule "higher is better", invert the sign.
        if key == "loss":
            x = -x

        # If the problem is minimization, lower real objective is better.
        # To compare internally with "higher is better", invert the sign.
        if self._direction == "minimize":
            x = -x

        return x

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        """
        Limit a number so it stays inside [lo, hi].

        Example:
            x = 12, lo = 0, hi = 10

        returns:
            10

        This is a safety helper used when converting values between skopt and
        GOW.
        """

        # If x is below the lower bound, return lo.
        # If x is above the upper bound, return hi.
        # Otherwise, return x unchanged.
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _get_direction(problem: ProblemConfig) -> str:
        """
        Read whether the problem is minimization or maximization.

        GOW may define this information in problem.objective.direction.

        Accepted values:

          - "minimize"
          - "maximize"

        Unlike the PSO reference implementation, this function only accepts the
        full words "minimize" and "maximize". It raises an error for unknown
        values instead of silently choosing a default.
        """

        # Default direction used if the problem does not explicitly define one.
        direction = "maximize"

        # Read the objective object from the problem, if it exists.
        obj = getattr(problem, "objective", None)

        # If an objective exists, try to read objective.direction.
        if obj is not None:
            direction = getattr(obj, "direction", direction) or direction

        # Normalize text: convert to lowercase and remove surrounding spaces.
        direction = str(direction).lower().strip()

        # Only two directions are accepted by this implementation.
        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"Unknown objective direction '{direction}' "
                "(expected 'minimize' or 'maximize')."
            )

        return direction
