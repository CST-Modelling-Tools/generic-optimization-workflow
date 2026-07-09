from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import cma

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class CMAESOptimizer(Optimizer):
    """
    CMA-ES optimizer adapted to GOW.

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    CMA-ES means Covariance Matrix Adaptation Evolution Strategy.

    It is a population-based stochastic optimizer. This means that it does not
    evaluate only one candidate at a time. Instead, it samples a group of
    candidates from a probability distribution and then uses their evaluation
    results to improve that distribution for the next generation.

    In CMA-ES:

      - The population is the group of candidates generated in one generation.
      - Each candidate is one possible solution to the optimization problem.
      - The search distribution is a multivariate normal distribution.
      - The mean of the distribution represents the current center of the search.
      - The sigma value controls the global step size of the search.
      - The covariance matrix controls the shape and orientation of the search.
      - Good candidates influence how the distribution moves and adapts.

    Common CMA-ES names:

      - x:
          candidate vector sampled by CMA-ES.

      - x0:
          initial mean of the CMA-ES search distribution.

      - sigma / sigma0:
          global step size. It controls the initial sampling radius around x0.

      - batch_size / population_size / popsize / lambda:
          number of candidates sampled per generation.

          In the GOW YAML, the public value is batch_size.
          Inside CMA-ES, that same value is used as population_size, because
          population size is the traditional CMA-ES name for this quantity.

      - covariance matrix:
          matrix that allows CMA-ES to learn correlations between parameters.

    This implementation uses the external `cma` Python package for the internal
    CMA-ES mechanics. Therefore, this file does not manually implement all
    mathematical CMA-ES updates. The `cma.CMAEvolutionStrategy` object handles
    the update of the mean, sigma, and covariance matrix internally.

    -------------------------------------------------------------------------
    NORMALIZED SEARCH SPACE
    -------------------------------------------------------------------------

    GOW works with real parameter values and their bounds.
    This wrapper makes CMA-ES search in normalized coordinates instead:

        normalized value = 0.0  -> lower bound
        normalized value = 1.0  -> upper bound

    Therefore, the internal CMA-ES search space is:

        [0, 1]^n

    where n is the number of optimizable parameters.

    For each parameter:

        real_value = lo + normalized_value * (hi - lo)

    This is useful because CMA-ES receives all parameters on a comparable scale,
    even if the original parameters have very different physical ranges.

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores the CMA-ES hyperparameters and prepares internal variables.
           It does not create the CMA-ES engine yet because the problem bounds
           are not known until ask(problem, n) receives the ProblemConfig.

      2. ask(problem, n)
           GOW calls this function to request new candidates.
           On the first call, ask() initializes CMA-ES from the problem bounds.
           Then it asks the external `cma` package for normalized vectors.

      3. _normalized_to_candidate(x)
           Converts each normalized vector from [0, 1]^n into a GOW candidate
           with real or integer parameter values.

      4. GOW evaluates the candidates outside this file.
           The optimizer does not compute the objective function directly.
           It only proposes candidates. The external evaluator computes their
           quality.

      5. tell(candidates, fitness)
           GOW calls this function to return the evaluation results.
           The results are converted into internal scores and then into CMA-ES
           losses, because the `cma` package expects a minimization loss.

      6. self._es.tell(...)
           The external CMA-ES engine receives the sampled vectors and their
           losses. It updates its internal mean, sigma, and covariance matrix.

      7. The ask() / tell() cycle repeats until max_generations is reached or
           until the internal CMA-ES stop criteria are triggered.

    -------------------------------------------------------------------------
    GOW INTEGRATION
    -------------------------------------------------------------------------

    GOW uses an ask/tell interface:

      - ask() produces candidates.
      - tell() receives results.

    In this implementation:

      - optimizer.batch_size defines how many candidates are requested per
        generation.
      - Internally, CMA-ES uses that same value as population_size.
      - population_size is not a user-facing YAML setting in this wrapper.
      - CMA-ES-specific settings such as sigma0 and max_generations can remain
        under optimizer.settings, because they belong to CMA-ES and not to the
        general GOW optimizer interface.
      - GOW must call ask(..., n=batch_size).
      - Real and integer optimizable parameters are supported.
      - Optimizable categorical parameters are not supported.
      - The internal score convention is: higher score is better.
      - The external `cma` package uses losses, where lower loss is better.
      - Therefore, valid internal scores are converted to losses using -score.
      - Invalid evaluations receive a very large loss, so CMA-ES treats them as
        very bad candidates.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        sigma0: float = 0.05,
        max_generations: int = 100,
        seed: int | None = None,
    ):
        """
        Store the initial CMA-ES configuration.

        This function runs only once, when the optimizer object is created.
        It does not create the CMA-ES engine yet. Initialization is delayed
        until _initialize_from_problem(), because the optimizer needs the
        problem bounds to build the normalized search space.

        CMA-ES-specific hyperparameters:

        batch_size:
            Public GOW configuration value that defines the number of
            candidates sampled in each generation.

            In CMA-ES literature this same quantity is usually called
            population_size, popsize, or lambda. To avoid duplicated YAML
            parameters, this wrapper exposes only batch_size to GOW and then
            uses it internally as population_size.

            Example:
                batch_size = 16

            means:
                16 candidates evaluated per generation
                internal CMA-ES population_size = 16

            In normal GOW execution, this value is read from the general
            optimizer.batch_size field in the YAML and reaches CMA-ES as n in
            ask(problem, n). If batch_size is passed directly to this
            constructor, the same relationship is used.

            Important YAML rule:
                population_size should not be placed in optimizer.settings.
                The population size is derived from optimizer.batch_size.

        sigma0:
            CMA-ES-specific initial global step size in normalized space.
            This value belongs to the CMA-ES configuration, so it can be passed
            through optimizer.settings in the YAML.

            Because this implementation searches inside [0, 1]^n, sigma0 is
            interpreted relative to that normalized range.

            Example:
                sigma0 = 0.05

            means:
                the initial search radius is small compared with the full
                normalized interval [0, 1].

        max_generations:
            CMA-ES-specific maximum number of completed ask/tell generations.
            This value can also be passed through optimizer.settings in the YAML.

        seed:
            Optional random seed passed to the external `cma` package.
            It helps make the same optimization run reproducible.
        """

        # ------------------------------------------------------------------
        # Basic hyperparameter validation
        # ------------------------------------------------------------------
        # These checks prevent impossible configurations before the optimizer
        # starts running.
        #
        # CMA-ES needs at least two candidates per generation because it learns
        # from comparing sampled candidates.
        #
        # The public GOW name is batch_size. Internally, that same value is
        # stored as population_size because the external `cma` package uses the
        # traditional CMA-ES population terminology.
        if batch_size is not None and batch_size < 2:
            raise ValueError("batch_size must be >= 2 for CMA-ES")

        # sigma0 is a step size, so it must be strictly positive.
        if sigma0 <= 0.0:
            raise ValueError("sigma0 must be > 0")

        # The run must have at least one generation.
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")

        # Number of candidates evaluated per generation.
        #
        # In the YAML, this value belongs to the main optimizer block as
        # optimizer.batch_size. It should not be duplicated inside
        # optimizer.settings as population_size.
        #
        # CMA-ES-specific values such as sigma0 and max_generations may still
        # come from optimizer.settings, because they are not general GOW fields.
        #
        # Depending on how the optimizer is created, batch_size may or may not
        # be passed directly to this constructor.
        #
        # If it is passed here, CMA-ES stores it immediately.
        # If it is not passed here, CMA-ES will infer it from n on the first
        # ask(problem, n) call.
        self.batch_size = int(batch_size) if batch_size is not None else None

        # Internal CMA-ES population size.
        # This is the same quantity as batch_size, but population_size is the
        # name expected by CMA-ES terminology and by the external `cma` package.
        self.population_size = self.batch_size

        # Initial global step size used by the external CMA-ES engine.
        self.sigma0 = float(sigma0)

        # Maximum number of completed generations allowed by this wrapper.
        self.max_generations = int(max_generations)

        # Optional seed forwarded to the `cma` package during initialization.
        self.seed = seed

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the external CMA-ES engine has already been created.
        # At the beginning this is False because ask() has not been called yet.
        self._initialized = False

        # Number of completed generations.
        # This is incremented in tell(), after evaluation results are received.
        self._generation = 0

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Names of the parameters that CMA-ES will optimize.
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

        # ------------------------------------------------------------------
        # External CMA-ES engine state
        # ------------------------------------------------------------------

        # Instance of cma.CMAEvolutionStrategy.
        # It remains None until _initialize_from_problem() creates it.
        self._es = None

        # Normalized vectors generated by the most recent ask() call.
        # tell() must pass these exact vectors back to the external CMA-ES
        # engine together with their losses.
        self._last_xs: List[List[float]] = []

        # Best internal score seen so far.
        # Internally, higher score is always better.
        self._best_score = None

        # Best GOW candidate seen so far, expressed in real parameter values.
        self._best_candidate = None

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
          2. If the internal population size has not been fixed yet, set it
             from n.
          3. Check that n matches the fixed internal population size.
          4. If CMA-ES has not been initialized yet, initialize it from problem.
          5. Ask the external `cma` package for normalized vectors.
          6. Store those vectors so tell() can later return their losses.
          7. Convert normalized vectors into GOW candidates.
          8. Return the candidate list.
        """

        # GOW passes the number of candidates requested for this generation as n.
        #
        # For CMA-ES, n defines the population size because the population is
        # exactly the group of candidates sampled in one generation.
        #
        # In the YAML, the user should configure only optimizer.batch_size.
        # The internal population_size is derived from that same value.
        #
        # If batch_size was not passed directly to __init__, the first ask()
        # call fixes the internal population size from n.
        if self.population_size is None:
            if n < 2:
                raise ValueError("batch_size must be >= 2 for CMA-ES")
            self.batch_size = int(n)
            self.population_size = int(n)

        # Once the population size is fixed, it must remain constant during the
        # run. CMA-ES cannot change the number of candidates in the middle of
        # an optimization because the external engine was configured with a
        # fixed popsize.
        if n != self.population_size:
            raise ValueError(
                "CMAESOptimizer requires ask(..., n=batch_size). "
                f"Got n={n}, batch_size={self.batch_size}, "
                f"internal population_size={self.population_size}."
            )

        # First call to ask(): the external CMA-ES engine does not exist yet.
        # It is created here because the problem provides bounds and initial
        # parameter values.
        if not self._initialized:
            self._initialize_from_problem(problem)

        # Ask the external CMA-ES engine for a new population.
        # Each x is a normalized vector in [0, 1]^n.
        xs = self._es.ask(number=self.population_size)

        # Store the normalized vectors from this generation.
        # tell() must pass the same vectors back to self._es.tell().
        self._last_xs = [list(x) for x in xs]

        # Convert normalized vectors into GOW candidates with real/int values.
        return [self._normalized_to_candidate(x) for x in self._last_xs]

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """
        Receive evaluation results and update the CMA-ES engine.

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

        CMA-ES update flow:

          1. Convert evaluator results into internal scores.
          2. Convert internal scores into losses for the `cma` package.
          3. Store the best candidate seen so far.
          4. Call self._es.tell(...) so CMA-ES can update its distribution.
          5. Count one completed generation.
        """

        # Reset diagnostic counters for this generation.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        # tell() only makes sense after ask(), because ask() creates the CMA-ES
        # engine and stores the normalized vectors in self._last_xs.
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); CMA-ES is not initialized.")

        # Every candidate must have exactly one fitness result.
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        # CMA-ES expects one result for every candidate in the population.
        # The population size is the internal CMA-ES name for GOW batch_size.
        if len(candidates) != self.population_size:
            raise ValueError(
                "CMAESOptimizer expects exactly batch_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.population_size}"
            )

        # The normalized vectors must come from the immediately previous ask().
        # Without them, the external CMA-ES engine would not know which sampled
        # vectors produced the reported losses.
        if not self._last_xs or len(self._last_xs) != self.population_size:
            raise RuntimeError("tell(): missing CMA-ES vectors from previous ask().")

        # Convert each evaluator result into an internal score.
        # The internal rule is always: higher is better.
        scores = [self._normalize_score(fdict) for fdict in fitness]

        # The external `cma` package minimizes losses.
        # This wrapper therefore converts each valid internal score into -score.
        losses = []

        # Loop over each score and the corresponding candidate.
        # zip(scores, candidates) keeps the same order returned by GOW.
        for score, cand in zip(scores, candidates):
            # A score of -inf means that the evaluation was invalid.
            # Give CMA-ES a very large loss so the candidate is treated as bad.
            if score == float("-inf"):
                losses.append(1e100)
            else:
                # Valid candidate: convert score into a minimization loss.
                losses.append(-score)

                # Keep a readable copy of the best GOW candidate seen so far.
                # The comparison uses the internal score convention:
                # higher is better.
                if self._best_score is None or score > self._best_score:
                    self._best_score = score
                    self._best_candidate = dict(cand)

        # Send the sampled normalized vectors and their losses back to CMA-ES.
        # This is where the external engine updates its mean, sigma, and
        # covariance matrix.
        self._es.tell(self._last_xs, losses)

        # When tell() finishes, one generation is considered complete.
        self._generation += 1

        # Clear the stored vectors so tell() cannot accidentally be called twice
        # for the same ask() output.
        self._last_xs = []

    def is_done(self) -> bool:
        """
        Return whether the optimizer should stop.

        CMA-ES stops when either:

          - this wrapper reaches max_generations; or
          - the external `cma` engine reports one of its own stop criteria.
        """

        # If CMA-ES was never initialized, no generation has been completed.
        if not self._initialized:
            return False

        # Stop if the configured generation limit is reached, or if the
        # external CMA-ES engine has triggered its internal stop criteria.
        return self._generation >= self.max_generations or bool(self._es.stop())

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return useful information about the current optimizer state.

        This function does not sample candidates and does not update CMA-ES.
        It only reports information.

        Main fields:

          - generation:
              current generation.

          - batch_size:
              public GOW value: number of candidates sampled per generation.

          - population_size_internal:
              same value as batch_size, using the traditional CMA-ES name.

          - sigma:
              current global step size reported by the external CMA-ES engine.

          - best_score_internal:
              best internal score found so far. Internally, higher is better.

          - best_candidate:
              best candidate found so far, in original GOW parameter values.

          - n_status_failed, n_missing_score, n_non_numeric, n_non_finite:
              counters for issues detected in evaluator results.
        """

        # Return a dictionary that GOW can expose as optimizer diagnostics.
        # If CMA-ES has not been initialized yet, sigma falls back to sigma0.
        return {
            "generation": self._generation,
            "batch_size": self.batch_size,
            "population_size_internal": self.population_size,
            "sigma": float(getattr(self._es, "sigma", self.sigma0)) if self._es is not None else self.sigma0,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
        }

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        """
        Create the external CMA-ES engine from the GOW problem definition.

        This function is called automatically from ask() the first time GOW
        requests candidates.

        Initialization steps:

          1. Read the objective direction: minimize or maximize.
          2. Read the optimizable parameters of the problem.
          3. Validate that each parameter has valid bounds.
          4. Store the type and bounds of each parameter.
          5. Build x0, the initial mean in normalized space.
          6. Configure the external `cma` package options.
          7. Create cma.CMAEvolutionStrategy(x0, sigma0, opts).
          8. Mark the optimizer as initialized.

        Important detail:

          The values from the YAML are used to construct x0.
          That means they define the initial mean of the search distribution.
          They are not automatically forced to be evaluated as the first exact
          candidate by this function.
        """

        # Store whether the real problem objective is minimize or maximize.
        self._direction = self._get_direction(problem)

        # GOW provides the parameters marked as optimizable.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for CMA-ES.")

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
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not (lo < hi):
                    raise ValueError(f"Real param '{name}' must have lo < hi (got {lo}, {hi})")
                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])
                if lo_i > hi_i:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})")
                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

            # --------------------------------------------------------------
            # Optimizable categorical parameter
            # --------------------------------------------------------------
            # CMA-ES works in a continuous numeric vector space.
            # Categories such as "red", "blue", or "green" do not have a
            # natural distance, covariance, or normalized coordinate here.
            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"CMA-ES does not support categorical param '{name}'. "
                    "Use RandomSearch or encode categoricals into numeric space first."
                )

            # --------------------------------------------------------------
            # Unknown parameter type
            # --------------------------------------------------------------
            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        # Number of optimizable dimensions.
        # This is the length of the vector that the external CMA-ES engine sees.
        dim = len(self._param_names)

        # Build x0: the initial CMA-ES mean in normalized space.
        #
        # The initial values come from problem.parameters[name].value.
        # Each value is clipped to its bounds and then mapped to [0, 1].
        x0 = []
        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            p = problem.parameters[name]
            val = float(p.value)
            val = self._clip(val, lo, hi)
            x0.append((val - lo) / (hi - lo))

        # Options passed directly to the external `cma` package.
        opts = {
            # Population size used by the CMA-ES engine.
            # This value comes from optimizer.batch_size in the GOW YAML.
            "popsize": self.population_size,

            # Bounds in normalized space.
            # Every dimension is constrained to the interval [0, 1].
            "bounds": [[0.0] * dim, [1.0] * dim],

            # Disable CMA-ES console output.
            "verbose": -9,

            # Small tolerances. These are internal CMA-ES stopping criteria.
            "tolx": 1e-12,
            "tolfun": 1e-12,
        }

        # If a seed was provided, pass it to the external CMA-ES engine.
        if self.seed is not None:
            opts["seed"] = int(self.seed)

        # Create the external CMA-ES optimizer.
        # x0 is the initial mean and sigma0 is the initial global step size.
        self._es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        # Reset generation state for the new optimization problem.
        self._generation = 0
        self._last_xs = []

        # Mark the optimizer as ready.
        self._initialized = True

    def _normalized_to_candidate(self, x: List[float]) -> Dict[str, Any]:
        """
        Convert one normalized CMA-ES vector into one GOW candidate.

        CMA-ES works internally with vectors in [0, 1]^n.
        GOW expects a dictionary with the original parameter names and values.

        Conversion for each dimension:

            1. Clip the normalized value to [0, 1].
            2. Convert it to the real parameter range:

                   real_value = lo + value * (hi - lo)

            3. If the parameter is integer, round the value.
            4. Clip again to ensure the final value stays inside bounds.
        """

        # Candidate dictionary that will be returned to GOW.
        cand: Dict[str, Any] = {}

        # Iterate over the normalized values and their parameter names.
        # zip() keeps the same order stored in self._param_names.
        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            # Keep the normalized coordinate inside [0, 1].
            value = self._clip(float(value), 0.0, 1.0)

            # Convert normalized coordinate to the original parameter range.
            real_value = lo + value * (hi - lo)

            # Remove tiny numerical deviations at the bounds.
            # This avoids returning values like 1.0000000000000002 when the
            # mathematical value should be exactly the upper bound.
            if abs(real_value - lo) < 1e-15:
                real_value = lo
            if abs(real_value - hi) < 1e-15:
                real_value = hi

            # Integer parameters are rounded after conversion to real space.
            # They are clipped again because rounding could move a value just
            # outside the allowed interval.
            if kind == "int":
                real_value = int(round(real_value))
                real_value = int(self._clip(float(real_value), lo, hi))

            # Store the converted value under the original parameter name.
            cand[name] = real_value

        return cand

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        """
        Convert one evaluator result into an internal score.

        The internal convention of this wrapper is always:

            higher score is better

        This function accepts several possible evaluator output keys:

          - fitness
          - objective
          - score
          - loss

        It also looks inside a nested "metrics" dictionary if needed.

        Invalid or unusable results return -inf. That value is later converted
        into a very large CMA-ES loss.
        """

        # If the evaluator reports a non-ok status, treat the result as invalid.
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        # val will store the numeric result.
        # key remembers which field provided it, because "loss" has special
        # sign handling below.
        val: Any = None
        key: str | None = None

        # First, look for score-like keys at the top level of the result.
        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        # If no top-level score was found, look inside fitness_dict["metrics"].
        if key is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("fitness", "objective", "score", "loss"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        # No usable score key was found.
        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        # Empty strings are treated as missing scores.
        if isinstance(val, str) and not val.strip():
            self._n_missing_score += 1
            return float("-inf")

        # Convert the score-like value to float.
        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        # Reject NaN and infinite values.
        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        # A loss is already a minimization quantity.
        # Convert it to the internal score convention: higher is better.
        if key == "loss":
            x = -x

        # If the real problem is minimization, smaller objective values are
        # better. Convert them to higher-is-better scores by changing the sign.
        if self._direction == "minimize":
            x = -x

        return x

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        """
        Return x limited to the interval [lo, hi].

        This helper prevents values from going outside the configured bounds.
        """

        # If x is below the lower bound, return the lower bound.
        # If x is above the upper bound, return the upper bound.
        # Otherwise, return x unchanged.
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _get_direction(problem: ProblemConfig) -> str:
        """
        Read and validate the objective direction from the GOW problem.

        Accepted directions are:

          - "minimize"
          - "maximize"

        If the problem does not explicitly define a direction, this wrapper uses
        "maximize" as the default.
        """

        # Default objective direction.
        direction = "maximize"

        # Try to read problem.objective.direction if it exists.
        obj = getattr(problem, "objective", None)
        if obj is not None:
            direction = getattr(obj, "direction", direction) or direction

        # Normalize the value so variants like " Minimize " become "minimize".
        direction = str(direction).lower().strip()

        # Only two directions are valid for this optimizer wrapper.
        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"Unknown objective direction '{direction}' (expected 'minimize' or 'maximize')."
            )

        return direction
