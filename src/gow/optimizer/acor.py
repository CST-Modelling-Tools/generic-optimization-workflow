from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class ACOROptimizer(Optimizer):
    """
    Ant Colony Optimization for Continuous Domains (ACOR) adapted to GOW.

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    ACOR is an ant-colony optimizer for continuous optimization problems.
    Classical ACO was created for problems where ants choose discrete components,
    such as edges in a graph. ACOR keeps the same ant-colony idea, but replaces
    the discrete pheromone table with a continuous model.

    In this implementation:

      - Each candidate is one possible solution to the optimization problem.
      - Each ant creates one new candidate per generation.
      - The best evaluated candidates are stored in a solution archive.
      - The solution archive acts as the pheromone memory of the colony.
      - New candidates are sampled around archive solutions using Gaussian
        distributions.
      - Better archive solutions are more likely to be selected as centers for
        new samples.

    Common ACOR names:

      - ant:
          one artificial agent that proposes one candidate solution.

      - candidate:
          one complete set of parameter values to be evaluated by GOW.

      - archive:
          the internal memory that stores the best candidates found so far.
          In ACOR, this archive plays the role of the pheromone model.

      - Gaussian kernel:
          the continuous probability model used to sample a new value around an
          archive value.

      - q:
          parameter that controls how strongly ACOR prefers the best archive
          entries when choosing a center for a new sample.

      - xi:
          parameter that controls the sampling width around the selected archive
          entry.

    -------------------------------------------------------------------------
    HOW ACOR SAMPLES A NEW CANDIDATE
    -------------------------------------------------------------------------

    After the first generation has been evaluated, the archive contains evaluated
    candidates sorted from best to worst. To create a new candidate, ACOR does
    the following:

      1. It assigns a probability weight to each archive entry.
         Better ranked entries receive larger weights.

      2. It randomly chooses one archive entry using those weights.
         This chosen entry becomes the center of the new sample.

      3. For each optimizable parameter, it computes a Gaussian width.
         The width depends on how far the selected archive value is from the
         other values stored in the archive.

      4. It samples a new normalized value around the selected archive value.

      5. It converts the normalized value back to the real parameter bounds
         expected by GOW.

    The optimizer works internally in normalized coordinates, where every
    optimizable parameter is represented between 0 and 1. This makes the same
    algorithm work for parameters with very different physical ranges.

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores the algorithm hyperparameters and prepares the internal
           variables where the ACOR state will be stored.

      2. ask(problem, n)
           GOW calls this function to request new candidates.
           On the first call, ask() initializes the parameter information and
           returns random candidates.
           On later calls, ask() samples candidates from the archive.

      3. GOW evaluates the candidates outside this file.
           The optimizer does not compute the objective function directly.
           It only proposes candidates. The external evaluator computes their
           quality.

      4. tell(candidates, fitness)
           GOW calls this function to return the evaluation results to the
           optimizer. Those results are used to update the archive.

      5. ask(problem, n) is called again.
           At this point, the archive already contains evaluated candidates, so
           ACOR can sample new candidates around promising regions.

      6. The ask() / tell() cycle repeats until max_generations is reached.

    -------------------------------------------------------------------------
    GOW INTEGRATION
    -------------------------------------------------------------------------

    GOW uses an ask/tell interface:

      - ask() produces candidates.
      - tell() receives results.

    In this implementation:

      - GOW's batch_size defines how many candidates are requested per
        generation.
      - In ACOR, that same value becomes the number of ants per generation.
      - In ACOR, that same value also becomes the archive size internally.
      - Therefore, archive_size is not a YAML parameter for the user.
      - The initial candidate from the YAML values is not inserted manually.
      - The first archive is built only from candidates sampled by ACOR.
      - ACOR compares candidates internally using the rule: higher score is
        better.
      - If the real objective is minimization, the sign is inverted internally.
      - Real and integer optimizable parameters are supported.
      - Optimizable categorical parameters are not supported.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        q: float = 0.1,
        xi: float = 0.85,
        max_generations: int | None = None,
        min_sigma: float = 1e-12,
        bound_strategy: str = "clip",
        seed: int | None = None,
    ):
        """
        Store the initial ACOR configuration.

        This function runs only once, when the optimizer object is created.
        It does not create the archive yet. It only stores the hyperparameters
        and prepares the internal variables that will be used during
        optimization.

        ACOR-specific hyperparameters:

        batch_size:
            Number of candidates requested by GOW in each generation.

            In ACOR, each candidate corresponds to one ant. Therefore,
            batch_size is interpreted internally as both:

              - the number of ants per generation;
              - the size of the solution archive.

            Example:
                batch_size = 100

            means:
                100 candidates evaluated per generation
                100 ants per generation
                100 archive entries kept internally

            In normal GOW execution, this value is read from the general
            optimizer.batch_size field in the YAML and reaches ACOR as n in
            ask(problem, n). If batch_size is passed directly to this
            constructor, the same relationship is used.

        q:
            Archive selection pressure.

            Lower value:
                ACOR gives more probability to the best archive entries.
                The search becomes more focused around the current best areas.

            Higher value:
                ACOR gives a more even probability to the archive entries.
                The search uses more archive solutions as possible centers.

        xi:
            Sampling spread around the selected archive solution.

            Higher value:
                new candidates can be farther from the selected archive entry.
                The search explores more.

            Lower value:
                new candidates stay closer to the selected archive entry.
                The search becomes more local.

        max_generations:
            Optional maximum number of generations. One generation is one
            complete evaluation round of the ants produced by ask().

        min_sigma:
            Minimum Gaussian width used when sampling a new value.
            This prevents the search from collapsing to a zero-width Gaussian.

        bound_strategy:
            Strategy used when a Gaussian sample falls outside [0, 1].

              - "clip":
                  cut the value to the nearest valid boundary.

              - "resample":
                  try to draw another value inside the valid range. If that
                  fails repeatedly, clip the value as a fallback.

        seed:
            Optional random seed. It makes the same run reproducible by
            generating the same sequence of random choices.
        """

        # ------------------------------------------------------------------
        # Basic hyperparameter validation
        # ------------------------------------------------------------------
        # These checks stop impossible or unclear configurations before the
        # optimization starts.
        #
        # ACOR needs at least two archive slots to keep a useful memory of the
        # search. Because archive_size is now controlled by batch_size, the
        # batch size must also be at least 2 when it is provided here.
        if batch_size is not None and batch_size < 2:
            raise ValueError("batch_size must be >= 2 for ACOR")
        if q <= 0.0:
            raise ValueError("q must be > 0")
        if xi <= 0.0:
            raise ValueError("xi must be > 0")
        if max_generations is not None and max_generations < 1:
            raise ValueError("max_generations must be >= 1 when provided")
        if min_sigma <= 0.0:
            raise ValueError("min_sigma must be > 0")

        # Normalize the bound strategy text so that values such as " Clip "
        # and "clip" are treated in the same way.
        bound_strategy = str(bound_strategy).lower().strip()
        if bound_strategy not in {"clip", "resample"}:
            raise ValueError("bound_strategy must be either 'clip' or 'resample'")

        # Number of candidates evaluated per generation.
        # In GOW, this value is configured as optimizer.batch_size in the YAML.
        #
        # Depending on how the optimizer is created, batch_size may or may not
        # be passed directly to this constructor.
        #
        # If it is not passed here, ACOR will infer it from n on the first
        # ask(problem, n) call.
        self.batch_size = int(batch_size) if batch_size is not None else None

        # In this GOW implementation, batch_size defines the archive size.
        # This removes archive_size from the user-facing YAML configuration.
        #
        # The value may be None during __init__ and will be fixed on the first
        # ask(problem, n) call if batch_size was not passed here.
        self.archive_size = self.batch_size

        # In ACOR, one ant produces one candidate.
        # Therefore, the number of ants is also the batch size internally.
        self.ants = self.batch_size

        # Hyperparameters that control ACOR sampling.
        self.q = float(q)
        self.xi = float(xi)
        self.max_generations = int(max_generations) if max_generations is not None else None
        self.min_sigma = float(min_sigma)
        self.bound_strategy = bound_strategy
        self.seed = seed

        # Optimizer-specific random generator.
        # Using self._rng instead of Python's global random generator makes the
        # run reproducible when a seed is provided.
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the parameter information has already been read from the GOW
        # problem configuration.
        self._initialized = False

        # Number of completed generations.
        # This is incremented in tell(), after evaluation results are received.
        self._generation = 0

        # Objective direction: "minimize" or "maximize".
        # It is read from problem in _initialize_from_problem().
        self._direction = "maximize"

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Names of the parameters that ACOR will optimize.
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

        # ------------------------------------------------------------------
        # Current ACOR archive state
        # ------------------------------------------------------------------

        # The archive stores the best evaluated candidates found so far.
        #
        # Each archive entry is a dictionary with:
        #
        #   "x":
        #       normalized vector used internally by ACOR.
        #
        #   "candidate":
        #       candidate dictionary using the real parameter values expected by
        #       GOW.
        #
        #   "score":
        #       internal score used for comparison.
        #
        # The internal score is always maximized. If the real problem is a
        # minimization problem, _normalize_score() multiplies the objective by
        # -1 before storing it.
        self._archive: List[Dict[str, Any]] = []

        # Best internal score seen during the complete run.
        self._best_score: float | None = None

        # Candidate associated with _best_score.
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
        self._n_invalid_candidates = 0

        # Checkpoint v1 is allowed only between complete generations.
        # ask() marks that an evaluation batch is in flight and tell()
        # clears the flag only after the archive has been updated.
        self._awaiting_tell = False

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
          2. If batch_size has not been fixed yet, set batch_size = n.
          3. Use that same value as the internal archive size.
          4. If the problem has not been initialized yet, read the parameter
             names, types, and bounds.
          5. If the archive is empty, return random initial candidates.
          6. If the archive already exists, sample new candidates from it.
          7. Return the candidate list to GOW.
        """

        # GOW passes the number of candidates requested for this generation as n.
        #
        # For ACOR, n defines the number of ants because each ant produces
        # exactly one candidate per generation.
        #
        # The same n also defines archive_size internally. This is why the YAML
        # does not need a separate archive_size parameter.
        if self.batch_size is None:
            if n < 2:
                raise ValueError("batch_size must be >= 2 for ACOR")
            self.batch_size = int(n)
            self.archive_size = int(n)
            self.ants = int(n)

        # Once batch_size is fixed, it must remain constant during the run.
        # The archive size depends on it, so changing n in the middle of the
        # optimization would also change the memory structure of ACOR.
        if n != self.batch_size:
            raise ValueError(
                "ACOROptimizer requires ask(..., n=batch_size). "
                f"Got n={n}, batch_size={self.batch_size}, archive_size={self.archive_size}."
            )

        # First call to ask(): parameter names, types, bounds, and objective
        # direction are read from the GOW problem configuration.
        if not self._initialized:
            self._initialize_from_problem(problem)

        # First generation: the archive is still empty because no candidate has
        # been evaluated yet.
        #
        # ACOR starts by generating random candidates inside the configured
        # bounds. It does not force the YAML value candidate into the first
        # generation.

        if not self._archive:
            candidates = self._initial_candidates(n)
        else:
            # Later generations sample around the ranked archive.
            candidates = [
                self._sample_candidate_from_archive()
                for _ in range(n)
            ]

        # From this point the RNG has already advanced. A checkpoint here
        # would not represent a complete generation because the candidates
        # have not yet been incorporated into the archive by tell().
        self._awaiting_tell = True

        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """
        Receive evaluation results and update the ACOR archive.

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

        In ACOR, tell() is where the archive is updated. The archive keeps only
        the best archive_size entries, where archive_size is equal to batch_size.
        """

        # Reset diagnostic counters for this generation.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0
        self._n_invalid_candidates = 0

        # tell() only makes sense after ask(), because ask() initializes the
        # problem information and proposes the candidates.
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); ACOR is not initialized.")

        # Every candidate must have exactly one fitness result.
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        # ACOR expects one result per ant.
        if self.batch_size is not None and len(candidates) != self.batch_size:
            raise ValueError(
                "ACOROptimizer expects exactly batch_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.batch_size}"
            )

        # Store the valid evaluated candidates from this generation before
        # merging them into the archive.
        new_entries: List[Dict[str, Any]] = []

        # Process candidate and fitness pairs one by one.
        for cand, fdict in zip(candidates, fitness):
            # Convert the evaluator output into an internal score.
            # Invalid results become -inf and are ignored.
            score = self._normalize_score(fdict)

            if score == float("-inf"):
                continue

            try:
                # Convert the real candidate values back into normalized [0, 1]
                # coordinates so the candidate can be stored in the archive.
                x = self._candidate_to_normalized(cand)
            except Exception:
                self._n_invalid_candidates += 1
                continue

            # Add the candidate to the list of valid entries from this
            # generation.
            new_entries.append(
                {
                    "x": x,
                    "candidate": dict(cand),
                    "score": score,
                }
            )

            # Keep a separate best-so-far candidate for diagnostics.
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_candidate = dict(cand)

        # Merge the new valid entries with the previous archive.
        if new_entries:
            if self.archive_size is None:
                raise RuntimeError("archive_size was not initialized before tell().")

            self._archive.extend(new_entries)

            # Sort from best to worst using the internal score.
            self._archive.sort(key=lambda row: row["score"], reverse=True)

            # Keep only the best archive_size entries.
            self._archive = self._archive[: self.archive_size]

        # One complete generation has finished.
        self._generation += 1

        # The whole generation is now committed to the archive.
        self._awaiting_tell = False


    # ------------------------
    # Checkpoint / resume
    # ------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return all state required to continue ACOR exactly.

        Checkpoint v1 is deliberately restricted to generation boundaries.
        ask() consumes randomness before the evaluator runs, therefore an
        outstanding ask() must never be persisted as a completed generation.
        """

        if self._awaiting_tell:
            raise RuntimeError(
                "ACOR checkpoint can only be created between generations, "
                "after tell() has completed."
            )

        if not self._initialized:
            raise RuntimeError(
                "ACOR checkpoint requires an initialized optimizer."
            )

        if self.batch_size is None or self.archive_size is None:
            raise RuntimeError(
                "ACOR checkpoint requires a fixed batch/archive size."
            )

        return {
            "schema_version": 1,
            "optimizer": "acor",

            "configuration": {
                "batch_size": self.batch_size,
                "q": self.q,
                "xi": self.xi,
                "max_generations": self.max_generations,
                "min_sigma": self.min_sigma,
                "bound_strategy": self.bound_strategy,
            },

            "initialized": self._initialized,
            "generation": self._generation,
            "direction": self._direction,

            "param_names": list(
                self._param_names
            ),
            "param_specs": dict(
                self._param_specs
            ),

            "archive_size": self.archive_size,
            "ants": self.ants,

            "archive": [
                {
                    "x": list(row["x"]),
                    "candidate": dict(row["candidate"]),
                    "score": row["score"],
                }
                for row in self._archive
            ],

            "best_score": self._best_score,
            "best_candidate": (
                None
                if self._best_candidate is None
                else dict(self._best_candidate)
            ),

            # A valid checkpoint is never mid-generation.
            "awaiting_tell": False,

            # Critical for deterministic continuation.
            "rng_state": self._rng.getstate(),

            "diagnostics": {
                "n_status_failed": self._n_status_failed,
                "n_missing_score": self._n_missing_score,
                "n_non_numeric": self._n_non_numeric,
                "n_non_finite": self._n_non_finite,
                "n_invalid_candidates": self._n_invalid_candidates,
            },
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
    ) -> None:
        """Restore a state previously returned by state_dict()."""

        if not isinstance(state, dict):
            raise TypeError(
                "ACOR checkpoint state must be a dictionary"
            )

        if state.get("schema_version") != 1:
            raise ValueError(
                "Unsupported ACOR checkpoint schema_version: "
                f"{state.get('schema_version')!r}"
            )

        if state.get("optimizer") != "acor":
            raise ValueError(
                "Checkpoint optimizer mismatch: expected 'acor', got "
                f"{state.get('optimizer')!r}"
            )

        # --------------------------------------------------------
        # Configuration compatibility
        # --------------------------------------------------------

        configuration = state.get(
            "configuration"
        )

        if not isinstance(configuration, dict):
            raise ValueError(
                "ACOR checkpoint is missing configuration"
            )

        checkpoint_batch_size = configuration.get(
            "batch_size"
        )

        if (
            isinstance(checkpoint_batch_size, bool)
            or not isinstance(checkpoint_batch_size, int)
            or checkpoint_batch_size < 2
        ):
            raise ValueError(
                "ACOR checkpoint batch_size must be an integer >= 2"
            )

        # ACOR can be constructed without batch_size because normal GOW
        # execution historically fixes it on the first ask(). During resume,
        # no new ask() must occur before restoration, so None may safely adopt
        # the persisted value.
        if (
            self.batch_size is not None
            and self.batch_size != checkpoint_batch_size
        ):
            raise ValueError(
                "ACOR checkpoint configuration mismatch for batch_size: "
                f"checkpoint={checkpoint_batch_size!r}, "
                f"current={self.batch_size!r}"
            )

        expected_configuration = {
            "q": self.q,
            "xi": self.xi,
            "max_generations": self.max_generations,
            "min_sigma": self.min_sigma,
            "bound_strategy": self.bound_strategy,
        }

        for key, current_value in expected_configuration.items():

            if key not in configuration:
                raise ValueError(
                    "ACOR checkpoint configuration "
                    f"is missing {key!r}"
                )

            checkpoint_value = configuration[
                key
            ]

            if checkpoint_value != current_value:
                raise ValueError(
                    "ACOR checkpoint configuration mismatch "
                    f"for {key}: "
                    f"checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"
                )

        # --------------------------------------------------------
        # General state
        # --------------------------------------------------------

        initialized = state.get(
            "initialized"
        )

        if initialized is not True:
            raise ValueError(
                "ACOR checkpoint must contain initialized=True"
            )

        generation = state.get(
            "generation"
        )

        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise ValueError(
                "ACOR checkpoint generation must be an integer >= 1"
            )

        direction = state.get(
            "direction"
        )

        if direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "ACOR checkpoint direction must be "
                "'minimize' or 'maximize'"
            )

        if state.get("awaiting_tell") is not False:
            raise ValueError(
                "ACOR checkpoint must represent a generation boundary"
            )

        # --------------------------------------------------------
        # Parameter metadata
        # --------------------------------------------------------

        param_names = state.get(
            "param_names"
        )

        if (
            not isinstance(param_names, list)
            or not param_names
        ):
            raise ValueError(
                "ACOR checkpoint param_names must be a non-empty list"
            )

        if (
            not all(
                isinstance(name, str)
                and bool(name)
                for name in param_names
            )
            or len(set(param_names)) != len(param_names)
        ):
            raise ValueError(
                "ACOR checkpoint contains invalid parameter names"
            )

        param_specs_raw = state.get(
            "param_specs"
        )

        if not isinstance(param_specs_raw, dict):
            raise ValueError(
                "ACOR checkpoint param_specs must be a dictionary"
            )

        param_specs: Dict[
            str,
            Tuple[
                str,
                Tuple[float, float],
            ],
        ] = {}

        for name in param_names:

            if name not in param_specs_raw:
                raise ValueError(
                    "ACOR checkpoint is missing parameter "
                    f"specification for {name!r}"
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
                    "Invalid ACOR parameter specification "
                    f"for {name!r}: {spec!r}"
                )

            kind = str(spec[0])

            try:
                lo = float(spec[1][0])
                hi = float(spec[1][1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid ACOR parameter bounds "
                    f"for {name!r}"
                ) from exc

            if (
                not math.isfinite(lo)
                or not math.isfinite(hi)
            ):
                raise ValueError(
                    "ACOR checkpoint parameter bounds "
                    "must be finite"
                )

            if kind == "real" and not lo < hi:
                raise ValueError(
                    f"Invalid real bounds for {name!r}"
                )

            if kind == "int" and lo > hi:
                raise ValueError(
                    f"Invalid integer bounds for {name!r}"
                )

            param_specs[name] = (
                kind,
                (lo, hi),
            )

        if set(param_specs_raw) != set(param_names):
            raise ValueError(
                "ACOR checkpoint param_specs keys do not match param_names"
            )

        # --------------------------------------------------------
        # Archive structure
        # --------------------------------------------------------

        archive_size = state.get(
            "archive_size"
        )

        ants = state.get(
            "ants"
        )

        if (
            archive_size != checkpoint_batch_size
            or ants != checkpoint_batch_size
        ):
            raise ValueError(
                "ACOR checkpoint archive_size/ants must equal batch_size"
            )

        archive_raw = state.get(
            "archive"
        )

        if not isinstance(archive_raw, list):
            raise ValueError(
                "ACOR checkpoint archive must be a list"
            )

        if len(archive_raw) > checkpoint_batch_size:
            raise ValueError(
                "ACOR checkpoint archive exceeds archive_size"
            )

        archive: List[
            Dict[str, Any]
        ] = []

        previous_score: float | None = None

        for row in archive_raw:

            if not isinstance(row, dict):
                raise ValueError(
                    "ACOR checkpoint archive entries must be dictionaries"
                )

            x_raw = row.get(
                "x"
            )

            if (
                not isinstance(x_raw, list)
                or len(x_raw) != len(param_names)
            ):
                raise ValueError(
                    "ACOR checkpoint archive vector has invalid dimension"
                )

            x: List[float] = []

            for value in x_raw:
                try:
                    x_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "ACOR checkpoint archive vector "
                        "contains a non-numeric value"
                    ) from exc

                if (
                    not math.isfinite(x_value)
                    or x_value < 0.0
                    or x_value > 1.0
                ):
                    raise ValueError(
                        "ACOR checkpoint normalized archive values "
                        "must be finite and inside [0, 1]"
                    )

                x.append(
                    x_value
                )

            candidate = row.get(
                "candidate"
            )

            if not isinstance(candidate, dict):
                raise ValueError(
                    "ACOR checkpoint archive candidate "
                    "must be a dictionary"
                )

            if not all(
                name in candidate
                for name in param_names
            ):
                raise ValueError(
                    "ACOR checkpoint archive candidate "
                    "is missing optimizable parameters"
                )

            score_raw = row.get(
                "score"
            )

            try:
                score = float(
                    score_raw
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ACOR checkpoint archive score "
                    "must be numeric"
                ) from exc

            if not math.isfinite(score):
                raise ValueError(
                    "ACOR checkpoint archive score "
                    "must be finite"
                )

            if (
                previous_score is not None
                and score > previous_score
            ):
                raise ValueError(
                    "ACOR checkpoint archive must be sorted "
                    "from best to worst"
                )

            previous_score = score

            archive.append(
                {
                    "x": x,
                    "candidate": dict(candidate),
                    "score": score,
                }
            )

        # --------------------------------------------------------
        # Best-so-far
        # --------------------------------------------------------

        best_score_raw = state.get(
            "best_score"
        )

        if best_score_raw is None:
            best_score = None
        else:
            try:
                best_score = float(
                    best_score_raw
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ACOR checkpoint best_score must be numeric or None"
                ) from exc

            if not math.isfinite(
                best_score
            ):
                raise ValueError(
                    "ACOR checkpoint best_score must be finite"
                )

        best_candidate_raw = state.get(
            "best_candidate"
        )

        if best_candidate_raw is None:
            best_candidate = None
        elif isinstance(
            best_candidate_raw,
            dict,
        ):
            best_candidate = dict(
                best_candidate_raw
            )
        else:
            raise ValueError(
                "ACOR checkpoint best_candidate "
                "must be a dictionary or None"
            )

        if (
            (best_score is None)
            != (best_candidate is None)
        ):
            raise ValueError(
                "ACOR checkpoint best_score and best_candidate "
                "must either both exist or both be None"
            )

        if archive and best_score is None:
            raise ValueError(
                "ACOR checkpoint with a non-empty archive "
                "requires best_score"
            )

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        diagnostics = state.get(
            "diagnostics"
        )

        if not isinstance(
            diagnostics,
            dict,
        ):
            raise ValueError(
                "ACOR checkpoint diagnostics must be a dictionary"
            )

        diagnostic_names = (
            "n_status_failed",
            "n_missing_score",
            "n_non_numeric",
            "n_non_finite",
            "n_invalid_candidates",
        )

        diagnostic_values: Dict[
            str,
            int,
        ] = {}

        for name in diagnostic_names:

            value = diagnostics.get(
                name
            )

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "ACOR checkpoint diagnostic "
                    f"{name!r} must be a non-negative integer"
                )

            diagnostic_values[
                name
            ] = value

        # --------------------------------------------------------
        # RNG state
        # --------------------------------------------------------

        rng_state = state.get(
            "rng_state"
        )

        probe_rng = random.Random()

        try:
            probe_rng.setstate(
                rng_state
            )
        except Exception as exc:
            raise ValueError(
                "ACOR checkpoint contains invalid RNG state"
            ) from exc

        # --------------------------------------------------------
        # Commit validated state
        # --------------------------------------------------------

        self.batch_size = checkpoint_batch_size
        self.archive_size = checkpoint_batch_size
        self.ants = checkpoint_batch_size

        self._initialized = True
        self._generation = generation
        self._direction = direction

        self._param_names = list(
            param_names
        )
        self._param_specs = param_specs

        self._archive = archive

        self._best_score = best_score
        self._best_candidate = best_candidate

        self._awaiting_tell = False

        self._n_status_failed = diagnostic_values[
            "n_status_failed"
        ]
        self._n_missing_score = diagnostic_values[
            "n_missing_score"
        ]
        self._n_non_numeric = diagnostic_values[
            "n_non_numeric"
        ]
        self._n_non_finite = diagnostic_values[
            "n_non_finite"
        ]
        self._n_invalid_candidates = diagnostic_values[
            "n_invalid_candidates"
        ]

        self._rng.setstate(
            rng_state
        )

    def is_done(self) -> bool:
        """
        Tell GOW whether ACOR has reached its generation limit.

        If max_generations is None, ACOR does not stop itself by generation
        count. In that case, the external GOW stopping condition controls the
        run.
        """

        if self.max_generations is None:
            return False
        return self._generation >= self.max_generations

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return useful internal information about the current ACOR state.

        Diagnostics are not used to drive the algorithm. They are useful for
        logs, debugging, and checking whether the evaluator is returning valid
        results.
        """

        return {
            "generation": self._generation,
            "batch_size": self.batch_size,
            "ants": self.ants,
            "archive_size": self.archive_size,
            "archive_size_current": len(self._archive),
            "q": self.q,
            "xi": self.xi,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
            "n_invalid_candidates": self._n_invalid_candidates,
        }

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        """
        Read the optimizable parameter information from the GOW problem.

        ACOR needs to know:

          - which parameters are optimizable;
          - whether each parameter is real or integer;
          - the lower and upper bound of each parameter;
          - whether the objective is minimized or maximized.

        The YAML parameter values are not used to inject an initial candidate.
        They may exist in the problem configuration, but ACOR does not force that
        exact point into the first generation.
        """

        # Read whether the external objective should be minimized or maximized.
        self._direction = self._get_direction(problem)

        # Get only the parameters marked as optimizable in the GOW problem.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for ACOR.")

        # Reset parameter containers before filling them.
        self._param_names = []
        self._param_specs = {}

        # Inspect each optimizable parameter.
        for name, p in params.items():
            if isinstance(p, RealParam):
                # Real parameters must have numeric bounds [lo, hi].
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")

                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not (lo < hi):
                    raise ValueError(f"Real param '{name}' must have lo < hi (got {lo}, {hi})")

                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            elif isinstance(p, IntParam):
                # Integer parameters are handled by sampling a continuous value
                # and then rounding it back to an integer candidate value.
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")

                lo_i, hi_i = int(p.bounds[0]), int(p.bounds[1])
                if lo_i > hi_i:
                    raise ValueError(f"Int param '{name}' must have lo <= hi (got {lo_i}, {hi_i})")

                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo_i), float(hi_i)))

            elif isinstance(p, CategoricalParam):
                # ACOR samples continuous Gaussian values. A categorical value
                # has no natural distance for that Gaussian sampling step, so it
                # is not supported here.
                raise ValueError(
                    f"ACOR does not support categorical param '{name}'. "
                    "Use numeric encoding or another optimizer for categorical variables."
                )

            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

        # Mark the optimizer as ready to generate candidates.
        self._initialized = True

    def _initial_candidates(self, n: int) -> List[Dict[str, Any]]:
        """
        Create the first generation of candidates.

        At the beginning, ACOR has no archive because no candidate has been
        evaluated yet. Therefore, the first generation is sampled randomly across
        the normalized search space.
        """

        # Create n random normalized vectors and convert each one to a GOW
        # candidate dictionary.
        return [self._normalized_to_candidate(self._random_x()) for _ in range(n)]

    def _random_x(self) -> List[float]:
        """
        Create one random normalized vector.

        Each element is between 0 and 1. The vector length is equal to the
        number of optimizable parameters.
        """

        return [self._rng.random() for _ in self._param_names]

    def _archive_weights(self) -> List[float]:
        """
        Compute the probability weight of each archive entry.

        The archive is already sorted from best to worst. The entry at rank 0 is
        the best entry. ACOR uses q to decide how concentrated the selection
        probability should be around the best archive entries.
        """

        k = len(self._archive)
        if k <= 0:
            raise RuntimeError("Cannot compute ACOR weights with an empty archive.")

        # q controls the width of the rank-based weighting curve.
        # A smaller denom makes the best ranks much more likely to be chosen.
        denom = self.q * k

        # Create one weight per archive rank.
        weights = [
            math.exp(-((rank ** 2) / (2.0 * denom * denom)))
            / (denom * math.sqrt(2.0 * math.pi))
            for rank in range(k)
        ]

        # Normalize the weights so they sum to 1.
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return [1.0 / k for _ in range(k)]

        return [w / total for w in weights]

    def _choose_archive_index(self) -> int:
        """
        Randomly choose one archive entry using the archive weights.

        The returned index points to the archive entry that will be used as the
        center for sampling a new candidate.
        """

        weights = self._archive_weights()
        r = self._rng.random()
        acc = 0.0

        # Accumulate probabilities until the random number falls inside one
        # interval. This is a simple weighted random choice.
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i

        # Floating-point rounding may leave a tiny probability gap at the end.
        # Returning the last index is a safe fallback.
        return len(weights) - 1

    def _sample_candidate_from_archive(self) -> Dict[str, Any]:
        """
        Sample one new candidate around the current archive.

        This is the main ACOR sampling step. It chooses one archive entry as the
        center and then samples each dimension around that center.
        """

        # Choose the archive entry that will act as the center of the Gaussian
        # sampling process.
        idx = self._choose_archive_index()
        center = self._archive[idx]["x"]
        k = len(self._archive)

        x_new: List[float] = []

        # Sample each parameter independently in normalized space.
        for dim, mu in enumerate(center):
            if k > 1:
                # Estimate the spread of this dimension by measuring the
                # average distance between the selected archive value and the
                # values stored in the rest of the archive.
                avg_distance = sum(
                    abs(float(row["x"][dim]) - float(mu))
                    for row in self._archive
                ) / float(k - 1)
                sigma = self.xi * avg_distance
            else:
                # This fallback is rarely used because batch_size must be >= 2,
                # but it keeps the method safe if the archive ever has one
                # valid entry due to invalid evaluator results.
                sigma = 0.5 * self.xi

            # Prevent sigma from becoming exactly zero.
            sigma = max(float(sigma), self.min_sigma)

            # Draw one bounded Gaussian sample for this dimension.
            x_new.append(self._sample_bounded_gaussian(float(mu), sigma))

        # Convert the normalized vector back to a GOW candidate dictionary.
        return self._normalized_to_candidate(x_new)

    def _sample_bounded_gaussian(self, mu: float, sigma: float) -> float:
        """
        Sample one normalized value from a Gaussian distribution.

        The valid normalized range is [0, 1]. If the raw Gaussian sample falls
        outside this range, the selected bound_strategy decides how to repair it.
        """

        if self.bound_strategy == "resample":
            # Try several times to obtain a value inside [0, 1].
            for _ in range(32):
                x = self._rng.gauss(mu, sigma)
                if 0.0 <= x <= 1.0:
                    return x

            # If repeated resampling fails, clip as a safe fallback.
            return self._clip(self._rng.gauss(mu, sigma), 0.0, 1.0)

        # Default strategy: sample once and clip to the valid range.
        return self._clip(self._rng.gauss(mu, sigma), 0.0, 1.0)

    def _normalized_to_candidate(self, x: List[float]) -> Dict[str, Any]:
        """
        Convert a normalized ACOR vector into a GOW candidate dictionary.

        ACOR stores and samples values in [0, 1]. GOW and the external evaluator
        need the real parameter values defined by the YAML bounds.
        """

        cand: Dict[str, Any] = {}

        # Convert each normalized value back to its parameter range.
        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            # Make sure the normalized value is valid.
            value = self._clip(float(value), 0.0, 1.0)

            # Linear conversion from normalized space to real space:
            #
            #   0.0 -> lower bound
            #   1.0 -> upper bound
            #
            # Any value between 0 and 1 maps proportionally inside the bounds.
            real_value = lo + value * (hi - lo)

            # Remove tiny numerical errors close to the boundaries.
            if abs(real_value - lo) < 1e-15:
                real_value = lo
            if abs(real_value - hi) < 1e-15:
                real_value = hi

            # Integer parameters are rounded after the continuous conversion.
            if kind == "int":
                real_value = int(round(real_value))
                real_value = int(self._clip(float(real_value), lo, hi))

            cand[name] = real_value

        return cand

    def _candidate_to_normalized(self, candidate: Mapping[str, Any]) -> List[float]:
        """
        Convert a GOW candidate dictionary back into normalized ACOR space.

        This is used in tell() so that evaluated candidates can be stored in the
        archive using the same internal representation used by ACOR sampling.
        """

        x: List[float] = []

        for name in self._param_names:
            if name not in candidate:
                raise KeyError(f"Candidate missing parameter '{name}'")

            kind, (lo, hi) = self._param_specs[name]
            val = float(candidate[name])

            # Integer candidates are rounded before normalization to keep their
            # archive representation consistent with the value that was really
            # evaluated.
            if kind == "int":
                val = float(int(round(val)))

            # Keep the value inside the configured bounds before normalizing.
            val = self._clip(val, lo, hi)
            x.append((val - lo) / (hi - lo))

        return x

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        """
        Convert an evaluator result into the internal ACOR score.

        ACOR always compares candidates as a maximization problem internally:

          - larger internal score is better;
          - invalid results become -inf;
          - minimization objectives are multiplied by -1.
        """

        # If the evaluator reports a non-ok status, the result is ignored.
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        key: str | None = None
        val: Any = None

        # Look for a common top-level score key.
        for k in ("objective", "score", "loss", "fitness"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        # Some evaluators return nested score dictionaries. If the selected
        # value is a dictionary, look inside it for the real numeric value.
        if isinstance(val, Mapping):
            nested = val
            key = None
            val = None
            for k in ("objective", "score", "loss", "fitness"):
                if k in nested:
                    key = k
                    val = nested[k]
                    break

        # Other evaluators place values inside a metrics dictionary.
        if val is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("objective", "score", "loss", "fitness"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        # If no usable score was found, ignore this result.
        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        if isinstance(val, str) and not val.strip():
            self._n_missing_score += 1
            return float("-inf")

        # Convert the value to a float so mathematical comparisons are possible.
        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        # NaN or infinity cannot be used as a valid score.
        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        # A value named loss is normally minimized, so invert its sign before
        # applying the objective direction.
        if key == "loss":
            x = -x

        # If the GOW objective is minimization, invert the objective so that
        # ACOR can still use "higher internal score is better".
        if self._direction == "minimize":
            x = -x

        return x

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        """
        Keep x inside the inclusive interval [lo, hi].

        This small helper avoids repeating the same boundary logic throughout
        the file.
        """

        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _get_direction(problem: ProblemConfig) -> str:
        """
        Read the objective direction from the GOW problem configuration.

        If no direction is found, maximize is used as the default. Only
        "minimize" and "maximize" are accepted.
        """

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
