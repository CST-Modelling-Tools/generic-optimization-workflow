from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch.quasirandom import SobolEngine

from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf

# BoTorch has two possible Expected Improvement implementations depending on
# the installed version.
#
# Preferred option:
#   qLogExpectedImprovement
#
# Fallback option:
#   qExpectedImprovement
#
# Both are acquisition functions. Their job is to decide where it is promising
# to evaluate next, using the Gaussian Process model fitted from previous data.
try:
    from botorch.acquisition.logei import qLogExpectedImprovement as AcquisitionFunction
except Exception:
    from botorch.acquisition.monte_carlo import qExpectedImprovement as AcquisitionFunction

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class SAASBOOptimizer(Optimizer):
    """
    Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO).

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    SAASBO is a Bayesian Optimization method designed for expensive black-box
    optimization problems, especially when the number of parameters is large.

    A black-box problem means that the optimizer cannot see the internal formula
    of the objective function. It can only propose candidates and wait for the
    external evaluator to return their objective values.

    Bayesian Optimization works with two main components:

      1. Surrogate model:
           A statistical model that tries to approximate the unknown objective
           function using all candidates evaluated so far.

      2. Acquisition function:
           A decision rule that uses the surrogate model to choose the next
           candidate or batch of candidates to evaluate.

    In this implementation, the surrogate model is a fully Bayesian Gaussian
    Process with a SAAS prior, provided by BoTorch.

    -------------------------------------------------------------------------
    WHAT MAKES SAASBO SPECIAL
    -------------------------------------------------------------------------

    In high-dimensional problems, not all parameters are usually equally
    important. Some parameters may strongly affect the objective, while many
    others may have little or no effect.

    SAASBO tries to learn this automatically.

    The name means:

      - Sparse:
          the model assumes that only a small subset of dimensions may be very
          important.

      - Axis-aligned:
          the important directions are assumed to correspond to original
          parameters, not arbitrary rotated combinations of parameters.

      - Subspace:
          the model tries to focus on the important part of the full search
          space.

    This does not remove parameters from the problem. The optimizer still works
    in the full parameter space. The SAAS prior simply helps the Gaussian Process
    pay more attention to the dimensions that appear to matter most.

    -------------------------------------------------------------------------
    HOW THIS IMPLEMENTATION WORKS
    -------------------------------------------------------------------------

    The optimizer has two phases:

      1. Initial design phase:
           Before there is enough information to fit a useful model, SAASBO
           generates candidates with a Sobol sequence. Sobol points are a
           structured way of spreading initial samples across the search space.

      2. Model-based phase:
           Once at least n_initial_points valid observations exist, SAASBO fits
           a SAAS Gaussian Process and optimizes an acquisition function to
           decide where to evaluate next.

    Important internal convention:

      - BoTorch acquisition functions are maximized.
      - To keep the internal logic simple, this optimizer always compares
        candidates using the rule: higher internal score is better.
      - If the real problem is a minimization problem, the objective value is
        multiplied by -1 internally.

    Example:

      Real minimization objective:
          objective = 8.7 is better than objective = 9.1

      Internal SAASBO score:
          score = -8.7 is better than score = -9.1

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores SAASBO hyperparameters and prepares the internal state.

      2. ask(problem, n)
           GOW calls this function to request n new candidates.
           If the optimizer does not have enough observations yet, it returns
           Sobol initial points. Otherwise, it fits the SAAS model and returns
           acquisition-optimized candidates.

      3. GOW evaluates the candidates outside this file.
           The optimizer does not compute the real objective function directly.
           It only proposes candidates.

      4. tell(candidates, fitness)
           GOW returns the evaluation results to the optimizer. The optimizer
           stores valid observations in its training history.

      5. ask(problem, n) is called again.
           The model is refitted using the accumulated history and a new batch
           of candidates is generated.

      6. The ask() / tell() cycle repeats until max_iterations is reached.

    -------------------------------------------------------------------------
    GOW INTEGRATION
    -------------------------------------------------------------------------

    GOW uses an ask/tell interface:

      - ask() produces candidates.
      - tell() receives evaluation results.

    In this implementation:

      - n is the number of candidates requested by GOW in one ask() call.
      - n becomes q in BoTorch's batch acquisition optimization.
      - max_iterations counts completed ask/tell rounds, not individual
        candidate evaluations.
      - Total evaluations are approximately max_iterations * batch_size, where
        batch_size is the n value passed by GOW to ask().

    Practical note:

      SAASBO is usually intended for expensive objectives and relatively small
      batches. Very large batch sizes can make acquisition optimization too
      expensive because BoTorch must optimize q candidates jointly.

    Supported optimizable parameters:

      - real parameters with bounds;
      - integer parameters with bounds.

    Not supported yet:

      - optimizable categorical parameters.

    Categorical parameters do not have a natural continuous distance, while this
    implementation relies on normalized continuous coordinates in [0, 1].
    """

    def __init__(
        self,
        *,
        n_initial_points: int = 20,
        max_iterations: int = 100,
        raw_samples: int = 256,
        num_restarts: int = 10,
        warmup_steps: int = 128,
        num_samples: int = 128,
        thinning: int = 16,
        max_tree_depth: int = 6,
        use_cuda: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Store the initial SAASBO configuration.

        This function runs once, when the optimizer object is created. It does
        not fit a model yet and it does not generate candidates yet. It only
        stores configuration values and initializes internal variables.

        SAASBO-specific hyperparameters:

        n_initial_points:
            Minimum number of valid observations required before the optimizer
            starts fitting the SAAS Gaussian Process.

            Before reaching this number, candidates are generated with Sobol
            sampling. This gives the model an initial set of observations spread
            across the search space.

            Important:
                In batched execution, ask(problem, n) must still return n
                candidates. Therefore, if n is larger than n_initial_points, the
                first Sobol batch can contain more than n_initial_points points.

        max_iterations:
            Maximum number of completed ask/tell rounds.

            One iteration means:
                ask() proposes a batch;
                GOW evaluates that batch;
                tell() stores the results.

            It is not the same as the number of individual objective
            evaluations when batch_size is greater than one.

        raw_samples:
            Number of initial random/Sobol samples used internally by BoTorch
            when searching for promising starting points for acquisition
            optimization.

            Higher value:
                more chances to find good acquisition starting points;
                slower acquisition optimization.

            Lower value:
                faster, but potentially less robust.

        num_restarts:
            Number of local optimization restarts used by optimize_acqf.

            The acquisition function can have many local optima. Multiple
            restarts improve the chance of finding a better acquisition maximum.

        warmup_steps:
            Number of NUTS warmup steps used when fitting the fully Bayesian GP.

            During warmup, the sampler adapts itself to the posterior geometry.
            These samples are not used as final posterior samples.

        num_samples:
            Number of post-warmup NUTS samples before thinning.

            These samples represent possible GP hyperparameter settings
            according to the posterior distribution.

        thinning:
            Keep only one sample every 'thinning' steps.

            Example:
                num_samples = 128
                thinning = 16

            means that only every 16th sample is retained by the fitted model.
            This reduces correlation and memory/computation cost.

        max_tree_depth:
            Maximum NUTS tree depth.

            Higher value:
                allows longer Hamiltonian trajectories;
                can improve sampling, but is slower.

            Lower value:
                faster sampling, but the sampler may explore less thoroughly.

        use_cuda:
            If True and a CUDA-capable GPU is available, model fitting and
            acquisition optimization are attempted on GPU.

            If CUDA is not available, the code automatically uses CPU.

        seed:
            Optional seed used for the Sobol initial design.
        """

        # ------------------------------------------------------------------
        # Basic hyperparameter validation
        # ------------------------------------------------------------------
        # SAASBO needs at least two initial points because a Gaussian Process
        # cannot learn useful variation from a single observation.
        if n_initial_points < 2:
            raise ValueError("n_initial_points must be >= 2 for SAASBO")

        # max_iterations must be positive because zero iterations would mean the
        # optimizer is already finished before proposing any candidate.
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        # Store all user-configurable hyperparameters as simple Python types.
        self.n_initial_points = int(n_initial_points)
        self.max_iterations = int(max_iterations)
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)

        # NUTS sampling configuration used when fitting the fully Bayesian model.
        self.warmup_steps = int(warmup_steps)
        self.num_samples = int(num_samples)
        self.thinning = int(thinning)
        self.max_tree_depth = int(max_tree_depth)

        # Device and reproducibility configuration.
        self.use_cuda = bool(use_cuda)
        self.seed = seed

        # Store any extra keyword arguments that may be passed by GOW or future
        # extensions. The current implementation does not use them directly.
        self.kwargs = kwargs

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the optimizer has already read the problem definition.
        # The problem is read lazily on the first ask() call.
        self._initialized = False

        # Number of completed ask/tell rounds.
        # This increases in tell(), after a batch has been evaluated.
        self._iteration = 0

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Names of the parameters that SAASBO will optimize.
        # Example:
        #   ["p0", "p1", "p2"]
        self._param_names: List[str] = []

        # Parameter specifications.
        #
        # The key is the parameter name.
        # The value is:
        #   (kind, (lower_bound, upper_bound))
        #
        # Example:
        #   self._param_specs["p0"] = ("real", (0.0, 10.0))
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}

        # Objective direction: "maximize" or "minimize".
        # It is read from the problem configuration in _initialize_from_problem().
        self._direction = "maximize"

        # Sobol generator used during the initial design phase.
        # It is created after the problem dimension is known.
        self._sobol: SobolEngine | None = None

        # ------------------------------------------------------------------
        # Training history used by the surrogate model
        # ------------------------------------------------------------------

        # Normalized candidate coordinates used for GP training.
        # Each inner list is a point in [0, 1]^D.
        self._train_x: List[List[float]] = []

        # Internal scores associated with _train_x.
        # The internal rule is always: higher score is better.
        self._train_y: List[float] = []

        # Normalized coordinates from the most recent ask() call.
        # tell() uses this to know exactly which points were just evaluated.
        self._last_xs: List[List[float]] = []

        # Best internal score found so far.
        self._best_score: float | None = None

        # Best candidate found so far, expressed in original GOW parameter units.
        self._best_candidate: Dict[str, Any] | None = None

        # ------------------------------------------------------------------
        # Diagnostic counters
        # ------------------------------------------------------------------
        # These counters help identify invalid or unusable evaluator results.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        """
        Generate candidates for GOW to evaluate.

        This function answers the question:

            "Which parameter sets should be evaluated now?"

        GOW calls ask() and expects a list of n candidate dictionaries.

        Example candidate:
            {"p0": 2.3, "p1": -0.001, "p2": 5}

        ask() flow:

          1. Check that GOW requested at least one candidate.
          2. If this is the first call, read the problem definition.
          3. Count the number of optimizable parameters.
          4. If there are not enough observations yet, draw Sobol points.
          5. Otherwise, fit the SAAS model and optimize the acquisition function.
          6. Convert normalized points in [0, 1]^D into real GOW candidates.
          7. Repair candidates to guarantee valid types and bounds.
          8. Store the normalized points so tell() can add them to history.
          9. Return the repaired candidates.
        """

        # GOW must request a positive number of candidates.
        if n < 1:
            raise ValueError("ask(..., n) requires n >= 1")

        # On the first ask() call, the optimizer reads:
        #   - the objective direction;
        #   - the optimizable parameter names;
        #   - the parameter types and bounds;
        #   - the search-space dimension.
        if not self._initialized:
            self._initialize_from_problem(problem)

        # Number of optimizable dimensions.
        # This is the D in the normalized domain [0, 1]^D.
        dim = len(self._param_names)

        # ------------------------------------------------------------------
        # Initial design phase
        # ------------------------------------------------------------------
        # If the optimizer has not collected enough valid observations yet, it
        # cannot fit a reliable Gaussian Process. In that case, it proposes
        # Sobol points.
        if len(self._train_x) < self.n_initial_points:
            unit_x = self._sobol.draw(n).double()

        # ------------------------------------------------------------------
        # Model-based phase
        # ------------------------------------------------------------------
        # Once enough observations exist, fit the SAAS GP and optimize the
        # acquisition function to obtain the next normalized candidates.
        else:
            unit_x = self._generate_saasbo_candidates(n=n, dim=dim)

        # Convert normalized vectors into candidates in the original parameter
        # ranges expected by GOW and the evaluator.
        raw_candidates = [
            self._unit_vector_to_candidate(x.tolist()) for x in unit_x
        ]

        # Safety repair:
        #   - clamp every value inside its bounds;
        #   - cast integer parameters back to int.
        repaired_candidates = [
            self._repair_candidate(problem, cand) for cand in raw_candidates
        ]

        # Store the normalized version of the candidates that were actually sent
        # to GOW. tell() will use exactly these points as training inputs.
        self._last_xs = [
            self._candidate_to_unit_vector(cand) for cand in repaired_candidates
        ]

        return repaired_candidates

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """
        Receive evaluation results and update the SAASBO history.

        This function answers the question:

            "How good were the candidates proposed by the last ask() call?"

        GOW evaluates candidates outside this optimizer. Then it calls tell()
        with two aligned lists:

          - candidates:
              the candidates that were evaluated;

          - fitness:
              the result dictionaries returned by the evaluator.

        Order matters:

          - candidates[0] corresponds to fitness[0]
          - candidates[1] corresponds to fitness[1]
          - etc.

        In SAASBO, tell() stores valid observations in the training dataset used
        by the Gaussian Process.
        """

        # Reset diagnostic counters for this generation/iteration.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        # tell() only makes sense after ask(), because ask() initializes the
        # problem and stores _last_xs.
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); SAASBO is not initialized.")

        # Every candidate must have exactly one returned fitness dictionary.
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        # _last_xs comes from the previous ask() call.
        # If its length does not match the current candidates, then tell() is
        # not receiving the same batch that ask() produced.
        if len(self._last_xs) != len(candidates):
            raise RuntimeError(
                "tell(): number of candidates does not match the previous ask() call."
            )

        # Convert evaluator outputs into internal scores.
        # Internal rule:
        #   higher score = better candidate.
        scores = [self._normalize_score(fdict) for fdict in fitness]

        # Store valid observations.
        # zip(...) walks through the normalized point, score, and candidate
        # together for each evaluated item in the batch.
        for x_unit, score, cand in zip(self._last_xs, scores, candidates):
            # -inf means the evaluator result was invalid or unusable.
            # Invalid points are not added to the GP training data.
            if score == float("-inf"):
                continue

            # Add one new training observation for the surrogate model.
            self._train_x.append(list(x_unit))
            self._train_y.append(float(score))

            # Update the best candidate found so far.
            if self._best_score is None or score > self._best_score:
                self._best_score = float(score)
                self._best_candidate = dict(cand)

        # One ask/tell round has now been completed.
        self._iteration += 1

        # Clear the last batch marker so the next tell() cannot accidentally be
        # applied twice to the same ask() output.
        self._last_xs = []

    def is_done(self) -> bool:
        """
        Return whether the optimizer should stop.

        SAASBO stops when the number of completed ask/tell rounds reaches
        max_iterations.
        """

        return self._iteration >= self.max_iterations

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return useful information about the current optimizer state.

        This function does not modify the optimizer. It only reports the current
        state for logging, debugging, or result inspection.

        Main fields:

          - iteration:
              number of completed ask/tell rounds.

          - n_observations:
              number of valid evaluated candidates stored in the GP dataset.

          - best_score_internal:
              best value using the internal convention: higher is better.

          - best_candidate:
              best candidate found so far in original GOW parameter units.

          - n_status_failed, n_missing_score, n_non_numeric, n_non_finite:
              counters for evaluator result issues.
        """

        return {
            "optimizer": "saasbo",
            "iteration": self._iteration,
            "n_observations": len(self._train_x),
            "n_initial_points": self.n_initial_points,
            "max_iterations": self.max_iterations,
            "raw_samples": self.raw_samples,
            "num_restarts": self.num_restarts,
            "warmup_steps": self.warmup_steps,
            "num_samples": self.num_samples,
            "thinning": self.thinning,
            "max_tree_depth": self.max_tree_depth,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
        }

    def _generate_saasbo_candidates(self, n: int, dim: int) -> torch.Tensor:
        """
        Fit the SAAS Gaussian Process and generate model-based candidates.

        This function is called only after the optimizer has collected at least
        n_initial_points valid observations.

        Steps:

          1. Select CPU or GPU.
          2. Convert the stored training data into PyTorch tensors.
          3. Build a fully Bayesian SAAS Gaussian Process.
          4. Fit the model using NUTS.
          5. Build the Expected Improvement acquisition function.
          6. Optimize the acquisition function inside [0, 1]^D.
          7. Return normalized candidate vectors.

        The returned tensor contains points in normalized space, not original
        parameter units.
        """

        # Use GPU only if requested and available. Otherwise use CPU.
        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )

        # BoTorch Gaussian Process models generally work best with double
        # precision for numerical stability.
        dtype = torch.double

        # Convert stored Python lists into tensors.
        # train_x has shape:
        #   number_of_observations x number_of_parameters
        train_x = torch.tensor(self._train_x, dtype=dtype, device=device)

        # train_y has shape:
        #   number_of_observations x 1
        train_y = torch.tensor(self._train_y, dtype=dtype, device=device).unsqueeze(-1)

        # This implementation assumes a nearly deterministic objective.
        # train_yvar is a very small observation noise variance.
        train_yvar = torch.full_like(train_y, 1e-6)

        # Build the fully Bayesian SAAS GP model.
        #
        # Important:
        #   train_x is already normalized to [0, 1]^D.
        #   train_y is internally standardized by Standardize(m=1).
        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            outcome_transform=Standardize(m=1),
        )

        # Fit the model using NUTS.
        # NUTS samples GP hyperparameters instead of estimating only one best
        # hyperparameter set. This is what makes the model fully Bayesian.
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            max_tree_depth=self.max_tree_depth,
            disable_progbar=True,
        )

        # Best observed internal score so far.
        # Expected Improvement needs this reference value to measure potential
        # improvement over the best known point.
        best_f = train_y.max()

        # Build the acquisition function.
        # The acquisition function is what tells BoTorch which points look
        # promising according to the fitted model.
        acqf = AcquisitionFunction(
            model=model,
            best_f=best_f,
        )

        # Bounds for acquisition optimization in normalized space.
        # Every parameter is represented between 0 and 1.
        bounds = torch.stack(
            [
                torch.zeros(dim, dtype=dtype, device=device),
                torch.ones(dim, dtype=dtype, device=device),
            ]
        )

        # Optimize the acquisition function.
        #
        # q=n means BoTorch will propose a batch of n candidates together.
        # raw_samples controls how many initial points are considered.
        # num_restarts controls how many local optimization starts are used.
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=n,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"maxiter": 100},
        )

        # Move candidates back to CPU and detach them from PyTorch's computation
        # graph. GOW only needs numeric values, not gradients.
        return candidates.detach().cpu()

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        """
        Read the GOW problem configuration and prepare the optimizer.

        This function is called automatically from the first ask() call.

        Initialization steps:

          1. Read the objective direction: minimize or maximize.
          2. Read all optimizable parameters.
          3. Validate that supported parameters have bounds.
          4. Store parameter names, types, and bounds.
          5. Reject optimizable categorical parameters.
          6. Create the Sobol generator using the number of dimensions.
          7. Mark the optimizer as initialized.
        """

        # Store whether the real objective must be maximized or minimized.
        self._direction = self._get_direction(problem)

        # GOW exposes only parameters marked as optimizable.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for SAASBO.")

        # Clear these structures before filling them for the current problem.
        self._param_names = []
        self._param_specs = {}

        # Loop over each optimizable parameter.
        for name, p in params.items():
            # --------------------------------------------------------------
            # Real-valued parameter
            # --------------------------------------------------------------
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")

                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not lo < hi:
                    raise ValueError(f"Real param '{name}' must have lo < hi")

                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")

                lo, hi = int(p.bounds[0]), int(p.bounds[1])
                if lo > hi:
                    raise ValueError(f"Int param '{name}' must have lo <= hi")

                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo), float(hi)))

            # --------------------------------------------------------------
            # Optimizable categorical parameter
            # --------------------------------------------------------------
            # SAASBO works in a continuous normalized domain [0, 1]^D.
            # Categories do not have a natural continuous distance, so they are
            # intentionally rejected here.
            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"SAASBO does not support optimizable categorical param '{name}' yet."
                )

            # --------------------------------------------------------------
            # Unknown parameter type
            # --------------------------------------------------------------
            else:
                raise TypeError(f"Unsupported optimizable parameter type for '{name}': {type(p)}")

        # Create the Sobol generator for the initial design.
        # The dimension must match the number of optimizable parameters.
        self._sobol = SobolEngine(
            dimension=len(self._param_names),
            scramble=True,
            seed=self.seed,
        )

        self._initialized = True

    def _unit_vector_to_candidate(self, x: List[float]) -> Dict[str, Any]:
        """
        Convert a normalized vector into a GOW candidate.

        Internally, SAASBO works in normalized coordinates:

            u in [0, 1]

        But GOW and the evaluator expect real parameter values inside the
        original parameter bounds.

        Conversion formula:

            real_value = lower_bound + u * (upper_bound - lower_bound)

        Example:

            bounds = [10, 20]
            u = 0.25

            real_value = 10 + 0.25 * (20 - 10) = 12.5
        """

        cand: Dict[str, Any] = {}

        # zip(x, self._param_names) walks through each normalized value together
        # with the parameter name it belongs to.
        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]

            # Clamp u to [0, 1] as a safety measure.
            u = min(1.0, max(0.0, float(value)))

            # Transform from normalized space to the real parameter range.
            real_value = lo + u * (hi - lo)

            # Integer parameters must be returned as int values.
            if kind == "int":
                cand[name] = int(round(real_value))
            else:
                cand[name] = float(real_value)

        return cand

    def _candidate_to_unit_vector(self, cand: Mapping[str, Any]) -> List[float]:
        """
        Convert a GOW candidate into normalized SAASBO coordinates.

        This is the inverse of _unit_vector_to_candidate().

        Conversion formula:

            u = (value - lower_bound) / (upper_bound - lower_bound)

        The result is clipped to [0, 1] for safety.
        """

        x: List[float] = []

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            val = float(cand[name])

            # Avoid division by zero for degenerate integer bounds.
            # In normal optimizable real parameters hi should be greater than lo.
            if hi == lo:
                u = 0.0
            else:
                u = (val - lo) / (hi - lo)

            # Store the clipped normalized value.
            x.append(min(1.0, max(0.0, float(u))))

        return x

    def _repair_candidate(self, problem: ProblemConfig, cand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair a candidate before returning it to GOW.

        The candidate should already be valid, but this function adds an extra
        safety layer:

          - values below the lower bound are moved to the lower bound;
          - values above the upper bound are moved to the upper bound;
          - integer parameters are rounded and returned as int.

        The problem argument is kept in the signature for compatibility and for
        possible future validation rules, even though this implementation uses
        the already stored parameter specifications.
        """

        repaired = dict(cand)

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            val = float(repaired[name])

            # Clamp inside bounds.
            val = min(hi, max(lo, val))

            # Return the correct type expected by GOW and the evaluator.
            if kind == "int":
                repaired[name] = int(round(val))
            else:
                repaired[name] = float(val)

        return repaired

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        """
        Convert an evaluator result into an internal SAASBO score.

        Evaluators may return results using different keys, for example:

          - fitness
          - objective
          - score
          - loss

        This optimizer needs one comparison rule:

            higher internal score = better candidate

        This function normalizes the evaluator output to that convention.

        Important cases:

          - If the evaluator failed, return -inf.
          - If the objective value is missing, return -inf.
          - If the value is not numeric, return -inf.
          - If the value is NaN or infinite, return -inf.
          - If the real objective is minimization, invert the sign.

        -inf means:
            worse than any valid result.
        """

        # Some evaluators return a status field.
        # If status exists and is not "ok", the result is treated as invalid.
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        # val will store the numeric value found.
        val: Any = None

        # key stores which kind of value was found.
        key: str | None = None

        # First, look for a direct value in the main result dictionary.
        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        # If no direct value exists, look inside a nested "metrics" dictionary.
        if key is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("fitness", "objective", "score", "loss"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        # No usable value was found.
        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        # Convert the value to float.
        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        # Reject NaN, +inf, and -inf.
        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        # A loss is normally better when it is smaller.
        # Invert it so that the internal rule remains: higher is better.
        if key == "loss":
            x = -x

        # For minimization problems, smaller real objective is better.
        # Invert it so that the internal rule remains: higher is better.
        if self._direction == "minimize":
            x = -x

        return x

    def _get_direction(self, problem: ProblemConfig) -> str:
        """
        Read whether the problem is maximization or minimization.

        GOW may define this information in:

            problem.objective.direction

        Accepted values in this implementation:

          - "maximize"
          - "minimize"

        If no direction is provided, this implementation currently uses
        "maximize" as the default.
        """

        # getattr reads an attribute without raising an error if it does not
        # exist. This keeps the optimizer robust to slightly different problem
        # object structures.
        direction = getattr(getattr(problem, "objective", None), "direction", None)

        # Current implementation default.
        if direction is None:
            return "maximize"

        # Normalize text to avoid issues with capitalization or extra spaces.
        direction = str(direction).lower().strip()

        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported objective direction: {direction}")

        return direction
