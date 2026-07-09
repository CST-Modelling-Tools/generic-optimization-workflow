from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Tuple

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class ParticleSwarmOptimizer(Optimizer):
    """
    Particle Swarm Optimization (PSO) with inertia weight.

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    PSO is a population-based optimization algorithm. This means that it does
    not work with a single candidate solution, but with a group of candidate
    solutions at the same time.

    In PSO:

      - The population is called the swarm.
      - Each individual in the swarm is called a particle.
      - Each particle represents one possible solution to the problem.
      - Each particle moves through the search space.
      - The movement is controlled by a velocity.
      - Each particle remembers the best position it has found so far.
      - The swarm remembers the best position found by any particle.

    Common PSO names:

      - position / x:
          current position of a particle.

      - velocity / v:
          displacement that indicates how the particle will move.

      - pbest:
          personal best position found by one particle.

      - gbest:
          global best position found by the whole swarm.

    This implementation uses the global-best PSO variant. That means all
    particles share the same global reference, gbest.

    -------------------------------------------------------------------------
    MOVEMENT RULE
    -------------------------------------------------------------------------

    At each generation, every particle updates its velocity and then its
    position. For each optimizable parameter, the rule is:

        v = w*v + c*r1*(pbest - x) + c*r2*(gbest - x)
        x = x + v

    where:

      - x is the current position.
      - v is the current velocity.
      - w is inertia_weight.
      - c is acceleration_coefficient.
      - r1 is a random number between 0 and 1.
      - r2 is another random number between 0 and 1.
      - pbest is the particle's personal best position.
      - gbest is the swarm's global best position.

    The velocity has three parts:

      1. Inertia:
           w*v

         Preserves part of the previous movement.

      2. Personal attraction:
           c*r1*(pbest - x)

         Pulls the particle toward the best position it found by itself.

      3. Social attraction:
           c*r2*(gbest - x)

         Pulls the particle toward the best position found by the swarm.

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores the algorithm hyperparameters and prepares the internal
           variables where the swarm state will be stored.

      2. ask(problem, n)
           GOW calls this function to request new candidates.
           On the first call, ask() initializes the swarm.
           Then it returns a list of candidates to evaluate.

      3. GOW evaluates the candidates outside this file.
           The optimizer does not compute the objective function directly.
           It only proposes candidates. The external evaluator computes their
           quality.

      4. tell(candidates, fitness)
           GOW calls this function to return the evaluation results to the
           optimizer. Those results are used to update pbest and gbest.

      5. ask(problem, n) is called again.
           At this point, the swarm already has pbest and gbest information,
           so the particles can move toward new positions.

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
      - In PSO, that same value becomes the swarm size.
      - Each particle produces one candidate per generation.
      - Therefore, batch_size and swarm_size are the same quantity internally.
      - PSO compares candidates internally using the rule: higher score is better.
      - If the real objective is minimization, the sign is inverted internally.
      - Real and integer optimizable parameters are supported.
      - Optimizable categorical parameters are not supported.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        max_generations: int = 50,
        inertia_weight: float = 0.729,
        acceleration_coefficient: float = 1.49445,
        velocity_clamp_fraction: float = 0.2,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Store the initial PSO configuration.

        This function runs only once, when the optimizer object is created.
        It does not create the swarm yet. It only stores the hyperparameters and
        prepares the internal variables that will be used during optimization.

        PSO-specific hyperparameters:

        batch_size:
            Number of candidates requested by GOW in each generation.

            In PSO, each candidate corresponds to one particle. Therefore,
            batch_size is interpreted internally as the swarm size.

            Example:
                batch_size = 100

            means:
                100 candidates evaluated per generation
                100 particles in the swarm

            In normal GOW execution, this value is read from the general
            optimizer.batch_size field in the YAML and reaches PSO as n in
            ask(problem, n). If batch_size is passed directly to this
            constructor, the same relationship is used.

        max_generations:
            Maximum number of generations. One generation is one complete
            evaluation round of the whole swarm.

        inertia_weight:
            Inertia weight. Controls how much of the previous velocity a
            particle keeps.

            Higher value:
                the particle tends to keep exploring.

            Lower value:
                the particle changes direction more easily and performs a more
                local search.

        acceleration_coefficient:
            Acceleration coefficient used for both attractions:

              - attraction toward pbest;
              - attraction toward gbest.

            A single coefficient is used to keep the particle's individual
            memory and the swarm's collective information balanced.

        velocity_clamp_fraction:
            Fraction of each parameter range used to compute the maximum
            allowed velocity.

            Example:
                bounds = [0, 10]
                velocity_clamp_fraction = 0.2

            then:
                range = 10 - 0 = 10
                maximum velocity = 0.2 * 10 = 2

            The velocity in that dimension is limited to [-2, 2].

        seed:
            Optional random seed. It makes the same run reproducible by
            generating the same sequence of random movements.
        """

        # ------------------------------------------------------------------
        # Basic hyperparameter validation
        # ------------------------------------------------------------------
        # These checks prevent impossible configurations before the algorithm
        # starts running.
        #
        # For example, it would not make sense to have zero particles or zero
        # generations.
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if inertia_weight < 0.0:
            raise ValueError("inertia_weight must be >= 0")
        if acceleration_coefficient < 0.0:
            raise ValueError("acceleration_coefficient must be >= 0")
        if velocity_clamp_fraction <= 0.0:
            raise ValueError("velocity_clamp_fraction must be > 0")

        # Number of candidates evaluated per generation.
        # In GOW, this value is configured as optimizer.batch_size in the YAML.
        #
        # Depending on how the optimizer is created, batch_size may or may not
        # be passed directly to this constructor.
        #
        # If it is passed here, PSO stores it immediately.
        # If it is not passed here, PSO will infer it from n on the first
        # ask(problem, n) call.
        self.batch_size = int(batch_size) if batch_size is not None else None

        # In PSO, the batch size is also the swarm size because each particle
        # produces exactly one candidate per generation.
        #
        # This value may be None during __init__ and will be fixed on the first
        # ask(problem, n) call.
        self.swarm_size = self.batch_size

        # Hyperparameters that control swarm movement.
        self.max_generations = int(max_generations)
        self.inertia_weight = float(inertia_weight)
        self.acceleration_coefficient = float(acceleration_coefficient)
        self.velocity_clamp_fraction = float(velocity_clamp_fraction)

        # Optimizer-specific random generator.
        # Using self._rng instead of Python's global random generator makes the
        # run reproducible when a seed is provided.
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the swarm has already been created.
        # At the beginning this is False because ask() has not been called yet.
        self._initialized = False

        # Number of completed generations.
        # This is incremented in tell(), after evaluation results are received.
        self._generation = 0

        # Objective direction: "minimize" or "maximize".
        # It is read from problem in _initialize().
        self._direction = "minimize"

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Names of the parameters that PSO will move.
        #
        # Example:
        #   ["p0", "p1", "p2"]
        self._param_names: List[str] = []

        # Specifications of each optimizable parameter.
        #
        # The key is the parameter name.
        # The value is a tuple with:
        #   (kind, lower_bound, upper_bound)
        #
        # Example:
        #   self._param_specs["p0"] = ("real", 0.0, 10.0)
        self._param_specs: Dict[str, Tuple[str, float, float]] = {}

        # ------------------------------------------------------------------
        # Current swarm state
        # ------------------------------------------------------------------

        # Current positions of all particles.
        #
        # This is a list because there are many particles.
        # Each item in the list is a dictionary.
        #
        # Example:
        #   self._positions[0] = {"p0": 2.1, "p1": 0.7}
        #
        # means particle 0 is currently at that position.
        self._positions: List[Dict[str, float]] = []

        # Current velocities of all particles.
        # It has the same structure as _positions.
        #
        # Example:
        #   self._velocities[0] = {"p0": 0.3, "p1": -0.1}
        #
        # means particle 0 will move +0.3 in p0 and -0.1 in p1.
        self._velocities: List[Dict[str, float]] = []

        # Personal best position of each particle.
        #
        # self._pbest_positions[0] stores the historical best position of
        # particle 0.
        self._pbest_positions: List[Dict[str, float]] = []

        # Personal best score of each particle.
        #
        # self._pbest_scores[0] stores the score of the historical best position
        # of particle 0.
        #
        # At the beginning it is None because there are no evaluations yet.
        self._pbest_scores: List[float | None] = []

        # Best position found by the whole swarm.
        # At the beginning this is None because no particle has been evaluated.
        self._gbest_position: Dict[str, float] | None = None

        # Score associated with gbest.
        self._gbest_score: float | None = None

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
          2. If the swarm size has not been fixed yet, set swarm_size = n.
          3. Check that n matches the fixed swarm size.
          4. If the swarm does not exist yet, create it with _initialize().
          5. If previous evaluations already exist, move the swarm with
             _move_swarm().
          6. Convert internal positions into GOW candidates.
          7. Return the candidate list.
        """

        # GOW passes the number of candidates requested for this generation as n.
        #
        # For PSO, n defines the swarm size because each particle produces
        # exactly one candidate per generation.
        #
        # If batch_size was not passed directly to __init__, the first ask()
        # call fixes the swarm size from n.
        if self.swarm_size is None:
            if n < 1:
                raise ValueError("batch_size must be >= 1")
            self.batch_size = int(n)
            self.swarm_size = int(n)

        # Once the swarm size is fixed, it must remain constant during the run.
        # PSO cannot change the number of particles in the middle of an
        # optimization.
        if n != self.swarm_size:
            raise ValueError(
                "ParticleSwarmOptimizer requires ask(..., n=batch_size). "
                f"Got n={n}, batch_size={self.batch_size}, swarm_size={self.swarm_size}."
            )

        # First call to ask(): particles do not exist yet.
        # Initial positions and velocities are created here.
        if not self._initialized:
            self._initialize(problem)

        # From the second generation onward, pbest and gbest have already been
        # computed from tell(). The swarm can therefore move before proposing
        # new candidates.
        if self._generation > 0:
            self._move_swarm()

        # self._positions stores internal positions as floats.
        # _candidate_from_position() converts them to the format expected by GOW.
        #
        # This line returns one candidate for each particle.
        return [self._candidate_from_position(pos) for pos in self._positions]

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """
        Receive evaluation results and update the swarm memory.

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

        In PSO, tell() is where the following memories are updated:

          - each particle's pbest;
          - the swarm's gbest.
        """

        # Reset diagnostic counters for this generation.
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        # tell() only makes sense after ask(), because ask() creates or moves
        # the swarm and proposes the candidates.
        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); PSO is not initialized.")

        # Every candidate must have exactly one fitness result.
        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: {len(candidates)} != {len(fitness)}"
            )

        # PSO expects one result per particle.
        if len(candidates) != self.swarm_size:
            raise ValueError(
                "ParticleSwarmOptimizer expects exactly swarm_size candidates per tell(): "
                f"got {len(candidates)}, expected {self.swarm_size}"
            )

        # Convert each evaluator result into an internal score.
        # The internal rule is always: higher is better.
        scores = [self._normalize_score(fdict) for fdict in fitness]

        # Loop over each particle and its score.
        # enumerate(scores) gives two values:
        #   i     -> particle index: 0, 1, 2, ...
        #   score -> internal score of that particle
        for i, score in enumerate(scores):
            # A score of -inf means the evaluation was not valid.
            # In that case, this particle does not update pbest or gbest.
            if score == float("-inf"):
                continue

            # Recover the position of the evaluated candidate.
            # This ensures pbest/gbest memory is stored in PSO's internal format.
            pos = self._position_from_candidate(candidates[i])

            # --------------------------------------------------------------
            # pbest update
            # --------------------------------------------------------------
            # old_score is the historical best score of this particle.
            old_score = self._pbest_scores[i]

            # If the particle never had a valid score, old_score is None.
            # If the new score is better than the previous one, update pbest.
            if old_score is None or score > old_score:
                self._pbest_positions[i] = dict(pos)
                self._pbest_scores[i] = score

            # --------------------------------------------------------------
            # gbest update
            # --------------------------------------------------------------
            # gbest stores the best solution found by any particle.
            if self._gbest_score is None or score > self._gbest_score:
                self._gbest_position = dict(pos)
                self._gbest_score = score

        # When tell() finishes, one generation is considered complete.
        self._generation += 1

    def is_done(self) -> bool:
        """
        Return whether the optimizer should stop.

        PSO stops when the number of completed generations reaches
        max_generations.
        """
        return self._generation >= self.max_generations

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return useful information about the current optimizer state.

        This function does not move the swarm and does not change the result.
        It only reports information.

        Main fields:

          - generation:
              current generation.

          - best_objective:
              best objective value found, expressed in the original problem
              scale.

          - best_internal_score:
              best internal score used by PSO. Internally, higher is better.

          - status_failed, missing_score, non_numeric, non_finite:
              counters for issues detected in evaluator results.
        """

        # If gbest does not exist yet, there is no valid best solution.
        if self._gbest_score is None:
            return {
                "generation": self._generation,
                "best_objective": None,
                "status_failed": self._n_status_failed,
                "missing_score": self._n_missing_score,
                "non_numeric": self._n_non_numeric,
                "non_finite": self._n_non_finite,
            }

        # For minimization problems, the objective was stored internally with
        # inverted sign. Here it is converted back to the original sign.
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
        """
        Create the initial swarm.

        This function is called automatically from ask() the first time GOW
        requests candidates.

        Initialization steps:

          1. Read the objective direction: minimize or maximize.
          2. Read the optimizable parameters of the problem.
          3. Validate that each parameter has valid bounds.
          4. Store the type and bounds of each parameter.
          5. Create swarm_size random positions.
          6. Create swarm_size random velocities.
          7. Initialize pbest as each particle's initial position.
          8. Leave pbest scores as None until fitness results arrive.

        Initialization uses only the problem bounds.
        """

        # Store whether the real problem objective is minimize or maximize.
        self._direction = self._get_direction(problem)

        # GOW provides the parameters marked as optimizable.
        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for Particle Swarm Optimization.")

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
                kind = "real"

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                kind = "int"

            # --------------------------------------------------------------
            # Optimizable categorical parameter
            # --------------------------------------------------------------
            # PSO works with positions, velocities, and distances.
            # That makes sense for real and integer numbers, but not for
            # optimizable categories such as "red", "blue", or "green".
            elif isinstance(p, CategoricalParam):
                raise TypeError(
                    f"ParticleSwarmOptimizer does not support categorical optimizable param '{name}'."
                )

            # --------------------------------------------------------------
            # Unknown parameter type
            # --------------------------------------------------------------
            else:
                raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

            # Bounds must make sense: lower bound must be smaller than upper bound.
            if not lo < hi:
                raise ValueError(f"Invalid bounds for '{name}': [{lo}, {hi}]")

            # Store the name so parameters are always traversed in the same order.
            self._param_names.append(name)

            # Store kind and bounds for initialization, movement, clamping, and
            # candidate conversion.
            self._param_specs[name] = (kind, lo, hi)

        # Initial positions and velocities will be stored here.
        self._positions = []
        self._velocities = []

        # Create one particle for each unit of swarm_size.
        # Each particle receives:
        #   - a random position inside the bounds;
        #   - a bounded random velocity.
        for _ in range(self.swarm_size):
            self._positions.append(self._random_position())
            self._velocities.append(self._random_velocity())

        # At the beginning, each particle's personal best position is its own
        # initial position. We do not yet know whether it is good or bad because
        # evaluation results arrive later through tell().
        self._pbest_positions = [dict(pos) for pos in self._positions]

        # None means: this particle does not have a valid score yet.
        self._pbest_scores = [None] * self.swarm_size

        # gbest also starts empty because nothing has been evaluated yet.
        self._gbest_position = None
        self._gbest_score = None

        # Mark the swarm as created.
        self._initialized = True

    def _move_swarm(self) -> None:
        """
        Move every particle by one generation.

        This function is the core of the PSO algorithm.

        For each particle:

          1. Take its current position.
          2. Take its current velocity.
          3. Take its pbest.
          4. Take the swarm gbest.
          5. For each parameter, compute a new velocity.
          6. Limit the velocity using velocity_clamp_fraction.
          7. Update the position with x = x + v.
          8. If the position leaves the bounds, apply clamp.

        Important:

          - The first loop traverses particles.
          - The second loop traverses dimensions/parameters.

        This is done because PSO updates position and velocity dimension by
        dimension.
        """

        # If gbest does not exist yet, no valid evaluation has selected a global
        # best position.
        #
        # In that case, there is no social direction for the swarm to follow.
        # A new random swarm is created instead.
        if self._gbest_position is None:
            self._positions = [self._random_position() for _ in range(self.swarm_size)]
            self._velocities = [self._random_velocity() for _ in range(self.swarm_size)]
            return

        # These lists will store the new swarm state.
        # We do not overwrite self._positions immediately because old positions
        # are still needed while computing the new ones.
        new_positions: List[Dict[str, float]] = []
        new_velocities: List[Dict[str, float]] = []

        # --------------------------------------------------------------
        # First loop: traverse particles
        # --------------------------------------------------------------
        for i in range(self.swarm_size):
            # Current state of particle i.
            pos = self._positions[i]
            vel = self._velocities[i]
            pbest = self._pbest_positions[i]

            # New position and velocity for this particle will be built here.
            new_pos: Dict[str, float] = {}
            new_vel: Dict[str, float] = {}

            # ----------------------------------------------------------
            # Second loop: traverse parameters/dimensions
            # ----------------------------------------------------------
            for name in self._param_names:
                # Get the type and bounds of this parameter.
                kind, lo, hi = self._param_specs[name]

                # Parameter range.
                # Example: if bounds = [2, 7], span = 5.
                span = hi - lo

                # Maximum allowed velocity in this dimension.
                vmax = self.velocity_clamp_fraction * span

                # Independent random numbers for the two attractions.
                # r1 affects the attraction toward pbest.
                # r2 affects the attraction toward gbest.
                r1 = self._rng.random()
                r2 = self._rng.random()

                # ------------------------------------------------------
                # New velocity calculation
                # ------------------------------------------------------
                # Part 1: inertia.
                #   Keeps part of the previous velocity.
                inertia_term = self.inertia_weight * vel[name]

                # Part 2: personal or cognitive component.
                #   Measures the distance between pbest and the current position.
                #   If pbest is to the right, it pulls to the right.
                #   If pbest is to the left, it pulls to the left.
                cognitive_term = self.acceleration_coefficient * r1 * (pbest[name] - pos[name])

                # Part 3: social component.
                #   Measures the distance between gbest and the current position.
                #   Pulls the particle toward the best known global solution.
                social_term = (
                    self.acceleration_coefficient
                    * r2
                    * (self._gbest_position[name] - pos[name])
                )

                # Resulting new velocity.
                v = inertia_term + cognitive_term + social_term

                # ------------------------------------------------------
                # Velocity limit
                # ------------------------------------------------------
                # Prevents a particle from making excessively large jumps in a
                # single generation.
                #
                # min(max(v, -vmax), vmax) means:
                #   - if v is smaller than -vmax, use -vmax;
                #   - if v is larger than  vmax, use  vmax;
                #   - if v is already inside the range, keep it unchanged.
                v = min(max(v, -vmax), vmax)

                # ------------------------------------------------------
                # Position update
                # ------------------------------------------------------
                # The new position is the previous position plus velocity.
                x = pos[name] + v

                # ------------------------------------------------------
                # Bound handling with clamp
                # ------------------------------------------------------
                # If x goes below the lower bound, place it at lo.
                # If x goes above the upper bound, place it at hi.
                # In both cases, reset velocity in this dimension to zero.
                if x < lo:
                    x = lo
                    v = 0.0
                elif x > hi:
                    x = hi
                    v = 0.0

                # ------------------------------------------------------
                # Integer parameter handling
                # ------------------------------------------------------
                # Internally, PSO computes continuous movements.
                # For an integer parameter, the final position is rounded.
                if kind == "int":
                    x = float(round(x))
                    x = min(max(x, lo), hi)

                # Store the new position and velocity of this dimension.
                new_pos[name] = float(x)
                new_vel[name] = float(v)

            # After all dimensions are processed, store the new particle state.
            new_positions.append(new_pos)
            new_velocities.append(new_vel)

        # Replace the old swarm state with the new state.
        self._positions = new_positions
        self._velocities = new_velocities

    def _random_position(self) -> Dict[str, float]:
        """
        Generate a random position inside the bounds.

        Returns a dictionary with one value for each optimizable parameter.

        Example:
            {"p0": 1.23, "p1": 8.0}
        """

        pos: Dict[str, float] = {}

        # Traverse each optimizable parameter.
        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]

            # Real parameter: draw a uniform real number between lo and hi.
            if kind == "real":
                pos[name] = self._rng.uniform(lo, hi)

            # Integer parameter: draw a uniform integer between lo and hi.
            else:
                pos[name] = float(self._rng.randint(int(lo), int(hi)))

        return pos

    def _random_velocity(self) -> Dict[str, float]:
        """
        Generate a random initial velocity.

        Velocity is also a dictionary with one value per parameter.

        Each velocity is drawn from:

            [-vmax, vmax]

        where:

            vmax = velocity_clamp_fraction * parameter_range
        """

        vel: Dict[str, float] = {}

        for name in self._param_names:
            _kind, lo, hi = self._param_specs[name]
            vmax = self.velocity_clamp_fraction * (hi - lo)
            vel[name] = self._rng.uniform(-vmax, vmax)

        return vel

    def _candidate_from_position(self, pos: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert an internal PSO position into a GOW candidate.

        PSO stores positions internally as floats so particles can move with
        continuous velocities.

        But GOW and the evaluator expect the correct parameter types:

          - real parameters as float;
          - integer parameters as int.

        This function performs that conversion.
        """

        cand: Dict[str, Any] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]

            # Ensure the value is inside bounds before returning it.
            x = min(max(pos[name], lo), hi)

            # If the parameter is integer, return it as int.
            if kind == "int":
                cand[name] = int(round(x))
            else:
                cand[name] = float(x)

        return cand

    def _position_from_candidate(self, cand: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert a GOW candidate into PSO's internal position format.

        This function is used in tell(), when evaluated candidates come back.

        PSO needs to store pbest and gbest as internal positions, meaning
        dictionaries of floats.
        """

        pos: Dict[str, float] = {}

        for name in self._param_names:
            kind, lo, hi = self._param_specs[name]

            # Convert the received value to float for internal use.
            x = float(cand[name])

            # Clamp inside bounds as a safety measure.
            x = min(max(x, lo), hi)

            # If the parameter was integer, round and clamp again.
            if kind == "int":
                x = float(round(x))
                x = min(max(x, lo), hi)

            pos[name] = float(x)

        return pos

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        """
        Convert the evaluator result into an internal score.

        Evaluators may return results using different keys, for example:

          - fitness
          - objective
          - score
          - loss

        PSO needs to compare all of them with one rule:

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

    def _get_direction(self, problem: ProblemConfig) -> str:
        """
        Read whether the problem is minimization or maximization.

        GOW may define this information in problem.objective.direction.

        Accepted values:

          - "minimize" or "min"
          - "maximize" or "max"

        If no clear direction is found, minimization is used by default.
        """

        # getattr reads an attribute without breaking the program if it does not
        # exist.
        objective = getattr(problem, "objective", None)
        direction = getattr(objective, "direction", None)

        if direction is None:
            return "minimize"

        # Convert to lowercase and strip spaces to accept small variations.
        direction = str(direction).lower().strip()

        if direction in {"minimize", "min"}:
            return "minimize"

        if direction in {"maximize", "max"}:
            return "maximize"

        # Safe default.
        return "minimize"
