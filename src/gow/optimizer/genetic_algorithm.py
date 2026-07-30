from __future__ import annotations

import random
from typing import Any, Dict, List

from gow.config.models import IntParam, ProblemConfig, RealParam
from .base import Optimizer


class GeneticAlgorithmOptimizer(Optimizer):
    """
    Genetic Algorithm (GA) optimizer for GOW.

    -------------------------------------------------------------------------
    GENERAL IDEA OF THE ALGORITHM
    -------------------------------------------------------------------------

    A Genetic Algorithm is a population-based optimization algorithm. This
    means that it does not work with one candidate solution at a time. Instead,
    it works with a group of candidate solutions.

    In GA:

      - The group of candidates is called the population.
      - Each candidate in the population is called an individual.
      - Each individual represents one possible solution to the problem.
      - The quality of each individual is measured by the evaluator.
      - Better individuals are more likely to be selected as parents.
      - Parents are combined to create new individuals.
      - Some new individuals are randomly modified by mutation.
      - The population is updated generation by generation.

    Common GA names:

      - population:
          complete group of candidate solutions.

      - individual:
          one candidate solution inside the population.

      - fitness / objective:
          numerical value returned by the evaluator. In this implementation,
          lower values are treated as better values.

      - selection:
          process used to choose parents from the current population.

      - crossover:
          process used to combine two parents and create a child.

      - mutation:
          process used to randomly modify a child after crossover.

      - elitism:
          mechanism that copies the best individuals directly to the next
          generation so they are not lost.

    -------------------------------------------------------------------------
    HOW TO READ THIS FILE
    -------------------------------------------------------------------------

    Main flow:

      1. __init__()
           Stores the GA hyperparameters and prepares the internal variables
           where the population and the fitness values will be stored.

      2. ask(problem, n)
           GOW calls this function to request candidates.
           On the first call, ask() uses GOW's batch_size to define the
           internal population size and then initializes a random population.
           On later calls, ask() creates the next generation using elitism,
           tournament selection, crossover, and mutation.

      3. GOW evaluates the candidates outside this file.
           The optimizer does not compute the objective function directly.
           It only proposes candidates. The external evaluator computes their
           quality.

      4. tell(candidates, fitness)
           GOW calls this function to return the evaluation results to the
           optimizer. Those results become the fitness values used to build the
           next generation.

      5. The ask() / tell() cycle repeats until the configured number of
         generations is reached.

    -------------------------------------------------------------------------
    GOW INTEGRATION
    -------------------------------------------------------------------------

    GOW uses an ask/tell interface:

      - ask() produces candidates.
      - tell() receives results.

    In this implementation:

      - GOW's batch_size defines how many candidates are evaluated in each
        generation.
      - One generation evaluates one complete population.
      - Therefore, the internal population_size is always equal to batch_size.
      - The user should configure batch_size in the YAML.
      - population_size is kept only as an internal GA concept and as a
        backwards-compatible optional alias.
      - The optimizer exposes both self.batch_size and self.population_size
        internally, but they always represent the same number.
      - Real and integer optimizable parameters are supported.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        population_size: int | None = None,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elite_fraction: float = 0.1,
        selection: str = "tournament",
        tournament_size: int = 3,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Store the initial GA configuration.

        This function runs only once, when the optimizer object is created.
        It does not create the population yet. It only stores the
        hyperparameters and prepares the internal variables that will be used
        during optimization.

        GA-specific hyperparameters:

        batch_size:
            Number of candidates requested by GOW in each generation.

            In GA, each generation evaluates one complete population.
            Therefore, batch_size is interpreted internally as the population
            size.

            Example:
                batch_size = 100

            means:
                100 candidates evaluated per generation
                100 individuals in the internal population
                internal population_size = 100

            In normal GOW execution, this value is read from the general
            optimizer.batch_size field in the YAML and reaches GA as n in
            ask(problem, n).

        population_size:
            Optional backwards-compatible alias for batch_size.

            New YAML files should not use population_size. If both batch_size
            and population_size are provided, they must have the same value.

        generations:
            Maximum number of generations. One generation is one complete
            evaluation round of the whole population.

        crossover_rate:
            Probability of applying crossover for each real-valued parameter.

            Higher value:
                children are more influenced by the combination of both
                parents.

            Lower value:
                children are more likely to copy values directly from one
                parent.

        mutation_rate:
            Probability of mutating each parameter of a child.

            Higher value:
                more random variation is introduced.

            Lower value:
                the algorithm preserves more of the parental structure.

        elite_fraction:
            Fraction of the best population copied directly into the next
            generation.

            This protects good solutions from being lost. Because this
            implementation uses max(1, ...), at least one elite individual is
            preserved even when elite_fraction is configured as 0.

        selection:
            Parent selection strategy.

            This implementation currently supports tournament selection. The
            explicit setting is kept so the YAML is easy to read and so future
            selection strategies can be added without changing the general
            structure.

        tournament_size:
            Number of individuals compared during tournament selection.

            A larger tournament increases selection pressure because good
            individuals have a higher chance of being selected. The configured
            value must not be larger than the population size because Python's
            random.sample() selects tournament members without repetition.

        seed:
            Optional random seed. It makes the GA's internal random choices
            reproducible when the problem definition, evaluator behavior,
            candidate order, and remaining execution conditions are unchanged.

        kwargs:
            Extra configuration values that may be passed by GOW.

            They are accepted here so the optimizer remains compatible with
            the common GOW configuration mechanism.
        """

        # ------------------------------------------------------------------
        # Batch size / population size relationship
        # ------------------------------------------------------------------
        # In classical GA terminology, the group of candidates is called the
        # population.
        #
        # In GOW terminology, the number of candidates evaluated in one
        # generation is called batch_size.
        #
        # For this GA, both concepts are the same practical quantity:
        #
        #   one batch = one complete population
        #
        # Therefore, the user-facing YAML parameter should be batch_size, and
        # the internal population_size is set from that same value.

        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")

        if population_size is not None and int(population_size) < 1:
            raise ValueError("population_size must be >= 1")

        if (
            batch_size is not None
            and population_size is not None
            and int(batch_size) != int(population_size)
        ):
            raise ValueError(
                "For GeneticAlgorithmOptimizer, batch_size and population_size "
                "must be the same value. Prefer configuring only batch_size "
                "in the YAML."
            )

        # Direct construction may pass batch_size or the old population_size
        # alias. Normal GOW execution usually provides batch_size later as n in
        # ask(problem, n), so both values can be None at this point.
        effective_size = batch_size if batch_size is not None else population_size

        # Number of candidates evaluated per generation.
        self.batch_size = int(effective_size) if effective_size is not None else None

        # Internal GA name for the same quantity.
        self.population_size = self.batch_size

        # ------------------------------------------------------------------
        # Main GA hyperparameters
        # ------------------------------------------------------------------

        if generations < 1:
            raise ValueError("generations must be >= 1")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0.0 <= elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be between 0 and 1")
        if tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")

        # Number of generations to run.
        self.generations = int(generations)

        # Probability of combining two parents for a real-valued parameter.
        self.crossover_rate = float(crossover_rate)

        # Probability of randomly modifying each parameter in a child.
        self.mutation_rate = float(mutation_rate)

        # Fraction of the best individuals copied directly to the next
        # generation.
        self.elite_fraction = float(elite_fraction)

        # Selection method. Currently, only tournament selection is implemented.
        self.selection = str(selection).lower().strip()
        if self.selection != "tournament":
            raise ValueError(
                "GeneticAlgorithmOptimizer currently supports only "
                "selection='tournament'."
            )

        # Number of individuals compared when selecting one parent.
        self.tournament_size = int(tournament_size)

        # Optimizer-specific random generator.
        # Using self._rng instead of Python's global random generator makes the
        # run reproducible when a seed is provided.
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # General optimizer state
        # ------------------------------------------------------------------

        # Whether the initial population has already been created.
        # At the beginning this is False because ask() has not been called yet.
        self._initialized = False

        # Number of completed generations.
        # This is incremented in tell(), after evaluation results are received.
        self._generation = 0

        # Current population.
        #
        # This is a list because there are many individuals.
        # Each item in the list is a dictionary.
        #
        # Example:
        #   self._population[0] = {"p0": 2.1, "p1": -0.001}
        #
        # means individual 0 has those parameter values.
        self._population = []

        # Fitness values associated with the current population.
        #
        # The order is the same as self._population:
        #   self._fitness[0] is the fitness of self._population[0]
        #   self._fitness[1] is the fitness of self._population[1]
        #   etc.
        self._fitness = []

        # ------------------------------------------------------------------
        # Optimizable parameter information
        # ------------------------------------------------------------------

        # Specifications of each optimizable parameter.
        #
        # The key is the parameter name.
        # The value is a tuple with:
        #   (kind, lower_bound, upper_bound)
        #
        # Example:
        #   self._param_specs["p0"] = ("real", 0.0, 10.0)
        self._param_specs = {}

        # Names of the parameters that GA will optimize.
        #
        # Example:
        #   ["p0", "p1", "p2"]
        self._param_names = []

    def ask(self, problem: ProblemConfig, n: int):
        """
        Generate candidates for GOW to evaluate.

        This function answers the question:

            "Which candidate solutions should be evaluated now?"

        GOW calls ask() and expects a list of candidates.
        Each candidate is a dictionary with parameter values.

        Example candidate:
            {"p0": 2.3, "p1": -0.001, "p2": 5}

        ask() flow:

          1. Read n, the batch size requested by GOW.
          2. Set the internal population_size equal to that batch size.
          3. If the population does not exist yet, create it with _initialize().
          4. If this is generation 0, return the initial random population.
          5. Otherwise, create a new population.
          6. Copy the best individuals directly into the new population
             using elitism.
          7. Fill the remaining positions using selection, crossover, and
             mutation.
          8. Return the new population as the candidate list.
        """

        # GOW passes n because all optimizers share the same ask(problem, n)
        # interface.
        #
        # In this GA, n is the user-facing batch_size. Since one GA generation
        # evaluates one complete population, n also defines the internal
        # population_size.
        self._sync_population_size_from_batch(n)

        # First call to ask(): the population does not exist yet.
        # The initial random population is created here.
        if not self._initialized:
            self._initialize(problem)

        # Generation 0 means the algorithm has just created the initial
        # population and has not received any fitness values yet.
        #
        # At this point, the best possible action is to evaluate the initial
        # population as it is.
        if self._generation == 0:
            return [dict(x) for x in self._population]

        # This list will contain the individuals of the next generation.
        new_population = []

        # ------------------------------------------------------------------
        # Elitism
        # ------------------------------------------------------------------
        # elite_count is the number of best individuals that will be copied
        # directly to the next generation.
        #
        # max(1, ...) guarantees that at least one elite individual survives.
        # Consequently, elite_fraction = 0 still preserves one elite; this
        # implementation does not provide a completely elitism-free mode.
        elite_count = max(1, int(self.population_size * self.elite_fraction))

        # Sort the current population from best to worst.
        #
        # In this implementation, lower fitness is better.
        # Therefore, reverse=False places the smallest value first.
        #
        # zip(self._population, self._fitness) creates pairs like:
        #   (individual, fitness_value)
        ranked = sorted(
            zip(self._population, self._fitness),
            key=lambda x: x[1],
            reverse=False,
        )

        # Copy only the individual dictionary from each elite pair.
        #
        # ranked[:elite_count] keeps only the first elite_count pairs.
        # The expression inside brackets builds a new list. For each pair x,
        # x[0] is the individual and dict(x[0]) creates an independent copy.
        elites = [dict(x[0]) for x in ranked[:elite_count]]

        # Add elite individuals to the new population before creating children.
        new_population.extend(elites)

        # ------------------------------------------------------------------
        # Reproduction
        # ------------------------------------------------------------------
        # Fill the rest of the population until it reaches population_size.
        while len(new_population) < self.population_size:

            # Select two parents from the current population.
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Combine the two parents to create one child.
            child = self._crossover(p1, p2)

            # Randomly modify the child according to mutation_rate.
            child = self._mutate(child)

            # Add the new child to the next generation.
            new_population.append(child)

        # Replace the old population with the new one.
        self._population = new_population

        # Return one candidate for each individual in the population.
        return [dict(x) for x in self._population]

    def _sync_population_size_from_batch(self, n: int) -> None:
        """
        Keep batch_size and population_size as the same quantity.

        GOW calls ask(problem, n), where n is the number of candidates requested
        for the next evaluation batch.

        For this GA:

            n = batch_size = population_size

        This function centralizes that rule. It makes the first ask() call set
        the internal population size from GOW's batch_size. After that, it
        verifies that the size remains constant during the optimization.
        """

        # Convert n to int so comparisons are stable even if another part of
        # the workflow passes an integer-like value.
        n = int(n)

        # A GA population must contain at least one individual.
        if n < 1:
            raise ValueError("batch_size must be >= 1")

        # If the optimizer was created without an explicit size, this is the
        # normal GOW path: the YAML batch_size arrives here as n.
        if self.population_size is None:
            self.batch_size = n
            self.population_size = n
            return

        # Once the population exists, its size must remain constant. GA cannot
        # evaluate 100 individuals in one generation and 50 in the next without
        # changing the meaning of selection, elitism, and reproduction.
        if n != self.population_size:
            raise ValueError(
                "GeneticAlgorithmOptimizer requires batch_size to be equal to "
                "the internal population_size for the whole run. "
                f"Got batch_size={n}, population_size={self.population_size}."
            )

        # Keep both internal names synchronized.
        self.batch_size = n
        self.population_size = n

    def tell(self, candidates, fitness):
        """
        Receive evaluation results and store the fitness values.

        This function answers the question:

            "How good was each candidate proposed by ask()?"

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

        This method relies on GOW to preserve that order and to provide one
        result per candidate. It does not independently verify the lengths or
        compare the received candidates with the current population.

        In this implementation, tell() extracts one numeric value from each
        fitness dictionary and stores it in self._fitness. Those values will be
        used by ask() to create the next generation.
        """

        # scores will store one numeric value per evaluated candidate.
        scores = []

        # Loop over every result returned by the evaluator.
        for f in fitness:

            # Some evaluators return the value using the key "fitness".
            if "fitness" in f:
                scores.append(float(f["fitness"]))

            # Other evaluators return the value using the key "objective".
            elif "objective" in f:
                scores.append(float(f["objective"]))

            # Other evaluators may return the value using the key "score".
            elif "score" in f:
                scores.append(float(f["score"]))

            # If no known key is found, keep the original implementation
            # behavior and assign -inf.
            #
            # Important: this optimizer treats lower values as better, so -inf
            # will be interpreted as the best possible score. This fallback can
            # therefore make a result with an unexpected format dominate parent
            # selection. The comment documents the existing behavior; the code
            # is intentionally left unchanged here.
            else:
                scores.append(float("-inf"))

        # Store the scores of the current population.
        self._fitness = scores

        # Once the results have been received, one generation is complete.
        self._generation += 1

    def is_done(self):
        """
        Return whether the optimizer should stop.

        GA stops when the number of completed generations reaches the
        configured number of generations.
        """

        return self._generation >= self.generations

    def diagnostics(self):
        """
        Return useful information about the current optimizer state.

        This function does not create candidates and does not change the
        population. It only reports information.

        Main fields:

          - generation:
              current generation.

          - best_fitness:
              best fitness value found in the current population.

        In this implementation, lower fitness is better.
        """

        # If there are no fitness values yet, there is nothing to report.
        if not self._fitness:
            return {}

        # Return the current generation and the best current fitness.
        return {
            "generation": self._generation,
            "best_fitness": min(self._fitness),
        }

    def _initialize(self, problem):
        """
        Create the initial population.

        This function is called automatically from ask() the first time GOW
        requests candidates.

        Initialization steps:

          1. Read the optimizable parameters of the problem.
          2. Store the type and bounds of each supported parameter.
          3. Store the parameter names.
          4. Create population_size random individuals.
          5. Initialize the fitness list.
          6. Mark the optimizer as initialized.

        Initialization uses only the problem bounds.
        """

        # GOW provides the parameters marked as optimizable.
        params = problem.optimizable_parameters()

        # Loop over all optimizable parameters.
        # name is the parameter name.
        # p is the parameter configuration object.
        for name, p in params.items():

            # --------------------------------------------------------------
            # Real-valued parameter
            # --------------------------------------------------------------
            if isinstance(p, RealParam):

                # Bounds define the allowed search interval.
                lo, hi = p.bounds

                # Store that this parameter is real and save its bounds.
                self._param_specs[name] = ("real", lo, hi)

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            elif isinstance(p, IntParam):

                # Bounds define the allowed integer search interval.
                lo, hi = p.bounds

                # Store that this parameter is integer and save its bounds.
                self._param_specs[name] = ("int", lo, hi)

        # Store the names of all parameters that were accepted above.
        self._param_names = list(self._param_specs.keys())

        # The population size should have been fixed from batch_size before
        # initialization. This safety check makes configuration errors easier
        # to understand.
        if self.population_size is None:
            raise RuntimeError(
                "GA population_size was not initialized from batch_size before _initialize()."
            )

        # Create the initial population.
        #
        # Each individual is created randomly inside the parameter bounds.
        self._population = [
            self._random_individual()
            for _ in range(self.population_size)
        ]

        # Initial placeholder fitness values.
        #
        # These values will be replaced when tell() receives real evaluation
        # results from GOW.
        self._fitness = [float("-inf")] * self.population_size

        # Mark the population as created.
        self._initialized = True

    def _random_individual(self):
        """
        Generate one random individual inside the bounds.

        An individual is a dictionary with one value for each optimizable
        parameter.

        Example:
            {"p0": 1.23, "p1": 8}
        """

        # ind means individual.
        # It will be filled parameter by parameter.
        ind = {}

        # Traverse each optimizable parameter.
        for name in self._param_names:

            # Get the type and bounds of this parameter.
            kind, lo, hi = self._param_specs[name]

            # Real parameter: draw a uniform real number between lo and hi.
            if kind == "real":
                ind[name] = self._rng.uniform(lo, hi)

            # Integer parameter: draw a uniform integer between lo and hi.
            else:
                ind[name] = self._rng.randint(int(lo), int(hi))

        return ind

    def _tournament_select(self):
        """
        Select one parent using tournament selection.

        Tournament selection works like this:

          1. Randomly choose tournament_size individuals from the population.
          2. Compare only those selected individuals.
          3. Return the best one as a parent.

        This method does not always select the global best individual. That is
        intentional. It gives good individuals a higher chance of reproducing,
        but still preserves some diversity.
        """

        # Randomly choose tournament_size positions from the population.
        # random.sample() chooses distinct positions, so tournament_size cannot
        # be larger than the number of individuals in the population.
        #
        # Example:
        #   if the population has 100 individuals and tournament_size = 3,
        #   this may return indexes like [7, 42, 91].
        idxs = self._rng.sample(
            range(len(self._population)),
            self.tournament_size,
        )

        # Select the best index among the sampled individuals.
        #
        # min(..., key=...) returns the sampled index whose associated fitness
        # is smallest. The short lambda expression means: for each index i,
        # compare self._fitness[i]. Because lower fitness is better, min() is
        # used rather than max().
        best_idx = min(idxs, key=lambda i: self._fitness[i])

        # Return a copy of the selected individual.
        #
        # dict(...) avoids modifying the original parent later by accident.
        return dict(self._population[best_idx])

    def _crossover(self, p1, p2):
        """
        Create one child by combining two parents.

        p1 and p2 are parent individuals.
        Each one is a dictionary of parameter values.

        For real-valued parameters:

          - With probability crossover_rate, the child receives an intermediate
            value between p1 and p2.
          - Otherwise, the child copies the value from p1.

        For integer-valued parameters:

          - The child copies the value from p1 or p2 with equal probability.

        The returned child is still inside the parameter bounds.
        """

        # child will store the new individual.
        child = {}

        # Build the child one parameter at a time.
        for name in self._param_names:

            # Get the type and bounds of this parameter.
            kind, lo, hi = self._param_specs[name]

            # --------------------------------------------------------------
            # Real-valued parameter
            # --------------------------------------------------------------
            if kind == "real":

                # Decide whether crossover is applied to this parameter.
                if self._rng.random() < self.crossover_rate:

                    # alpha is a random weight between 0 and 1.
                    #
                    # It controls how much the child is influenced by each
                    # parent.
                    alpha = self._rng.random()

                    # Arithmetic crossover.
                    #
                    # If alpha is close to 1, the child is closer to p1.
                    # If alpha is close to 0, the child is closer to p2.
                    value = alpha * p1[name] + (1.0 - alpha) * p2[name]

                # If crossover is not applied, copy the value from p1.
                else:
                    value = p1[name]

                # Clamp the value inside the allowed bounds.
                child[name] = min(max(value, lo), hi)

            # --------------------------------------------------------------
            # Integer-valued parameter
            # --------------------------------------------------------------
            else:

                # For integer parameters, choose the value from one of the two
                # parents with equal probability.
                child[name] = (
                    p1[name]
                    if self._rng.random() < 0.5
                    else p2[name]
                )

        return child

    def _mutate(self, child):
        """
        Randomly modify a child.

        Mutation introduces new variation into the population. Without
        mutation, the algorithm could only recombine values already present in
        the current population.

        For each parameter:

          - A random number is compared with mutation_rate.
          - If mutation is applied, the parameter is modified.

        For real-valued parameters:

          - A small Gaussian change is added around the current value.
          - The standard deviation is 5% of the parameter range.

        For integer-valued parameters:

          - A new random integer is drawn inside the bounds.
        """

        # Traverse every parameter of the child.
        for name in self._param_names:

            # Decide whether this parameter mutates.
            if self._rng.random() < self.mutation_rate:

                # Get the type and bounds of this parameter.
                kind, lo, hi = self._param_specs[name]

                # ----------------------------------------------------------
                # Real-valued parameter
                # ----------------------------------------------------------
                if kind == "real":

                    # Parameter range.
                    # Example: if bounds = [2, 7], span = 5.
                    span = hi - lo

                    # Mutation scale.
                    #
                    # sigma is the standard deviation of the Gaussian noise.
                    # 0.05 means 5% of the parameter range.
                    sigma = 0.05 * span

                    # Add a local random perturbation around the current value.
                    value = child[name] + self._rng.gauss(0.0, sigma)

                    # Keep the mutated value inside the allowed bounds.
                    child[name] = min(max(value, lo), hi)

                # ----------------------------------------------------------
                # Integer-valued parameter
                # ----------------------------------------------------------
                else:

                    # For integer parameters, mutation replaces the value with
                    # a random integer inside the allowed bounds.
                    child[name] = self._rng.randint(
                        int(lo),
                        int(hi),
                    )

        return child
