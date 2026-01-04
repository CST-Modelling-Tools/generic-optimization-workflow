# Detailed Explanation of the example problem.yaml (toy-sphere)

This document explains, line by line and concept by concept, the following
problem configuration:

id: toy-sphere

objective:
  direction: minimize

parameters:
  x:
    type: real
    value: 0.1
    bounds: [-5.0, 5.0]

  y:
    type: real
    value: -0.2
    bounds: [-5.0, 5.0]

  n:
    type: int
    value: 5
    bounds: [1, 10]
    optimizable: false

  mode:
    type: categorical
    value: a
    choices: [a, b, c]

evaluator:
  command: ["{python}", "../../tests/toy_eval.py"]
  timeout_s: 60
  env:
    OMP_NUM_THREADS: "1"

optimizer:
  name: random_search
  seed: 123
  max_evaluations: 20
  batch_size: 4

context:
  note: toy example (YAML)

----------------------------------------------------------------
1. Problem identity
----------------------------------------------------------------

id: toy-sphere

This is the unique identifier of the optimization problem.

Purpose:
- Used to name result directories
- Used in results.jsonl and summary.json
- Allows multiple problems to coexist in the same results folder

With this id, all results will be written under:

results/toy-sphere/

Best practices:
- Use lowercase
- Use hyphens instead of spaces
- Keep it stable across runs so results accumulate coherently

----------------------------------------------------------------
2. Objective
----------------------------------------------------------------

objective:
  direction: minimize

This defines the optimization goal.

direction can be:
- minimize
- maximize

Meaning:
- The optimizer will treat smaller objective values as better
- GOW itself does not compute the objective
- The evaluator must return a numeric objective value
- The direction only affects ranking and optimizer logic

Typical examples:
- minimize loss, error, cost, runtime
- maximize accuracy, efficiency, profit

----------------------------------------------------------------
3. Parameters section (core of the problem)
----------------------------------------------------------------

parameters:
  ...

This section defines all parameters passed to the evaluator.
Some parameters are optimized, others are fixed.

Every parameter has:
- a name (x, y, n, mode)
- a type
- a value (default / initial / fixed)
- optionally bounds or choices
- optionally optimizable: false

All parameters (optimized or not) will appear in:
- input.json written for each candidate
- params field inside result.json and results.jsonl

----------------------------------------------------------------
3.1 Continuous parameters (x and y)
----------------------------------------------------------------

x:
  type: real
  value: 0.1
  bounds: [-5.0, 5.0]

y:
  type: real
  value: -0.2
  bounds: [-5.0, 5.0]

These define two continuous decision variables.

type: real
- Indicates a floating-point parameter

value:
- Default or initial value
- Used for:
  - initial candidate (depending on optimizer)
  - manual evaluation (gow evaluate)
  - documentation purposes

bounds:
- Lower and upper limits for optimization
- The optimizer will sample values inside these bounds

Because bounds are present and optimizable is not set to false:
- x and y are treated as optimizable parameters

Typical use cases:
- Geometry dimensions
- Continuous hyperparameters
- Physical coefficients

----------------------------------------------------------------
3.2 Integer parameter (n)
----------------------------------------------------------------

n:
  type: int
  value: 5
  bounds: [1, 10]
  optimizable: false

This is an integer parameter.

type: int
- Parameter must be an integer

bounds:
- Defines allowed values, even if not optimized
- Useful for validation and documentation

optimizable: false
- This explicitly tells GOW and the optimizer:
  - Do NOT change this parameter
  - Always pass value = 5 to the evaluator

Why include bounds if not optimizable?
- Documentation
- Validation
- Future-proofing (you may later decide to optimize it)

Typical use cases:
- Fixed discretization level
- Number of iterations
- Grid resolution

----------------------------------------------------------------
3.3 Categorical parameter (mode)
----------------------------------------------------------------

mode:
  type: categorical
  value: a
  choices: [a, b, c]

This is a categorical parameter.

type: categorical
- Parameter can only take one of a finite set of values

choices:
- Enumerates all allowed values

value:
- Default / initial category

Since optimizable is not set to false:
- mode IS optimized
- The optimizer will select among ["a", "b", "c"]

Typical use cases:
- Algorithm selection
- Solver mode
- Model architecture choice
- Feature toggles with more than two options

Note:
- Internally, some optimizers treat categorical values specially
- Others may encode them as integers

----------------------------------------------------------------
4. Evaluator
----------------------------------------------------------------

evaluator:
  command: ["{python}", "../../tests/toy_eval.py"]
  timeout_s: 60
  env:
    OMP_NUM_THREADS: "1"

This section defines how a candidate is evaluated.

The evaluator is responsible for:
- Reading parameters from input.json
- Running the computation
- Writing output.json
- Returning objective and metrics

----------------------------------------------------------------
4.1 command
----------------------------------------------------------------

command: ["{python}", "../../tests/toy_eval.py"]

This defines the executable command.

Key points:
- It is a list, not a string
- Each element is one argument
- "{python}" is a placeholder resolved by GOW to the current Python interpreter

Effectively, this runs something like:

python ../../tests/toy_eval.py

inside the candidate work directory.

Why this design:
- Cross-platform (no shell-specific syntax)
- Explicit argument handling
- Easy substitution of interpreters or binaries

You could also use:
- Compiled executables
- Bash scripts
- Docker wrappers
- MPI launchers

----------------------------------------------------------------
4.2 timeout_s
----------------------------------------------------------------

timeout_s: 60

This is a safety mechanism.

Meaning:
- If the evaluator runs longer than 60 seconds
- It is terminated
- The candidate is marked as failed

Important for:
- External solvers that may hang
- Protecting large batch runs
- CI and automated experiments

----------------------------------------------------------------
4.3 env
----------------------------------------------------------------

env:
  OMP_NUM_THREADS: "1"

Environment variables set for the evaluator process.

In this example:
- Forces single-threaded execution for OpenMP programs

Typical use cases:
- Control parallelism
- Set library paths
- Configure licenses
- Set reproducibility flags

Values must be strings.

----------------------------------------------------------------
5. Optimizer
----------------------------------------------------------------

optimizer:
  name: random_search
  seed: 123
  max_evaluations: 20
  batch_size: 4

This section controls how candidates are generated.

----------------------------------------------------------------
5.1 name
----------------------------------------------------------------

name: random_search

Selects the optimizer implementation.

Examples of possible optimizers:
- random_search
- grid_search
- bayesian
- cma_es
- custom_optimizer

The optimizer must implement:
- ask(problem, n)
- tell(candidates, fitnesses)

----------------------------------------------------------------
5.2 seed
----------------------------------------------------------------

seed: 123

Sets the random seed.

Purpose:
- Reproducibility
- Debugging
- Fair comparison between runs

Same problem.yaml + same seed = same candidate sequence (for deterministic optimizers).

----------------------------------------------------------------
5.3 max_evaluations
----------------------------------------------------------------

max_evaluations: 20

Total number of candidate evaluations for this run.

This is a hard limit.

Resulting layout:
- 20 candidate folders under runs/<run_id>/
- 20 entries appended to results.jsonl

----------------------------------------------------------------
5.4 batch_size
----------------------------------------------------------------

batch_size: 4

Number of candidates proposed at once.

Meaning:
- Optimizer generates 4 candidates
- They are evaluated
- Results are collected
- Optimizer is updated
- Next batch is generated

Why batching matters:
- Parallel execution (especially with FireWorks)
- Some optimizers benefit from batch updates
- Controls job submission granularity

----------------------------------------------------------------
6. Context
----------------------------------------------------------------

context:
  note: toy example (YAML)

Context is arbitrary metadata.

Characteristics:
- Not optimized
- Not interpreted by the optimizer
- Passed through to the evaluator if needed
- Stored for traceability

Typical uses:
- Notes
- Dataset version
- Experiment tags
- Author
- Git commit hash
- Simulation scenario description

Context helps turn results into self-describing experiments.

----------------------------------------------------------------
7. Resulting filesystem layout
----------------------------------------------------------------

With this configuration, a local run produces:

results/
  toy-sphere/
    results.jsonl          (all runs combined)
    summary.json           (latest run summary)
    runs/
      <run_id>/
        results.jsonl      (this run only)
        summary.json
        c000000/
          input.json
          output.json
          result.json
          stdout.txt
          stderr.txt
        c000001/
        ...

Each candidate directory is fully self-contained and reproducible.

----------------------------------------------------------------
8. Why this YAML is powerful
----------------------------------------------------------------

This single file allows you to:
- Change optimizers without touching code
- Swap evaluators (Python, C++, simulation, ML training)
- Add or remove parameters
- Mix continuous, integer, categorical variables
- Control execution environment
- Run locally or with FireWorks
- Archive and reproduce experiments

In practice, this makes GOW suitable for:
- Research experiments
- Engineering optimization
- Hyperparameter tuning
- Simulation-based design
- Automated benchmarking
