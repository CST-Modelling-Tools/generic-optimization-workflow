# Problem Configuration (problem.yaml) – Detailed Guide

This document describes the structure of `problem.yaml` and how to use it to define
a wide range of optimization problems with the Generic Optimization Workflow (GOW).

The core idea:
- You describe WHAT you want to optimize (objective, parameters, evaluation method).
- GOW handles candidate generation, evaluation, and result storage.
- Your “problem” can be anything as long as it can be evaluated from parameters.

----------------------------------------------------------------
# 1. Conceptual Model
----------------------------------------------------------------

A GOW optimization problem consists of:

1) Problem identity
   - A stable id used to name result folders.

2) Parameters
   - The variables the optimizer can change.
   - Can include both:
     - Optimizable parameters (decision variables)
     - Runtime/static parameters (constants, modes, settings)

3) Objective
   - The quantity to minimize or maximize.

4) Evaluation
   - How a candidate is evaluated (usually an external script or program).
   - The evaluator reads candidate parameters and produces a structured result.

5) Optimizer configuration
   - Which optimizer to use.
   - How many evaluations to run and in what batch size.

The YAML is designed to be flexible enough for:
- toy benchmark functions (sphere, ackley, etc.)
- ML hyperparameter tuning
- simulation-based optimization
- expensive external workflows
- multi-metric evaluation with a single scalar objective
- constraints and feasibility checks

----------------------------------------------------------------
# 2. Typical YAML Skeleton
----------------------------------------------------------------

A typical `problem.yaml` includes:

- id: <string>
- parameters: <definitions>
- objective: <direction + metric>
- evaluator: <how to run evaluations>
- optimizer: <how to generate candidates>

Exact keys may vary slightly depending on your current schema, but the concepts
below apply regardless of small naming differences.

----------------------------------------------------------------
# 3. Problem Identity
----------------------------------------------------------------

id:
- A short, filesystem-friendly identifier for the problem.
- Used in the results folder layout:

  results/<problem_id>/...

Rules of thumb:
- Use lowercase, hyphen-separated names.
- Keep it stable over time.

Example values:
- toy-sphere
- turbine-design
- xgboost-hpo
- cfd-calibration

----------------------------------------------------------------
# 4. Parameters
----------------------------------------------------------------

Parameters define the values the optimizer and evaluator will work with.

There are two broad categories:

A) Optimizable (decision variables)
- The optimizer will propose values for these.

B) Runtime/static parameters
- Not optimized, but included in every evaluation.
- Useful for modes, dataset paths, solver settings, seeds, flags, etc.

A single parameter definition typically includes:
- type: the expected data type (float, int, bool, str)
- value: default / fixed value (if not optimizable)
- optimizable: true/false (or implied by presence of bounds)
- bounds or choices: the allowed search space
- (optional) metadata: description, units, etc.

----------------------------------------------------------------
# 4.1 Continuous Parameters (float)
----------------------------------------------------------------

Use for real-valued decision variables:

- Example: learning rate, geometric dimension, continuous coefficient.

Typical elements:
- type: float
- value: default
- optimizable: true
- bounds: [min, max]

Notes:
- The optimizer samples values in the bounds.
- Your evaluator should handle float inputs robustly.

----------------------------------------------------------------
# 4.2 Integer Parameters (int)
----------------------------------------------------------------

Use for discrete integer decision variables:

- Example: number of layers, tree depth, iterations.

Typical elements:
- type: int
- value: default
- optimizable: true
- bounds: [min, max]

Notes:
- Optimizer must generate integer values or values that can be cast safely.
- Your evaluator should validate constraints (e.g., min > 0).

----------------------------------------------------------------
# 4.3 Categorical Parameters (string / enum)
----------------------------------------------------------------

Use for categorical choices:

- Example: model type, solver method, augmentation mode.

Typical elements:
- type: str
- value: default
- optimizable: true
- choices: [a, b, c]

Notes:
- If your optimizer supports categorical variables, it should sample from choices.
- If not, you may encode categories numerically in the optimizer and decode in
  the evaluator.

----------------------------------------------------------------
# 4.4 Boolean Parameters
----------------------------------------------------------------

Use for feature toggles:

- Example: use_dropout, normalize_inputs, enable_cache.

Typical elements:
- type: bool
- value: true/false
- optimizable: true
- (no bounds; domain is {true, false})

Notes:
- Some optimizers treat booleans as categorical with two choices.

----------------------------------------------------------------
# 4.5 Fixed / Runtime Parameters (not optimized)
----------------------------------------------------------------

These are constants that still get passed to the evaluator.

Examples:
- dataset_path
- solver_executable
- mesh_file
- simulation_mode
- seed
- parallelism settings

Typical elements:
- type: ...
- value: ...
- optimizable: false

These values appear in:
- input.json written for the evaluation
- stored result records (params field)

----------------------------------------------------------------
# 4.6 Parameter Metadata (recommended)
----------------------------------------------------------------

Even if not required, metadata makes problems self-describing:

Common metadata fields:
- description: human readable explanation
- units: "m", "s", "kg", "%"
- tags: ["geometry", "solver", "hpo"]
- group: "model" / "solver" / "data"

This metadata does not affect optimization, but helps:
- documentation
- UI integration later
- analysis scripts

----------------------------------------------------------------
# 5. Objective
----------------------------------------------------------------

The objective defines:
- direction: minimize or maximize
- what the evaluator output field corresponds to the objective

Key idea:
- Your evaluator can compute many metrics.
- The objective is a single scalar used for ranking.

Common patterns:
- minimize "loss"
- maximize "accuracy"
- minimize "runtime"
- maximize "profit"

The evaluator is responsible for producing:
- status: ok / failed
- objective: numeric scalar
- metrics: dict of additional values (optional)
- constraints: dict (optional)
- artifacts: dict (optional)
- error: string (optional)

GOW stores all of this into results.jsonl.

----------------------------------------------------------------
# 6. Evaluator Definition
----------------------------------------------------------------

The evaluator describes HOW a candidate is evaluated.

Typical design:
- GOW writes input.json (parameters + runtime context) into the candidate workdir.
- An external program/script is executed inside that workdir.
- The external program writes output.json (results).
- GOW parses output.json and records fitness.

This approach allows:
- Python scripts
- compiled binaries
- bash pipelines
- simulations (CFD, FEM, SPICE, etc.)
- ML training runs
- remote execution wrappers

Important: the evaluator should always produce machine-readable output.

----------------------------------------------------------------
# 6.1 Workdir and Files
----------------------------------------------------------------

For each candidate, GOW uses a work directory like:

results/<problem_id>/runs/<run_id>/<candidate_id>/

Typical files inside:
- input.json
- output.json
- stdout.txt
- stderr.txt
- result.json (GOW summary of the evaluation)

This makes each candidate reproducible and debuggable.

----------------------------------------------------------------
# 6.2 Failures and Robustness
----------------------------------------------------------------

Your evaluator should handle failures gracefully.

If evaluation fails:
- objective may be omitted or null
- status should be "failed"
- error should describe what happened

GOW will still record the candidate and mark it as failed.

This is important for:
- expensive simulations that can diverge
- invalid parameter regions
- infeasible designs

----------------------------------------------------------------
# 7. Constraints (Optional but Supported)
----------------------------------------------------------------

Constraints are typically handled inside the evaluator.

Common approaches:
A) Hard constraints:
- If constraints violated, status="failed" and no objective.

B) Soft constraints:
- Add penalties to objective.

C) Report constraints separately:
- Provide constraints dict in fitness so analysis can filter by feasibility.

Example constraint metrics:
- mass <= 10
- max_stress <= 250
- stability_margin >= 1.2

Even if the optimizer is unaware of constraints, you can enforce feasibility via
the evaluator and post-filter results.

----------------------------------------------------------------
# 8. Multiple Metrics (Optional)
----------------------------------------------------------------

Many problems produce many metrics:
- objective is one scalar
- metrics contains everything else

Example metrics:
- accuracy, precision, recall
- runtime, memory usage
- stress, deformation, drag
- cost breakdowns

Store everything you care about so later you can:
- analyze tradeoffs
- plot Pareto fronts
- audit stability

----------------------------------------------------------------
# 9. Optimizer Configuration
----------------------------------------------------------------

Optimizer configuration typically includes:
- name: which optimizer to use
- seed: random seed
- max_evaluations: total evaluations
- batch_size: how many candidates proposed at once

Batch size matters:
- Local run: mostly affects structure, not parallelism unless you parallelize yourself.
- FireWorks run: batch size controls how many evaluations are submitted before launching.

You can implement multiple optimizers (random search, Bayesian opt, CMA-ES, etc.).
Each optimizer typically needs:
- bounds/choices for parameters
- direction (minimize/maximize)
- ask(n): propose candidates
- tell(candidates, fitnesses): update internal state

----------------------------------------------------------------
10. Flexibility and Advanced Use Cases
----------------------------------------------------------------

The YAML-based approach allows advanced patterns:

A) External workflows
- Evaluate candidates using docker, SLURM, remote scripts, etc.

B) Multi-stage evaluation
- Your evaluator can run multiple stages and aggregate into one objective.

C) Conditional logic in evaluation
- Example: if solver="A", run program A, else run program B.

D) Parameter transformations
- Example: optimize log_lr but convert to lr inside evaluator.

E) Mixed variable types
- float + int + categorical + boolean in one problem.

F) Reproducibility
- Keep problem.yaml + results folders = fully reproducible experiments.

----------------------------------------------------------------
# 11. Recommended User Workflow
----------------------------------------------------------------

1) Create a project folder outside the repository
2) Create problem.yaml
3) Run:

   gow run problem.yaml --run-id <name>

or with FireWorks:

   gow fw run problem.yaml --launchpad my_launchpad.yaml

4) Inspect:
- results/<problem_id>/results.jsonl
- results/<problem_id>/summary.json
- per-candidate workdirs for debugging

