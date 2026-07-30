# BO in GOW

## Optimizer User Guide and YAML Configuration

*User guide focused on YAML configuration and execution in GOW.*

## Contents

1. [What Bayesian Optimization Is](#1-what-bayesian-optimization-is)
2. [Main Idea: Surrogate Model and Acquisition Function](#2-main-idea-surrogate-model-and-acquisition-function)
3. [From the Idea to the Algorithm: How BO Works Inside GOW](#3-from-the-idea-to-the-algorithm-how-bo-works-inside-gow)
4. [When BO Makes Sense](#4-when-bo-makes-sense)
5. [How a BO Execution Is Controlled in GOW](#5-how-a-bo-execution-is-controlled-in-gow)
6. [How to Configure the YAML](#6-how-to-configure-the-yaml)
   - [6.1 `objective` Block](#61-objective-block)
   - [6.2 `parameters` Block](#62-parameters-block)
   - [6.3 `evaluator` Block](#63-evaluator-block)
   - [6.4 `optimizer` Block](#64-optimizer-block)
7. [Configurable BO Parameters](#7-configurable-bo-parameters)
   - [7.1 `base_estimator` Options](#71-base_estimator-options)
   - [7.2 `acquisition_function` Options](#72-acquisition_function-options)
   - [7.3 `acq_optimizer` Options](#73-acq_optimizer-options)
   - [7.4 `batch_strategy` Options](#74-batch_strategy-options)
8. [Practical Recommendations](#8-practical-recommendations)
9. [Commented Base YAML](#9-commented-base-yaml)
10. [Quick Reading of the BO-GOW Flow](#10-quick-reading-of-the-bo-gow-flow)
11. [Final Summary](#11-final-summary)

---

## 1. What Bayesian Optimization Is

Bayesian Optimization, or **BO**, is a model-based optimization method designed especially for problems in which evaluating one solution can be expensive in terms of time, computational resources, or experimentation.

Each candidate represents a complete combination of values for the optimizable parameters defined in the YAML. GOW sends that candidate to the external evaluator and receives an objective value indicating the quality of the solution.

Unlike population-based optimizers such as GA or PSO, BO does not evolve a population or move particles. It uses previously obtained results to build an approximate model of the objective function and decide which candidate should be evaluated next.

> **Question answered by BO**
>
> Which parameter combination should be evaluated next to find a better solution while using as few expensive evaluations as possible?

---

## 2. Main Idea: Surrogate Model and Acquisition Function

The central idea of BO is to learn progressively how the objective function behaves without evaluating every possible point.

### Surrogate model

The surrogate model is a cheaper approximation of the real objective function. It is built from previously evaluated candidates and their results.

It does not replace the external evaluator and does not determine the final result. It only tries to predict which regions appear promising and where uncertainty is greater.

The current implementation can use a Gaussian Process or tree-based models, depending on the value configured in `base_estimator`.

### Acquisition function

The acquisition function uses the model predictions to decide which point should be evaluated next. Its task is to balance two needs:

- **Exploitation:** search near regions that have already produced good results.
- **Exploration:** test less-known regions that may contain a better solution.

### Initial points

Before the model can guide the search, BO needs initial observations.

During this phase, candidates are generated inside the defined `bounds`. Once enough information is available, the surrogate model and acquisition function begin guiding candidate selection.

---

## 3. From the Idea to the Algorithm: How BO Works Inside GOW

Inside GOW, BO does not calculate the objective function. Its responsibility is to propose candidates and learn from the results it receives.

The implementation uses the `ask`/`tell` interface from `scikit-optimize`:

- Through `ask`, BO gives GOW the candidates that must be evaluated.
- Through `tell`, GOW returns the obtained results to BO.

On the first request, BO reads the parameter types and `bounds` and constructs the search space.

The `value` fields in the YAML remain reference values, but they are **not automatically inserted as initial candidates**.

The flow is:

1. BO generates one or more candidates inside the `bounds`.
2. GOW sends the candidates to the external evaluator.
3. The evaluator calculates the real objective value.
4. GOW returns the results to BO.
5. BO incorporates the new observations and updates the surrogate model.
6. The acquisition function identifies new promising regions.
7. The cycle continues until one of the execution limits is reached.

The current implementation supports optimizable parameters of type `real` and `int`. Optimizable categorical parameters are rejected. It also supports both minimization and maximization objectives.

---

## 4. When BO Makes Sense

BO is especially suitable when:

- Each evaluation requires a simulation, an intensive calculation, model training, or an expensive experiment.
- The objective function behaves as a black box and derivatives are not available.
- The parameters can be represented as bounded numerical values.
- The evaluation budget is limited.
- Information from previous evaluations should be used to decide what to evaluate next.
- Good solutions are required with fewer evaluations than a massive or random search.

BO is usually especially useful for low- or moderate-dimensional spaces and relatively limited evaluation budgets.

BO is not usually the best option when the objective function is very cheap and extremely large numbers of evaluations can be performed, when the problem depends mainly on categorical parameters, or when very large and fully independent batches are required.

---

## 5. How a BO Execution Is Controlled in GOW

The execution is controlled by general GOW parameters and two parameters specific to this BO implementation:

| Parameter | YAML location | What it controls |
|---|---|---|
| `max_evaluations` | `optimizer` | Maximum total number of candidates that GOW can evaluate. |
| `batch_size` | `optimizer` | Number of candidates requested in each cycle. |
| `max_iterations` | `optimizer.settings` | Maximum number of complete `ask`/`tell` cycles. |
| `n_initial_points` | `optimizer.settings` | Number of initial observations before relying mainly on the model. |

### BO does not use generations

BO is not an evolutionary algorithm.

In this implementation, one **iteration** is a complete cycle in which BO proposes a batch, GOW evaluates it, and the results are returned to the optimizer.

> **Essential relationship**
>
> `planned evaluations = batch_size × max_iterations`

To prevent one limit from stopping the execution before the other, it is recommended to configure:

```text
max_evaluations = batch_size × max_iterations
```

When `max_evaluations` is lower, GOW exhausts the evaluation budget before all iterations are completed.

When `max_evaluations` is higher, BO may reach `max_iterations` while leaving part of the budget unused.

### Relationship with `n_initial_points`

`n_initial_points` counts evaluated candidates, not iterations.

These points are included in `max_evaluations`; they are not added as an extra budget.

For example, with:

```yaml
max_evaluations: 100
batch_size: 1

settings:
  n_initial_points: 20
  max_iterations: 100
```

the first 20 evaluations form the initial phase, and 80 evaluations remain for model-guided search.

With:

```yaml
batch_size: 5

settings:
  n_initial_points: 20
```

the optimizer will have received approximately 20 observations after four complete batches.

---

## 6. How to Configure the YAML

The YAML describes the objective, the parameters that may vary, the external evaluator, and the optimizer configuration.

### 6.1 `objective` Block

```yaml
objective:
  metric: metrics.objective
  direction: minimize
```

`direction` indicates whether the objective must be minimized or maximized.

The implementation accepts only:

```yaml
direction: minimize
```

or:

```yaml
direction: maximize
```

`metric` identifies the path of the objective value inside the result produced by the evaluator.

### 6.2 `parameters` Block

```yaml
parameters:
  x0:
    type: real
    value: 0.5
    bounds: [0.0, 1.0]

  x1:
    type: int
    value: 10
    bounds: [5, 15]
```

Each optimizable parameter must define:

- `type`: this implementation supports `real` and `int`.
- `value`: a reference value; it is not automatically inserted as an initial point.
- `bounds`: the limits inside which BO may generate candidates.

For `real` parameters, the lower bound must be lower than the upper bound.

For `int` parameters, the implementation accepts equal bounds, although in that case there is no real search in that dimension.

Optimizable categorical parameters are not supported.

### 6.3 `evaluator` Block

```yaml
evaluator:
  command:
    ["/path/to/the/evaluator"]
  timeout_s: 600
```

This block identifies the external program responsible for calculating the real result of each candidate.

The evaluator must return a finite numerical value compatible with the metric defined by the problem.

### 6.4 `optimizer` Block

```yaml
optimizer:
  name: bayesian_optimization
  seed: 123
  max_evaluations: 100
  batch_size: 1

  settings:
    n_initial_points: 20
    base_estimator: GP
    acquisition_function: EI
    acq_optimizer: auto
    batch_strategy: cl_min
    max_iterations: 100
```

The main block contains the general GOW execution controls.

The `settings` sub-block contains the hyperparameters specific to the BO implementation.

---

## 7. Configurable BO Parameters

The following table separates the general GOW controls from the hyperparameters specific to Bayesian Optimization.

| Parameter | YAML location | What it controls | Usage guide |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | Must be `bayesian_optimization`. |
| `seed` | `optimizer` | Fixes the pseudo-random sequence. | Use it to compare configurations. Keep the YAML, evaluator, dependency versions, and result order unchanged. |
| `max_evaluations` | `optimizer` | Limits the total GOW evaluation budget. | To complete every cycle, make it equal to `batch_size × max_iterations`. |
| `batch_size` | `optimizer` | Defines how many candidates are requested per iteration. | Use `1` for sequential operation. Use small batches only when parallelism is important. |
| `n_initial_points` | `optimizer.settings` | Defines how many initial observations BO receives. | Must be at least `1`. A very low value gives the model little information; a very high value consumes much of the budget. |
| `base_estimator` | `optimizer.settings` | Selects the surrogate model. | Directly supported options are `GP`, `RF`, `ET`, and `GBRT`. |
| `acquisition_function` | `optimizer.settings` | Defines the rule used to select promising points. | Main options are `EI`, `PI`, `LCB`, and `gp_hedge`. `EI` is a balanced starting option. |
| `acq_optimizer` | `optimizer.settings` | Defines how the acquisition function is optimized. | Use `auto` as a general option, `sampling` with tree-based models, and `lbfgs` mainly with `GP`. |
| `batch_strategy` | `optimizer.settings` | Controls batch generation before real results are known. | Options are `cl_min`, `cl_mean`, and `cl_max`. It matters only when `batch_size > 1`. |
| `max_iterations` | `optimizer.settings` | Limits complete `ask`/`tell` cycles. | Must be at least `1`. Together with `batch_size`, it determines the required number of evaluations. |

### 7.1 `base_estimator` Options

| Value | Model | Practical interpretation |
|---|---|---|
| `GP` | Gaussian Process | Classical BO formulation. Models both prediction and uncertainty. Suitable for small or moderate budgets and low- or moderate-dimensional spaces. |
| `RF` | Random Forest | An ensemble of trees. It can adapt well to nonlinear or irregular relationships and can handle more observations than a GP with lower internal cost. |
| `ET` | Extra Trees | Trees with greater randomization. It provides a flexible and fast model for irregular functions. |
| `GBRT` | Gradient Boosted Regression Trees | Trees built sequentially to correct previous errors and model complex relationships. |

The underlying library also provides `DUMMY`, but this performs random sampling without building a surrogate model. It is not recommended as a normal BO configuration.

### 7.2 `acquisition_function` Options

| Value | Name | Practical interpretation |
|---|---|---|
| `EI` | Expected Improvement | Favors points with a high expected improvement over the best known result. It balances prediction and uncertainty. |
| `PI` | Probability of Improvement | Favors points with a high probability of improvement, even when the expected improvement may be small. |
| `LCB` | Lower Confidence Bound | Combines a favorable prediction with uncertainty to balance exploitation and exploration. |
| `gp_hedge` | Adaptive combination | Considers `EI`, `PI`, and `LCB` and progressively adapts which one is used. |

`scikit-optimize` also recognizes `EIps` and `PIps`, but they require the objective value and evaluation time to be returned together.

The current wrapper sends one numerical loss per candidate, so these options must not be used.

Internal parameters such as `xi`, `kappa`, and `eta` are also not currently exposed in the YAML.

### 7.3 `acq_optimizer` Options

| Value | Behavior | Usage guide |
|---|---|---|
| `auto` | Automatically selects the method according to the surrogate model. | Recommended for a general configuration. |
| `sampling` | Evaluates the acquisition function over sampled points and selects the best one. | Appropriate for `RF`, `ET`, and `GBRT`. |
| `lbfgs` | Optimizes the acquisition function with L-BFGS. | Appropriate mainly with `GP`. It should not be forced with models that do not provide gradients. |

### 7.4 `batch_strategy` Options

When `batch_size` is greater than `1`, BO must propose several candidates before knowing the real result of the first one.

To do this, it uses **constant liar** strategies.

| Value | Temporary result used |
|---|---|
| `cl_min` | Temporarily uses the best observed internal value. |
| `cl_mean` | Temporarily uses the mean of the observed values. |
| `cl_max` | Temporarily uses the worst observed internal value. |

With:

```yaml
batch_size: 1
```

only one candidate is requested per cycle, so `batch_strategy` does not produce a practical difference.

---

## 8. Practical Recommendations

### 8.1 Start with a truly sequential search

To take full advantage of BO learning after every evaluation, start with:

```yaml
batch_size: 1
```

When the evaluator supports parallel execution and total runtime is more important, test small batches afterward.

### 8.2 Align the budget and the iterations

Keep:

```text
max_evaluations = batch_size × max_iterations
```

so that both stopping criteria represent the same budget.

### 8.3 Reserve budget for the guided phase

`n_initial_points` must provide enough observations to begin modeling the problem, but it should not consume the entire budget.

When `n_initial_points` is equal to or greater than `max_evaluations`, the run may end without a useful model-guided phase.

### 8.4 Choose the model and acquisition optimizer together

Safe combinations include:

```yaml
base_estimator: GP
acq_optimizer: auto
```

```yaml
base_estimator: ET
acq_optimizer: auto
```

```yaml
base_estimator: RF
acq_optimizer: sampling
```

Avoid manually combining a tree-based model with:

```yaml
acq_optimizer: lbfgs
```

because those models do not provide the required gradients.

### 8.5 Define informative `bounds`

BO can only propose values inside the defined intervals.

Bounds that are too narrow may exclude a better solution. Bounds that are too wide make it harder for the initial observations to describe the search space.

### 8.6 Use `seed` to compare configurations

A fixed `seed` makes it easier to compare surrogate models, acquisition functions, initial-point counts, and batch sizes.

To evaluate robustness, repeat the final configuration with different seeds.

### 8.7 Validate a short run first

Before launching an expensive optimization, verify with a small budget that:

- The evaluator receives all required parameters.
- The evaluator returns the expected numerical metric.
- The optimization direction is correct.
- Candidates respect the `bounds`.
- The execution limits are coherent.
- The surrogate model and acquisition optimizer are compatible.

### 8.8 Avoid unsupported options

Do not add:

- Optimizable categorical parameters.
- `EIps` or `PIps`.
- `xi`, `kappa`, or `eta`.
- Custom model options that this implementation does not forward to `scikit-optimize`.

---

## 9. Commented Base YAML

This example is generic and must be adapted to the parameters, metric, and evaluator path of each problem.

```yaml
id: continuous-problem-bo

objective:
  metric: metrics.objective
  direction: minimize            # BO will search for the lowest value.

parameters:
  x0:
    type: real                   # Continuous parameter.
    value: 0.5                   # Reference value; not used as an initial point.
    bounds: [0.0, 1.0]           # Allowed interval.

  x1:
    type: int                    # Integer parameter.
    value: 10
    bounds: [5, 15]

evaluator:
  command:
    ["/path/to/the/evaluator"]   # External program that calculates the objective.
  timeout_s: 600                 # Maximum time allowed per evaluation.

optimizer:
  name: bayesian_optimization    # Selects BO.
  seed: 123                      # Seed for reproducibility.
  max_evaluations: 100           # Total GOW evaluation budget.
  batch_size: 1                  # One candidate per iteration.

  settings:
    n_initial_points: 20         # Initial observations.
    base_estimator: GP           # GP, RF, ET, or GBRT.
    acquisition_function: EI     # EI, PI, LCB, or gp_hedge.
    acq_optimizer: auto          # auto, sampling, or lbfgs.
    batch_strategy: cl_min       # Relevant only when batch_size > 1.
    max_iterations: 100          # 100 × 1 = 100 evaluations.
```

---

## 10. Quick Reading of the BO-GOW Flow

1. GOW reads the YAML and prepares the problem, evaluator, and optimizer.
2. BO reads the parameter types and `bounds`.
3. GOW requests a batch of size `batch_size` from BO.
4. BO generates initial candidates or model-guided candidates.
5. GOW sends each candidate to the external evaluator.
6. The evaluator returns one objective value for each candidate.
7. GOW returns the results to BO.
8. BO incorporates the observations and updates the surrogate model.
9. The acquisition function identifies new promising regions.
10. The process continues until `max_evaluations` or `max_iterations` is reached.

---

## 11. Final Summary

Bayesian Optimization is a model-based optimizer intended for problems in which every evaluation is expensive and the available budget must be used efficiently.

In the YAML, the user configures the objective direction, parameters and `bounds`, external evaluator, evaluation budget, batch size, initial points, surrogate model, acquisition function, acquisition optimizer, batch strategy, and maximum number of iterations.

GOW coordinates candidate generation, external execution, and result collection. The evaluator calculates the real objective value. BO uses those results to update its model and decide which candidates should be evaluated next.

> **Recommended starting configuration**
>
> Use `batch_size: 1`, `base_estimator: GP`, `acquisition_function: EI`, `acq_optimizer: auto`, and an `n_initial_points` value that leaves enough budget for the model-guided phase.
