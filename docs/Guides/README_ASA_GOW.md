# ASA in GOW

## Guide 2: Using the Optimizer and Configuring the YAML File

## Contents

1. What ASA Is
2. Main Inspiration: From Material Cooling to Optimization
3. From the Idea to the Algorithm: How ASA Works Within GOW
4. When It Makes Sense to Use ASA
5. How an ASA Run Is Controlled in GOW
6. How to Configure the YAML File
7. Configurable ASA Parameters
8. Practical Recommendations
9. Commented Base YAML File
10. Quick Overview of the ASA-GOW Flow
11. Final Summary

# 1. What ASA Is

ASA stands for **Adaptive Simulated Annealing**. It is a stochastic global-search optimizer designed to adjust numerical parameters within defined limits. It can work with real-valued and integer variables, and it is especially useful when the objective function is nonlinear, has several local optima, or does not provide derivatives.

In GOW, ASA does not calculate the objective function. The optimizer proposes candidates, GOW sends them to an external evaluator, and then returns the results. ASA uses those evaluations to decide which solution should become the next search state and how to progressively reduce its level of exploration.

> **Question ASA answers**
> Which combination of numerical values within these limits produces the best result while allowing broad exploration at the beginning and more focused refinement afterward?

# 2. Main Inspiration: From Material Cooling to Optimization

Simulated annealing is inspired by the thermal treatment of materials. When a material is hot, its particles can reorganize freely. If it is cooled slowly, their movement decreases and the system tends to stabilize in a lower-energy structure.

In optimization, temperature is a control mechanism. At the beginning, a high temperature allows broad movements and makes it possible to occasionally accept worse candidates. This ability helps the optimizer leave regions that appear locally good but are not the best global solution.

As the temperature decreases, ASA becomes more selective: it accepts fewer deteriorations and concentrates the search around promising solutions. The adaptive part of the method allows the search to be readjusted during the run and a soft restart to be applied when no improvement is observed for an extended period.

# 3. From the Idea to the Algorithm: How ASA Works Within GOW

Within GOW, each candidate is a complete combination of values for the parameters defined in the YAML file. The `value` fields provide the initial reference, and the `bounds` fields define the permitted search space. The implementation does not automatically add the candidate formed exactly by the `value` fields as a separate evaluation; ASA uses those values as the starting point from which it generates proposals.

The general flow is as follows:

1. ASA generates a batch of candidates within the `bounds`.
2. GOW sends each candidate to the external evaluator.
3. The evaluator returns an objective value for each candidate.
4. ASA decides which proposals to accept as the new search state and separately preserves the best solution found.
5. The temperatures decrease, and the search gradually moves from exploratory movements to more local adjustments.
6. When appropriate, ASA readjusts the search through reannealing or applies a soft restart around the best candidate.

```text
ask()  -> ASA proposes a batch of candidates
GOW    -> evaluates those candidates
tell() -> ASA receives the results and updates its search
```

# 4. When It Makes Sense to Use ASA

ASA is appropriate when the problem is described by numerical parameters with clear limits and the evaluator can return an objective measure of quality for each combination.

This optimizer is suitable when:

- the objective function may have several local optima;
- derivatives are unavailable or the evaluator behaves as a black box;
- the search must combine global exploration and local refinement;
- the parameters are real-valued, integer-valued, or a mixture of both;
- a stochastic search is acceptable and a sufficient evaluation budget is available.

ASA is not usually the first choice for purely categorical problems, simple convex problems with reliable gradients, or situations where the evaluation budget is too small to allow a meaningful exploration and cooling phase.

# 5. How an ASA Run Is Controlled in GOW

The run is controlled with `max_evaluations`, `batch_size`, and `max_iterations`. The first two are configured at the main `optimizer` level, while `max_iterations` is written inside `settings`.

- `max_evaluations` defines the maximum number of candidates that GOW can evaluate.
- `batch_size` defines how many candidates GOW requests in each batch.
- `max_iterations` defines how many complete generation, evaluation, and update cycles ASA can perform.

In this integration, `max_iterations` must not be selected as an independent budget. It must be calculated by dividing the total number of evaluations by the batch size:

```text
max_iterations = max_evaluations / batch_size
```

For example, with `max_evaluations: 1000` and `batch_size: 25`, the correct value is `max_iterations: 40`. It is recommended that `max_evaluations` be a multiple of `batch_size` to avoid an incomplete final batch and to keep all three limits consistent.

> **Important note about `batch_size`**
> Candidates in the same batch are generated before their results are received. A large batch can make better use of parallel execution, but ASA updates its search less frequently. A small batch allows more frequent adaptation, although it provides less parallelism.

# 6. How to Configure the YAML File

The YAML file must describe the objective, the optimizable parameters, the external evaluator, and the ASA configuration. The main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

## 6.1 `objective` Block

```yaml
objective:
  direction: minimize
```

`direction` indicates whether the metric must be minimized or maximized. For error, cost, or distance problems, `minimize` is normally used. When a larger metric represents a better result, use `maximize`.

## 6.2 `parameters` Block

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

- `type`: the numerical type of the parameter. This implementation supports `real` and `int`;
- `value`: the initial reference from which ASA begins generating proposals;
- `bounds`: the lower and upper limits of the space in which ASA can search.

Categorical parameters are not directly supported. Each `value` must lie within its `bounds` and represent a valid starting point for the evaluator.

## 6.3 `evaluator` Block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

This block indicates which external program will evaluate each candidate and how long an evaluation may take. ASA proposes values; the evaluator calculates the metric, and GOW returns the result to the optimizer.

## 6.4 `optimizer` Block

```yaml
optimizer:
  name: asa
  seed: 123
  max_evaluations: 1000
  batch_size: 25
  settings:
    max_iterations: 40
    initial_temperature: 1.0
    final_temperature: 0.001
    reanneal_interval: 64
    restart_interval: 240
    restart_sigma: 0.012
```

At the main level, ASA is selected and the general GOW budget is defined. Inside `settings`, only the ASA parameters that the user must configure in this implementation are included.

# 7. Configurable ASA Parameters

The following table lists the parameters that the user should review. General GOW parameters and ASA-specific hyperparameters are distinguished by their location in the YAML file.

| Parameter | YAML location | What it controls | Usage guidance |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | It must be `asa` to use this implementation. |
| `seed` | `optimizer` | Fixes the pseudorandom sequence. | Use a fixed value when a run must be repeated under equivalent conditions. |
| `max_evaluations` | `optimizer` | Defines GOW's total evaluation budget. | Choose it according to the cost of the evaluator and the available time. |
| `batch_size` | `optimizer` | Defines how many candidates are evaluated per batch. | Larger values favor parallelism; smaller values allow ASA to update more frequently. |
| `max_iterations` | `settings` | Limits the number of complete ASA cycles. | It must be exactly `max_evaluations / batch_size`. It must not be selected independently. |
| `initial_temperature` | `settings` | Defines the initial exploration level and how easily worse candidates can be accepted. | It must be greater than `0`. A higher value makes the initial phase more exploratory; a lower value makes it more conservative. Base value: `1.0`. |
| `final_temperature` | `settings` | Defines the minimum temperature of the final phase. | It must be greater than `0` and no greater than `initial_temperature`. Lower values produce a stricter final phase. Base value: `0.001`. |
| `reanneal_interval` | `settings` | Indicates how often, in evaluations, ASA readjusts the search. | A shorter interval adapts more frequently. Use `0` to disable periodic reannealing. Base value: `64`. |
| `restart_interval` | `settings` | Defines how many evaluations without improvement trigger a soft restart. | It must be smaller than the evaluation budget if it is expected to take effect. Use `0` to disable it. Base value: `240`. |
| `restart_sigma` | `settings` | Controls the size of the restart around the best candidate. | It is interpreted in normalized space. `0.012` is approximately `1.2%` of each parameter's range. It only takes effect when a restart occurs. |

# 8. Practical Recommendations

## 8.1 Keep Evaluations, Batch Size, and Iterations Consistent

Always calculate `max_iterations` as `max_evaluations` divided by `batch_size`. For example, 200 evaluations with batches of 10 require 20 iterations. If one of the three values is changed, the other two must be reviewed.

```text
max_evaluations = batch_size x max_iterations
```

## 8.2 Choose `batch_size` According to the Execution Mode

A high `batch_size` can be useful when many evaluations are executed in parallel, but ASA receives information less frequently. A low `batch_size` favors more continuous adaptation. The best balance depends on the cost of the evaluator and the available resources.

## 8.3 Define `value` and `bounds` With Physical Meaning

ASA can search only within the `bounds`. Limits that are too narrow prevent alternative solutions from being explored; excessively broad limits may require many more evaluations. The `value` fields must be valid and provide a reasonable initial reference, although they do not have to represent the best known solution.

## 8.4 Adjust the Temperatures First

If the search becomes local too early, `initial_temperature` can be increased or `final_temperature` can be reduced cautiously. If ASA continues exploring too much near the end, a higher `final_temperature` can be used. Always keep `final_temperature` less than or equal to `initial_temperature`, and avoid changing several parameters at the same time.

## 8.5 Configure Intervals That Can Take Effect

`reanneal_interval` and `restart_interval` are expressed in evaluations. They must be consistent with `max_evaluations`. For example, `restart_interval: 240` cannot be activated in a run limited to 200 evaluations. `restart_sigma` will also have no effect if restart is disabled or the stagnation interval is never reached.

## 8.6 Use `seed` to Compare Configurations

A fixed seed helps compare configuration changes under the same pseudorandom sequence. Because ASA is stochastic, its general behavior should be assessed through several runs with different seeds, comparing both the best results and their stability.

# 9. Commented Base YAML File

This example shows a generic and internally consistent configuration. It must be adapted to the actual parameters, evaluator, and evaluation budget of the problem.

```yaml
id: continuous-problem-asa

objective:
  direction: minimize              # Use maximize when a larger value is better.

parameters:
  x0:
    type: real                     # Real-valued numerical parameter.
    value: 0.5                     # ASA's initial reference.
    bounds: [0.0, 1.0]             # Permitted range.
  x1:
    type: int                      # Integer parameters are also supported.
    value: 10
    bounds: [5, 15]

evaluator:
  command:
    ["/path/to/evaluator"]         # Program that calculates the metric.
  timeout_s: 600                   # Maximum time per evaluation.

optimizer:
  name: asa                        # Selects Adaptive Simulated Annealing.
  seed: 123                        # Seed for reproducibility.
  max_evaluations: 1000            # GOW's total evaluation budget.
  batch_size: 25                   # Candidates per batch.
  settings:
    max_iterations: 40             # 1000 / 25 = 40 cycles.
    initial_temperature: 1.0       # Exploration at the beginning.
    final_temperature: 0.001       # Minimum final temperature.
    reanneal_interval: 64          # Readjustment every 64 evaluations.
    restart_interval: 240          # Restart after 240 evaluations without improvement.
    restart_sigma: 0.012           # Normalized restart size.
```

# 10. Quick Overview of the ASA-GOW Flow

1. GOW reads the YAML file and validates the problem.
2. GOW creates ASA using the configured budget and parameters.
3. ASA generates a batch of candidates within the `bounds`.
4. GOW sends each candidate to the external evaluator.
5. The evaluator returns the objective value of each candidate.
6. GOW provides the results to ASA.
7. ASA accepts or rejects proposals, preserves the best candidate, and updates its search.
8. When appropriate, ASA applies reannealing or a soft restart.
9. The process continues until `max_evaluations` or `max_iterations` is reached.

# 11. Final Summary

ASA is a global-search optimizer for numerical parameters that combines an exploratory initial phase with a more selective final phase. In the YAML file, the user defines the objective, the parameters and their `bounds`, the evaluator, the GOW budget, and the six visible ASA settings. GOW coordinates candidate generation and evaluation; ASA decides how to continue the search based on the results it receives.
