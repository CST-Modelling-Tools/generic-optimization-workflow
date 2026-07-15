# PSO in GOW

## Guide 2: Using the Optimizer and Configuring the YAML

## Contents

1. What PSO Is
2. Inspiration or Main Idea of the Algorithm
3. From the Idea to the Algorithm: How PSO Works Within GOW
4. When It Makes Sense to Use PSO
5. How a PSO Run Is Controlled in GOW
6. How to Configure the YAML
7. Configurable PSO Parameters
8. Practical Recommendations
9. Commented Base YAML
10. Quick Overview of the PSO-GOW Flow
11. Final Summary

# 1. What PSO Is

PSO stands for Particle Swarm Optimization. It is a population-based stochastic optimizer that searches for good solutions using a set of candidates called particles.

Each particle represents a complete combination of values for the optimizable parameters. The set of particles forms the swarm. During the run, the swarm explores the ranges defined in the YAML and progressively concentrates the search around the solutions that have produced the best results.

In this GOW implementation, PSO can optimize numerical parameters of type `real` and `int`. Categorical parameters are not optimized directly.

# 2. Inspiration or Main Idea of the Algorithm

PSO is inspired by the collective behavior of groups such as bird flocks or fish schools. No individual knows the best direction in advance, but the group can orient itself by sharing information about the areas that have been most promising.

The metaphor is transferred to optimization through two types of memory:

- Each particle keeps the best position it has found, known as `pbest`.
- The swarm keeps the best position found by any of its particles, known as `gbest`.

To generate new candidates, each particle combines the direction in which it was already moving, its own experience, and the information shared by the swarm. In this way, PSO maintains a balance between exploring new areas and refining promising ones.

> **Core idea**
>
> Particles are not replaced through crossover or mutation. They move through the search space using their velocity, their best personal experience, and the best collective experience.

# 3. From the Idea to the Algorithm: How PSO Works Within GOW

Within GOW, PSO is responsible for proposing candidates, while the external evaluator is responsible for calculating the quality of each one.

1. PSO generates a batch of particles within the defined bounds.
2. GOW sends each candidate to the external evaluator.
3. The evaluator returns an objective value for each candidate.
4. PSO updates the `pbest` of each particle and the `gbest` of the swarm.
5. The particles move and form the next batch of candidates.
6. The process continues until one of the configured execution limits is reached.

In this implementation, each particle produces one candidate per generation. Therefore, `batch_size` also determines the swarm size.

# 4. When It Makes Sense to Use PSO

PSO makes sense when the problem is defined by numerical parameters with clear limits and the result can only be known by running an evaluator.

It is usually a suitable option when:

- The objective function is nonlinear, complex, or does not provide derivatives.
- There are several promising regions or possible local optima.
- Several candidates can be evaluated per batch.
- The parameters are real-valued or integer-valued and have physically valid bounds.

It is usually not the best option when most variables are categorical, when the budget allows only very few candidates to be evaluated, or when the problem requires a method that guarantees an exact solution.

> **Question answered by PSO**
>
> Which combination of numerical values within these ranges produces the best result according to my objective function?

# 5. How a PSO Run Is Controlled in GOW

The run is controlled mainly through `max_evaluations`, `batch_size`, and `max_generations`.

- `max_evaluations` defines the maximum evaluation budget allowed by GOW.
- `batch_size` defines how many candidates are evaluated in each batch and, in PSO, how many particles the swarm contains.
- `max_generations` defines how many complete generations PSO can perform.

One generation is equivalent to evaluating the entire swarm once. Therefore, the natural relationship between the three values is:

```text
max_evaluations = batch_size × max_generations
```

For example, with `max_evaluations: 1000` and `batch_size: 20`, a consistent run will have `max_generations: 50`.

GOW can stop the run because of `max_evaluations`, and PSO can stop it because of `max_generations`. If both limits are not consistent, the first one reached will stop the run. To avoid confusion, `max_evaluations` should be a multiple of `batch_size`, and `max_generations` should be exactly the result of that division.

# 6. How to Configure the YAML

The YAML must describe the objective, the optimizable parameters, the external evaluator, and the optimizer configuration. For PSO, the main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

## 6.1 `objective` Block

```yaml
objective:
  direction: minimize
```

`direction` indicates whether the objective must be minimized or maximized. For error, cost, or loss problems, `minimize` is normally used. For metrics that should increase, use `maximize`.

## 6.2 `parameters` Block

```yaml
parameters:
  x0:
    type: real
    value: 0.5
    bounds: [0.0, 1.0]

  x1:
    type: int
    value: 8
    bounds: [1, 20]
```

Each optimizable parameter must define its type, a reference value, and its bounds.

- `type` can be `real` or `int` for parameters that PSO will optimize.
- `value` is the reference value of the problem; in this implementation, it is not automatically inserted as an initial particle.
- `bounds` defines the allowed interval and must contain a lower limit smaller than the upper limit.

## 6.3 `evaluator` Block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

The external evaluator receives each candidate and returns the objective value. PSO does not calculate the objective function directly; GOW coordinates communication between the optimizer and the evaluator.

## 6.4 `optimizer` Block

```yaml
optimizer:
  name: pso
  seed: 123
  max_evaluations: 1000
  batch_size: 20
  settings:
    max_generations: 50
    inertia_weight: 0.729
    acceleration_coefficient: 1.49445
    velocity_clamp_fraction: 0.2
```

The fields `name`, `seed`, `max_evaluations`, and `batch_size` are part of the general execution control. The `settings` sub-block contains only the PSO hyperparameters exposed to the user by this implementation.

# 7. Configurable PSO Parameters

The following table includes only the parameters that the user must configure or review in the YAML. Decisions fixed internally in the implementation are not presented as configurable options.

| Parameter | Where It Goes | What It Controls | Usage Guide |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | Must be `pso`. |
| `seed` | `optimizer` | Fixes the pseudo-random sequence. | Use an integer when you need to repeat a run under equivalent conditions. |
| `max_evaluations` | `optimizer` | Defines the maximum GOW evaluation budget. | Must be a positive integer and should be a multiple of `batch_size`. |
| `batch_size` | `optimizer` | Defines candidates per batch and particles in the swarm. | A larger value increases diversity per generation but consumes more evaluations in each cycle. |
| `max_generations` | `settings` | Defines the maximum number of PSO generations. | For a consistent run, use `max_evaluations / batch_size`. |
| `inertia_weight` | `settings` | Controls how much previous movement each particle retains. | Increasing it makes particles retain more momentum; reducing it allows them to change direction more easily. |
| `acceleration_coefficient` | `settings` | Controls the strength of attraction toward `pbest` and `gbest`. | Increasing it makes particles respond more strongly to the best known positions. |
| `velocity_clamp_fraction` | `settings` | Limits the maximum step as a fraction of each parameter range. | A value of `1` represents 100% of the range; `0.2` represents 20%. Smaller values produce shorter movements. |

# 8. Practical Recommendations

## 8.1 Keep the Budget and Generations Consistent

Calculate `max_generations` from `max_evaluations` and `batch_size`. This prevents GOW and PSO from working with different limits and makes it easier to interpret how long the run will last.

## 8.2 Choose `batch_size` According to the Evaluator Cost

A larger swarm tests more areas per generation, but each generation requires more evaluations. When the evaluator is expensive, the batch size should be chosen together with the total budget and the available execution capacity.

## 8.3 Define Useful and Realistic Bounds

PSO can only search within the bounds. Bounds that are too narrow may exclude good solutions, while bounds that are too wide increase the space that the swarm must explore. The bounds should represent values that are valid for the real problem.

## 8.4 Use `seed` to Compare Runs

PSO uses random movements. Setting `seed` makes it possible to repeat the same pseudo-random sequence as long as the YAML, evaluator, evaluation order, and environment remain unchanged.

## 8.5 Adjust One Hyperparameter at a Time

To understand the effect of `inertia_weight`, `acceleration_coefficient`, and `velocity_clamp_fraction`, change one of them and compare the results while keeping the others fixed. Changing all of them at once makes it difficult to identify what caused an improvement or deterioration.

# 9. Commented Base YAML

The following example is generic and ready to be adapted to another problem. The names, values, bounds, evaluator path, and budget must be replaced according to the case.

```yaml
id: continuous-problem-pso

objective:
  direction: minimize  # Use maximize when the objective should increase.

parameters:
  x0:
    type: real          # Continuous parameter.
    value: 0.5          # Reference value of the problem.
    bounds: [0.0, 1.0] # Allowed search interval.

  x1:
    type: int           # Integer parameter.
    value: 8
    bounds: [1, 20]

evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600        # Maximum time allowed for one evaluation.

optimizer:
  name: pso             # Selects Particle Swarm Optimization.
  seed: 123             # Optional seed for reproducibility.
  max_evaluations: 1000 # Total budget controlled by GOW.
  batch_size: 20        # Particles and candidates per generation.

  settings:
    max_generations: 50               # 1000 / 20 = 50 generations.
    inertia_weight: 0.729             # Previous movement retained.
    acceleration_coefficient: 1.49445 # Attraction toward pbest and gbest.
    velocity_clamp_fraction: 0.2      # Maximum step: 20% of the range.
```

# 10. Quick Overview of the PSO-GOW Flow

> **Execution flow**
>
> 1. GOW reads the YAML.
> 2. GOW selects PSO and requests a batch of candidates.
> 3. PSO creates or moves the particles in the swarm.
> 4. GOW sends the candidates to the external evaluator.
> 5. The evaluator returns a result for each candidate.
> 6. GOW sends those results to PSO.
> 7. PSO updates `pbest`, `gbest`, and prepares the next generation.
> 8. The process continues until `max_evaluations` or `max_generations` is reached.

# 11. Final Summary

PSO is used to search for good combinations of numerical parameters through a swarm of candidates that learns from both individual and collective experience. The user defines the objective, the parameters and their bounds, the evaluator, the evaluation budget, the batch size, and the hyperparameters exposed in `settings`. GOW coordinates the execution and evaluations; PSO generates candidates, interprets the results, and moves the swarm until the configured limit is reached.
