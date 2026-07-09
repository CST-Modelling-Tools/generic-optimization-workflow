# PSO in GOW

## PSO Guide in GOW

**Optimizer behavior and YAML configuration**

**Document objective.** Explain how the PSO optimizer works inside GOW and how the YAML file is configured to use it in continuous or integer numerical problems. This guide focuses on the idea of the method, the swarm metaphor, and the optimizer-specific hyperparameters that modify the search.

## Contents

1. [What problem PSO solves](#1-what-problem-pso-solves)
2. [How PSO communicates with GOW](#2-how-pso-communicates-with-gow)
3. [Swarm, batches, and generations](#3-swarm-batches-and-generations)
4. [How PSO interprets the YAML parameters](#4-how-pso-interprets-the-yaml-parameters)
5. [What information PSO remembers during the search](#5-what-information-pso-remembers-during-the-search)
6. [How PSO generates new candidates](#6-how-pso-generates-new-candidates)
7. [How to interpret the evaluation result](#7-how-to-interpret-the-evaluation-result)
8. [How to configure the YAML in GOW](#8-how-to-configure-the-yaml-in-gow)
9. [Practical recommendations](#9-practical-recommendations)
10. [Appendix A. Conceptually commented base YAML](#appendix-a-conceptually-commented-base-yaml)
11. [Appendix B. Quick reading of the ask/tell flow](#appendix-b-quick-reading-of-the-asktell-flow)

## 1. What problem PSO solves

PSO stands for **Particle Swarm Optimization**. It is a population-based optimizer inspired by the collective behavior of a swarm. Instead of working with a single solution, PSO works with several particles at the same time.

Each particle represents a complete candidate: a set of values for all the parameters to be optimized. If the problem has 15 parameters, one particle contains 15 values, one for each parameter.

The metaphor is simple: several particles move through the search space. Each one remembers the best position it has found, and the swarm remembers the best position found by any particle. Using these two memories, PSO decides how to move the particles in the next generation.

```text
several particles test different regions of the search space
        ↓
each particle remembers where it performed best
        ↓
the swarm remembers the best global position found
        ↓
the particles move using those memories
        ↓
the swarm moves toward promising regions
```

**Key idea**

PSO proposes candidates. The external evaluator calculates the quality of each candidate. GOW returns that result to the optimizer. PSO updates the individual memory of each particle, the global memory of the swarm, and prepares the next generation.

## 2. How PSO communicates with GOW

Communication between GOW and the optimizer mainly occurs through the `ask/tell` flow. This flow separates candidate generation from external evaluation.

GOW calls `ask()` and requests a batch of candidates.

PSO returns the current swarm positions as candidates.

GOW runs the external evaluator for each candidate.

GOW calls `tell()` and sends the results back to the optimizer.

PSO interprets the results, updates `pbest`, updates `gbest`, and prepares the movement for the next generation.

**Important point in this implementation**

The `batch_size` parameter defines how many candidates GOW requests per batch. In PSO, each candidate corresponds to one particle, so `batch_size` also defines the swarm size during the execution.

## 3. Swarm, batches, and generations

This PSO implementation does handle an explicit notion of internal generation. One generation is equivalent to evaluating a complete batch of particles.

If `batch_size` is `20`, each generation evaluates 20 candidates.

```text
ask()
↓
PSO proposes the current swarm positions
↓
GOW evaluates all particles
↓
tell()
↓
PSO updates pbest, gbest, and the internal generation counter
```

The approximate number of evaluations controlled by PSO is:

```text
approximate evaluations = batch_size x max_generations
```

For example, if `batch_size` is `20` and `max_generations` is `50`, PSO will perform approximately `1000` evaluations, as long as GOW does not stop the execution earlier due to another external criterion.

**PSO and batch size**

PSO works through internal generations. Each generation evaluates the whole swarm. In this implementation, the batch size indicates how many particles are evaluated in each generation.

## 4. How PSO interprets the YAML parameters

PSO reads the optimizable parameters defined in the YAML and stores each parameter name, type, and limits. This information defines the space where the particles can move.

This implementation works directly in the real scale of each parameter. It does not transform all parameters into a normalized `[0, 1]` space to move the particles.

This is important because velocity is calculated using the real range of each parameter. For each dimension, the velocity limit is obtained as a fraction of the allowed range:

```text
vmax = velocity_clamp_fraction x (hi - lo)
```

This PSO implementation only optimizes numerical parameters: `real` and `int`. `categorical` parameters are not supported as optimizable variables because PSO needs to calculate numerical positions, velocities, and distances. However, a categorical parameter may appear in the YAML as a fixed value if it has `optimizable: false`.

| Field | Interpretation |
|---|---|
| `type` | Type of the parameter. This PSO implementation supports optimizable `real` and `int` parameters. `categorical` parameters can only be used as fixed values if they have `optimizable: false`. |
| `value` | Reference value defined in the YAML for the problem. PSO performs the search mainly using the parameter `bounds`; this value should only be interpreted as an initial candidate if an advanced `warm_start` option is enabled. |
| `bounds` | Allowed range where PSO can search. It must have a lower and upper limit. |

## 5. What information PSO remembers during the search

PSO does not move the particles completely at random. During optimization, the swarm keeps two types of memory:

**Individual memory:** each particle remembers the best position it has found so far. This position is known as `pbest`.

**Collective memory:** the swarm remembers the best position found by any particle. This position is known as `gbest`.

The difference between both memories is fundamental:

```text
pbest
↓
individual memory of each particle

gbest
↓
collective memory of the whole swarm
```

When GOW returns the results of a generation, PSO checks whether any particle improved its own best position. If it did, PSO updates its `pbest`. If that particle also found a solution better than all known solutions in the swarm, PSO also updates `gbest`.

These two memories guide the movement of the particles in the following generations.

## 6. How PSO generates new candidates

The first generation and the following generations are not built in the same way.

### 6.1 First generation

The first generation is created before having evaluated results. Therefore, there are still no useful `pbest` or `gbest` values.

In the first call to `ask()`, PSO creates the initial particles randomly inside the `bounds` defined for each parameter. It also creates random initial velocities, limited by `velocity_clamp_fraction`.

At the beginning, each particle keeps its own initial position as `pbest`, but it still does not have a valid score. That score arrives later, when GOW evaluates the candidates and calls `tell()`.

```text
YAML bounds
↓
create random initial positions
↓
create random initial velocities
↓
return candidates to GOW
```

### 6.2 Later generations

After the first evaluation, PSO already knows which particles performed best. From that point on, each new generation is produced by moving the swarm.

```text
current position
↓
calculate new velocity
↓
limit velocity with velocity_clamp_fraction
↓
apply movement
↓
correct position if it goes outside the bounds
↓
return new candidate to GOW
```

### 6.3 Velocity control and limits

After calculating the new velocity, the code applies a maximum limit called `vmax`.

```text
vmax = velocity_clamp_fraction x (hi - lo)
```

If `velocity_clamp_fraction` is high, the particle can take large steps. If it is low, the particle moves with smaller steps.

If a particle leaves the allowed range, this implementation places the position at the corresponding limit and resets the velocity of that dimension to zero.

```text
position below the lower bound
↓
position = lower bound
↓
velocity in that dimension = 0

position above the upper bound
↓
position = upper bound
↓
velocity in that dimension = 0
```

For integer parameters, PSO internally calculates movement as a continuous value, but when returning the candidate to GOW, the value is rounded and kept inside its `bounds`.

## 7. How to interpret the evaluation result

During execution, PSO proposes candidates and GOW evaluates them using the external evaluator defined in the YAML. At the end of the optimization, the user must review the best candidate found and the associated objective value.

The final result indicates which parameter combination obtained the best performance according to the metric defined in the problem.

In minimization problems, the best result is the candidate with the lowest objective value. In maximization problems, it is the candidate with the highest objective value.

Therefore, before running PSO, it is important to correctly define the objective direction:

```yaml
objective:
  direction: minimize
```

## 8. How to configure the YAML in GOW

The YAML must describe the problem, the optimizable parameters, the external evaluator, and the optimizer configuration. For PSO, the main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

### 8.1 `objective` block

```yaml
objective:
  direction: minimize
```

This block indicates whether the objective should be minimized or maximized. In an error, cost, or distance problem, `minimize` is normally used. In a problem where the goal is to increase a performance metric, `maximize` is used.

### 8.2 `parameters` block

```yaml
parameters:
  x0:
    type: real
    value: 0.5
    bounds: [0.0, 1.0]
  x1:
    type: real
    value: 10.0
    bounds: [5.0, 15.0]
```

Each optimizable parameter must define its type, a reference value, and its `bounds`. PSO uses the `bounds` to generate initial positions, limit velocities, and keep particles inside the allowed space.

### 8.3 `evaluator` block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

This block indicates which external program will evaluate each candidate. PSO does not calculate the objective; the evaluator calculates it and GOW returns the result to the optimizer.

### 8.4 `optimizer` block: general GOW parameters

The `optimizer` block indicates which optimizer GOW will use and defines the general execution parameters. These fields are not exclusive to PSO: they also appear when configuring other optimizers inside GOW.

The general GOW parameters must be written directly inside `optimizer`, at the same indentation level:

```yaml
optimizer:
  name: pso
  seed:
  max_evaluations:
  batch_size:
```

#### `name`

Selects the optimizer to use. For this Particle Swarm Optimization implementation, use:

```yaml
name: pso
```

#### `seed`

Defines the random seed of the execution. Using the same seed allows repeating an execution under equivalent conditions.

#### `max_evaluations`

Defines the maximum number of evaluations that GOW will allow during the optimization. In PSO, it should be configured consistently with `batch_size` and `max_generations`, because one generation corresponds to evaluating the whole swarm.

```text
max_evaluations = batch_size x max_generations
```

#### `batch_size`

`batch_size` defines the number of candidates that GOW requests per batch. In PSO, each candidate corresponds to one particle; therefore, this parameter also defines how many particles the swarm has during the execution.

It controls the number of particles, the number of candidates per generation, and the initial diversity of the swarm.

If it is increased, the swarm tests more regions in each generation. This improves diversity and can help in problems with many local minima, but it consumes more evaluations per generation.

If it is reduced, each generation is cheaper, but the swarm explores fewer regions at the same time.

### 8.5 `settings` block: PSO-specific hyperparameters

The `settings` sub-block contains the hyperparameters specific to the selected algorithm. In this case, the PSO-specific parameters.

These hyperparameters modify how the swarm moves, how many internal generations it can perform, and how the particle velocity is limited.

```yaml
settings:
  max_generations:
  inertia_weight:
  acceleration_coefficient:
  velocity_clamp_fraction:
```

#### 8.5.1 `max_generations`

`max_generations` defines the maximum number of internal PSO generations.

One generation corresponds to evaluating the whole swarm. Therefore, this parameter controls how many times the swarm can move and learn from its results.

**What it controls:** number of PSO internal cycles, number of times `pbest` and `gbest` are updated, and opportunities to improve the search.

If it is increased, the swarm has more time to move toward promising regions and refine results.

If it is reduced, the execution ends earlier and may remain in a very initial phase.

It is useful to modify it when the total evaluation budget should be increased or reduced.

```text
max_evaluations ≈ batch_size x max_generations
```

#### 8.5.2 `inertia_weight`

`inertia_weight` controls how much of the previous velocity a particle keeps. In simple terms, it indicates how much the particle continues moving in the direction it was already following.

It controls the memory of the previous movement, the continuity of displacement, and the balance between exploration and local search.

If it is increased, the particles keep more momentum. This can favor exploration because particles move further and may cross wide regions of the search space. If it is increased too much, they may oscillate, hit the `bounds`, or take longer to stabilize.

If it is reduced, particles lose momentum and become more dependent on `pbest` and `gbest`. This can favor a more local and stable search, but if it is reduced too much, the swarm may move very little and stagnate.

As a practical reference, the value `1.0` can be interpreted as keeping 100% of the previous velocity. Values lower than `1.0` progressively reduce that conservation, while `0.0` means that the particle does not keep any previous velocity. In contrast, values greater than `1.0` amplify the previous velocity.

#### 8.5.3 `acceleration_coefficient`

The `acceleration_coefficient` parameter controls the intensity with which particles adjust their velocity toward the best known positions. In this simplified implementation, the same value is used for the attraction toward each particle's personal best position (`pbest`) and toward the swarm's global best position (`gbest`).

As a practical reference, the value `2.0` can be interpreted as an average correction close to 100% of the distance toward the reference position, either `pbest` or `gbest`. This happens because the attraction term is internally multiplied by a random number between `0` and `1`, whose average value can be considered approximately `0.5`.

Therefore, values lower than `2.0` produce a more moderate attraction, while values greater than `2.0` can make the particle more likely to overshoot the reference position. The value `0.0` disables the attraction toward the best known positions, so the particle uses neither its personal memory nor the swarm's global information to adjust its velocity.

#### 8.5.4 `velocity_clamp_fraction`

`velocity_clamp_fraction` defines the maximum allowed velocity as a fraction of the range of each parameter. This parameter controls the maximum step length that a particle can take in each dimension.

**What it controls:** fraction of the parameter range, maximum allowed velocity, and maximum particle jump size.

```text
vmax = velocity_clamp_fraction x (hi - lo)
```

For example, if a parameter has `bounds [0.0, 100]` and `velocity_clamp_fraction` is `0.2`, the maximum velocity for that parameter will be `20.0`.

If it is increased, particles can take larger jumps. This favors exploration, but it can also make particles hit the limits more often.

If it is reduced, particles move with smaller steps. This favors a finer search, but if it is reduced too much, the swarm may advance slowly or become trapped.

It is useful to modify it when the swarm jumps too much between regions or when it moves so little that it cannot explore.

## 9. Practical recommendations

### 9.1 Plan evaluations with `batch_size` and `max_generations`

A simple way to plan the budget is:

```text
max_evaluations = batch_size x max_generations
```

For example, with `batch_size` equal to `30` and `max_generations` equal to `100`, the natural budget will be approximately `3000` evaluations.

`max_evaluations` remains a general GOW criterion. If it is configured below that natural budget, GOW may stop the execution before completing all internal PSO generations.

For PSO, it is recommended to avoid incomplete final batches. The clearest option is to configure `max_evaluations` as an exact multiple of `batch_size`.

### 9.2 Adjust only a few parameters at first

It is not convenient to change all hyperparameters at the same time. To understand PSO behavior, it is better to modify one or two parameters and compare results.

If the swarm explores too little, `batch_size` can be increased, `inertia_weight` can be slightly increased, or `velocity_clamp_fraction` can be increased carefully.

If the swarm jumps too much or sticks too often to the limits, `velocity_clamp_fraction` or `inertia_weight` can be reduced.

If the swarm converges too quickly to a region that does not improve, `acceleration_coefficient` can be reduced or `batch_size` can be increased to improve diversity.

If the swarm moves very slowly toward promising regions, `acceleration_coefficient` or `velocity_clamp_fraction` can be slightly increased.

### 9.3 Keep physically reasonable bounds

If the `bounds` are too narrow, the optimizer is limited. If they are too wide, the swarm may need many generations to find promising regions.

The `bounds` should represent allowed and reasonable values for the real problem, not just large ranges for convenience.

## Appendix A. Conceptually commented base YAML

The following YAML is a conceptual example. The numerical values shown are illustrative and must be adapted to each optimization problem.

```yaml
id: continuous-problem-pso

objective:
  direction: minimize # Change to maximize if the objective should increase.

parameters:
  x0:
    type: real # PSO supports real numerical parameters.
    value: 0.5 # Reference value defined in the YAML.
    bounds: [0.0, 1.0] # Range where PSO can search.
  x1:
    type: real
    value: 10.0
    bounds: [5.0, 15.0]

evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600 # Maximum time allowed for one evaluation.

optimizer:
  name: pso # Selects the PSO optimizer.
  seed: 123 # Seed for reproducibility.
  max_evaluations: 1000 # Total evaluation budget.
  batch_size: 20 # Number of particles/candidates per generation.
  settings:
    max_generations: 50 # Maximum number of internal generations.
    inertia_weight: 0.7 # Weight of the previous velocity.
    acceleration_coefficient: 1.5 # Attraction toward pbest and gbest.
    velocity_clamp_fraction: 0.2 # Maximum velocity relative to the range.
```

## Appendix B. Quick reading of the ask/tell flow

```text
ask(problem, n)
↓
if PSO is not initialized yet, use n as the batch/swarm size
↓
read parameters, types, bounds, and objective direction
↓
create initial positions and velocities inside bounds
↓
return the current positions as candidates
↓
GOW evaluates the candidates with the external evaluator
↓
tell(candidates, results)
↓
PSO interprets each result according to objective.direction
↓
update the pbest of each particle if it improves
↓
update gbest if a new global best solution appears
↓
increment the internal generation counter
↓
in the next call to ask(), move the swarm using velocity, pbest, and gbest
```
