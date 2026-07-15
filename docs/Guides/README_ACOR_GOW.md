# ACOR in GOW

## Guide 2: optimizer usage and YAML configuration

## Contents

1. What ACOR is
2. Biological inspiration: how ants find good paths
3. From the metaphor to the algorithm: how ACOR works in GOW
4. When it makes sense to use ACOR
5. How an ACOR run is controlled in GOW
6. How to configure the YAML
7. Configurable ACOR parameters
8. Practical recommendations
9. Commented base YAML
10. Quick reading of the ACOR-GOW flow

## 1. What ACOR is

ACOR means *Ant Colony Optimization for Continuous Domains*. It is a variant of ant colony optimization designed for problems where the variables to be adjusted are numerical values within a range.

In GOW, ACOR generates candidates formed by a complete combination of values for the optimizable parameters. GOW evaluates those candidates with an external evaluator and returns the result to the optimizer. With that information, ACOR decides which regions of the search space seem more promising.

## 2. Biological inspiration: how ants find good paths

Ant colony optimization is inspired by the behavior of real ants when they search for food. At the beginning, ants do not know which path is the best between the nest and the food source, so they explore different routes.

When an ant finds a useful path, it leaves a chemical signal called pheromone. That pheromone can be detected by other ants, which will have a higher probability of following paths where the signal is stronger. If that path leads to a good food source, more ants travel along it and the signal is reinforced.

In an optimization algorithm, this idea is transferred artificially. The ants are not real insects, but solution candidates. Each candidate represents a possible answer to the problem being optimized. After evaluating those candidates, the algorithm identifies which ones were better and uses that information to guide the following search.

In classical ant colony algorithms, this metaphor is usually applied to discrete problems, such as choosing paths in a graph or building routes between cities. ACOR adapts the same idea to continuous problems: instead of choosing among fixed paths, it searches inside numerical ranges.

## 3. From the metaphor to the algorithm: how ACOR works in GOW

The biological idea of the ant colony is transferred to the algorithm through candidates, evaluations, and memory of good solutions.

In ACOR, an artificial ant represents a solution candidate: a complete combination of values for the parameters to be optimized. Each candidate is built inside the limits defined in the YAML.

The general flow inside GOW is simple: ACOR generates a group of candidates, GOW sends those candidates to the external evaluator, and the evaluator returns a result for each one. That result indicates how good each candidate was according to the objective function defined by the user.

After receiving the evaluations, ACOR keeps the best solutions in an internal memory. That memory plays the role of pheromone: it does not store physical paths, but evaluated solutions that help guide the search.

From that memory, ACOR generates new candidates around the most promising solutions. The best solutions have greater influence, so the optimizer stops searching completely at random and starts concentrating on regions where good results have already been found.

1. ACOR generates candidates inside the defined ranges.
2. GOW evaluates those candidates with the external evaluator.
3. ACOR keeps the best solutions found.
4. ACOR generates new candidates around promising solutions.
5. The process repeats until the configured evaluation budget is reached.

## 4. When it makes sense to use ACOR

ACOR makes sense when the optimization problem is defined by numerical parameters that can vary within a range. That is, when continuous values such as coefficients, dimensions, constants, weights, angles, correction factors, or other real parameters need to be adjusted.

This optimizer is suitable when:

- the parameters have clear limits defined through `bounds`;
- the external evaluator can receive a complete candidate and return an objective value;
- the best combination of values is not known in advance;
- the goal is to explore the search space and, at the same time, progressively concentrate the search around the solutions that have produced better results.

ACOR is not mainly intended for categorical parameters without numerical order, such as names, labels, or classes. In this implementation, categorical parameters are not directly supported as optimizable variables.

**Question answered by ACOR**

> Which combination of numerical values within these ranges produces the best result according to my objective function?

## 5. How an ACOR run is controlled in GOW

For the GOW user, the most important thing is to control the run with two values: `max_evaluations` and `batch_size`.

- `max_evaluations` defines the total evaluation budget that GOW will allow to run.
- `batch_size` defines how many candidates GOW requests in each batch.

ACOR is not configured mainly through generations in the same sense as a genetic algorithm or differential evolution. In GOW, each batch follows this cycle: ACOR proposes candidates, GOW evaluates them, and ACOR updates its internal memory with the results.

```text
ask() -> ACOR proposes a batch of candidates
GOW -> evaluates those candidates
tell() -> ACOR receives results and updates its memory
```

In practice, each batch can be understood as an optimization cycle. Therefore, if the user wants to estimate how many practical cycles a run will have, the following can be used:

```text
approximate practical cycles = max_evaluations / batch_size
```

For example, with `max_evaluations: 10000` and `batch_size: 50`, GOW will work with approximately 200 batches. It is recommended that `max_evaluations` be a multiple of `batch_size` to avoid an incomplete final batch.

## 6. How to configure the YAML

The YAML must describe the objective, the optimizable parameters, the external evaluator, and the optimizer configuration. For ACOR, the main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

### 6.1 Objective block

```yaml
objective:
  direction: minimize
```

This block indicates whether the objective must be minimized or maximized. In error, cost, or loss problems, `minimize` is normally used. If the goal is to increase a performance metric, `maximize` is used.

### 6.2 Parameters block

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

Each optimizable parameter must have a type, a value, and limits. ACOR uses the `bounds` to know inside which range it can search.

- `type` indicates the parameter type. This implementation supports `real` and `int`.
- `value` is the reference value defined in the YAML.
- `bounds` defines the allowed range where ACOR can search.

### 6.3 Evaluator block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

This block indicates which external program will evaluate each candidate. ACOR does not calculate the objective by itself; GOW calls the evaluator and then returns the result to the optimizer.

### 6.4 Optimizer block

```yaml
optimizer:
  name: acor
  seed: 123
  max_evaluations: 10000
  batch_size: 50
  settings:
    q: 0.1
    xi: 0.85
    bound_strategy: clip
```

This block selects ACOR and defines how the search will run. The most important parameters are explained in the following section.

## 7. Configurable ACOR parameters

The following table summarizes the parameters that the user normally needs to review when configuring ACOR in GOW.

| Parameter | Where it goes | What it controls | Usage guide |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | Must be `acor` to use this implementation. |
| `seed` | `optimizer` | Fixes the optimizer's pseudorandom sequence. | Use it when you need reproducibility. If the environment or evaluation order changes, full reproducibility may be affected. |
| `max_evaluations` | `optimizer` | Defines the total evaluation budget. | It should be chosen according to the cost of the evaluator and the available time. |
| `batch_size` | `optimizer` | Defines how many candidates are evaluated per batch. | In ACOR, it can be understood as the number of ants per batch. |
| `q` | `settings` | Controls how much weight the best solutions in the archive receive. | It must be greater than 0. It is normally used between 0 and 1. Small values concentrate selection more strongly on the best solutions. |
| `xi` | `settings` | Controls the sampling amplitude around good solutions. | It must be greater than 0. It is normally used between 0 and 1. Larger values explore farther away. |
| `bound_strategy` | `settings` | Defines what to do if a sample falls outside the allowed range. | `clip` cuts it to the limit; `resample` tries to sample again before clipping. |

## 8. Practical recommendations

### 8.1 Use max_evaluations and batch_size as the main control

For a user guide, the clearest approach is to control ACOR with a total evaluation budget and a batch size. The user does not need to directly configure a number of generations.

```text
max_evaluations = batch_size x number_of_batches
```

### 8.2 Define coherent bounds

ACOR can only search inside the `bounds` defined in the YAML. If the limits are too narrow, the search is restricted. If they are too broad, the optimizer may need many evaluations to find promising regions.

### 8.3 Use seed when you need reproducibility

The seed controls the optimizer's pseudorandom sequence. To reproduce a run, the YAML, the `batch_size`, the `bounds`, the `ask`/`tell` order, and the conditions of the external evaluator must remain the same.

### 8.5 Interpret diagnostics carefully

ACOR may use an internal score to rank candidates uniformly, both in minimization and maximization. As a user, you should normally interpret the real objective value reported by GOW. The internal score is an auxiliary measure of the optimizer.

## 9. Commented base YAML

This example shows a generic configuration for a continuous problem. It must be adapted to the evaluator and to the real parameters of the problem.

```yaml
id: continuous-problem-acor

objective:
  direction: minimize  # Use maximize if the objective must increase.

parameters:
  x0:
    type: real
    value: 0.5
    bounds: [0.0, 1.0]
  x1:
    type: real
    value: 10.0
    bounds: [5.0, 15.0]

evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600

optimizer:
  name: acor
  seed: 123
  max_evaluations: 10000
  batch_size: 50
  settings:
    archive_size: 50
    q: 0.1
    xi: 0.85
    include_initial_candidate: false
    bound_strategy: clip
```

## 10. Quick reading of the ACOR-GOW flow

1. GOW reads the YAML.
2. GOW calls ACOR and asks it for a batch of candidates.
3. ACOR generates candidates inside the `bounds`.
4. GOW sends each candidate to the external evaluator.
5. The evaluator returns the objective value.
6. GOW gives those results to ACOR.
7. ACOR keeps the best solutions in its internal memory.
8. ACOR uses that memory to generate the next batch.
9. The process continues until `max_evaluations` is reached.

## Final summary

ACOR is useful for optimizing continuous numerical parameters. The user defines the ranges and the evaluation budget in the YAML; ACOR is responsible for exploring the search space and using the best solutions found to guide the following batches of candidates.
