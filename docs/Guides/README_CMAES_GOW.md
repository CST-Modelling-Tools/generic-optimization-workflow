# CMA-ES in GOW

**Optimizer Usage and YAML Configuration Guide**

## Contents

1. What CMA-ES is
2. Core idea of the algorithm: a distribution that learns
3. From the idea to the algorithm: how CMA-ES works within GOW
4. When it makes sense to use CMA-ES
5. How to control a CMA-ES run in GOW
6. How to configure the YAML
   1. `objective` block
   2. `parameters` block
   3. `evaluator` block
   4. `optimizer` block
7. Configurable CMA-ES parameters
8. Practical recommendations
9. Commented base YAML
10. Quick overview of the CMA-ES-GOW flow
11. Final summary

## 1. What CMA-ES is

CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy. It is a stochastic evolutionary strategy for numerical optimization, especially useful when several real-valued parameters must be adjusted and derivatives of the objective function are not available.

Instead of testing unrelated points, CMA-ES maintains a search distribution. In each generation, it produces a population of candidates, observes which candidates obtain the best results, and adapts the distribution to guide subsequent generations toward more promising regions.

This ability to learn relationships between parameters makes CMA-ES especially useful for nonlinear, nonconvex, poorly scaled problems or problems in which variables influence one another.

## 2. Core idea of the algorithm: a distribution that learns

The central idea of CMA-ES is to search using a multivariate normal distribution that changes during the run. Conceptually, this distribution is described by three elements:

- The mean indicates the current center of the search.
- Sigma controls the overall scale of the movements around that center.
- The covariance matrix adjusts the shape and orientation of the search, allowing the optimizer to learn directions and relationships between parameters.

At the beginning, the distribution is built around the `value` entries defined in the YAML. After evaluating a population, the best candidates influence the new center and the shape of the distribution. In this way, CMA-ES can expand, reduce, or redirect exploration according to the information obtained during the run.

> **Key idea**
> CMA-ES does not only learn where it should search. It also learns what search scale to use and in which directions it is useful to move several parameters together.

## 3. From the idea to the algorithm: how CMA-ES works within GOW

Within GOW, CMA-ES is integrated through a generation-based cycle. The optimizer proposes a complete population, GOW evaluates each candidate using the external program, and then returns the results so that CMA-ES can update its search distribution.

1. GOW requests a batch of candidates according to `batch_size`.
2. CMA-ES generates that population around its current distribution.
3. GOW sends each candidate to the external evaluator.
4. The evaluator returns an objective value for each candidate.
5. GOW delivers the results to CMA-ES.
6. CMA-ES updates the mean, sigma, and covariance before generating the next population.

In this integration, a generation is considered complete only after the results of the entire population have been received. Therefore, `batch_size` also acts internally as the population size and must remain constant throughout the run.

```text
ask()  -> CMA-ES proposes batch_size candidates
GOW    -> evaluates the complete population
tell() -> CMA-ES receives the results and updates the search
```

## 4. When it makes sense to use CMA-ES

CMA-ES makes sense when the problem is mainly composed of numerical parameters with defined `bounds`, and the goal is to find a combination that minimizes or maximizes an objective function calculated by an external evaluator.

It is usually a good option when:

- the variables are real-valued or can reasonably be treated as numerical;
- the objective function is nonlinear, nonconvex, or does not provide derivatives;
- relationships may exist between parameters and learning coordinated movements is useful;
- there is enough evaluation budget to assess several populations.

It is not usually the best choice when the problem is mainly categorical or combinatorial. This implementation supports `real` and `int` parameters, but it does not support optimizable categorical parameters. Integer values are generated from a continuous space and then rounded, so CMA-ES remains essentially a continuous method.

> **Question CMA-ES answers**
> Which combination of numerical values within these ranges produces the best result, and which directions in the search space should be explored more intensively?

## 5. How to control a CMA-ES run in GOW

The run is controlled through three related values: `max_evaluations`, `batch_size`, and `max_generations`.

- `max_evaluations` is the maximum evaluation budget allowed by GOW.
- `batch_size` is the number of candidates evaluated in each generation. In this implementation, it also defines the internal CMA-ES population size.
- `max_generations` is the maximum number of complete generations allowed by the optimizer.

```text
evaluations associated with the generation limit = batch_size x max_generations
```

For example, with `batch_size: 16` and `max_generations: 100`, the optimizer limit corresponds to 1,600 evaluations. To make both controls coincide, `max_evaluations: 1600` can be configured.

The run may end earlier if GOW reaches `max_evaluations` or if the internal CMA-ES library activates one of its stopping criteria. If `max_evaluations` is greater than `batch_size x max_generations`, CMA-ES will stop because of the generation limit before consuming the entire GOW budget.

`batch_size` must be at least 2 and must remain constant. It is also advisable for `max_evaluations` to be a multiple of `batch_size` so that the optimizer always works with complete populations.

## 6. How to configure the YAML

The YAML must describe the objective, the optimizable parameters, the external evaluator, and the optimizer configuration. For CMA-ES, the main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

### 6.1 `objective` block

```yaml
objective:
  direction: minimize
```

`direction` indicates whether the objective value must be minimized or maximized. For error, cost, or loss problems, `minimize` is normally used. When the goal is to increase a performance metric, `maximize` is used. This direction should be defined explicitly.

### 6.2 `parameters` block

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

Each optimizable parameter must have `type`, `value`, and `bounds`.

- `type` indicates the parameter type. This implementation supports `real` and `int`.
- `value` is used to build the initial mean of the CMA-ES distribution.
- `bounds` defines the allowed interval in which the optimizer can search.

The implementation converts each parameter into a normalized coordinate between 0 and 1. This allows parameters with very different real-world scales to be handled internally on a common scale. Candidates are converted back to their real values before being sent to the evaluator.

Integer parameters are generated internally as continuous values, converted to their real range, and rounded to the nearest permitted integer. Categorical parameters are not directly supported.

> **Important**
> The `value` entries build the initial mean of CMA-ES. The exact candidate written in the YAML is not automatically evaluated as the first candidate.

### 6.3 `evaluator` block

```yaml
evaluator:
  command: ["/path/to/evaluator"]
  timeout_s: 600
```

This block specifies which external program evaluates each candidate. CMA-ES does not calculate the objective function itself. GOW runs the evaluator and returns the result to the optimizer.

### 6.4 `optimizer` block

```yaml
optimizer:
  name: cmaes
  seed: 123
  max_evaluations: 1600
  batch_size: 16
  settings:
    sigma0: 0.05
    max_generations: 100
```

The general execution parameters are placed directly inside `optimizer`. CMA-ES-specific hyperparameters are placed inside `settings`.

`population_size` must not be added to the YAML. In this implementation, the population size is obtained directly from `batch_size` to avoid having two different configurations for the same quantity.

## 7. Configurable CMA-ES parameters

The following table includes only the parameters exposed by this integration and required to configure the run.

| Parameter | YAML location | What it controls | Usage guidance |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | Use `cmaes` for this implementation. |
| `seed` | `optimizer` | Fixes the pseudorandom sequence. | Use an integer when reproducibility is required. Keep the YAML, environment, and evaluation order unchanged as well. |
| `max_evaluations` | `optimizer` | Defines the maximum budget allowed by GOW. | Choose it according to the evaluator cost. It should preferably be a multiple of `batch_size`. |
| `batch_size` | `optimizer` | Defines the number of candidates per generation and the internal population size. | It must be `>= 2` and remain constant throughout the run. Do not configure `population_size` separately. |
| `sigma0` | `settings` | Defines the initial distribution scale in the normalized `[0, 1]` space. | It must be `> 0`. The default value is `0.05`. Increasing it broadens the initial search; reducing it makes the search more local. |
| `max_generations` | `settings` | Limits the number of complete generations. | It must be `>= 1`. The default value is `100`. Adjust it consistently with `batch_size` and `max_evaluations`. |

The internal adaptation of the mean, sigma, covariance, and evolution paths is managed by the `cma` library. These details are not configured in the YAML for this implementation.

## 8. Practical recommendations

### 8.1 Align the budget with the generations

Before running the optimization, calculate `batch_size x max_generations`. To make GOW and CMA-ES stop at approximately the same point, configure `max_evaluations` with that result. Avoid budgets that leave an incomplete population.

### 8.2 Choose `sigma0` according to confidence in the initial point

`sigma0` is interpreted in the normalized `[0, 1]` space. A value of `0.05` represents a small initial scale relative to the complete range. A value of `1.0` corresponds to the full normalized scale, but it does not mean that every candidate will move exactly across the entire range.

When the `value` entries represent a good reference, it usually makes sense to begin with a more local search. When they are only a technical starting point, increasing `sigma0` may be useful to explore farther away.

### 8.3 Define useful `bounds` with positive width

CMA-ES can search only within the defined `bounds`. Bounds that are too narrow may prevent improvements, while excessively wide bounds may require many more evaluations. Each optimizable parameter must have a lower bound that is smaller than its upper bound for normalization to be valid.

### 8.4 Use `seed` when reproducibility is required

The seed helps reproduce the candidate sequence. To compare runs, the `bounds`, `batch_size`, `max_generations`, evaluator, and environment conditions must also remain unchanged.

### 8.5 Ensure valid evaluator results

Each candidate must return a finite numerical objective value. When an evaluation fails, the result is missing, or a non-numerical value appears, this implementation penalizes it with a very large loss so that CMA-ES treats it as a very poor solution. This penalty should not be relied on as normal behavior.

### 8.6 Evaluate an exact reference separately when necessary

The YAML `value` entries define the initial center of the distribution, but they do not guarantee that this exact combination will be evaluated. When a comparison with a specific reference is required, evaluate it explicitly through the GOW evaluation workflow.

## 9. Commented base YAML

This example shows a generic and consistent configuration for a continuous problem. It must be adapted to the evaluator and the actual parameters of the problem.

```yaml
id: continuous-problem-cmaes

objective:
  direction: minimize          # Use maximize if the objective must increase.

parameters:
  x0:
    type: real                 # CMA-ES supports real and int parameters.
    value: 0.5                 # Used to build the initial mean.
    bounds: [0.0, 1.0]         # Allowed search interval.
  x1:
    type: real
    value: 10.0
    bounds: [5.0, 15.0]

evaluator:
  command: ["/path/to/evaluator"]
  timeout_s: 600               # Maximum time allowed for one evaluation.

optimizer:
  name: cmaes                  # Selects CMA-ES.
  seed: 123                    # Seed for reproducibility.
  max_evaluations: 1600        # Maximum budget managed by GOW.
  batch_size: 16               # Candidates per generation and internal population size.
  settings:
    sigma0: 0.05               # Initial scale in the normalized space.
    max_generations: 100       # Limit of complete generations.
```

## 10. Quick overview of the CMA-ES-GOW flow

1. GOW reads the YAML and obtains the objective, parameters, evaluator, and CMA-ES configuration.
2. GOW calls the optimizer and requests `batch_size` candidates.
3. CMA-ES generates a complete population in the normalized space and converts it to the real values defined by the `bounds`.
4. GOW sends each candidate to the external evaluator.
5. The evaluator returns the objective value for each candidate.
6. GOW delivers the population and its results to CMA-ES.
7. CMA-ES updates the mean, sigma, and covariance, completing one generation.
8. The process continues until `max_evaluations`, `max_generations`, or an internal stopping criterion is reached.

## 11. Final summary

CMA-ES optimizes numerical parameters through a search distribution that learns where to explore, at what scale, and in which directions. The YAML defines the objective, the `value` entries and `bounds`, the evaluator, the budget, `batch_size`, `sigma0`, and `max_generations`. GOW coordinates the external evaluations, while CMA-ES adapts the search using their results. In this implementation, `batch_size` also defines the internal population, and the `value` entries establish the initial mean rather than a candidate that is automatically evaluated.
