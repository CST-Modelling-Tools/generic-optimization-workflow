# GA in GOW

**User guide for the optimizer and YAML configuration**

## Contents

1. What GA is
2. Main inspiration: evolution and natural selection
3. From the idea to the algorithm: how GA works within GOW
4. When it makes sense to use GA
5. How a GA execution is controlled in GOW
6. How to configure the YAML
   - 6.1 `objective` block
   - 6.2 `parameters` block
   - 6.3 `evaluator` block
   - 6.4 `optimizer` block
7. Configurable GA parameters
8. Practical recommendations
9. Commented base YAML
10. Quick overview of the GA-GOW workflow
11. Final summary

## 1. What GA is

GA stands for Genetic Algorithm. It is a population-based optimizer: instead of working with a single solution, it maintains and evaluates a complete set of candidates called a population.

Each candidate represents a complete combination of values for the optimizable parameters defined in the YAML. GA compares the results of these candidates and uses the best ones as the basis for building new solutions through selection, crossover, mutation, and elitism.

This GA implementation in GOW works with numerical parameters of type `real` and `int`, uses the `bounds` as search limits, and considers lower objective values to be better. Therefore, it is intended for minimization problems.

> **Question answered by GA**
> Which combination of numerical values within these ranges produces the lowest value of my objective function?

## 2. Main inspiration: evolution and natural selection

GA is inspired by natural evolution. In a biological population, better-adapted individuals are more likely to pass on their characteristics. In optimization, each individual is a candidate, and its fitness is measured using the value returned by the external evaluator.

The metaphor is translated into five simple ideas:

- **Population:** the set of candidates evaluated in one generation.
- **Selection:** choosing promising candidates to use as parents.
- **Crossover:** combining information from two parents to create a child.
- **Mutation:** randomly modifying some parameters to introduce variation.
- **Elitism:** directly preserving a proportion of the best candidates.

The goal is not to copy biological evolution exactly, but to balance two needs: exploiting regions that have already produced good results and continuing to explore new combinations.

## 3. From the idea to the algorithm: how GA works within GOW

Within GOW, GA does not calculate the objective function. Its responsibility is to propose candidates and use the received results to build the next population.

The first population is generated randomly within the `bounds`. The `value` fields in the YAML are retained as reference values for the problem, but they are not automatically inserted as an initial candidate in this implementation.

After evaluating a population, GA ranks the candidates, considering the lowest objective value to be the best. Based on this information, it preserves elites, selects parents through tournaments, combines their values, and applies mutations before providing the next generation to GOW.

1. GA generates a complete population within the `bounds`.
2. GOW sends each candidate to the external evaluator.
3. The evaluator returns an objective value for each candidate.
4. GA identifies the best results and preserves the elites.
5. GA creates new candidates through selection, crossover, and mutation.
6. The cycle repeats until the generation limit or GOW evaluation budget is reached.

## 4. When it makes sense to use GA

GA is suitable when the problem can be represented using bounded numerical parameters and the evaluator can receive a complete candidate and return a single objective value.

It is particularly appropriate when:

- the objective function is complex, nonlinear, discontinuous, or has no available derivatives;
- several regions of the search space should be explored at the same time;
- there is enough budget to evaluate populations over several generations;
- evaluations from the same population can be run in parallel;
- a global strategy that combines exploration and exploitation of good solutions is required.

GA is not usually the best option when each evaluation is extremely expensive and only a few trials can be performed, when categorical parameters are required without a suitable numerical encoding, or when the objective must be maximized without first adapting this implementation. For very small budgets, a model-based method may make better use of each evaluation.

## 5. How a GA execution is controlled in GOW

The execution is controlled through general GOW parameters and one limit specific to the Genetic Algorithm:

- `max_evaluations`: the maximum number of candidates that GOW may evaluate.
- `batch_size`: the number of candidates requested and evaluated in each generation.
- `generations`: the maximum number of generations that GA may complete.

> **Essential relationship**
> In this implementation, one generation evaluates a complete population. Therefore:
> `batch_size = candidates per generation = internal population size`

```text
planned evaluations = batch_size × generations
```

For example, with `batch_size: 100` and `generations: 500`, GA requires 50,000 evaluations to complete all generations.

`max_evaluations` and `generations` are two active limits. The execution ends when GOW exhausts the evaluation budget or when GA completes its generations. To avoid premature termination or leaving evaluations unused, it is recommended to configure them consistently:

```text
max_evaluations = batch_size × generations
```

`batch_size` must remain constant throughout the execution. There is no need to add `population_size` to the YAML: it exists only as a compatibility alias and represents exactly the same value as `batch_size`.

## 6. How to configure the YAML

The YAML describes the objective, the parameters that may vary, the external evaluator, and the optimizer configuration. For GA, the main blocks are `objective`, `parameters`, `evaluator`, and `optimizer`.

### 6.1 `objective` block

```yaml
objective:
  direction: minimize
```

This block indicates the optimization direction. The current GA implementation compares candidates by considering the lower value to be better, so `direction: minimize` must be used. Configuring `maximize` does not change the internal selection logic and would produce inconsistent behavior.

### 6.2 `parameters` block

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

Each optimizable parameter must define its type, a reference value, and its permitted limits.

- `type`: this implementation supports `real` and `int`.
- `value`: reference value for the problem; it is not automatically inserted into the initial population.
- `bounds`: the interval within which GA may generate, combine, and mutate values.

Categorical parameters are not directly supported as genes in this version.

### 6.3 `evaluator` block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

This block indicates which external program will evaluate each candidate. GOW runs the evaluator and provides GA with the resulting objective value. The evaluator must return a numerical result compatible with the metric defined by the problem.

### 6.4 `optimizer` block

```yaml
optimizer:
  name: genetic_algorithm
  seed: 123
  max_evaluations: 10000
  batch_size: 50
  settings:
    generations: 200
    elite_fraction: 0.05
    crossover_rate: 0.9
    mutation_rate: 0.2
    tournament_size: 3
```

The main block contains GOW's general execution controls. The `settings` sub-block contains the hyperparameters specific to GA.

## 7. Configurable GA parameters

The following table separates GOW's general controls from the hyperparameters specific to the Genetic Algorithm.

| Parameter | YAML location | What it controls | Usage guidance |
|---|---|---|---|
| `name` | `optimizer` | Selects the optimizer. | It must be `genetic_algorithm` to use this implementation. |
| `seed` | `optimizer` | Fixes the pseudorandom sequence. | Use it for reproducible executions. Keep the YAML, evaluator, and evaluation order unchanged as well. |
| `max_evaluations` | `optimizer` | Limits GOW's total evaluation budget. | It should match `batch_size × generations` when all generations are intended to be completed. |
| `batch_size` | `optimizer` | Defines the candidates per generation and the population size. | It must be an integer ≥ 1 and remain constant throughout the execution. |
| `generations` | `settings` | Defines the maximum number of GA generations. | Integer ≥ 1. Together with `batch_size`, it determines the evaluations required to complete the execution. |
| `elite_fraction` | `settings` | Determines what proportion of the candidates with the best results passes directly to the next generation. These candidates are preserved without crossover or mutation, so they protect the best solutions already found. | **How it is applied:** the number of elites depends on `elite_fraction` and the population size defined by `batch_size`.<br><br>**Low values:** preserve few candidates and leave more room for creating new children. They maintain greater diversity, but a good solution may be lost more easily.<br><br>**High values:** protect more good solutions, but reduce the number of new candidates and may cause the population to become repetitive or converge too early.<br><br>**Edge cases:** the implementation preserves at least one elite even when `0` is configured. With `1`, the entire population is copied unchanged and no new children are generated. |
| `crossover_rate` | `settings` | Controls, for each real-valued parameter of a child, the probability of combining the values of both parents. When crossover is applied, the child receives an intermediate value between them; when it is not applied, it retains the value from the first parent. | **How it is applied:** it is evaluated independently for each real-valued parameter, not once for the complete candidate.<br><br>**Low values:** produce children that are more similar to the first parent and reduce the mixing of information. The search changes more slowly and depends more heavily on mutation.<br><br>**High values:** more frequently generate intermediate values between promising parents. They favor refinement within good regions, although they do not add diversity beyond what the parents provide.<br><br>**Interpretation:** `0` disables the combination of real-valued parameters; `1` attempts to combine every real-valued parameter of each child. |
| `mutation_rate` | `settings` | Controls the probability of modifying each parameter of a child after crossover. For real-valued parameters, it introduces a local change; for integer parameters, it may choose another permitted value within the `bounds`. | **How it is applied:** the probability is checked parameter by parameter. For example, with 15 parameters and `mutation_rate = 0.2`, approximately 3 parameters per child will be modified on average.<br><br>**Low values:** better preserve the information inherited from the parents, but may leave the population without diversity and cause stagnation.<br><br>**High values:** increase exploration because more parameters are changed, but may destroy good combinations and make the search too random.<br><br>**Implementation detail:** real-valued mutation is local and uses a fixed scale equivalent to 5% of the parameter range; `mutation_rate` determines how many parameters mutate, not the size of each change. |
| `tournament_size` | `settings` | Defines how many candidates are randomly selected and compared each time GA needs to choose a parent. The candidate with the best result within that group wins the tournament and is used for reproduction. | **How it is applied:** a new tournament is performed every time a parent must be selected. It must be an integer between `1` and `batch_size`.<br><br>**Low values:** make selection more random. They maintain diversity and allow non-dominant candidates to reproduce, but apply less pressure toward the best solutions.<br><br>**High values:** increase the probability of selecting the best candidates. They may accelerate initial improvement, but can also reduce diversity and encourage premature convergence.<br><br>**Extreme cases:** with `1`, the parent is selected almost randomly. With a value close to `batch_size`, the best candidates from the entire population almost always win. |

> **About `selection`**
> The implementation exposes the `selection` parameter, but it currently accepts only `tournament`. Because it does not provide a real choice, the base YAML omits it and uses the default value.

> **About `population_size`**
> `population_size` is retained as a compatibility alias. In new YAML files, only `batch_size` should be configured because both represent the same size.

## 8. Practical recommendations

### 8.1 Align the budget and generations

First calculate the available budget and divide it between population size and number of generations. Maintain the equality `max_evaluations = batch_size × generations` so that both stopping criteria coincide.

### 8.2 Interpret `batch_size` as the population size

With a fixed budget, a larger `batch_size` provides more diversity and more candidates that can be parallelized per generation, but leaves fewer generations. A smaller `batch_size` allows more evolutionary cycles, although each population contains fewer alternatives and `tournament_size` becomes more limited.

### 8.3 Define useful `bounds`

GA can search only within the `bounds`. Limits that are too narrow exclude possible solutions; excessively wide limits disperse the population and may require many evaluations. The `value` field does not correct poorly defined `bounds` and is not used as an initial candidate.

### 8.4 Balance preservation and variation

`elite_fraction` protects good solutions, `tournament_size` controls selection pressure, `crossover_rate` mixes information from the parents, and `mutation_rate` introduces variation. These parameters should be interpreted together: too much preservation can make the population repetitive, while too much mutation can destroy useful structures.

`mutation_rate` is applied to each parameter, not once to the complete candidate. For example, in a problem with 20 parameters, a rate of `0.2` means that, on average, approximately four parameters of each child may mutate.

### 8.5 Use `seed` to compare configurations

A fixed `seed` makes it easier to compare hyperparameter changes under the same pseudorandom sequence. To assess the robustness of the final result, the configuration should subsequently be repeated with different seeds.

### 8.6 Validate a short execution first

Before launching a large budget, use a few generations to verify that the evaluator receives the candidates, returns the expected metric, and that `max_evaluations`, `batch_size`, and `generations` are consistent.

### 8.7 Avoid parameters that this version does not use

Do not add `include_initial_candidate` or `mutation_scale` to the YAML for this implementation: they are not active options. Do not configure `population_size` separately either. For selection, the tournament method is already applied by default.

## 9. Commented base YAML

This example is generic and must be adapted to the parameters, metric, and evaluator path of each problem.

```yaml
id: continuous-problem-ga

objective:
  direction: minimize          # This GA implementation treats lower values as better.

parameters:
  x0:
    type: real                 # Continuous numerical parameter.
    value: 0.5                 # Reference; it is not inserted as an initial candidate.
    bounds: [0.0, 1.0]         # Permitted interval for search and mutation.
  x1:
    type: int                  # Integer numerical parameter.
    value: 10
    bounds: [5, 15]

evaluator:
  command:
    ["/path/to/evaluator"]     # External program that calculates the objective.
  timeout_s: 600               # Maximum time permitted for one evaluation.

optimizer:
  name: genetic_algorithm      # Selects GA.
  seed: 123                    # Seed for reproducibility.
  max_evaluations: 10000       # GOW's total budget.
  batch_size: 50               # Population and candidates per generation.
  settings:
    generations: 200           # 50 × 200 = 10,000 evaluations.
    elite_fraction: 0.05       # Fraction of the best candidates preserved.
    crossover_rate: 0.9        # Crossover applied per real-valued parameter.
    mutation_rate: 0.2         # Mutation applied parameter by parameter.
    tournament_size: 3         # Candidates compared to select each parent.
```

## 10. Quick overview of the GA-GOW workflow

1. GOW reads the YAML and prepares the problem, evaluator, and optimizer.
2. GOW requests a batch of size `batch_size` from GA.
3. GA creates the random initial population or builds a new generation.
4. GOW sends each candidate to the external evaluator.
5. The evaluator returns an objective value for each candidate.
6. GOW provides the results to GA in the same order as the candidates.
7. GA preserves elites and generates new children through tournament selection, crossover, and mutation.
8. The process continues until `generations` or `max_evaluations` is reached.

## 11. Final summary

GA is a population-based optimizer useful for searching for combinations of numerical parameters within defined `bounds`. In the YAML, the user configures the minimization objective, parameters, evaluator, budget, population size, and evolutionary hyperparameters. GOW coordinates the external evaluation of the candidates; GA uses the results to preserve good solutions and build new generations through tournament selection, crossover, and mutation.
