# ASA Guide in GOW

*Optimizer behavior and YAML configuration*

Document goal. Explain how the ASA optimizer works inside GOW and how to configure the YAML file to use it with continuous or integer numerical problems. This guide focuses on the method, the workflow with GOW, and the practical configuration of the parameters.

## Contents

- 1. What problem ASA solves
- 2. How ASA communicates with GOW
- 3. Batches, iterations, and evaluation budget
- 4. How ASA interprets YAML parameters
- 5. What information ASA remembers during the search
- 6. How ASA generates new candidates
- 7. How the evaluation result is interpreted
- 8. How to configure the YAML in GOW
- 9. ASA-specific parameters in `settings`
- 10. Practical recommendations
- Appendix A. Conceptually commented base YAML
- Appendix B. Quick reading of the ask/tell flow

## 1. What problem ASA solves

ASA stands for Adaptive Simulated Annealing. It is an adaptive variant of simulated annealing designed to search for good solutions in numerical parameter spaces. In GOW, ASA does not compute the objective function directly: it proposes candidates, GOW evaluates them with an external evaluator, and ASA uses those results to continue the search.

Simulated annealing is an optimization technique inspired by the slow cooling of materials. In a real material, heating and slowly cooling allows particles to reorganize into a more stable structure. In optimization, that idea translates into starting with a freer search and ending with a stricter search.

The difference from a purely greedy search is that simulated annealing does not accept only better candidates. At the beginning, when the temperature is high, it can also accept some worse candidates. This helps escape local minima. As the temperature decreases, the algorithm accepts fewer worsening moves and focuses on refining the solution found.

> **Key idea**
>
> ASA proposes candidates. The external evaluator computes the quality of each candidate. GOW returns that result to the optimizer. ASA then decides whether to accept, reject, or record the candidate as the best solution found.

## 2. How ASA communicates with GOW

Communication between GOW and the optimizer mainly happens through the ask/tell flow. This flow separates candidate generation from external evaluation.

```text
GOW calls ask()
        ↓
ASA returns a batch of candidates
        ↓
GOW runs the external evaluator for each candidate
        ↓
GOW calls tell()
        ↓
ASA interprets results, accepts or rejects candidates, and updates the search
```

This means ASA does not need to know the internal details of the evaluator. Its task is to propose parameter combinations. GOW's task is to send those combinations to the external evaluator and return the numerical result to ASA.

## 3. Batches, iterations, and evaluation budget

ASA does not work with population generations in the same sense as PSO or a genetic algorithm. There is no swarm and no population with individual memory. What exists is a batch-based cycle inside the GOW workflow.

```text
ask()
        ↓
ASA proposes a batch of candidates
        ↓
GOW evaluates those candidates with the external evaluator
        ↓
tell()
        ↓
ASA processes results, accepts or rejects candidates, and updates the search
```

In the GOW configuration, the main controls should be `max_evaluations` and `batch_size`. The first defines how many evaluations are allowed in total. The second defines how many candidates are requested per batch.

For example, if `max_evaluations` is 1000 and `batch_size` is 25, GOW will work with approximately 40 batches, as long as no other external stopping criterion ends the run earlier.

In ASA, a batch should not be interpreted as a population generation. It should be understood as an evaluation batch inside GOW's ask/tell flow.

> **ASA and batch mode**
>
> Candidates in the same batch are generated before their results are received. ASA cannot update its search between candidates in the same batch because the results arrive together in `tell()`. This is a practical adaptation for GOW and for parallel executions.

## 4. How ASA interprets YAML parameters

ASA takes the optimizable parameters defined in the YAML from three elements: the initial value, the lower bound, and the upper bound. These elements define the search space where the optimizer can propose candidates.

Although each parameter may have a different scale, ASA works internally with a normalized representation between 0 and 1. This makes it possible to handle parameters with very different ranges in a comparable way. For example, a parameter ranging from 0 to 1 and another ranging from 100 to 500 are transformed to the same internal scale before movements are generated.

The YAML `value` entries are used as the initial reference for the search. That initial value is not considered good or bad by itself; it only serves as the starting point. Its quality is known only when GOW evaluates it with the external evaluator.

ASA supports real and integer parameters. For integer parameters, the movement is generated internally as a continuous value and then rounded to the nearest allowed integer. Categorical parameters are not directly supported by this implementation.

## 5. What information ASA remembers during the search

ASA does not work with a swarm or a solution archive like other optimizers. It mainly keeps a current candidate, the best candidate found, and a set of temperatures that control the search.

The current state is the candidate from which ASA generates new proposals. When a proposal is accepted, it becomes the new starting point. If the proposal is rejected, ASA keeps the previous candidate and continues searching from there.

The best candidate found is stored separately. This is important because ASA may temporarily accept worse candidates in order to explore, but it does not lose the best solution found so far.

Temperatures control the balance between exploration and exploitation. The cost temperature influences the probability of accepting worse candidates. The parameter temperatures influence the size of the jumps used to generate new candidates.

Reannealing allows the search to be readjusted during the run. Instead of always keeping the same movement scales, ASA can adapt its behavior according to the acceptance rate or the observed parameter sensitivity.

## 6. How ASA generates new candidates

Once ASA has the parameters represented internally, it generates new candidates by modifying part of them. It does not necessarily change all parameters in every proposal. The number of parameters modified depends on `mutation_parameter_fraction_start` and `mutation_parameter_fraction_end`.

At the beginning of the search, ASA modifies a larger fraction of parameters to favor exploration. This makes it possible to test broader combinations and move more freely through the search space. As the run progresses, the fraction of modified parameters decreases, favoring a more local and finer search.

The starting point for the first candidates is the set of `value` entries defined in the YAML. The `include_initial_candidate` parameter decides whether those values are evaluated as an exact candidate. If `include_initial_candidate` is `true`, ASA sends that initial candidate directly to GOW for evaluation. If `include_initial_candidate` is `false`, ASA does not evaluate that exact candidate, but starts by generating variations around that initial reference.

> **Important difference from other optimizers**
>
> `include_initial_candidate` only controls whether the YAML `value` entries are sent as an exact candidate for evaluation. It does not, by itself, define how the rest of the first batch is generated. In ASA, even if `include_initial_candidate` is `false`, the `value` entries still act as the initial reference for generating variations.

After receiving evaluations, ASA generates new proposals using the accepted current candidate as a reference. If a proposal is accepted, it becomes the new point from which the next proposals are generated. If it is rejected, ASA keeps the previous candidate as the reference.

The jump shape depends on `generating_distribution`. With `ingber_asa`, ASA uses a method-specific distribution capable of combining small movements with the possibility of larger jumps. This favors exploration and helps avoid premature freezing. With `gaussian`, jumps are generated with a normal distribution, which produces more local movements around the current point.

Finally, every generated candidate is kept inside the bounds defined by the YAML, converted to real values, and sent to GOW. GOW evaluates it and ASA uses that result to decide whether to accept it, reject it, or record it as the best candidate found.

```text
current reference
        ↓
selection of parameters to modify
        ↓
jump generation
        ↓
candidate inside bounds
        ↓
evaluation by GOW
        ↓
acceptance, rejection, or best-candidate update
```

## 7. How the evaluation result is interpreted

GOW can work with minimization or maximization problems. For that reason, ASA needs to interpret the result returned by the evaluator consistently with the direction defined in the YAML.

If the problem is a maximization problem, a higher objective value means a better solution. In that case, ASA can use the evaluated value directly as a quality measure.

If the problem is a minimization problem, a lower objective value means a better solution. To handle both cases with the same internal logic, ASA transforms the minimization value so that internally it can keep using the rule: higher internal score means better candidate.

```text
Minimization problem

Candidate A
        ↓
real objective = 8.0
        ↓
internal score = -8.0

Candidate B
        ↓
real objective = 5.0
        ↓
internal score = -5.0

Since -5.0 is greater than -8.0, ASA considers candidate B better.
```

In addition to the internal score, ASA uses the idea of cost to apply simulated annealing logic. In that logic, lower cost means a better solution. This makes it possible to decide whether a candidate directly improves the current state or whether, even if it is worse, it can be temporarily accepted with a probability that depends on temperature.

## 8. How to configure the YAML in GOW

The YAML must describe the problem, the optimizable parameters, the external evaluator, and the optimizer configuration. For ASA, the most important blocks are objective, parameters, evaluator, and optimizer.

### 8.1 Objective block

```yaml
objective:
  direction: minimize
```

This block indicates whether the objective should be minimized or maximized. In a problem involving error, cost, or distance, `minimize` is normally used. In a problem where the goal is to increase a performance metric, `maximize` is used.

### 8.2 Parameters block

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

Each optimizable parameter must have a type, an initial value, and `bounds`. ASA uses the `bounds` to represent each parameter internally in the normalized range [0, 1] and to return candidates inside the allowed range.

| Field | Interpretation |
| --- | --- |
| `type` | Parameter type. This implementation supports `real` and `int`. |
| `value` | Reference value or initial value defined in the YAML. It can be emitted as an exact candidate if `include_initial_candidate` is `true`. It also serves as the initial base for generating variations. |
| `bounds` | Allowed range where ASA can search. It must have a lower and an upper bound. |

Categorical parameters are not directly supported by this ASA implementation. If a problem has categorical parameters, use another optimizer or a carefully justified numerical encoding.

### 8.3 Evaluator block

```yaml
evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600
```

This block indicates which external program will evaluate each candidate. ASA does not compute the objective; the evaluator computes it and GOW returns the result to the optimizer.

### 8.4 Optimizer block

The optimizer block controls how ASA runs inside GOW. General execution parameters should remain at the main level of the optimizer block. ASA-specific hyperparameters should be placed inside `settings`.

## 9. ASA-specific parameters in `settings`

The `settings` block contains the hyperparameters that modify ASA's internal behavior. These parameters control how candidates are generated, how temperatures are cooled, how worse candidates are accepted, how the search is readjusted, and when a soft restart is applied.

In a base ASA configuration, the main parameters can be organized as follows:

```yaml
settings:
  generating_distribution: ingber_asa
  temperature_schedule: ingber_asa
  acceptance_rule: metropolis_exp
  reannealing_method: hybrid

  initial_temperature: 1.0
  final_temperature: 0.001

  parameter_quench_factor: 1.0
  cost_quench_factor: 1.0
  asa_temperature_scale: 1.0

  mutation_parameter_fraction_start: 0.55
  mutation_parameter_fraction_end: 0.18

  reanneal_interval: 64
  target_acceptance_rate: 0.25

  restart_interval: 240
  restart_sigma: 0.012
  include_initial_candidate: true

  # Only needed when using temperature_schedule: geometric
  # cooling_rate: 0.985
```

### 9.1 `generating_distribution`

Defines the way ASA generates the jumps used to build new candidates. It does not control how many parameters are modified, and it does not decide whether a candidate is accepted or rejected. It only controls how the change applied to the selected parameters is calculated.

With `ingber_asa`, ASA uses a method-specific distribution designed to combine local movements with the possibility of wider jumps. This option favors exploration and helps prevent the search from getting trapped too early in a local region.

With `gaussian`, jumps are generated with a normal distribution. This produces candidates more concentrated around the current point and favors a more local and fine-grained search.

- **Accepted range:** `ingber_asa`, `gaussian`.
- **Recommended starting value:** `ingber_asa`.

Use `ingber_asa` when you want to preserve global exploration capability. Use `gaussian` when you want a more local search around a region that already looks promising.

### 9.2 `temperature_schedule`

Defines how ASA's internal temperatures are cooled during optimization. These temperatures control how much freedom the algorithm has to explore and when it starts focusing more on a local search.

In this implementation, the accepted values are `ingber_asa` and `geometric`.

With `ingber_asa`, ASA uses a schedule specific to the Adaptive Simulated Annealing approach. This cooling takes the problem dimension into account and works together with parameters such as `parameter_quench_factor`, `cost_quench_factor`, and `asa_temperature_scale`. It is the most representative option for the ASA method and is recommended as the starting point.

With `geometric`, the temperature decreases by being multiplied by `cooling_rate`. It is a simpler option and is similar to what is commonly called exponential or geometric cooling.

- **Accepted range:** `ingber_asa`, `geometric`.
- **Recommended starting value:** `ingber_asa`.

Use `ingber_asa` to preserve ASA's intended behavior. Use `geometric` when you want a simpler cooling process controlled directly by `cooling_rate`.

#### 9.2.1 `cooling_rate`

`cooling_rate` is only used when `temperature_schedule: geometric`. In that case, the temperature is multiplied by this value at each update.

If `cooling_rate` is very close to 1, cooling is slower and ASA preserves exploration for longer. If it is reduced, the temperature decreases faster and ASA becomes local earlier.

- **Accepted range:** greater than 0 and less than or equal to 1.
- **Practical initial range:** 0.95 to 0.995.
- **Recommended starting value:** 0.985.

Use values closer to 1.0 when you want slower cooling. Use lower values when you want ASA to focus earlier.

### 9.3 `acceptance_rule`

Defines the rule ASA uses to decide whether an evaluated candidate is accepted or rejected as the new starting point of the search.

If the candidate improves the objective value, ASA accepts it. If the candidate worsens the objective value, ASA can still accept it with some probability. That probability depends on how much the candidate worsens and on the current cost temperature.

`metropolis_exp` uses an exponential probability. It is the option closest to the classic Simulated Annealing logic and is recommended as the starting point. It allows some worse candidates to be accepted when the temperature is high, which helps prevent the search from getting trapped too early.

`asa_logistic` uses a logistic probability. It also allows worse candidates to be accepted, but it is usually more conservative than `metropolis_exp`. It can be tested when the search accepts too many worsening moves or when a smoother and more controlled transition is desired.

- **Accepted range:** `metropolis_exp`, `asa_logistic`.
- **Recommended starting value:** `metropolis_exp`.

Use `metropolis_exp` to preserve ASA's main behavior. Use `asa_logistic` when you want a more moderate acceptance of worsening moves.

### 9.4 `reannealing_method`

Defines how ASA readjusts its search behavior during the run. It is not the mechanism that directly generates candidates, and it is not the rule that accepts or rejects solutions. Its function is to inspect how the search is behaving and correct internal scales so ASA does not become too rigid, too scattered, or treat all parameters as if they were equally important.

The idea is that ASA not only cools the temperature automatically, but can also recalibrate the search according to what it observes. This matters because, in a real problem, not all parameters behave the same way.

With `acceptance_rate`, ASA looks at the recent acceptance rate. If it accepts very few candidates, it reduces the movement scale to search more carefully. If it accepts too many, it can slightly expand the scale.

With `historical_sensitivity`, ASA tries to estimate which parameters are more sensitive. If a parameter appears very sensitive, ASA reduces its temperature to move it more carefully. If a parameter appears less sensitive, ASA can allow wider movements.

With `hybrid`, ASA combines the two previous strategies. It adjusts the search according to the recent acceptance rate and also according to the observed sensitivity of each parameter. This is the recommended starting option.

With `none`, ASA does not apply adaptive reannealing. It can be useful for controlled tests or debugging, but normally it is not the best option for a real run.

- **Accepted range:** `hybrid`, `acceptance_rate`, `historical_sensitivity`, `none`.
- **Recommended starting value:** `hybrid`.

Use `hybrid` for normal runs. Use `acceptance_rate` when you want ASA to adjust mainly according to how many candidates it is accepting. Use `historical_sensitivity` when you want to give more importance to differences in parameter behavior. Use `none` only for tests or comparisons.

### 9.5 `initial_temperature`

Controls how much freedom ASA has at the start of the optimization. A high initial temperature allows a more exploratory search; a low initial temperature makes ASA start in a stricter and more local way.

This parameter affects ASA's initial ability to accept some worse candidates and, when `generating_distribution: ingber_asa` is used, it also influences the initial size of parameter jumps.

If it is increased, ASA starts hotter. This allows more exploration and acceptance of some early setbacks, which can help escape local minima. If it is increased too much, the search may become too permissive and take longer to focus.

If it is reduced, ASA starts colder. This makes the search stricter from the beginning. It can be useful when the problem is well bounded or when a more local search is desired, but if it is reduced too much ASA may lose exploration and get trapped early.

- **Accepted range:** greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 0.5 to 2.0.
- **Broad experimental range:** 0.1 to 10.0.
- **Recommended starting value:** 1.0.

Additional restriction: `final_temperature` must be less than or equal to `initial_temperature`. Use 1.0 as the starting point. Use lower values when a more controlled initial search is desired. Use higher values when more initial exploration is desired. Values greater than 10.0 should be treated as experimental.

### 9.6 `final_temperature`

Defines the minimum temperature ASA is allowed to cool toward. If `initial_temperature` represents the initial freedom, `final_temperature` represents the final freedom level.

If it is reduced, ASA ends colder. This favors a finer and stricter search, but if it is reduced too much it can freeze the search and make ASA reject almost any worsening move.

If it is increased, ASA ends less cold. This preserves more movement freedom until the end, but if it is increased too much it can prevent the search from focusing and refining a promising solution.

- **Accepted range:** greater than 0 and less than or equal to `initial_temperature`.
- **Upper limit in the code:** `initial_temperature`.
- **Practical initial range:** 0.0001 to 0.01.
- **Broad experimental range:** 0.000001 to 0.1.
- **Recommended starting value:** 0.001.

Use 0.001 as the starting point. Use lower values when a stricter final search is desired. Use higher values when more exploration should be preserved into later stages.

### 9.7 `parameter_quench_factor`

Controls how the temperatures associated with the parameters are cooled when `temperature_schedule: ingber_asa` is used.

These temperatures affect the size of jumps. If this value is reduced, ASA cools parameter temperatures earlier and movements become more local sooner. If this value is increased, ASA preserves higher parameter temperatures for longer and maintains more exploration capability.

This parameter should not be interpreted as a percentage. `parameter_quench_factor: 1.0` does not mean 10%, 1%, or that ASA moves a specific fraction of the range. It should be understood as the recommended baseline value for cooling parameter temperatures.

- **Accepted range:** greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 0.5 to 3.0.
- **Broad experimental range:** 0.25 to 5.0.
- **Recommended starting value:** 1.0.

Use values lower than 1.0 when you want movements to become local sooner. Use values higher than 1.0 when you want to preserve more exploration for longer. Values greater than 5.0 should be considered experimental.

### 9.8 `cost_quench_factor`

Controls how the cost temperature is cooled when `temperature_schedule: ingber_asa` is used.

The cost temperature affects the probability of accepting worse candidates. If this value is reduced, ASA cools that temperature earlier and becomes stricter sooner. If this value is increased, ASA preserves the possibility of accepting some worsening moves for longer.

- **Accepted range:** greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 0.5 to 3.0.
- **Broad experimental range:** 0.25 to 5.0.
- **Recommended starting value:** 1.0.

Use values lower than 1.0 when you want ASA to become stricter sooner. Use values higher than 1.0 when you want to preserve probabilistic exploration for longer.

### 9.9 `asa_temperature_scale`

A global factor that modifies the overall speed of ASA cooling.

If it is increased, temperatures decrease faster. If it is reduced, temperatures decrease more slowly.

- **Accepted range:** greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 0.5 to 3.0.
- **Broad experimental range:** 0.25 to 5.0.
- **Recommended starting value:** 1.0.

Use values greater than 1.0 when you want to speed up cooling. Use values lower than 1.0 when you want to preserve high temperatures for more evaluations.

### 9.10 `mutation_parameter_fraction_start`

Defines the fraction of parameters ASA attempts to modify at the beginning of the search.

For example, `mutation_parameter_fraction_start: 0.55` means that, at the beginning, ASA modifies approximately 55% of the parameters in each candidate.

If it is increased, each candidate changes more parameters at once. If it is reduced, each candidate changes fewer parameters from the start.

- **Accepted range:** greater than 0 and less than or equal to 1.
- **Practical initial range:** 0.3 to 0.8.
- **Recommended starting value:** 0.55.

Use higher values when a more global initial search is desired. Use lower values when a more localized initial search is desired.

### 9.11 `mutation_parameter_fraction_end`

Defines the fraction of parameters ASA attempts to modify toward the end of the search.

For example, `mutation_parameter_fraction_end: 0.18` means that, toward the end, ASA modifies approximately 18% of the parameters in each candidate.

If it is increased, ASA preserves more global movements for longer. If it is reduced, ASA ends with more local movements.

Usually, `mutation_parameter_fraction_start` should be greater than `mutation_parameter_fraction_end`, because ASA should start with broader changes and end with finer adjustments.

- **Accepted range:** greater than 0 and less than or equal to 1.
- **Practical initial range:** 0.05 to 0.3.
- **Recommended starting value:** 0.18.

Use lower values when a finer final phase is desired. Use higher values when you want to prevent the final search from becoming too local.

### 9.12 `reanneal_interval`

Controls how often, in number of evaluations, ASA applies reannealing. Reannealing is the mechanism that recalibrates the search according to the recent acceptance rate and/or the observed parameter sensitivity, depending on `reannealing_method`.

If it is reduced, ASA readjusts the search more frequently. This allows it to react earlier, but if the value is too small it may adjust based on too little information and become sensitive to noise.

If it is increased, ASA waits for more evaluations before readjusting. This provides a more stable basis for decisions, but if it is increased too much ASA may take too long to correct a poorly calibrated search.

```text
evaluation 1
        ↓
evaluation 2
        ↓
...
        ↓
evaluation reanneal_interval
        ↓
ASA applies reannealing
```

- **Accepted range:** functionally, integer greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** between 2 and 5 times `batch_size`.
- **Recommended starting value:** 64 when `batch_size` is close to 32.

A value of 0 or negative does not activate periodic reannealing. It must be lower than `max_evaluations` if it should happen at least once. For large runs, it is better to scale it with the batch size.

### 9.13 `target_acceptance_rate`

Controls the target acceptance rate used by ASA during acceptance-based reannealing.

This parameter does not force the optimizer to accept exactly that proportion of candidates. It works as an internal reference. ASA compares the recent acceptance rate with this value and, if it is accepting too little or too much, it readjusts the search scale.

```text
ASA reviews the latest evaluated candidates
        ↓
calculates how many were accepted
        ↓
compares that rate with target_acceptance_rate
        ↓
adjusts the search scale if needed
```

If it is increased, ASA uses a higher acceptance rate as the reference. This can make the search more flexible and tolerate more movement, but if it is increased too much it can become less selective.

If it is reduced, ASA uses a lower acceptance rate as the reference. This favors a stricter search, but if it is reduced too much the algorithm may accept too few candidates and move with difficulty.

- **Accepted range:** greater than 0 and less than 1.
- **Practical initial range:** 0.15 to 0.35.
- **Broad experimental range:** 0.10 to 0.50.
- **Recommended starting value:** 0.25.

Use values close to 0.25 for normal runs. Use higher values when ASA is rejecting too much and a more flexible search is desired. Use lower values when ASA is accepting too much and a more selective search is desired.

### 9.14 `restart_interval`

Defines how many evaluations may pass without improvement before applying a soft restart around the best candidate found.

For example, `restart_interval: 240` means ASA may restart near the best candidate if enough evaluations pass without improvement.

If it is reduced, restarts happen more frequently. If it is increased, ASA waits longer before restarting.

- **Accepted range:** greater than or equal to 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 100 to 1000.
- **Recommended starting value:** 240.

Use lower values when you want to react earlier to stagnation. Use higher values when you want to give the search more room before restarting. Use 0 when you want to disable the soft restart.

### 9.15 `restart_sigma`

Controls the amplitude of the noise used in the soft restart around the best candidate.

This parameter does not control the normal jumps of all candidates. It is only used when ASA detects stagnation and applies a soft restart near the best candidate found.

The value is interpreted in normalized [0, 1] space. For example, `restart_sigma: 0.012` means the restart is performed with a typical perturbation close to 1.2% of each parameter's normalized range.

If it is increased, the restart moves farther away from the best candidate. This allows ASA to explore a wider area around the best solution found. If it is reduced, the restart stays closer to the best candidate and behaves more like a local search.

- **Accepted range:** greater than 0.
- **Upper limit in the code:** not defined.
- **Practical initial range:** 0.005 to 0.05.
- **Recommended starting value:** 0.012.

Use higher values when you want the restart to explore more around the best candidate. Use lower values when you want the restart to be more local.

### 9.16 `include_initial_candidate`

Defines whether the YAML `value` entries are sent as an exact candidate for evaluation.

If it is `true`, ASA sends the `value` entries as an exact candidate for GOW to evaluate. If it is `false`, ASA does not evaluate that exact candidate as the first proposal.

In ASA, `include_initial_candidate: false` does not mean the first batch is completely random inside the `bounds`. The YAML `value` entries still work as the initial reference for generating the first variations.

- **Accepted range:** `true` or `false`.
- **Recommended starting value:** depends on how the `value` entries are used.

Use `true` when the `value` entries represent an important reference that should be explicitly evaluated. Use `false` when the `value` entries are only a technical starting point and evaluating that exact candidate is not necessary.

## 10. Practical recommendations

### 10.1 Use `max_evaluations` and `batch_size` as the main controls

For GOW, the clearest way to control ASA is with a total evaluation budget and a batch size. The practical number of batches is obtained by dividing `max_evaluations` by `batch_size`.

```text
max_evaluations = batch_size × number_of_batches
```

It is recommended that `max_evaluations` be a multiple of `batch_size`. This avoids incomplete final batches and makes result interpretation cleaner.

### 10.2 Start with the main ASA variant

For a first run, the most coherent choice is to use `generating_distribution: ingber_asa` and `temperature_schedule: ingber_asa`. That combination represents the main intention of the ASA optimizer implemented here.

### 10.3 Adjust temperatures before changing too many parameters

If ASA becomes too local, you can increase `initial_temperature`, decrease `asa_temperature_scale`, or carefully increase the quenching factors. If ASA explores too much and takes too long to stabilize, you can increase `asa_temperature_scale` or moderately reduce the quenching factors.

### 10.4 Choose the parameter fraction according to the search stage

A high initial fraction allows several parameters to change at once. A low final fraction helps make more local adjustments. A reasonable configuration is to start around 0.5 and end between 0.1 and 0.25.

### 10.5 Interpret `include_initial_candidate` carefully

Using `include_initial_candidate: true` can be useful when you want to explicitly evaluate a reference solution defined by the YAML `value` entries. That reference does not automatically dominate the search. It only matters if, after evaluation, it becomes the accepted state or the best candidate found.

In ASA, `include_initial_candidate: false` does not mean a completely random start. It means the exact candidate is not evaluated, but the `value` entries still serve as the initial reference for generating variations.

### 10.6 Keep `bounds` coherent

ASA can only search inside the `bounds` defined in the YAML. If the `bounds` are too narrow, the search is limited. If they are too wide, it may need many more evaluations to find promising regions.

### 10.7 Check the evaluator contract

The evaluator must return a numerical result compatible with the direction defined in `objective.direction`. In minimization problems, the lower value should represent a better solution. In maximization problems, the higher value should represent a better solution.

## Appendix A. Conceptually commented base YAML

```yaml
id: continuous-problem-asa

objective:
  direction: minimize        # Change to maximize if the objective should increase.

parameters:
  x0:
    type: real               # ASA supports real numerical parameters.
    value: 0.5               # Reference value or initial YAML value.
    bounds: [0.0, 1.0]       # Range where ASA can search.

  x1:
    type: real
    value: 10.0
    bounds: [5.0, 15.0]

evaluator:
  command:
    ["/path/to/evaluator"]
  timeout_s: 600             # Maximum time allowed for one evaluation.

optimizer:
  name: asa                  # Selects the ASA optimizer.
  seed: 123                  # Seed for reproducibility.
  max_evaluations: 1000      # Total evaluation budget.
  batch_size: 25             # Candidates per batch requested by GOW.

  settings:
    generating_distribution: ingber_asa  # Main ASA distribution.
    temperature_schedule: ingber_asa     # Dimension-aware cooling.
    acceptance_rule: metropolis_exp      # Classic probabilistic acceptance rule.
    reannealing_method: hybrid           # Adapts by acceptance and sensitivity.

    initial_temperature: 1.0             # Initial temperature.
    final_temperature: 0.001             # Minimum final temperature.

    parameter_quench_factor: 1.0         # Cooling of parameter jumps.
    cost_quench_factor: 1.0              # Cooling of worse-candidate acceptance.
    asa_temperature_scale: 1.0           # Global ASA cooling scale.

    mutation_parameter_fraction_start: 0.55  # Initial fraction of mutated parameters.
    mutation_parameter_fraction_end: 0.18    # Final fraction of mutated parameters.

    reanneal_interval: 64                # How often, in evaluations, adaptation occurs.
    target_acceptance_rate: 0.25         # Target acceptance rate.

    restart_interval: 240                # Soft restart if there is no improvement.
    restart_sigma: 0.012                 # Restart noise in normalized space.
    include_initial_candidate: true      # Evaluates the value entries as an exact candidate.

    # Only if using temperature_schedule: geometric
    # cooling_rate: 0.985                # Geometric cooling factor.
```

## Appendix B. Quick reading of the ask/tell flow

```text
ask(problem, n)
        ↓
ASA prepares the internal parameter representation if it does not exist yet
        ↓
if applicable, it returns the exact candidate defined by the YAML value entries
        ↓
it generates variations by modifying a fraction of parameters
        ↓
it keeps every candidate inside the bounds
        ↓
it returns candidates to GOW in real parameter values
        ↓
GOW evaluates the candidates with the external evaluator
        ↓
tell(candidates, results)
        ↓
ASA interprets each result according to objective.direction
        ↓
if a candidate improves, ASA accepts it
        ↓
if a candidate worsens, ASA may accept it with a temperature-dependent probability
        ↓
ASA keeps the best candidate found
        ↓
ASA cools the search according to the configured schedule
        ↓
ASA applies reannealing if applicable
        ↓
ASA applies a soft restart if there is stagnation
```
