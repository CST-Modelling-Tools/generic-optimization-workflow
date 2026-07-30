# PSO in GOW

## Usage, Configuration, Implementation, and Scientific Traceability Guide

---

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Implemented PSO variant](#2-implemented-pso-variant)
3. [When to use PSO](#3-when-to-use-pso)
4. [Requirements and functional scope](#4-requirements-and-functional-scope)
5. [Quick start](#5-quick-start)
6. [Optimizer configuration](#6-optimizer-configuration)
7. [`ask()`-evaluation-`tell()` flow](#7-askevaluationtell-flow)
8. [Integration-specific behavior](#8-integration-specific-behavior)
9. [Mathematical foundations](#9-mathematical-foundations)
10. [Scientific and technical traceability](#10-scientific-and-technical-traceability)
11. [Validation](#11-validation)
12. [Known limitations](#12-known-limitations)
13. [References](#13-references)

---

# 1. Purpose and scope

The first sections present the practical use of the optimizer. The later sections describe its mathematical foundations, its integration into GOW, its scientific traceability, and the scope of the validation performed.

For the general GOW architecture, the contract with external evaluators, the configuration structure, provenance identifiers, and the organization of execution files, see the [GOW architecture and usage documentation](https://cst-modelling-tools.github.io/generic-optimization-workflow-blog/gow-architecture-and-usage).

---

# 2. Implemented PSO variant

The implementation available in GOW corresponds to a **global-best Particle Swarm Optimizer with inertia weight**, fixed swarm size, synchronous generational updates, symmetric cognitive and social acceleration, velocity clamping, and explicit bound handling.

The original global-best formulation, based on particles with position and velocity, personal-best memory (`pbest`), and attraction toward the global-best position (`gbest`), was introduced by **James Kennedy and Russell C. Eberhart (1995)**. The inertia-weight term included in the velocity equation comes from the modification proposed by **Yuhui Shi and Russell C. Eberhart (1998)**. The implemented variant can therefore be described as **Kennedy and Eberhart's global-best PSO with the inertia-weight modification of Shi and Eberhart**.

It is a population-based, stochastic, single-objective optimizer for bounded numerical variables. Each particle stores its own best position, `pbest`, and all particles use the same best position found by the complete swarm, `gbest`.

The implemented velocity and position rules are:

\[
v_{i,d}^{(g+1)}
=
w\,v_{i,d}^{(g)}
+
c\,r_{1,i,d}^{(g)}
\left(p_{i,d}^{(g)}-x_{i,d}^{(g)}\right)
+
c\,r_{2,i,d}^{(g)}
\left(g_d^{(g)}-x_{i,d}^{(g)}\right),
\]

\[
x_{i,d}^{(g+1)}
=
x_{i,d}^{(g)}+v_{i,d}^{(g+1)}.
\]

The same configurable coefficient, `acceleration_coefficient`, is used for the cognitive and social terms. Independent pseudo-random values are generated for both attractions in every particle and dimension.

The implementation does **not** expose:

- separate cognitive and social coefficients;
- a local-best topology;
- a configurable neighborhood structure;
- a time-varying or adaptive inertia schedule;
- a separate constriction-factor parameter;
- an adaptive swarm size;
- automatic restarts.


---

# 3. When to use PSO

PSO is appropriate for black-box problems in which:

- the optimizable variables are numerical and bounded;
- derivatives are unavailable, unreliable, or too expensive to obtain;
- the objective may be nonlinear, non-convex, or non-separable;
- several regions of the search space may be promising;
- the evaluator can process several candidates per generation;
- a stochastic approximate solution is acceptable;
- the budget is sufficient to evaluate multiple complete generations.

In GOW, PSO can optimize real-valued variables directly and integer variables through continuous movement followed by rounding. The method is especially useful when the result of a candidate can only be known by running an external simulation, model, or executable.

It is generally not the most suitable option when:

- most optimizable variables are categorical or combinatorial;
- the evaluation budget is too small to initialize and update a useful swarm;
- an exact mathematical guarantee is required;
- several simultaneous objectives must be optimized as a Pareto front;
- general constraints require a specialized feasibility mechanism;
- strong multimodality requires explicit diversity preservation or automatic restarts;
- the integer domain is very small and discrete movement must be modeled exactly.


---

# 4. Requirements and functional scope

## 4.1 Variable types

The current PSO implementation in GOW supports the following optimizable parameter types:

| Parameter type | Current support | Treatment |
|---|---:|---|
| `RealParam` | Yes | Continuous position and velocity inside the configured bounds |
| `IntParam` | Yes | Continuous velocity update followed by position rounding and bound enforcement |
| `CategoricalParam` | No | An optimizable categorical parameter is rejected explicitly |

Fixed parameters may remain in the problem configuration with `optimizable: false`. They are supplied to the evaluator by the GOW execution layer but do not form part of the swarm position.

Every optimizable numerical parameter must define two valid bounds with:

\[
\text{lower bound}<\text{upper bound}.
\]

The `value` field is a reference or runtime value in the problem configuration. The PSO adapter does not use it to insert a predefined initial particle. Initial positions are sampled from the bounds.

## 4.2 Problem type

| Characteristic | Current support |
|---|---:|
| Black box | Yes |
| Objective | Single, with `minimize` or `maximize` direction |
| Lower and upper bounds for each optimizable variable | Required |
| Direct gradient use | No |
| General nonlinear equality or inequality constraints | No specific PSO mechanism |
| Complete generational batches | Required |
| Parallel evaluation through the GOW execution layer | Compatible |

---

# 5. Quick start

## 5.1 Minimal YAML example

The following example defines a bounded problem with one real and one integer variable:

```yaml
id: numerical-problem-pso

parameters:
  x:
    type: real
    value: 0.0
    bounds: [-5.0, 5.0]
    optimizable: true

  n:
    type: int
    value: 5
    bounds: [1, 20]
    optimizable: true

evaluator:
  command:
    - "/path/to/evaluator"
  timeout_s: 60

objective:
  direction: minimize

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

In this example:

```text
20 particles per generation × 50 generations = 1000 evaluations
```

Each particle produces exactly one candidate per generation. Therefore, `batch_size` is also the swarm size.

## 5.2 Expected evaluator result

The recommended format for a successful evaluation is:

```json
{
  "status": "ok",
  "objective": 0.03125,
  "metrics": {
    "reported_metric": 0.03125
  },
  "constraints": {},
  "artifacts": {}
}
```

For a failed evaluation:

```json
{
  "status": "failed",
  "objective": null,
  "metrics": {},
  "error": "The simulator did not converge."
}
```

The recommended comparison key is `objective`. The objective direction must be declared in the YAML as `minimize` or `maximize`; the user must not manually invert its sign.

## 5.3 Execution

From the virtual environment and the appropriate directory:

```bash
gow run path/to/problem.yaml
```

The organization of the results and the complete evaluator contract are described in the general GOW documentation indicated in Section 1.

---

# 6. Optimizer configuration

## 6.1 Exposed parameters

- `name` (`optimizer.name`): selects PSO and must have the value `pso`.
- `seed` (`optimizer.seed`): initializes the optimizer-specific pseudo-random generator.
- `max_evaluations` (`optimizer.max_evaluations`): total evaluation budget managed by the GOW executor.
- `batch_size` (`optimizer.batch_size`): number of candidates per generation and number of particles in the swarm.
- `max_generations` (`optimizer.settings.max_generations`): maximum number of complete generations stored by the PSO adapter.
- `inertia_weight` (`optimizer.settings.inertia_weight`): fraction of the previous velocity retained in the next movement.
- `acceleration_coefficient` (`optimizer.settings.acceleration_coefficient`): shared strength of the attraction toward `pbest` and `gbest`.
- `velocity_clamp_fraction` (`optimizer.settings.velocity_clamp_fraction`): maximum absolute velocity as a fraction of each variable range.

The implementation does not require a separate `swarm_size` field. In this integration:

```text
swarm_size = batch_size
```

The implementation also does not expose separate parameters named `cognitive_coefficient`, `social_coefficient`, `constriction_factor`, `topology`, or `maximum_velocity`.

## 6.2 `batch_size`

`batch_size` defines both:

- the number of candidates requested by GOW in each generation;
- the number of particles maintained by PSO.

It must be at least `1` and remain constant throughout the execution. Once the first call to `ask(problem, n)` fixes the swarm size, any later request with a different value of `n` is rejected.

A larger swarm can cover more regions in one generation and may use parallel resources effectively, but it consumes more evaluations per generation. A smaller swarm permits more generations under the same total budget, although it represents less diversity at each update.

The choice must therefore be made together with:

- the dimension of the problem;
- the evaluation cost;
- the available parallel capacity;
- the total evaluation budget;
- the number of complete generations required for learning.

## 6.3 `inertia_weight`

`inertia_weight` controls how much of the previous velocity is retained:

```yaml
settings:
  inertia_weight: 0.729
```

Its contribution is:

\[
w\,v_{i,d}^{(g)}.
\]

In general:

- increasing it preserves more momentum and can extend exploration;
- reducing it makes particles respond more quickly to `pbest` and `gbest`;
- a value of `0` removes memory of the previous velocity;
- the value must be greater than or equal to `0`.

In the current adapter, the inertia weight remains constant for the complete execution. No automatic schedule decreases or increases it over generations.

## 6.4 `acceleration_coefficient`

`acceleration_coefficient` is used for both attraction terms:

```yaml
settings:
  acceleration_coefficient: 1.49445
```

Its contributions are:

\[
c\,r_1(pbest-x)
\]

and

\[
c\,r_2(gbest-x).
\]

The coefficient must be greater than or equal to `0`.

Increasing it makes particles respond more strongly to their best personal position and to the best position found by the swarm. Reducing it weakens both attractions and leaves a greater relative role to the inertia term.

Because a single shared coefficient is used, the current YAML cannot give different deterministic weights to individual and collective memory. Their instantaneous effects still differ because `r1` and `r2` are generated independently.

## 6.5 `velocity_clamp_fraction`

`velocity_clamp_fraction` limits the absolute velocity separately in every dimension:

```yaml
settings:
  velocity_clamp_fraction: 0.2
```

For a variable with bounds \([l_d,u_d]\), the maximum velocity is:

\[
v_{\max,d}
=
\texttt{velocity\_clamp\_fraction}\,(u_d-l_d).
\]

The computed velocity is then restricted to:

\[
-v_{\max,d}
\leq
v_{i,d}
\leq
v_{\max,d}.
\]

Examples:

- `1.0` represents a maximum step equal to 100% of the variable range;
- `0.2` represents 20% of the variable range;
- `0.05` represents 5% of the variable range.

A smaller fraction produces shorter movements and a more local search. A larger fraction permits wider movements but can increase contact with the bounds and reduce fine control. The value must be strictly positive.

## 6.6 `seed`

The seed initializes a dedicated `random.Random` instance inside the optimizer. With the same seed and equivalent execution conditions, PSO generates the same initial positions, initial velocities, and subsequent pseudo-random factors.

To reproduce an execution, the following must also remain constant:

- the YAML file;
- the parameter order and bounds;
- the evaluator and its dependencies;
- the evaluation order;
- the returned objective values;
- the batching and execution environment;
- the PSO implementation version.

A seed does not represent a quality setting. Different seeds generate different, but valid, stochastic trajectories of the same configured algorithm.

## 6.7 Budget, generations, and termination

`max_evaluations` sets the maximum evaluation budget controlled by the GOW executor. PSO stores its own limit in `max_generations` and exposes it through `is_done()`.

Because PSO requires a complete and fixed swarm at every cycle, the recommended relationship is:

\[
\texttt{max\_evaluations}
=
\texttt{batch\_size}
\times
\texttt{max\_generations}.
\]

Example:

```yaml
optimizer:
  max_evaluations: 1600
  batch_size: 40

  settings:
    max_generations: 40
```

```text
40 particles × 40 generations = 1600 evaluations
```

`max_evaluations` must be a multiple of `batch_size`. Otherwise, an executor that requests a smaller final batch will conflict with the fixed swarm size required by the adapter.

The immediate application of `max_generations` depends on whether the selected executor checks `optimizer.is_done()` before requesting another generation. For clear and portable behavior, the budget and generation count should remain consistent instead of relying on only one of them.

## 6.8 Practical configuration recommendations

### Keep the budget and generations consistent

Calculate `max_generations` from `max_evaluations` and `batch_size`. This prevents the executor and optimizer from working with different effective limits.

### Define useful and realistic bounds

PSO can only search inside the configured intervals. Bounds that are too narrow may exclude good solutions, while unnecessarily broad bounds enlarge the search region and the velocity scale.

### Choose the swarm size together with the budget

Increasing `batch_size` without increasing `max_evaluations` reduces the number of completed generations. Swarm diversity and generational learning must be balanced.

### Change one movement parameter at a time

When studying `inertia_weight`, `acceleration_coefficient`, or `velocity_clamp_fraction`, change one parameter while holding the others fixed. This makes the observed effect interpretable.


---

# 7. `ask()`-evaluation-`tell()` flow

PSO is a generational optimizer. Each generation follows this cycle:

```text
ask()
  ↓
particle positions converted into GOW candidates
  ↓
external evaluation of the complete swarm
  ↓
collection of results in the same order
  ↓
tell()
  ↓
update of pbest, gbest, diagnostics, and generation count
  ↓
next ask() moves the complete swarm
```

## 7.1 Deferred initialization

The swarm is created when the first call to `ask(problem, n)` is received, because the following are available at that point:

- the objective direction;
- the optimizable parameter definitions;
- their types and bounds;
- the number of candidates requested by GOW.

Initialization uses the bounds, not the YAML `value` entries.

For every particle:

- the initial position is sampled inside the allowed interval of every variable;
- the initial velocity is sampled uniformly from the corresponding interval \([-v_{\max,d},v_{\max,d}]\);
- the initial `pbest` position is set to the initial position;
- the `pbest` score remains undefined until the first valid evaluation;
- `gbest` remains undefined until at least one valid evaluation is received.

## 7.2 Cycle rules

The intended cycle is:

1. one call to `ask()` for the complete swarm;
2. evaluation of every returned candidate;
3. one call to `tell()` with the same complete swarm and its results;
4. a new call to `ask()` to move the particles and generate the next population.

The implementation checks that:

- the swarm has been initialized before `tell()`;
- `candidates` and `fitness` have the same length;
- the number of candidates is exactly the fixed swarm size;
- every generation is returned as one complete batch.

The first `ask()` returns the randomly initialized positions. From the second generation onward, `ask()` calls the particle-movement rule before returning the candidates.

If every evaluation received so far is invalid and no `gbest` exists, the next movement cannot use a social attractor. In that case, the adapter creates new random positions and velocities for the swarm.

## 7.3 Importance of order

Order is part of the contract:

```text
candidates[i] ↔ fitness[i] ↔ particle i
```

`tell()` updates the personal memory of particle `i` using `candidates[i]` and `fitness[i]`. Reordering the fitness list would assign results to the wrong particles and corrupt `pbest` updates.

The swarm update is synchronous at the generation level. All candidates are evaluated before the next set of positions is generated. During movement, new positions and velocities are first stored in separate lists, so no particle uses another particle's partially updated current position.

---

# 8. Integration-specific behavior

- PSO operates directly in each variable's physical interval. Unlike optimizers that use an internal normalized domain, this adapter computes distances and velocities using the configured numerical bounds.
- GOW supplies the objective direction. Internally, PSO converts comparisons to the rule `higher internal score = better candidate`.
- For `minimize`, an `objective` value is sign-inverted internally; for `maximize`, it is used without that direction inversion.
- The recommended evaluator key is `objective`. The compatibility parser also recognizes `fitness`, `score`, and `loss`, but `objective` is the verified and unambiguous interface for the current adapter.
- A result with a non-`ok` status, a missing comparison value, a non-numeric value, `NaN`, or infinity receives an internal score of negative infinity and cannot update `pbest` or `gbest`.
- Diagnostic counters record failed status values, missing scores, non-numeric values, and non-finite values for the latest completed generation.
- Real candidates are returned as `float` values.
- Integer candidates are rounded and returned as `int` values.
- Fixed parameters are not moved by PSO; the GOW execution layer combines them with each optimizable candidate before evaluation.
- The configured `value` of an optimizable parameter is not inserted automatically into the initial swarm.
- When a position exceeds a bound, it is placed on the violated bound and the velocity in that dimension is reset to zero.

For implementation details, internal validations, and the handling of exceptional cases, see the `src/gow/optimizer/particle_swarm.py` adapter, which is documented through comments in the source code.

---

# 9. Mathematical foundations

This section presents the equations needed to identify the implemented variant and relate the YAML parameters to particle behavior.

## 9.1 Notation

- \(D\): number of optimizable numerical variables;
- \(S\): swarm size, equal to `batch_size`;
- \(g\): completed generation index;
- \(i\): particle index, \(i=1,\ldots,S\);
- \(d\): variable or dimension index, \(d=1,\ldots,D\);
- \(x_{i,d}^{(g)}\): position of particle \(i\) in dimension \(d\);
- \(v_{i,d}^{(g)}\): velocity of particle \(i\) in dimension \(d\);
- \(p_{i,d}^{(g)}\): personal-best position of particle \(i\);
- \(g_d^{(g)}\): global-best position found by the swarm;
- \(w\): `inertia_weight`;
- \(c\): shared `acceleration_coefficient`;
- \(r_{1,i,d}^{(g)},r_{2,i,d}^{(g)}\): independent uniform random values in \([0,1)\);
- \(l_d,u_d\): lower and upper bounds of dimension \(d\).

## 9.2 Swarm initialization

For a real-valued variable:

\[
x_{i,d}^{(0)}
\sim
U(l_d,u_d).
\]

For an integer-valued variable, an integer is sampled uniformly from the inclusive integer interval and stored internally as a floating-point position.

The velocity limit is:

\[
v_{\max,d}
=
\alpha\,(u_d-l_d),
\]

where \(\alpha\) is `velocity_clamp_fraction`.

The initial velocity is sampled as:

\[
v_{i,d}^{(0)}
\sim
U(-v_{\max,d},v_{\max,d}).
\]

Each initial position is copied into the particle's `pbest` location, but it has no valid `pbest` score until the initial candidate has been evaluated successfully.

## 9.3 Personal and global best

The adapter converts each valid evaluator result into an internal score \(s_i^{(g)}\) such that larger is better.

The personal best of particle \(i\) is replaced when:

\[
s_i^{(g)}
>
s_{pbest,i}^{(g-1)}.
\]

The global best is replaced when:

\[
s_i^{(g)}
>
s_{gbest}^{(g-1)}.
\]

Strict comparison is used. An equal score does not replace an existing memory.

`pbest` represents individual memory: the best location found by one particle. `gbest` represents collective memory: the best location found by any particle in the complete swarm.

## 9.4 Velocity update

The new velocity is calculated as:

\[
v_{i,d}^{(g+1)}
=
\underbrace{w\,v_{i,d}^{(g)}}_{\text{inertia}}
+
\underbrace{c\,r_{1,i,d}^{(g)}
\left(p_{i,d}^{(g)}-x_{i,d}^{(g)}\right)}_{\text{cognitive attraction}}
+
\underbrace{c\,r_{2,i,d}^{(g)}
\left(g_d^{(g)}-x_{i,d}^{(g)}\right)}_{\text{social attraction}}.
\]

The three terms have different roles:

- **inertia** preserves part of the previous direction and step;
- **cognitive attraction** pulls the particle toward its own best experience;
- **social attraction** pulls it toward the best experience shared by the swarm.

The cognitive and social terms have the same deterministic coefficient, but use independent random multipliers.

## 9.5 Velocity clamping

After the velocity is computed, the adapter applies:

\[
v_{i,d}^{(g+1)}
\leftarrow
\min\left(
\max\left(v_{i,d}^{(g+1)},-v_{\max,d}\right),
 v_{\max,d}
\right).
\]

This limits the largest displacement that can be produced in a single generation and ties the movement scale to the physical range of each variable.

## 9.6 Position update and bound handling

The tentative new position is:

\[
x_{i,d}^{(g+1)}
=
x_{i,d}^{(g)}+v_{i,d}^{(g+1)}.
\]

If the result lies below the lower bound:

\[
x_{i,d}^{(g+1)}=l_d,
\qquad
v_{i,d}^{(g+1)}=0.
\]

If it lies above the upper bound:

\[
x_{i,d}^{(g+1)}=u_d,
\qquad
v_{i,d}^{(g+1)}=0.
\]

This is a boundary-clamping strategy with velocity reset in the violated dimension. It guarantees that every candidate returned to the evaluator remains inside the configured bounds.

## 9.7 Integer parameters

For an integer variable, PSO first computes the same continuous velocity and tentative position used for real variables. The position is then rounded:

\[
x_{i,d}^{(g+1)}
\leftarrow
\operatorname{round}\left(x_{i,d}^{(g+1)}\right),
\]

and clamped again to the integer bounds.

This is an integration adaptation rather than a dedicated discrete PSO formulation. The internal velocity remains continuous, while the evaluated candidate is integer-valued.

## 9.8 Exploration and exploitation

Exploration and exploitation emerge from the interaction of:

- random initialization;
- retained velocity;
- stochastic attraction toward `pbest`;
- stochastic attraction toward `gbest`;
- swarm size;
- velocity limits;
- variable bounds.

A stronger inertia contribution and larger allowed velocity generally permit wider movement. Stronger attraction can accelerate movement toward known good regions. However, excessive concentration around one `gbest` can reduce diversity and cause premature convergence.

The fixed global-best topology tends to share improvements rapidly across the swarm. This can produce fast convergence, but may be less robust than local-neighborhood variants on strongly multimodal landscapes.

## 9.9 Use of objective values

PSO does not calculate gradients and does not fit a probabilistic model of the objective. It uses objective values to decide whether one position is better than a stored personal or global best.

The magnitude of the difference does not directly scale the velocity. Once `pbest` and `gbest` locations have been selected, movement depends on geometric distances to those locations, the movement coefficients, and the random factors.

Therefore, strictly order-preserving transformations of valid objective values retain the same best-position comparisons, provided that they do not alter ties, validity, or numerical behavior.

---

# 10. Scientific and technical traceability

Traceability distinguishes three complementary levels.

## 10.1 Scientific literature

The GOW implementation reproduces a classical **global-best Particle Swarm Optimization variant with inertia weight**.

The original formulation comes from **James Kennedy and Russell C. Eberhart (1995)**. The following elements from that work are implemented:

- representation of solutions as particles;
- position and velocity vectors;
- memory of the personal-best position, `pbest`;
- stochastic attraction toward personal experience;
- the global-best model, in which every particle uses the best position found by the swarm, `gbest`.

The inertia-weight term comes from **Yuhui Shi and Russell C. Eberhart (1998)**. This term controls the influence of the previous velocity on the next movement and contributes to the balance between global exploration and local refinement.

The correspondence between these formulations and the GOW code can be observed directly in the velocity and position update equations and in the management of `pbest` and `gbest`.
## 10.2 Mathematical implementation

The particle mathematics is executed directly by the GOW adapter:

```text
src/gow/optimizer/particle_swarm.py
```

The implementation uses the Python standard library, principally:

- `random.Random` for optimizer-specific pseudo-random generation;
- `math.isfinite` for validation of evaluator values.

The adapter itself is responsible for:

- initial position sampling;
- initial velocity sampling;
- `pbest` and `gbest` storage;
- velocity calculation;
- velocity clamping;
- position movement;
- boundary clamping and velocity reset;
- integer rounding;
- invalid-evaluation handling;
- generation and diagnostic state.

No third-party PSO library performs these updates.

## 10.3 Integration into GOW

The GOW integration is responsible for:

- reading the problem and optimizer configuration;
- selecting the optimizable parameters;
- rejecting unsupported optimizable types;
- using `batch_size` as the swarm size;
- coordinating the `ask()`-evaluation-`tell()` cycle;
- invoking the external evaluator;
- preserving candidate-result order;
- converting minimization and maximization into the internal comparison rule;
- combining fixed runtime parameters with particle candidates;
- managing the evaluation budget;
- recording results and optimizer diagnostics.

> In summary, the literature defines the PSO principles, `particle_swarm.py` executes the particle mathematics, and GOW integrates the optimizer into the external evaluation flow.

---

# 11. Validation

## 11.1 Technical integrity

The PSO integration in GOW was evaluated through a formal campaign of **600 executions** on noiseless COCO/BBOB:

- 8 functions;
- dimensions 2, 3, 5, 10, and 20;
- 5 COCO problem instances;
- 3 repetitions per instance;
- 15 executions per function-dimension combination;
- explicit seeds;
- complete `ask()`-evaluation-`tell()` cycles;
- complete swarms;
- a budget of \(100{,}000 \times D\) evaluations;
- no external restarts.

The campaign verified:

- 600 planned executions and 600 verified executions;
- all 40 function-dimension combinations;
- equality between the internal evaluation count and the count recorded by COCO;
- compliance with the evaluation budget;
- consistency of the generation count;
- use of complete swarms;
- reproducibility through explicit seeds;
- absence of non-numeric or non-finite results;
- complete generation of JSON files, logs, and native COCO results;
- absence of final technical errors.

The base PSO integration was accepted as **technically correct within the evaluated scope**.

## 11.2 Behavioral consistency

The campaign used a configuration comparable to the global-best PSO published by **El-Abd and Kamel (2009)**:

- 40 particles;
- inertia weight \(w=0.792\);
- acceleration coefficient \(c=1.4944\) for both the cognitive and social components;
- maximum velocity equal to 50% of the range of each dimension;
- absorbing boundary handling, placing the position on the bound and resetting the velocity;
- a formal budget of \(100{,}000 \times D\) evaluations.

The comparison used success rate, *Expected Running Time* (ERT), and the change in behavior as dimension increased. Across the 40 function-dimension combinations:

- 27 had exactly the same success rate as the reference;
- 8 differed by only one execution out of 15;
- 35 of 40 had either the same success rate or a maximum difference of one success;
- the close-agreement rate was 87.5%;
- the dimension at which robustness began to decline matched the reference for 7 of the 8 functions.

The aggregate pattern was also consistent with the reference: high robustness on Sphere, good behavior in low dimension, and progressive degradation as dimension increased on multimodal, rotated, or non-separable functions. The remaining differences are compatible with the stochastic nature of the algorithm and with the use of different implementations, programming languages, and pseudo-random number generators.

The comparison does not require identical individual trajectories. Its purpose is to verify that the GOW implementation reproduces the aggregate behavior of the reference variant. The results support this correspondence and show no experimental indication of an incorrect implementation.

## 11.3 Functional benchmark integrated into GOW

PSO was also evaluated through the two-dimensional benchmark integrated into the standard GOW flow. Eight functions were executed with seed `123` and a budget of `400` evaluations per function.

PSO completed all eight executions correctly and produced results consistent with the reference values, including:

- Sphere: `0.001330286`, with reference optimum `0`;
- Beale: `0.002145938`, with reference optimum `0`;
- Goldstein-Price: `3.053915676`, with reference optimum `3`;
- McCormick: `-1.913145493`, compared with the approximate reference value `-1.913222955`.

Results directed toward the optimum regions were also obtained for Rosenbrock, Rastrigin, Ackley, and Himmelblau. The benchmark confirmed:

- complete execution of the standard GOW flow;
- compliance with the evaluation budget;
- generation of candidates within the configured bounds;
- functional updating of `pbest` and `gbest`;
- correct generation of results and files;
- consistent behavior on unimodal, multimodal, and non-separable landscapes.

---

# 12. Known limitations

## 12.1 Numerical variables only

The adapter supports optimizable real and integer variables. It rejects optimizable categorical variables because the implemented movement rule requires numerical distances and velocities.

## 12.2 Integer treatment is approximate

Integer parameters use continuous PSO movement followed by rounding. This is not a dedicated discrete or binary PSO. On narrow integer intervals, several different internal positions may produce the same evaluated integer and reduce effective diversity.

## 12.3 Single objective

The implementation compares one scalar objective. It does not construct a Pareto front or implement multi-objective PSO.

## 12.4 Fixed global-best topology

All particles use one shared `gbest`. Local-best neighborhoods, rings, random topologies, and adaptive information graphs are not exposed.

The global-best topology can spread improvements quickly, but it can also concentrate the swarm prematurely around a local optimum.

## 12.5 Fixed movement parameters

`inertia_weight`, `acceleration_coefficient`, and `velocity_clamp_fraction` remain constant throughout the run. The adapter does not implement time-varying inertia, adaptive coefficients, self-adaptation, or automatic parameter schedules.

## 12.6 Shared cognitive and social coefficient

The cognitive and social components use the same configurable acceleration coefficient. Their deterministic weights cannot be tuned independently in the current YAML interface.

## 12.7 Boundary bias

A particle that leaves the domain is placed directly on the violated bound and its velocity in that dimension is set to zero. This guarantees feasibility, but may accumulate particles on boundaries or reduce movement near them.

Other boundary-handling strategies described in the PSO literature are not available through this adapter.

## 12.8 Premature convergence and strong multimodality

The implementation has no automatic restart, mutation, niching, repulsion, or diversity-restoration mechanism. Once particles become concentrated around the same `gbest`, escape from a poor basin may be difficult.

## 12.9 Scaling with dimensionality

The arithmetic cost per generation is approximately proportional to swarm size times dimension. More importantly, the volume of a bounded search space grows rapidly with dimension, and a fixed swarm and budget may provide insufficient coverage.

The implementation does not adapt the swarm size automatically to dimension.

## 12.10 Fixed swarm size and complete generations

The swarm size cannot change during a run. Every `ask()` and `tell()` must use exactly `batch_size` candidates.

A budget that leaves a partial final batch is incompatible with the adapter. Configure `max_evaluations` as a multiple of `batch_size`.

## 12.11 Standard executor termination

The adapter exposes `is_done()` when `max_generations` is reached, but the selected executor must consult that state to stop immediately. The standard budget loop is primarily governed by `max_evaluations`, so both limits should be configured consistently.

## 12.12 Invalid evaluations

An invalid evaluation cannot update `pbest` or `gbest`. This allows the run to continue when some evaluations fail, but invalid points provide no useful search information.

If all evaluations fail and no `gbest` exists, the next generation is randomly reinitialized. Persistent evaluator failure therefore prevents meaningful learning.

## 12.13 Evaluator comparison key

The verified and recommended key is `objective`. Alternative compatibility keys should not replace the documented evaluator contract without a dedicated integration test, particularly when combining a `loss` field with the configured objective direction.

---

# 13. References

1. Kennedy, J. and Eberhart, R. C. (1995). **Particle Swarm Optimization**. *Proceedings of the IEEE International Conference on Neural Networks*, vol. 4, pp. 1942-1948. [IEEE Xplore](https://ieeexplore.ieee.org/document/488968) · [DOI: 10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).

2. Shi, Y. and Eberhart, R. C. (1998). **A Modified Particle Swarm Optimizer**. *Proceedings of the IEEE International Conference on Evolutionary Computation*, pp. 69-73. [IEEE Xplore](https://ieeexplore.ieee.org/document/699146) · [DOI: 10.1109/ICEC.1998.699146](https://doi.org/10.1109/ICEC.1998.699146).

3. El-Abd, M. and Kamel, M. S. (2009). **Black-Box Optimization Benchmarking for Noiseless Function Testbed Using Particle Swarm Optimization**. *Proceedings of GECCO 2009*, pp. 2269-2274. [Reference paper](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsGECCO09/p2269-elabd.pdf) · [Official BBOB results archive](https://coco-platform.org/testsuites/bbob/data-archive.html).

4. CST Modelling Tools. **GOW: Architecture, Evaluator Contract, and Provenance**. [GOW architecture and usage documentation](https://cst-modelling-tools.github.io/generic-optimization-workflow-blog/gow-architecture-and-usage).

5. Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021). **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**. *Optimization Methods and Software*, 36(1), 114-144. [DOI: 10.1080/10556788.2020.1808977](https://doi.org/10.1080/10556788.2020.1808977) · [Official COCO platform](https://coco-platform.org/) · [Official BBOB suite description](https://coco-platform.org/testsuites/bbob/overview.html) · [Official BBOB results archive](https://coco-platform.org/testsuites/bbob/data-archive.html) · [Public `numbbo/coco` repository](https://github.com/numbbo/coco).

---

## Final summary

The documented implementation is a bounded, single-objective, global-best PSO with constant inertia weight, one shared acceleration coefficient, independent cognitive and social random factors, velocity clamping, synchronous generations, and a fixed swarm size.

GOW executes the external evaluations and coordinates the `ask()`-evaluation-`tell()` cycle. The PSO adapter directly implements initialization, movement, memory updates, objective comparison, bound handling, integer rounding, and diagnostics.

The user configures numerical variables with valid bounds, the objective direction, the external evaluator, the seed, the evaluation budget, `batch_size`, `max_generations`, `inertia_weight`, `acceleration_coefficient`, and `velocity_clamp_fraction`.

The current scope is single-objective optimization of bounded real and integer variables, with complete fixed-size generations, no categorical movement, no local topology, no adaptive parameter schedule, and no automatic restarts.
