# CMA-ES in GOW

## Usage, Configuration, Implementation, and Scientific Traceability Guid

---

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Implemented CMA-ES variant](#2-implemented-cma-es-variant)
3. [When to use CMA-ES](#3-when-to-use-cma-es)
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

# 2. Implemented CMA-ES variant

The implementation available in GOW corresponds to **Active CMA-ES**, a continuous, single-objective, non-elitist variant of CMA-ES with weighted recombination, cumulative step-size adaptation, and rank-one and rank-\(\mu\) covariance matrix updates. The general CMA-ES formulation is based mainly on Hansen (2006), while active covariance adaptation was introduced by Jastrebski and Arnold (2006).

It is intended exclusively for continuous real-valued parameters. The internal mathematical operations are provided by the `cma` library, while GOW is responsible for integrating them into the `ask()`-`tell()` flow.

---

# 3. When to use CMA-ES

CMA-ES is appropriate for black-box problems in which:

- the optimizable variables are continuous;
- derivatives are unavailable or unreliable;
- the function may be nonlinear or non-convex;
- there are substantial scale differences between variables;
- there may be correlations or interactions between variables;
- the problem may be ill-conditioned or non-separable;
- sufficient budget is available to evaluate several complete generations.

In GOW, CMA-ES serves as a general continuous optimizer for low- or medium-dimensional problems. Its full covariance matrix allows it to adapt both the scales and the principal search directions.

It is generally not the most suitable option when:

- the problem mainly contains integer, Boolean, or categorical variables;
- the structure is essentially combinatorial;
- several simultaneous objectives are required;
- the evaluation budget is extremely small;
- the dimension is so high that the internal cost of a full matrix becomes prohibitive;
- a restart policy is needed to address strong multimodality and has not been added externally.

---

# 4. Requirements and functional scope

## 4.1 Variable types

The current CMA-ES implementation in GOW optimizes only continuous real-valued parameters (`RealParam`). Although extensions of the CMA-ES family exist for integer or mixed variables, they are not included in this implementation. Integer and categorical parameters may only appear as fixed values with `optimizable: false`.

## 4.2 Problem type

| Characteristic | Current support |
|---|---:|
| Black box | Yes |
| Objective | Single, with `minimize` or `maximize` direction |
| Lower and upper bounds for each variable | Yes |
| Specific handling of general black-box constraints | No |

---

# 5. Quick start

## 5.1 Minimal YAML example

The following example defines a continuous two-variable problem:

```yaml
id: sphere-cmaes

parameters:
  x:
    type: real
    value: 0.0
    bounds: [-5.0, 5.0]
    optimizable: true

  y:
    type: real
    value: 0.0
    bounds: [-5.0, 5.0]
    optimizable: true

evaluator:
  command:
    - "/path/to/evaluator"
  timeout_s: 60

objective:
  direction: minimize

optimizer:
  name: cmaes
  seed: 123
  max_evaluations: 120
  batch_size: 6

  settings:
    sigma0: 0.05
    max_generations: 20
```

In this example:

```text
6 candidates per generation × 20 generations = 120 evaluations
```

## 5.2 Expected evaluator result

The recommended format for a successful evaluation is:

```json
{
  "status": "ok",
  "objective": 0.03125,
  "metrics": {
    "sphere": 0.03125
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

## 5.3 Execution

From the virtual environment and the appropriate directory:

```bash
gow run path/to/problem.yaml
```

The organization of the results and the complete evaluator contract are described in the general GOW documentation indicated in Section 1.

---

# 6. Optimizer configuration

## 6.1 Exposed parameters

- `name` (`optimizer.name`): selects CMA-ES and must have the value `cmaes`.
- `seed` (`optimizer.seed`): seed passed to the CMA-ES engine.
- `max_evaluations` (`optimizer.max_evaluations`): total budget managed by the GOW executor.
- `batch_size` (`optimizer.batch_size`): number of candidates per generation and population size \(\lambda\).
- `sigma0` (`optimizer.settings.sigma0`): initial global step size in normalized space.
- `max_generations` (`optimizer.settings.max_generations`): generation limit stored by the adapter.

A separate parameter named `population_size` must not be added: in this integration, `batch_size` directly determines the population size \(\lambda\).

## 6.2 `batch_size`

`batch_size` defines the number of candidates in each generation and corresponds to the CMA-ES population size \(\lambda\). It must be at least `2` and remain constant throughout the execution.

The usual recommendation may be used as an initial guideline:

\[
\lambda = 4 + \left\lfloor 3\ln(n) \right\rfloor,
\]

where \(n\) is the number of optimizable continuous variables.

GOW does not calculate this value automatically. The user must set it through `batch_size` in the YAML file.

A larger population may increase exploration and facilitate parallel evaluation, but it consumes more evaluations per generation. A smaller population allows more generations to be completed with the same budget, although less selection information is available in each update.

## 6.3 `sigma0`

`sigma0` is the initial global step size in the normalized domain \([0,1]^n\):

```yaml
settings:
  sigma0: 0.05
```

In general:

- a small value starts a more local search around the initial mean;
- a larger value increases the dispersion of the first population;
- the value must be strictly positive.

The choice depends on the quality of the initial point, the width of the bounds, the problem dimension, and the available budget.

## 6.4 `seed`

The seed is passed to the CMA-ES engine. To reproduce an execution, the following must also remain constant:

- the version of the `cma` dependency;
- the YAML file;
- the evaluation order;
- the evaluator and its dependencies;
- the way each batch is executed;
- the numerical environment.

## 6.5 Budget, generations, and termination

`max_evaluations` sets the maximum evaluation budget for the execution. Because CMA-ES works with complete populations, this value must be a multiple of `batch_size`.

When `max_generations` is also configured, both limits should remain consistent:

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
  batch_size: 16
  settings:
    max_generations: 100
```

```text
16 × 100 = 1600
```

The execution may also reach internal `pycma` stopping criteria. Their immediate application depends on whether the executor checks the optimizer state before requesting a new population.

---

# 7. `ask()`-evaluation-`tell()` flow

CMA-ES is a generational optimizer. Each generation follows this cycle:

```text
ask()
  ↓
normalized CMA-ES candidates
  ↓
conversion to GOW physical values
  ↓
external evaluation of all candidates
  ↓
collection of objectives in the same order
  ↓
tell()
  ↓
update of mean, sigma, and covariance
```

## 7.1 Deferred initialization

The CMA-ES object is created when the first call to `ask(problem, n)` is received, because the following are available at that point:

- the optimizable parameters;
- their bounds;
- their initial values;
- the objective direction;
- the population size requested by GOW.

## 7.2 Cycle rules

The implementation requires:

1. one call to `ask()`;
2. evaluation of the complete population;
3. a single call to `tell()` with that population;
4. a new call to `ask()` only after the previous `tell()`.

Two consecutive calls to `ask()` are not allowed because the correspondence with the normalized vectors that must be returned to the engine would be lost.

`tell()` checks that:

- a previous population exists;
- the number of candidates matches the number of results;
- exactly one complete population is received;
- the normalized vectors from the last `ask()` are still available.

## 7.3 Importance of order

Order is part of the contract:

```text
candidates[i] ↔ fitness[i]
```

GOW must preserve the same order from candidate generation to the call to `tell()`. The adapter returns to the engine the exact vectors generated by the last `ask()` and the corresponding losses in the same order.

---

# 8. Integration-specific behavior

- GOW automatically converts minimization and maximization problems to the format expected by `pycma`; the user must not manually change the sign of the objective.
- The recommended key for the optimizable result is `objective`. When `loss` is used, it is always interpreted as a quantity to be minimized.
- Missing, non-numeric, or non-finite evaluations are penalized so that the execution can continue.
- CMA-ES works internally in the normalized domain \([0,1]^n\). The `bounds` determine the conversion to physical space, and the `value` entries define the initial mean.

For implementation details, internal validations, and the handling of exceptional cases, see the `src/gow/optimizer/cmaes.py` adapter, which is documented through comments in the source code.

---

# 9. Mathematical foundations

This section presents the essential equations needed to identify the variant. The concrete numerical implementation is the responsibility of `pycma`.

## 9.1 Notation

- \(n\): number of optimizable continuous variables;
- \(g\): generation;
- \(\lambda\): number of candidates in a generation;
- \(\mu\): number of candidates selected for positive recombination;
- \(\mathbf m^{(g)}\): distribution mean;
- \(\sigma^{(g)}\): global step size;
- \(\mathbf C^{(g)}\): covariance matrix;
- \(w_i\): recombination weights;
- \(\mu_{\mathrm{eff}}=(\sum_i w_i^2)^{-1}\): effective selection mass.

## 9.2 Candidate sampling

CMA-ES generates each candidate from a multivariate normal distribution:

\[
\mathbf x_k^{(g+1)}
=
\mathbf m^{(g)}
+
\sigma^{(g)}
\left(\mathbf C^{(g)}\right)^{1/2}
\mathbf z_k,
\qquad
\mathbf z_k\sim\mathcal N(\mathbf 0,\mathbf I),
\]

for \(k=1,\ldots,\lambda\).

The mean determines the center; \(\sigma\) controls the global scale; \(\mathbf C\) controls the shape and orientation of the distribution.

## 9.3 Selection and recombination

After the \(\lambda\) candidates have been evaluated, they are ranked according to the objective. The new mean is obtained through a weighted combination of the best candidates:

\[
\mathbf m^{(g+1)}
=
\sum_{i=1}^{\mu}
 w_i\mathbf x_{i:\lambda}^{(g+1)},
\qquad
\sum_{i=1}^{\mu}w_i=1.
\]

The best candidates normally receive larger positive weights.

## 9.4 Evolution path and rank-one update

CMA-ES accumulates successive mean displacements in an evolution path. Schematically:

\[
\mathbf p_c^{(g+1)}
=
(1-c_c)\mathbf p_c^{(g)}
+
h_\sigma
\sqrt{c_c(2-c_c)\mu_{\mathrm{eff}}}
\frac{\mathbf m^{(g+1)}-\mathbf m^{(g)}}{\sigma^{(g)}}.
\]

The rank-one term uses \(\mathbf p_c\mathbf p_c^\mathsf T\) to increase variance in directions that have produced coherent movements over several generations.

## 9.5 Rank-\(\mu\) update

The selected steps from the current generation contribute through outer products:

\[
\mathbf y_{i:\lambda}
=
\frac{\mathbf x_{i:\lambda}^{(g+1)}-\mathbf m^{(g)}}{\sigma^{(g)}}.
\]

A simplified form of the combined update is:

\[
\mathbf C^{(g+1)}
=
(1-c_1-c_\mu)\mathbf C^{(g)}
+
c_1\mathbf p_c^{(g+1)}\mathbf p_c^{(g+1)\mathsf T}
+
c_\mu\sum_i w_i
\mathbf y_{i:\lambda}\mathbf y_{i:\lambda}^{\mathsf T}.
\]

The rank-one term accumulates historical information. The rank-\(\mu\) term uses the selected population from the current generation.

## 9.6 Active covariance adaptation

Active CMA-ES also uses information from poorly ranked candidates. Negative weights reduce variance in directions associated with unfavorable results, with safeguards to preserve a valid covariance matrix.

In GOW, this variant is explicitly requested through:

```python
"CMA_active": True
```

## 9.7 Cumulative step-size adaptation

*Cumulative Step-Size Adaptation* (CSA) maintains a second path, usually denoted by \(\mathbf p_\sigma\), in isotropic coordinates:

\[
\mathbf p_\sigma^{(g+1)}
=
(1-c_\sigma)\mathbf p_\sigma^{(g)}
+
\sqrt{c_\sigma(2-c_\sigma)\mu_{\mathrm{eff}}}
\mathbf C^{-1/2}
\frac{\mathbf m^{(g+1)}-\mathbf m^{(g)}}{\sigma^{(g)}}.
\]

The step size is updated by comparing the path length with the expected length of a standard normal distribution:

\[
\sigma^{(g+1)}
=
\sigma^{(g)}
\exp\left(
\frac{c_\sigma}{d_\sigma}
\left(
\frac{\lVert\mathbf p_\sigma^{(g+1)}\rVert}
{\mathbb E\lVert\mathcal N(\mathbf 0,\mathbf I)\rVert}
-1
\right)
\right).
\]

Persistently aligned movements tend to increase \(\sigma\); oscillations or paths that are too short tend to reduce it.

## 9.8 Use of ranking

The updates depend mainly on the ordering of the candidates, not on the absolute differences between objective values. This property provides invariance under strictly increasing transformations of the objective, provided that they preserve the ranking.

---

# 10. Scientific and technical traceability

Traceability distinguishes three complementary levels.

## 10.1 Scientific literature

The literature defines the foundations of CMA-ES:

- sampling from a multivariate normal distribution;
- weighted selection and recombination;
- evolution paths;
- cumulative step-size adaptation;
- rank-one update;
- rank-\(\mu\) update;
- active covariance adaptation.

The general formulation is based mainly on the work of Hansen and Hansen-Ostermeier. Active adaptation is based on the work of Jastrebski and Arnold.

## 10.2 Mathematical implementation: `pycma`

The `pycma` library executes the internal mathematical operations of the algorithm through:

```text
cma.CMAEvolutionStrategy
```

The dependency used by the integration is:

```toml
cma==4.4.4
```

The version is pinned to preserve the internal options and the verified behavior of the integration.

`pycma` is responsible for:

- generating the population;
- selecting and recombining candidates;
- adapting the mean and step size;
- updating the covariance matrix;
- applying the rank-one, rank-\(\mu\), and active updates;
- handling the normalized bounds through `BoundTransform`.

## 10.3 Integration into GOW

The adapter is located at:

```text
src/gow/optimizer/cmaes.py
```

GOW does not reimplement the internal equations. The adapter is responsible for:

- reading the problem configuration;
- selecting the optimizable continuous parameters;
- normalizing and denormalizing their values;
- constructing the initial mean;
- calling `ask()` and `tell()`;
- executing the external evaluator;
- converting the objective into the loss expected by `pycma`;
- penalizing invalid evaluations;
- preserving the order between candidates and results;
- controlling the fixed population size and the execution budget.

> In summary, the literature defines the algorithm, `pycma` executes it, and GOW integrates it into the optimization flow.

---

# 11. Validation

## 11.1 Technical integrity

The base CMA-ES integration in GOW was evaluated through a formal campaign of **600 executions** on noiseless COCO/BBOB:

- 8 functions;
- dimensions 2, 3, 5, 10, and 20;
- 15 executions per function-dimension combination;
- explicit seeds;
- `ask()`-evaluation-`tell()` cycle;
- complete populations;
- normalized bounds;
- no external restarts.

The campaign verified:

- budget control;
- correspondence between the internal and COCO counts;
- the use of complete populations;
- reproducibility through seeds;
- correct generation of the campaign results and files;
- the absence of final technical errors.

The base integration was accepted as technically correct.

## 11.2 Behavioral consistency

The pattern of results was consistent with the expected behavior of a full-matrix CMA-ES without external restarts:

- high robustness on Sphere, Rosenbrock, and Ellipsoid;
- reduced robustness on strongly multimodal functions such as Rastrigin, Griewank-Rosenbrock, and Gallagher.

This deterioration on certain functions corresponds to the characteristics of the evaluated variant and configuration. By itself, it does not constitute evidence of an incorrect implementation.

The campaign does not exactly reproduce the historical reference experiments because there are differences in initialization, step size, bounds, seeds, budgets, and restart policy. The comparison is therefore used to verify qualitative behavioral consistency, not to require identical success rates.

## 11.3 Functional benchmark integrated into GOW

CMA-ES was also evaluated through the two-dimensional benchmark incorporated into GOW. Eight functions were executed with seed `123` and a budget of `400` evaluations per function. CMA-ES successfully completed all executions and produced results consistent with the reference values, including \(5.046\times10^{-8}\) on Sphere, \(1.049\times10^{-6}\) on Beale, and \(3.000081202\) on Goldstein-Price.

This benchmark confirms the basic operation of CMA-ES within the standard GOW flow.

---

# 12. Known limitations

## 12.1 Continuous variables only

The adapter rejects optimizable integer and categorical variables. Parameters of these types may only appear as fixed values with `optimizable: false`.

## 12.2 Full covariance matrix

CMA-ES learns an \(n\times n\) matrix. Storage and many internal operations scale quadratically with the dimension. Matrix decomposition may have a cubic cost per call, although practical implementations amortize it by performing it less frequently.

For this reason, this optimizer may cease to be suitable when the dimension is very high.

## 12.3 Strong multimodality and restarts

A single execution may converge to a local basin. The current implementation does not include automatic restart strategies such as IPOP or BIPOP.

IPOP restarts CMA-ES while progressively increasing the population size. BIPOP alternates executions with small and large populations. These strategies are not part of the current adapter.

## 12.4 Single objective

The implementation processes one scalar comparison quantity. It does not implement multi-objective optimization.

## 12.5 Fixed population

The batch size cannot change during the execution. A budget that leaves a partial final population is incompatible with the adapter.

## 12.6 Standard executor termination

The adapter exposes its termination state, but the executor must check it to apply the internal `pycma` stopping criteria immediately. While the loop is governed mainly by `max_evaluations`, the budget and number of generations should remain consistent.

## 12.7 Invalid evaluations

Penalization with a large loss allows the execution to continue, but it provides no useful information to the optimizer. If an entire population fails, the result is a flat generation with no real learning capacity.

---

# 13. References

1. Hansen, N. (2006). **The CMA Evolution Strategy: A Comparing Review**. In *Towards a New Evolutionary Computation*, Studies in Fuzziness and Soft Computing, vol. 192, pp. 75-102.

2. Hansen, N. (2023). **The CMA Evolution Strategy: A Tutorial**. Version compiled on March 13, 2023.

3. Hansen, N. and Ostermeier, A. (2001). **Completely Derandomized Self-Adaptation in Evolution Strategies**. *Evolutionary Computation*, 9(2), 159-195.

4. Jastrebski, G. A. and Arnold, D. V. (2006). **Improving Evolution Strategies through Active Covariance Matrix Adaptation**. *IEEE Congress on Evolutionary Computation*, pp. 2814-2821.

5. CST Modelling Tools. **GOW: Architecture, Evaluator Contract, and Provenance**. [GOW architecture and usage documentation](https://cst-modelling-tools.github.io/generic-optimization-workflow-blog/gow-architecture-and-usage).

6. Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021). **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**. *Optimization Methods and Software*, 36(1), 114-144. [Official COCO platform](https://coco-platform.org/) · [Official BBOB suite description](https://coco-platform.org/testsuites/bbob/overview.html) · [Public `numbbo/coco` repository](https://github.com/numbbo/coco).

---

## Final summary

The documented implementation is a continuous, single-objective, full-matrix Active CMA-ES. `pycma` executes the mathematical updates, while GOW normalizes the domain, coordinates the `ask()`-evaluation-`tell()` cycle, transforms the results, and controls the execution.

The user configures real-valued variables with bounds, the objective direction, the external evaluator, the seed, the budget, `batch_size`, `sigma0`, and `max_generations`.

The current scope is continuous, single-objective, with per-variable bounds, a fixed population, and no automatic restarts.
