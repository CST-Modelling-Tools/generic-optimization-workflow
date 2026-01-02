# generic-optimization-workflow

Generic optimization framework for scientific workflows using FireWorks.

This repository focuses on **optimization orchestration** (candidate generation, scheduling, provenance, result collection),
while **scientific evaluation workflows** live in independent projects and are executed as external programs.

---

## Core idea

- The **optimizer loop** proposes candidate parameter sets.
- Each candidate is evaluated by a **scientific evaluator**.
- The evaluator returns one or more **fitness metrics** (and optionally a scalar objective).
- The optimizer updates its state and proposes the next candidates.

This design keeps the scientific code and its dependencies isolated and versioned independently from the optimization framework.

---

## Scientific Evaluator Contract (External Executable)

Scientific workflows are implemented as **external executables** (often C++ for performance) and can be maintained in
separate repositories. The optimization framework treats evaluators as black boxes that follow this contract:

### 1) CLI interface

Evaluators MUST be runnable as a command-line program that accepts an input JSON file and produces an output JSON file.

Recommended standard:

```bash
<evaluator_exe> --input input.json --output output.json
```

(Additional flags are allowed, but --input and --output must be supported.)

### 2) Working directory layout

Each evaluation runs in its own working directory (created by the optimization framework), for example:

```bash
runs/<problem_id>/<run_id>/<candidate_id>/
  input.json
  output.json
  stdout.txt
  stderr.txt
  logs/
  artifacts/
```

The optimizer writes input.json.

The evaluator writes output.json and any additional artifacts.

The framework captures stdout/stderr for debugging.

### 3) Input JSON (input.json)

At minimum:

run_id (string)

candidate_id (string)

params (object): the runtime parameter values for this candidate

context (object): optional metadata (datasets, location, seeds, etc.)

Example:

```json
{
  "run_id": "7c3f3a2a-7c40-4c7b-b9c6-5b02f3b6c6d0",
  "candidate_id": "c000123",
  "params": {
    "k1": 0.123,
    "k2": -1.2,
    "n_terms": 14
  },
  "context": {
    "problem": "sunpos",
    "reference_dataset": "mica_2026a",
    "seed": 42
  }
}
```

Notes:

params should contain already-cast values (floats/ints/strings) ready for computation.

If you maintain a richer parameter schema elsewhere (types, bounds, etc.), this should be resolved before writing input.json

### 4) Output JSON (output.json)

Evaluators MUST produce an output JSON file with:

status: "ok" or "failed"

metrics: object of named metrics (floats/ints)

objective: optional scalar (float) for single-objective optimization

constraints: optional object (e.g., runtime, feasibility flags)

artifacts: optional object of relative paths to produced files

error: optional error message if status="failed"

Example:

```json
{
  "status": "ok",
  "metrics": {
    "avg_error_deg": 0.42,
    "max_error_deg": 1.90,
    "min_error_deg": 0.02
  },
  "objective": 0.42,
  "constraints": {
    "runtime_s": 58.1
  },
  "artifacts": {
    "log": "logs/run.log",
    "details_csv": "artifacts/errors.csv"
  }
}
```

### 5) Failure behaviour

Evaluators SHOULD attempt to write output.json even when they fail. For failures:

Set status to "failed"

Include an error message if possible

Leave metrics empty or partial

Example:

```json
{
  "status": "failed",
  "metrics": {},
  "objective": null,
  "error": "Numerical solver did not converge."
}
```

The optimization framework can then decide how to handle failures (retry, penalize, skip, etc.).

### 6) Evaluator-internal orchestration is allowed

An evaluator executable may internally run multiple programs/steps, for example:

generate a configuration / geometry

run one or more simulations (possibly parallel)

post-process results

compute metrics and write output.json

This internal structure is completely up to the evaluator. Only the input/output contract above must be respected.

## Licensing note

This repository provides the optimization orchestration framework. Scientific evaluators can live in separate repositories
and may be licensed independently, as long as they comply with the evaluator contract.