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

## Toy Example

The repository ships with a small toy example in `examples/toy/optimization_specs.yaml`
that evaluates a simple sphere objective using `x` and `y`. The user-facing examples
below use that toy framing so the provenance and evaluator contract stay consistent
with the code you can run locally.

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
<outdir>/runs/<run_id>/<candidate_id>/
  input.json
  output.json
  stdout.txt
  stderr.txt
  logs/
  artifacts/
```

GOW writes input.json.

The evaluator writes output.json and any additional artifacts.

The framework captures stdout/stderr for debugging.

GOW currently keeps one work directory per logical candidate. If a caller re-executes the
same candidate with a higher attempt index, the provenance identifiers distinguish attempts,
but the directory layout is intentionally unchanged.

### 3) Provenance identifiers

- `run_id` identifies the optimization run.
- `candidate_id` identifies the logical candidate proposed by the optimizer.
- `attempt_id` identifies one concrete execution attempt of that candidate.
- `candidate_local_id` preserves the legacy local label `g<generation>_c<candidate>`.
- `candidate_index` is the zero-based global candidate/evaluation sequence number within a run.

Canonical formats:

```text
candidate_id      = r7c3f3a2a_g000002_c000014
candidate_local_id = g000002_c000014
attempt_id        = r7c3f3a2a_g000002_c000014_a000
```

Why this matters:

- `candidate_id` stays human-readable while remaining globally unique across runs.
- `attempt_id` preserves provenance if the same candidate is retried or manually re-executed.
- `candidate_local_id` lets existing tooling keep using the older generation/candidate label.

Identifier semantics:

- `generation_id` is the zero-based generation/batch number.
- the `c......` field in `candidate_local_id` and `candidate_id` uses the run-global `candidate_index`
- `r7c3f3a2a_g000002_c000014` therefore means generation `2`, global candidate index `14`

### 4) Input JSON (input.json)

At minimum:

run_id (string)

candidate_id (string)

candidate_local_id (string, optional legacy/local label)

attempt_id (string)

params (object): the runtime parameter values for this candidate

context (object): optional metadata (datasets, location, seeds, etc.)

Example:

```json
{
  "run_id": "7c3f3a2a-7c40-4c7b-b9c6-5b02f3b6c6d0",
  "candidate_id": "r7c3f3a2a_g000002_c000014",
  "candidate_local_id": "g000002_c000014",
  "attempt_id": "r7c3f3a2a_g000002_c000014_a000",
  "params": {
    "x": 0.5,
    "y": -0.25,
    "n": 5,
    "mode": "a"
  },
  "context": {
    "note": "toy example using Differential Evolution"
  }
}
```

Notes:

params should contain already-cast values (floats/ints/strings) ready for computation.

`candidate_id` is the primary provenance identifier for the logical candidate. `attempt_id`
identifies the specific execution attempt. The first attempt is always `a000`.

If you maintain a richer parameter schema elsewhere (types, bounds, etc.), this should be resolved before writing input.json

### 5) Output JSON (output.json)

Evaluators MUST produce an output JSON file with:

status: "ok" or "failed"

metrics: object of named metrics (floats/ints)

objective: scalar used by the optimizer for single-objective optimization

constraints: optional object (e.g., runtime, feasibility flags)

artifacts: optional object of relative paths to produced files

error: optional error message if status="failed"

Example:

```json
{
  "status": "ok",
  "metrics": {
    "sphere": 0.3125
  },
  "objective": 0.3125,
  "artifacts": {}
}
```

Notes:

- `metrics` is the general container for evaluator outputs.
- `objective` is the single scalar value the optimizer compares when selecting the best candidate.
- successful optimizable evaluations should provide `objective`.
- failed evaluations may omit `objective` or set it to `null`.
- `objective` may duplicate one metric, but it has a distinct role in optimization and provenance.

### 6) GOW Provenance Records

GOW stores richer provenance alongside evaluator output in `result.json` and `results.jsonl`.
These records include:

- `failure_kind`: machine-readable failure category when GOW detects a failure mode such as
  `missing_output`, `timeout`, `nonzero_exit`, or `invalid_output`
- `started_at` / `finished_at`: UTC timestamps in ISO 8601 format
- `wall_time_s`: elapsed execution time in seconds
- `evaluator`: a small execution snapshot with the resolved command, timeout, and extra args

### 7) Failure behaviour

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

### 8) Current attempt semantics

GOW does not currently implement an automatic retry policy in the local or FireWorks runners.

Current behavior:

- automatic optimization runs create one attempt per candidate, so generated attempts are `a000`
- manual re-execution can use a higher attempt index explicitly
- `gow evaluate` auto-generates a canonical candidate id when `--generation-id` and `--candidate-index` are provided
- if those metadata are omitted, `gow evaluate` falls back to the explicit non-canonical id `manual`
- JSONL append logic is keyed by `attempt_id` when available, so repeated executions can be preserved as separate attempts

### 9) Evaluator-internal orchestration is allowed

An evaluator executable may internally run multiple programs/steps, for example:

generate a configuration / geometry

run one or more simulations (possibly parallel)

post-process results

compute metrics and write output.json

This internal structure is completely up to the evaluator. Only the input/output contract above must be respected.

## Licensing note

This repository provides the optimization orchestration framework. Scientific evaluators can live in separate repositories
and may be licensed independently, as long as they comply with the evaluator contract.
