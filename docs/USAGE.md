# Generic Optimization Workflow (GOW) – User Guide

This document explains how to use the Generic Optimization Workflow (GOW) framework
to define, run, and analyze optimization problems.

The framework itself is problem-agnostic. Users are expected to create their own
project folders outside the GOW repository and run optimizations from there.

----------------------------------------------------------------
# 1. Installation
----------------------------------------------------------------

Clone the repository and install it in editable mode:

```bash
  git clone <repository-url>
  cd generic-optimization-workflow
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
```

(Optional, for FireWorks support)

```bash
  pip install -e ".[fireworks]"
```

----------------------------------------------------------------
# 2. Recommended Project Layout
----------------------------------------------------------------

Each optimization project should live OUTSIDE the GOW repository.

Example:

```bash
  my-optimization-project/
  ├── problem.yaml
  └── results/
```  

- problem.yaml defines the optimization problem
- results/ will be created automatically by GOW
- the GOW repository itself stays clean and free of run artifacts

----------------------------------------------------------------
# 3. Running a Local Optimization (No FireWorks)
----------------------------------------------------------------

From inside your project directory:

```bash
  gow run problem.yaml
```

This will create the following structure:

```bash
  results/
    <problem_id>/
      results.jsonl
      summary.json
      runs/
        <run_id>/
          results.jsonl
          summary.json
          c000000/
          c000001/
          ...
```

If you want to control the run id:

```bash
  gow run problem.yaml --run-id myrun1
```

----------------------------------------------------------------
# 4. Evaluating a Single Candidate (Debugging)
----------------------------------------------------------------

You can evaluate a single candidate without running a full optimization loop:

```bash
  gow evaluate problem.yaml --run-id debug --candidate-id test1 -p x=0.2 -p y=0.4
```

This is useful for:
- testing external evaluators
- debugging parameter mappings
- validating problem.yaml

----------------------------------------------------------------
# 5. Viewing the Best Result
----------------------------------------------------------------

To inspect the best candidate from a run:

```bash
  gow best results/<problem_id>/runs/<run_id>
```

To show more than one result:

```bash
  gow best results/<problem_id>/runs/<run_id> --top 5
```

----------------------------------------------------------------
# 6. FireWorks-Based Execution
----------------------------------------------------------------

FireWorks enables distributed and parallel execution using a MongoDB-backed queue.

Before using FireWorks, ensure:
- MongoDB is running
- A FireWorks launchpad YAML file exists

Example:

```bash
  my_launchpad.yaml
```

----------------------------------------------------------------
# 7. FireWorks: Single Evaluation
----------------------------------------------------------------

Submit a single evaluation to FireWorks:

```bash
  gow fw evaluate problem.yaml --launchpad my_launchpad.yaml -p x=0.1 -p y=0.2
```

To submit AND immediately launch:

```bash
  gow fw evaluate problem.yaml --launchpad my_launchpad.yaml --launch
```

Artifacts will appear under:

```bash
  results/
    <problem_id>/
      runs/
      results.jsonl
```

Launcher directories are written to:

```bash
  results/launchers/
```

----------------------------------------------------------------
# 8. FireWorks: Full Optimization Loop
----------------------------------------------------------------

Submit and run a full optimization loop:

```bash
  gow fw run problem.yaml --launchpad my_launchpad.yaml
```

This will:
- submit candidates in batches
- launch them automatically
- collect results
- feed results back to the optimizer

All artifacts remain under the project’s results/ directory.

----------------------------------------------------------------
# 9. Environment Variable Override
----------------------------------------------------------------

You may override the base results directory using:

```bash
  export GOW_OUTDIR=/path/to/results
```

This is useful for:
- shared filesystems
- cluster environments
- keeping results separate from configs

----------------------------------------------------------------
# 10. Output Files
----------------------------------------------------------------

results.jsonl
- One JSON record per evaluated candidate
- Append-only
- Suitable for streaming and post-processing

summary.json
- Best candidate found
- Objective direction
- Run metadata

----------------------------------------------------------------
# 11. Philosophy and Design Principles
----------------------------------------------------------------

- GOW is a framework, not a project runner
- Problem definitions live with the user
- Results never pollute the repository
- Local and distributed execution share the same layout
- FireWorks is optional

