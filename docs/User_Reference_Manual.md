# Generic Optimization Workflow (GOW)

## User Reference Manual

------------------------------------------------------------------------

## 1. Overview

The **Generic Optimization Workflow (GOW)** provides a structured and
reproducible way to define and execute optimization problems.

A problem configuration:

-   Defines parameters (real, int, categorical)
-   Specifies how candidates are evaluated
-   Configures the optimizer
-   Defines objective direction
-   Stores optional contextual metadata

This manual uses a **toy polynomial regression example** to illustrate
the configuration format.

------------------------------------------------------------------------

## 2. Toy Example Problem

We optimize a simple polynomial model:

f(x) = k1 \* x + k2 \* x\^2

The optimizer searches for the best parameters:

-   `k1`
-   `k2`
-   `n_terms`

to minimize the mean squared error on a synthetic dataset.

------------------------------------------------------------------------

## 3. Evaluator Contract

Every external evaluator must follow this contract.

### Input (`input.json`)

``` json
{
  "run_id": "7c3f3a2a-7c40-4c7b-b9c6-5b02f3b6c6d0",
  "candidate_id": "c000123",
  "params": {
    "k1": 0.123,
    "k2": -1.2,
    "n_terms": 14
  },
  "context": {
    "problem": "toy_polynomial_regression",
    "dataset": "synthetic_v1",
    "seed": 42
  }
}
```

### Output (`output.json`)

``` json
{
  "objective": 0.04231,
  "metrics": {
    "mse": 0.04231
  },
  "artifacts": {}
}
```

### Required Fields

  Field       Required   Description
  ----------- ---------- -----------------------------
  objective   Yes        Scalar objective value
  metrics     No         Optional diagnostic metrics
  artifacts   No         Optional output artifacts

------------------------------------------------------------------------

## 4. Configuration Schema

The following Pydantic schema defines the structure of a GOW problem.

``` python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


ParamType = Literal["real", "int", "categorical"]


class BaseParam(BaseModel):
    type: ParamType
    value: Any
    description: Optional[str] = None
    optimizable: bool = True


class RealParam(BaseParam):
    type: Literal["real"] = "real"
    value: float
    bounds: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.optimizable and self.bounds is None:
            raise ValueError("RealParam.bounds is required when optimizable is true")
        return self


class IntParam(BaseParam):
    type: Literal["int"] = "int"
    value: int
    bounds: Optional[List[int]] = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.optimizable and self.bounds is None:
            raise ValueError("IntParam.bounds is required when optimizable is true")
        return self


class CategoricalParam(BaseParam):
    type: Literal["categorical"] = "categorical"
    value: str
    choices: Optional[List[str]] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_choices(self):
        if self.optimizable:
            if not self.choices:
                raise ValueError("CategoricalParam.choices is required when optimizable is true")
            if self.value not in self.choices:
                raise ValueError("CategoricalParam.value must be one of choices")
        return self


Parameter = Union[RealParam, IntParam, CategoricalParam]


class ExternalEvaluatorConfig(BaseModel):
    command: List[str]
    timeout_s: int = 600
    extra_args: List[str] = []
    env: Dict[str, str] = {}


ObjectiveDirection = Literal["minimize", "maximize"]


class ObjectiveConfig(BaseModel):
    direction: ObjectiveDirection = "minimize"


class OptimizerConfig(BaseModel):
    name: str = "random_search"
    seed: Optional[int] = None
    max_evaluations: int = 100
    batch_size: int = 4
    settings: Dict[str, Any] = {}


class ProblemConfig(BaseModel):
    id: str
    parameters: Dict[str, Parameter]
    evaluator: ExternalEvaluatorConfig
    objective: ObjectiveConfig = ObjectiveConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    context: Dict[str, Any] = {}
    source_path: Optional[Path] = None

    def runtime_params(self):
        return {name: p.value for name, p in self.parameters.items()}

    def optimizable_parameters(self):
        return {
            name: p
            for name, p in self.parameters.items()
            if getattr(p, "optimizable", True)
        }
```

------------------------------------------------------------------------

## 5. Example `project_config.json`

``` json
{
  "id": "toy-polynomial-regression",
  "parameters": {
    "k1": {
      "type": "real",
      "value": 0.0,
      "bounds": [-5.0, 5.0],
      "optimizable": true,
      "description": "Linear coefficient"
    },
    "k2": {
      "type": "real",
      "value": 0.0,
      "bounds": [-5.0, 5.0],
      "optimizable": true,
      "description": "Quadratic coefficient"
    },
    "n_terms": {
      "type": "int",
      "value": 2,
      "bounds": [1, 10],
      "optimizable": true,
      "description": "Number of polynomial terms"
    }
  },
  "evaluator": {
    "command": ["{python}", "toy_evaluator.py"],
    "timeout_s": 300
  },
  "objective": {
    "direction": "minimize"
  },
  "optimizer": {
    "name": "random_search",
    "seed": 42,
    "max_evaluations": 50,
    "batch_size": 4
  },
  "context": {
    "problem": "toy_polynomial_regression",
    "dataset": "synthetic_v1",
    "seed": 42
  }
}
```

------------------------------------------------------------------------

## 6. Summary

This manual:

-   Uses a clear toy regression example
-   Demonstrates the structured parameter system
-   Documents the evaluator contract
-   Aligns with the fully structured configuration format

The Generic Optimization Workflow is designed to be simple, extensible,
and fully reproducible.