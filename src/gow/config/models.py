from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# -------------------------
# Parameter definitions
# -------------------------

ParamType = Literal["real", "int", "categorical"]


class BaseParam(BaseModel):
    """
    Generic parameter definition.

    'value' holds the current/initial value.
    If 'optimizable' is False, the optimizer should treat it as fixed.
    """
    type: ParamType
    value: Any
    description: Optional[str] = None
    optimizable: bool = True


class RealParam(BaseParam):
    type: Literal["real"] = "real"
    value: float
    bounds: List[float] = Field(..., min_length=2, max_length=2)


class IntParam(BaseParam):
    type: Literal["int"] = "int"
    value: int
    bounds: List[int] = Field(..., min_length=2, max_length=2)


class CategoricalParam(BaseParam):
    type: Literal["categorical"] = "categorical"
    value: str
    choices: List[str] = Field(..., min_length=1)


Parameter = Union[RealParam, IntParam, CategoricalParam]


# -------------------------
# Evaluator configuration
# -------------------------

class ExternalEvaluatorConfig(BaseModel):
    """
    External evaluator command that follows the evaluator contract:
    - reads input.json
    - writes output.json
    """
    command: List[str] = Field(
        ...,
        min_length=1,
        description="Evaluator command as a list, e.g. ['{python}', 'path/to/eval.py'] or ['my_eval_exe']",
    )
    timeout_s: int = Field(600, ge=1)
    extra_args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


# -------------------------
# Objective configuration
# -------------------------

ObjectiveDirection = Literal["minimize", "maximize"]


class ObjectiveConfig(BaseModel):
    """
    Objective configuration for selecting the best candidate.

    - direction: whether lower ('minimize') or higher ('maximize') objective is better.
    """
    direction: ObjectiveDirection = Field("minimize")


# -------------------------
# Optimizer configuration
# -------------------------

class OptimizerConfig(BaseModel):
    """
    Minimal optimizer config. Optimizer-specific settings can go in 'settings'.
    """
    name: str = Field("random_search", description="Optimizer id (e.g., random_search, bo, ga).")
    seed: Optional[int] = None
    max_evaluations: int = Field(100, ge=1)
    batch_size: int = Field(4, ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Problem configuration
# -------------------------

class ProblemConfig(BaseModel):
    """
    Top-level configuration for an optimization problem.
    """
    id: str = Field(..., description="Problem identifier, e.g. sunpos-mica or heliostat-optics.")
    parameters: Dict[str, Parameter]
    evaluator: ExternalEvaluatorConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    context: Dict[str, Any] = Field(default_factory=dict, description="Problem-specific metadata.")
    source_path: Optional[Path] = Field(default=None, exclude=True)

    def runtime_params(self) -> Dict[str, Any]:
        """
        Return a simple dict of param_name -> value suitable for writing into evaluator input.json.
        """
        return {name: p.value for name, p in self.parameters.items()}

    def optimizable_parameters(self) -> Dict[str, Parameter]:
        """
        Return only parameters marked optimizable.
        """
        return {name: p for name, p in self.parameters.items() if getattr(p, "optimizable", True)}
