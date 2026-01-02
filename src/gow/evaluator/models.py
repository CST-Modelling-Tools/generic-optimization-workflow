from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class FitnessResult(BaseModel):
    """
    Typed representation of evaluator output.json.

    - status: "ok" or "failed"
    - metrics: dict of named metrics (floats/ints)
    - objective: optional scalar objective (float)
    - constraints: optional dict (runtime, feasibility, etc.)
    - artifacts: optional dict of relative paths (files produced)
    - error: optional error message
    """
    status: str = Field(..., pattern=r"^(ok|failed)$")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    objective: Optional[float] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
