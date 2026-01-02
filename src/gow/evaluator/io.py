from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import FitnessResult


def write_input_json(
    path: str | Path,
    *,
    run_id: str,
    candidate_id: str,
    params: Dict[str, Any],
    context: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    payload = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "params": params,
        "context": context or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_output_json(path: str | Path) -> FitnessResult:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return FitnessResult.model_validate(data)
