from __future__ import annotations

import json
from pathlib import Path
from .models import ProblemConfig


def load_problem_config(path: str | Path) -> ProblemConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProblemConfig.model_validate(data)