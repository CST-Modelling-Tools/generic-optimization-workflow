from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import ProblemConfig


def _load_data(path: Path) -> Any:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_problem_config(path: str | Path) -> ProblemConfig:
    path = Path(path)
    data = _load_data(path)
    cfg = ProblemConfig.model_validate(data)
    cfg.source_path = path.resolve()
    return cfg