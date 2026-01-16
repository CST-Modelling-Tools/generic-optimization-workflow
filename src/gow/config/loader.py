from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import ProblemConfig


def _load_data(path: Path) -> Any:
    """
    Load a config file into a Python object.

    - JSON: returns parsed JSON
    - YAML: returns parsed YAML (empty file -> {})
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        # yaml.safe_load returns None for empty files -> normalize to {}
        return yaml.safe_load(text) or {}

    raise ValueError(f"Unsupported config format: {path.suffix} (expected .yaml/.yml/.json)")


def load_problem_config(path: str | Path) -> ProblemConfig:
    """
    Load and validate a GOW ProblemConfig from YAML/JSON.

    Always runs Pydantic validation (including param validators),
    so optimizers can assume required fields exist for optimizable params.
    """
    path = Path(path).expanduser().resolve()
    data = _load_data(path)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping/object, got {type(data).__name__}: {path}")

    cfg = ProblemConfig.model_validate(data)
    cfg.source_path = path
    return cfg