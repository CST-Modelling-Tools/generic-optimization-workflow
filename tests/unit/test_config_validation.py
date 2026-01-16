from __future__ import annotations

from pathlib import Path

import pytest

from gow.config import load_problem_config


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "problem.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_valid_config_from_yaml(tmp_path: Path) -> None:
    """
    A minimal, valid optimization problem should load without errors.
    """
    yaml_text = """
id: toy-problem
objective:
  direction: minimize
parameters:
  x:
    type: real
    value: 0.0
    bounds: [-1.0, 1.0]
evaluator:
  command: ["{python}", "dummy_eval.py"]
optimizer:
  name: random_search
  max_evaluations: 1
"""
    cfg_path = _write_yaml(tmp_path, yaml_text)
    cfg = load_problem_config(cfg_path)

    assert cfg.id == "toy-problem"
    assert "x" in cfg.parameters


def test_optimizable_parameter_requires_bounds(tmp_path: Path) -> None:
    """
    Optimizable numeric parameters must define bounds.
    """
    yaml_text = """
id: bad-problem
objective:
  direction: minimize
parameters:
  x:
    type: real
    value: 0.0
evaluator:
  command: ["{python}", "dummy_eval.py"]
optimizer:
  name: random_search
"""
    cfg_path = _write_yaml(tmp_path, yaml_text)

    with pytest.raises(ValueError):
        load_problem_config(cfg_path)


def test_non_optimizable_parameter_may_omit_bounds(tmp_path: Path) -> None:
    """
    Fixed parameters (optimizable: false) are allowed to omit bounds.
    """
    yaml_text = """
id: fixed-param-problem
objective:
  direction: minimize
parameters:
  n:
    type: int
    value: 5
    optimizable: false
evaluator:
  command: ["{python}", "dummy_eval.py"]
optimizer:
  name: random_search
"""
    cfg_path = _write_yaml(tmp_path, yaml_text)
    cfg = load_problem_config(cfg_path)

    assert cfg.parameters["n"].optimizable is False


def test_categorical_parameter_requires_choices(tmp_path: Path) -> None:
    """
    Categorical parameters must define a list of choices.
    """
    yaml_text = """
id: bad-categorical
objective:
  direction: minimize
parameters:
  mode:
    type: categorical
    value: a
evaluator:
  command: ["{python}", "dummy_eval.py"]
optimizer:
  name: random_search
"""
    cfg_path = _write_yaml(tmp_path, yaml_text)

    with pytest.raises(ValueError):
        load_problem_config(cfg_path)


def test_unknown_parameter_type_is_rejected(tmp_path: Path) -> None:
    """
    Unknown parameter types should fail fast.
    """
    yaml_text = """
id: unknown-type
objective:
  direction: minimize
parameters:
  x:
    type: not-a-type
    value: 1
evaluator:
  command: ["{python}", "dummy_eval.py"]
optimizer:
  name: random_search
"""
    cfg_path = _write_yaml(tmp_path, yaml_text)

    with pytest.raises(ValueError):
        load_problem_config(cfg_path)