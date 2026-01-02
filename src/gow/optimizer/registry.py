from __future__ import annotations

from typing import Callable, Dict

from .base import Optimizer
from .random_search import RandomSearchOptimizer


def make_optimizer(name: str, *, seed: int | None = None) -> Optimizer:
    name = name.lower().strip()
    if name in {"random", "random_search", "rand"}:
        return RandomSearchOptimizer(seed=seed)
    raise ValueError(f"Unknown optimizer: {name}")