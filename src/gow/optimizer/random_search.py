from __future__ import annotations

import random
from typing import Any, Dict, List

from gow.config.models import ProblemConfig, RealParam, IntParam, CategoricalParam
from .base import Optimizer


class RandomSearchOptimizer(Optimizer):
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        params = problem.optimizable_parameters()
        out: List[Dict[str, Any]] = []

        for _ in range(n):
            cand: Dict[str, Any] = {}
            for name, p in params.items():
                if isinstance(p, RealParam):
                    if not p.bounds or len(p.bounds) != 2:
                        raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")
                    lo, hi = p.bounds
                    cand[name] = self._rng.uniform(lo, hi)

                elif isinstance(p, IntParam):
                    if not p.bounds or len(p.bounds) != 2:
                        raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")
                    lo, hi = p.bounds
                    cand[name] = self._rng.randint(lo, hi)

                elif isinstance(p, CategoricalParam):
                    if not p.choices:
                        raise ValueError(f"Optimizable categorical param '{name}' missing choices=[...]")
                    cand[name] = self._rng.choice(p.choices)

                else:
                    raise TypeError(f"Unsupported parameter type for {name}: {type(p)}")

            out.append(cand)

        return out

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        return