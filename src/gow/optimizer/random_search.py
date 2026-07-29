from __future__ import annotations

from typing import Any, Dict, List

import random

from gow.config.models import (
    CategoricalParam,
    IntParam,
    ProblemConfig,
    RealParam,
)

from .base import Optimizer


class RandomSearchOptimizer(Optimizer):
    """Uniform random-search optimizer with resumable RNG state."""

    STATE_SCHEMA_VERSION = 1

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def ask(
        self,
        problem: ProblemConfig,
        n: int,
    ) -> List[Dict[str, Any]]:
        params = problem.optimizable_parameters()
        out: List[Dict[str, Any]] = []

        for _ in range(n):
            candidate: Dict[str, Any] = {}

            for name, parameter in params.items():
                if isinstance(parameter, RealParam):
                    if not parameter.bounds or len(parameter.bounds) != 2:
                        raise ValueError(
                            f"Optimizable real param '{name}' missing bounds=[lo,hi]"
                        )

                    lower, upper = parameter.bounds
                    candidate[name] = self._rng.uniform(lower, upper)

                elif isinstance(parameter, IntParam):
                    if not parameter.bounds or len(parameter.bounds) != 2:
                        raise ValueError(
                            f"Optimizable int param '{name}' missing bounds=[lo,hi]"
                        )

                    lower, upper = parameter.bounds
                    candidate[name] = self._rng.randint(lower, upper)

                elif isinstance(parameter, CategoricalParam):
                    if not parameter.choices:
                        raise ValueError(
                            f"Optimizable categorical param '{name}' missing choices=[...]"
                        )

                    candidate[name] = self._rng.choice(parameter.choices)

                else:
                    raise TypeError(
                        f"Unsupported parameter type for {name}: {type(parameter)}"
                    )

            out.append(candidate)

        return out

    def tell(
        self,
        candidates: List[Dict[str, Any]],
        fitness: List[Dict[str, Any]],
    ) -> None:
        return

    def state_dict(self) -> Dict[str, Any]:
        """Capture the exact Python random-generator state."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the exact Python random-generator state."""
        if not isinstance(state, dict):
            raise TypeError("Random-search checkpoint state must be a dictionary")

        schema_version = state.get("schema_version")
        if schema_version != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported random-search checkpoint schema version: "
                f"{schema_version!r}"
            )

        if "rng_state" not in state:
            raise ValueError(
                "Random-search checkpoint does not contain 'rng_state'"
            )

        try:
            self._rng.setstate(state["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid random-search RNG checkpoint state"
            ) from exc