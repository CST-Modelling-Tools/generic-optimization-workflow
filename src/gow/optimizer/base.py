from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Optimizer(ABC):
    """Base class for all optimizers used by GOW."""

    @abstractmethod
    def ask(self, problem, n: int) -> List[Dict[str, Any]]:
        """Return n candidate parameter dicts."""
        raise NotImplementedError

    @abstractmethod
    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """Update optimizer state from evaluated candidates and their fitness dicts."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Optional termination hook.

        Optimizers that have their own internal stopping criterion
        (e.g. max generations) can override this. Default: never done.
        """
        return False

    def diagnostics(self) -> Dict[str, Any]:
        """Optional diagnostic information for logging/debugging.

        Should return a small JSON-serializable dict (no huge populations).
        Default: empty dict.
        """
        return {}