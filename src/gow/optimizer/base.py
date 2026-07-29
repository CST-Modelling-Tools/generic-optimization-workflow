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
    def tell(
        self,
        candidates: List[Dict[str, Any]],
        fitness: List[Dict[str, Any]],
    ) -> None:
        """Update optimizer state from evaluated candidates and their fitness dicts."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Optional termination hook.

        Optimizers that have their own internal stopping criterion
        can override this. Default: never done.
        """
        return False

    def diagnostics(self) -> Dict[str, Any]:
        """Return small JSON-serializable diagnostic information."""
        return {}

    def state_dict(self) -> Dict[str, Any]:
        """Return the complete state required to resume the optimizer.

        Stateful optimizers must override this method. The returned object
        must contain everything necessary to continue producing the same
        candidate sequence after a pause.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint persistence"
        )

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a state previously produced by state_dict()."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint restoration"
        )