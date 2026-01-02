from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from gow.config.models import ProblemConfig


class Optimizer(ABC):
    @abstractmethod
    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        """Return n candidate param dictionaries (optimizable params only)."""
        raise NotImplementedError

    @abstractmethod
    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        """Update optimizer state from evaluated candidates and their fitness dicts."""
        raise NotImplementedError