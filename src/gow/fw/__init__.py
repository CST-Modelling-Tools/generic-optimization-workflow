# gow/fw/__init__.py
from __future__ import annotations

"""
FireWorks backend (optional dependency).

Importing this package will raise a clear RuntimeError if FireWorks isn't installed.
"""

from .launchpad import load_launchpad
from .workflow import SingleEvalSpec, build_single_evaluate_workflow

__all__ = [
    "load_launchpad",
    "SingleEvalSpec",
    "build_single_evaluate_workflow",
]