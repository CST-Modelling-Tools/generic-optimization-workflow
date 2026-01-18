from __future__ import annotations

import pytest


def test_result_namespace_does_not_exist() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("gow.result")


def test_results_namespace_does_not_exist() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("gow.results")