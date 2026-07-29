from __future__ import annotations

from gow.config.models import ProblemConfig
from gow.optimizer.random_search import RandomSearchOptimizer


def _build_problem() -> ProblemConfig:
    return ProblemConfig.model_validate(
        {
            "id": "checkpoint-random-search",
            "objective": {"direction": "minimize"},
            "parameters": {
                "x": {
                    "type": "real",
                    "value": 0.0,
                    "bounds": [-10.0, 10.0],
                },
                "iterations": {
                    "type": "int",
                    "value": 5,
                    "bounds": [1, 20],
                },
                "mode": {
                    "type": "categorical",
                    "value": "balanced",
                    "choices": ["fast", "balanced", "accurate"],
                },
            },
            "evaluator": {
                "command": ["python", "dummy_evaluator.py"],
                "timeout_s": 30,
            },
            "optimizer": {
                "name": "random_search",
                "seed": 123,
                "max_evaluations": 20,
                "batch_size": 4,
            },
        }
    )


def test_random_search_state_round_trip_preserves_candidate_sequence() -> None:
    problem = _build_problem()

    original = RandomSearchOptimizer(seed=123)

    # Avanzamos el generador para simular una campaña ya comenzada.
    original.ask(problem, 7)

    checkpoint_state = original.state_dict()

    # La semilla inicial del objeto restaurado no debe importar:
    # load_state_dict() debe sustituirla por el estado exacto del checkpoint.
    restored = RandomSearchOptimizer(seed=999)
    restored.load_state_dict(checkpoint_state)

    expected_candidates = original.ask(problem, 10)
    resumed_candidates = restored.ask(problem, 10)

    assert resumed_candidates == expected_candidates