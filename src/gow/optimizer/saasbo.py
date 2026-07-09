from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch.quasirandom import SobolEngine

from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf

try:
    from botorch.acquisition.logei import qLogExpectedImprovement as AcquisitionFunction
except Exception:
    from botorch.acquisition.monte_carlo import qExpectedImprovement as AcquisitionFunction

from gow.config.models import CategoricalParam, IntParam, ProblemConfig, RealParam
from .base import Optimizer


class SAASBOOptimizer(Optimizer):
    """
    SAASBO optimizer adapted to GOW using BoTorch.

    GOW interface:
      - ask(problem, n) returns n candidate parameter dicts.
      - tell(candidates, fitness) updates the internal history.

    Internal convention:
      - BoTorch acquisition maximizes.
      - For minimize objectives, we store score = -objective.
      - Therefore, higher internal score is always better.
    """

    def __init__(
        self,
        *,
        n_initial_points: int = 20,
        max_iterations: int = 100,
        raw_samples: int = 256,
        num_restarts: int = 10,
        warmup_steps: int = 128,
        num_samples: int = 128,
        thinning: int = 16,
        max_tree_depth: int = 6,
        use_cuda: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        if n_initial_points < 2:
            raise ValueError("n_initial_points must be >= 2 for SAASBO")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        self.n_initial_points = int(n_initial_points)
        self.max_iterations = int(max_iterations)
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)

        self.warmup_steps = int(warmup_steps)
        self.num_samples = int(num_samples)
        self.thinning = int(thinning)
        self.max_tree_depth = int(max_tree_depth)

        self.use_cuda = bool(use_cuda)
        self.seed = seed
        self.kwargs = kwargs

        self._initialized = False
        self._iteration = 0

        self._param_names: List[str] = []
        self._param_specs: Dict[str, Tuple[str, Tuple[float, float]]] = {}
        self._direction = "maximize"

        self._sobol: SobolEngine | None = None

        self._train_x: List[List[float]] = []
        self._train_y: List[float] = []
        self._last_xs: List[List[float]] = []

        self._best_score: float | None = None
        self._best_candidate: Dict[str, Any] | None = None

        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

    def ask(self, problem: ProblemConfig, n: int) -> List[Dict[str, Any]]:
        if n < 1:
            raise ValueError("ask(..., n) requires n >= 1")

        if not self._initialized:
            self._initialize_from_problem(problem)

        dim = len(self._param_names)

        if len(self._train_x) < self.n_initial_points:
            unit_x = self._sobol.draw(n).double()
        else:
            unit_x = self._generate_saasbo_candidates(n=n, dim=dim)

        raw_candidates = [
            self._unit_vector_to_candidate(x.tolist()) for x in unit_x
        ]

        repaired_candidates = [
            self._repair_candidate(problem, cand) for cand in raw_candidates
        ]

        self._last_xs = [
            self._candidate_to_unit_vector(cand) for cand in repaired_candidates
        ]

        return repaired_candidates

    def tell(self, candidates: List[Dict[str, Any]], fitness: List[Dict[str, Any]]) -> None:
        self._n_status_failed = 0
        self._n_missing_score = 0
        self._n_non_numeric = 0
        self._n_non_finite = 0

        if not self._initialized:
            raise RuntimeError("tell() called before first ask(); SAASBO is not initialized.")

        if len(candidates) != len(fitness):
            raise ValueError(
                f"tell(): candidates and fitness lengths differ: "
                f"{len(candidates)} != {len(fitness)}"
            )

        if len(self._last_xs) != len(candidates):
            raise RuntimeError(
                "tell(): number of candidates does not match the previous ask() call."
            )

        scores = [self._normalize_score(fdict) for fdict in fitness]

        for x_unit, score, cand in zip(self._last_xs, scores, candidates):
            if score == float("-inf"):
                continue

            self._train_x.append(list(x_unit))
            self._train_y.append(float(score))

            if self._best_score is None or score > self._best_score:
                self._best_score = float(score)
                self._best_candidate = dict(cand)

        self._iteration += 1
        self._last_xs = []

    def is_done(self) -> bool:
        return self._iteration >= self.max_iterations

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "optimizer": "saasbo",
            "iteration": self._iteration,
            "n_observations": len(self._train_x),
            "n_initial_points": self.n_initial_points,
            "max_iterations": self.max_iterations,
            "raw_samples": self.raw_samples,
            "num_restarts": self.num_restarts,
            "warmup_steps": self.warmup_steps,
            "num_samples": self.num_samples,
            "thinning": self.thinning,
            "max_tree_depth": self.max_tree_depth,
            "best_score_internal": self._best_score,
            "best_candidate": self._best_candidate,
            "n_status_failed": self._n_status_failed,
            "n_missing_score": self._n_missing_score,
            "n_non_numeric": self._n_non_numeric,
            "n_non_finite": self._n_non_finite,
        }

    def _generate_saasbo_candidates(self, n: int, dim: int) -> torch.Tensor:
        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )
        dtype = torch.double

        train_x = torch.tensor(self._train_x, dtype=dtype, device=device)
        train_y = torch.tensor(self._train_y, dtype=dtype, device=device).unsqueeze(-1)
        train_yvar = torch.full_like(train_y, 1e-6)

        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            outcome_transform=Standardize(m=1),
        )

        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            max_tree_depth=self.max_tree_depth,
            disable_progbar=True,
        )



        best_f = train_y.max()

        acqf = AcquisitionFunction(
            model=model,
            best_f=best_f,
        )

        bounds = torch.stack(
            [
                torch.zeros(dim, dtype=dtype, device=device),
                torch.ones(dim, dtype=dtype, device=device),
            ]
        )

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=n,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"maxiter": 100},
        )

        return candidates.detach().cpu()

    def _initialize_from_problem(self, problem: ProblemConfig) -> None:
        self._direction = self._get_direction(problem)

        params = problem.optimizable_parameters()
        if not params:
            raise ValueError("No optimizable parameters found for SAASBO.")

        self._param_names = []
        self._param_specs = {}

        for name, p in params.items():
            if isinstance(p, RealParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable real param '{name}' missing bounds=[lo,hi]")

                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if not lo < hi:
                    raise ValueError(f"Real param '{name}' must have lo < hi")

                self._param_names.append(name)
                self._param_specs[name] = ("real", (lo, hi))

            elif isinstance(p, IntParam):
                if not p.bounds or len(p.bounds) != 2:
                    raise ValueError(f"Optimizable int param '{name}' missing bounds=[lo,hi]")

                lo, hi = int(p.bounds[0]), int(p.bounds[1])
                if lo > hi:
                    raise ValueError(f"Int param '{name}' must have lo <= hi")

                self._param_names.append(name)
                self._param_specs[name] = ("int", (float(lo), float(hi)))

            elif isinstance(p, CategoricalParam):
                raise ValueError(
                    f"SAASBO does not support optimizable categorical param '{name}' yet."
                )

            else:
                raise TypeError(f"Unsupported optimizable parameter type for '{name}': {type(p)}")

        self._sobol = SobolEngine(
            dimension=len(self._param_names),
            scramble=True,
            seed=self.seed,
        )

        self._initialized = True

    def _unit_vector_to_candidate(self, x: List[float]) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}

        for value, name in zip(x, self._param_names):
            kind, (lo, hi) = self._param_specs[name]
            u = min(1.0, max(0.0, float(value)))
            real_value = lo + u * (hi - lo)

            if kind == "int":
                cand[name] = int(round(real_value))
            else:
                cand[name] = float(real_value)

        return cand

    def _candidate_to_unit_vector(self, cand: Mapping[str, Any]) -> List[float]:
        x: List[float] = []

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            val = float(cand[name])

            if hi == lo:
                u = 0.0
            else:
                u = (val - lo) / (hi - lo)

            x.append(min(1.0, max(0.0, float(u))))

        return x

    def _repair_candidate(self, problem: ProblemConfig, cand: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(cand)

        for name in self._param_names:
            kind, (lo, hi) = self._param_specs[name]
            val = float(repaired[name])
            val = min(hi, max(lo, val))

            if kind == "int":
                repaired[name] = int(round(val))
            else:
                repaired[name] = float(val)

        return repaired

    def _normalize_score(self, fitness_dict: Mapping[str, Any]) -> float:
        status = fitness_dict.get("status")
        if status is not None and str(status).lower() != "ok":
            self._n_status_failed += 1
            return float("-inf")

        val: Any = None
        key: str | None = None

        for k in ("fitness", "objective", "score", "loss"):
            if k in fitness_dict:
                key = k
                val = fitness_dict[k]
                break

        if key is None:
            metrics = fitness_dict.get("metrics")
            if isinstance(metrics, Mapping):
                for k in ("fitness", "objective", "score", "loss"):
                    if k in metrics:
                        key = k
                        val = metrics[k]
                        break

        if val is None:
            self._n_missing_score += 1
            return float("-inf")

        try:
            x = float(val)
        except (TypeError, ValueError):
            self._n_non_numeric += 1
            return float("-inf")

        if not math.isfinite(x):
            self._n_non_finite += 1
            return float("-inf")

        if key == "loss":
            x = -x

        if self._direction == "minimize":
            x = -x

        return x

    def _get_direction(self, problem: ProblemConfig) -> str:
        direction = getattr(getattr(problem, "objective", None), "direction", None)
        if direction is None:
            return "maximize"

        direction = str(direction).lower().strip()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported objective direction: {direction}")

        return direction
