from __future__ import annotations

import json
from pathlib import Path

import pytest

from gow.candidate_ids import format_candidate_id
from gow.cli import _wait_for_candidate_results


class _FakeFW:
    def __init__(self, state: str) -> None:
        self.state = state


class _FakeLaunchPad:
    def __init__(self, states: dict[int, str]) -> None:
        self._states = dict(states)

    def get_fw_by_id(self, fw_id: int):
        return _FakeFW(self._states[fw_id])


def _write_result(outdir: Path, run_id: str, candidate_id: str, objective: float = 0.0) -> None:
    workdir = outdir / "runs" / run_id / candidate_id
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_id": candidate_id,
        "fitness": {"status": "ok", "objective": objective},
    }
    (workdir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_wait_for_candidate_results_returns_when_results_appear(tmp_path: Path, monkeypatch) -> None:
    outdir = tmp_path / "out"
    run_id = "queue-run"
    c1 = format_candidate_id(0, 0, run_id=run_id)
    c2 = format_candidate_id(0, 1, run_id=run_id)

    sleep_calls = {"count": 0}

    def fake_sleep(seconds: int) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            _write_result(outdir, run_id, c1, 1.0)
        elif sleep_calls["count"] == 2:
            _write_result(outdir, run_id, c2, 2.0)

    monkeypatch.setattr("gow.cli.time.sleep", fake_sleep)

    lp = _FakeLaunchPad({101: "RUNNING"})
    _wait_for_candidate_results(
        lp=lp,
        results_dir=outdir,
        run_id=run_id,
        candidate_ids=[c1, c2],
        fw_ids=[101],
        poll_seconds=1,
        timeout_seconds=10,
    )

    assert (outdir / "runs" / run_id / c1 / "result.json").exists()
    assert (outdir / "runs" / run_id / c2 / "result.json").exists()
    assert sleep_calls["count"] == 2


def test_wait_for_candidate_results_raises_when_terminal_fireworks_finish_without_results(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    run_id = "queue-run"
    c1 = format_candidate_id(0, 0, run_id=run_id)
    c2 = format_candidate_id(0, 1, run_id=run_id)

    _write_result(outdir, run_id, c1, 1.0)
    lp = _FakeLaunchPad({201: "FIZZLED", 202: "COMPLETED"})

    with pytest.raises(RuntimeError, match="Missing result.json"):
        _wait_for_candidate_results(
            lp=lp,
            results_dir=outdir,
            run_id=run_id,
            candidate_ids=[c1, c2],
            fw_ids=[201, 202],
            poll_seconds=1,
            timeout_seconds=10,
        )
