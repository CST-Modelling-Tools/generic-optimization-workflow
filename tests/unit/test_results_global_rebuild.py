from __future__ import annotations

import json
from pathlib import Path

from gow.fw.tasks import append_run_result_record, rebuild_problem_results_jsonl, rebuild_run_results_jsonl, verify_run_results_complete


def _record(run_id: str, candidate_id: str, attempt_index: int = 0):
    return {
        "problem_id": "prob",
        "run_id": run_id,
        "candidate_id": candidate_id,
        "attempt_id": f"{candidate_id}--{attempt_index}",
        "attempt_index": attempt_index,
        "fitness": {"status": "failed", "objective": None},
    }


def test_persist_records_then_rebuild_run_and_global(tmp_path: Path):
    outdir = tmp_path / "results"
    append_run_result_record(outdir, "run-a", _record("run-a", "c1"))
    append_run_result_record(outdir, "run-b", _record("run-b", "c2"))

    assert not (outdir / "results.jsonl").exists()
    assert not (outdir / "runs" / "run-a" / "results.jsonl").exists()

    run_rebuilt = rebuild_run_results_jsonl(outdir, "run-a")
    assert run_rebuilt.exists()
    run_lines = [json.loads(line) for line in run_rebuilt.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [line["candidate_id"] for line in run_lines] == ["c1"]

    rebuild_run_results_jsonl(outdir, "run-b")
    rebuilt = rebuild_problem_results_jsonl(outdir)
    lines = [json.loads(line) for line in rebuilt.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [line["candidate_id"] for line in lines] == ["c1", "c2"]


def test_verify_run_results_complete_counts_result_json_files(tmp_path: Path):
    outdir = tmp_path / "results"
    append_run_result_record(outdir, "run-a", _record("run-a", "c1", 0))
    append_run_result_record(outdir, "run-a", _record("run-a", "c2", 0))
    ok, count = verify_run_results_complete(outdir, "run-a", 2)
    assert ok is True
    assert count == 2
