from __future__ import annotations

import json
import tarfile
from pathlib import Path

from gow.fw.tasks import append_run_result_record, rebuild_problem_results_jsonl, rebuild_run_results_jsonl, verify_run_results_complete
from gow.postprocess import archive_generation_workdirs, finalize_generation, generation_results_path, merge_runs


def _record(run_id: str, candidate_id: str, generation_id: int, candidate_index: int, objective: float | None):
    status = "ok" if objective is not None else "failed"
    return {
        "problem_id": "prob",
        "run_id": run_id,
        "candidate_id": candidate_id,
        "attempt_id": f"{candidate_id}_a000",
        "attempt_index": 0,
        "generation_id": generation_id,
        "candidate_index": candidate_index,
        "params": {"x": candidate_index},
        "fitness": {"status": status, "objective": objective},
    }


def _jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_finalize_generation_builds_summary_and_run_results_from_generation_shards(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    run_id = "run-a"
    candidate_ids = ["r123_g000000_c000000", "r123_g000000_c000001"]

    append_run_result_record(outdir, run_id, _record(run_id, candidate_ids[0], 0, 0, 1.2))
    append_run_result_record(outdir, run_id, _record(run_id, candidate_ids[1], 0, 1, 0.4))

    gen_path = finalize_generation(
        outdir=outdir,
        run_id=run_id,
        problem_id="prob",
        generation_id=0,
        candidate_ids=candidate_ids,
        max_evaluations=2,
        direction="minimize",
        completed_generations=1,
        final=True,
    )
    assert gen_path == generation_results_path(outdir, run_id, 0)
    assert gen_path.exists()

    run_results = rebuild_run_results_jsonl(outdir, run_id)
    rows = _jsonl_rows(run_results)
    assert [r["candidate_id"] for r in rows] == candidate_ids

    problem_results = rebuild_problem_results_jsonl(outdir)
    assert problem_results.exists()

    ok, count = verify_run_results_complete(outdir, run_id, 2)
    assert ok is True
    assert count == 2

    summary = json.loads((outdir / "runs" / run_id / "summary.json").read_text(encoding="utf-8"))
    assert summary["best"]["candidate_id"] == candidate_ids[1]
    assert summary["evaluations_done"] == 2

    summary_rows = _jsonl_rows(outdir / "runs" / run_id / "summary.jsonl")
    assert len(summary_rows) == 1
    assert summary_rows[0]["generation_id"] == 0


def test_merge_runs_and_archive_generation(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    run_a = "run-a"
    run_b = "run-b"
    c1 = "r111_g000000_c000000"
    c2 = "r222_g000000_c000001"
    append_run_result_record(outdir, run_a, _record(run_a, c1, 0, 0, 1.0))
    append_run_result_record(outdir, run_b, _record(run_b, c2, 0, 1, 2.0))
    finalize_generation(outdir=outdir, run_id=run_a, problem_id="prob", generation_id=0, candidate_ids=[c1], max_evaluations=1, direction="minimize", completed_generations=1, final=True)
    finalize_generation(outdir=outdir, run_id=run_b, problem_id="prob", generation_id=0, candidate_ids=[c2], max_evaluations=1, direction="minimize", completed_generations=1, final=True)
    rebuild_run_results_jsonl(outdir, run_a)
    rebuild_run_results_jsonl(outdir, run_b)

    merged = merge_runs(outdir=outdir, target_run_id="merged", source_run_ids=[run_a, run_b])
    merged_rows = _jsonl_rows(merged)
    assert [r["candidate_id"] for r in merged_rows] == [c1, c2]

    archive_path = archive_generation_workdirs(outdir=outdir, run_id=run_a, generation_id=0, candidate_ids=[c1], delete_source=True)
    assert archive_path.exists()
    assert not (outdir / "runs" / run_a / c1).exists()
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.startswith(c1 + "/") for name in names)
