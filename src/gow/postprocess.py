from __future__ import annotations

import gzip
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from gow.candidate_ids import parse_candidate_id
from gow.layout import candidate_workdir, run_root


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def generation_results_dir(outdir: Path | str, run_id: str) -> Path:
    return run_root(outdir, run_id) / "generations"


def generation_results_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_results_dir(outdir, run_id) / f"g{int(generation_id):06d}.jsonl"


def generation_archive_dir(outdir: Path | str, run_id: str) -> Path:
    return run_root(outdir, run_id) / "archives"


def generation_archive_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_archive_dir(outdir, run_id) / f"g{int(generation_id):06d}.tar.gz"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _unique_key(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    rid = record.get("run_id")
    aid = record.get("attempt_id")
    cid = record.get("candidate_id")
    return (
        str(rid) if rid is not None else None,
        str(aid) if aid is not None else None,
        str(cid) if cid is not None else None,
    )


def _sort_key(obj: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        obj.get("generation_id") if isinstance(obj.get("generation_id"), int) else 10**12,
        obj.get("candidate_index") if isinstance(obj.get("candidate_index"), int) else 10**12,
        str(obj.get("candidate_id", "")),
        obj.get("attempt_index") if isinstance(obj.get("attempt_index"), int) else 10**12,
    )


def _best_from_records(records: Iterable[dict[str, Any]], direction: str) -> dict[str, Any] | None:
    best_obj: float | None = None
    best_record: dict[str, Any] | None = None
    for rec in records:
        fit = rec.get("fitness")
        if not isinstance(fit, dict) or fit.get("status") != "ok":
            continue
        obj = fit.get("objective")
        try:
            obj_val = float(obj)
        except Exception:
            continue
        if best_obj is None:
            best_obj = obj_val
            best_record = rec
            continue
        if direction == "maximize" and obj_val > best_obj:
            best_obj = obj_val
            best_record = rec
        elif direction != "maximize" and obj_val < best_obj:
            best_obj = obj_val
            best_record = rec
    if best_record is None:
        return None
    return {
        "candidate_id": best_record.get("candidate_id"),
        "generation_id": best_record.get("generation_id"),
        "objective": best_obj,
        "params": best_record.get("params"),
        "attempt_id": best_record.get("attempt_id"),
    }


def verify_generation_results_complete(
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
) -> tuple[bool, int, list[str]]:
    outdir = Path(outdir).expanduser().resolve()
    missing: list[str] = []
    count = 0
    for candidate_id in candidate_ids:
        result_path = candidate_workdir(outdir, run_id, candidate_id) / "result.json"
        if result_path.exists():
            count += 1
        else:
            missing.append(candidate_id)
    return not missing, count, missing


def rebuild_generation_results_jsonl(
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    gen_path = generation_results_path(outdir, run_id, generation_id)
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str | None, str | None, str | None]] = set()
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        result_path = candidate_workdir(outdir, run_id, candidate_id) / "result.json"
        obj = _load_json(result_path)
        if not isinstance(obj, dict):
            continue
        key = _unique_key(obj)
        if key in seen:
            continue
        seen.add(key)
        if obj.get("generation_id") is None:
            parts = parse_candidate_id(candidate_id)
            if parts is not None:
                obj["generation_id"] = parts.generation_id
                obj.setdefault("candidate_index", parts.candidate_index)
                obj.setdefault("candidate_local_id", parts.candidate_local_id)
        records.append(obj)

    records.sort(key=_sort_key)
    with gen_path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return gen_path


def _read_generation_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(summary_path))


def update_run_summary(
    *,
    outdir: Path | str,
    run_id: str,
    problem_id: str,
    max_evaluations: int,
    direction: str,
    generation_id: int,
    generation_results_file: Path,
    expected_count: int,
    actual_count: int,
    completed_generations: int,
    final: bool,
) -> tuple[Path, Path, Path]:
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root(outdir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    generation_records = list(_iter_jsonl(generation_results_file))
    generation_best = _best_from_records(generation_records, direction)

    summary_jsonl_path = run_dir / "summary.jsonl"
    rows = _read_generation_summary_rows(summary_jsonl_path) if summary_jsonl_path.exists() else []
    rows = [row for row in rows if int(row.get("generation_id", -1)) != int(generation_id)]
    row = {
        "timestamp": _now_iso(),
        "problem_id": problem_id,
        "run_id": run_id,
        "generation_id": int(generation_id),
        "expected_count": int(expected_count),
        "actual_count": int(actual_count),
        "generation_results_file": str(generation_results_file),
        "generation_best": generation_best,
    }
    rows.append(row)
    rows.sort(key=lambda obj: int(obj.get("generation_id", 10**12)))
    with summary_jsonl_path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    best_so_far = _best_from_records(
        ({
            "candidate_id": r.get("generation_best", {}).get("candidate_id"),
            "generation_id": r.get("generation_best", {}).get("generation_id"),
            "params": r.get("generation_best", {}).get("params"),
            "attempt_id": r.get("generation_best", {}).get("attempt_id"),
            "fitness": {"status": "ok", "objective": r.get("generation_best", {}).get("objective")},
        } for r in rows if isinstance(r.get("generation_best"), dict)),
        direction,
    )

    evaluations_done = sum(int(r.get("actual_count", 0)) for r in rows)
    summary = {
        "problem_id": problem_id,
        "run_id": run_id,
        "max_evaluations": int(max_evaluations),
        "objective": {"direction": direction},
        "best": best_so_far,
        "results_file": str(outdir / "results.jsonl"),
        "run_results_file": str(run_dir / "results.jsonl"),
        "run_root": str(run_dir),
        "outdir": str(outdir),
        "summary_jsonl": str(summary_jsonl_path),
        "completed_generations": int(completed_generations),
        "evaluations_done": evaluations_done,
        "finalized": bool(final),
        "updated_at": _now_iso(),
    }
    run_summary = run_dir / "summary.json"
    problem_summary = outdir / "summary.json"
    run_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    problem_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_jsonl_path, run_summary, problem_summary


def finalize_generation(
    *,
    outdir: Path | str,
    run_id: str,
    problem_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
    max_evaluations: int,
    direction: str,
    completed_generations: int,
    final: bool = False,
) -> Path:
    ok, actual_count, missing = verify_generation_results_complete(outdir, run_id, generation_id, candidate_ids)
    if not ok:
        raise RuntimeError(
            f"Generation {generation_id} of run {run_id} incomplete: missing result.json for {len(missing)} candidate(s): {', '.join(missing[:10])}"
        )
    gen_path = rebuild_generation_results_jsonl(outdir, run_id, generation_id, candidate_ids)
    update_run_summary(
        outdir=outdir,
        run_id=run_id,
        problem_id=problem_id,
        max_evaluations=max_evaluations,
        direction=direction,
        generation_id=generation_id,
        generation_results_file=gen_path,
        expected_count=len(candidate_ids),
        actual_count=actual_count,
        completed_generations=completed_generations,
        final=final,
    )
    return gen_path


def iter_generation_shards(outdir: Path | str, run_id: str) -> Iterable[Path]:
    gen_dir = generation_results_dir(outdir, run_id)
    if not gen_dir.exists():
        return
    for path in sorted(gen_dir.glob("g*.jsonl")):
        if path.is_file():
            yield path


def merge_runs(
    *,
    outdir: Path | str,
    target_run_id: str,
    source_run_ids: Sequence[str],
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    target_run = run_root(outdir, target_run_id)
    target_run.mkdir(parents=True, exist_ok=True)
    target_results = target_run / "results.jsonl"

    seen: set[tuple[str | None, str | None, str | None]] = set()
    rows: list[dict[str, Any]] = []
    for run_id in source_run_ids:
        path = run_root(outdir, run_id) / "results.jsonl"
        for obj in _iter_jsonl(path):
            key = _unique_key(obj)
            if key in seen:
                continue
            seen.add(key)
            rows.append(obj)
    rows.sort(key=_sort_key)
    with target_results.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return target_results


def archive_generation_workdirs(
    *,
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
    delete_source: bool = False,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    archive_path = generation_archive_path(outdir, run_id, generation_id)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as tar:
        for candidate_id in candidate_ids:
            workdir = candidate_workdir(outdir, run_id, candidate_id)
            if not workdir.exists():
                continue
            tar.add(workdir, arcname=workdir.relative_to(run_root(outdir, run_id)).as_posix())
    if delete_source:
        import shutil

        for candidate_id in candidate_ids:
            workdir = candidate_workdir(outdir, run_id, candidate_id)
            if workdir.exists():
                shutil.rmtree(workdir)
    return archive_path
