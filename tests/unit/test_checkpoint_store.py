from __future__ import annotations

from pathlib import Path

import pytest

from gow.checkpoint import CheckpointCorruptionError, CheckpointStore


def test_checkpoint_round_trip_preserves_manifest_and_optimizer_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run-001"
    store = CheckpointStore(run_dir)

    manifest = {
        "schema_version": 1,
        "run_id": "run-001",
        "problem_id": "checkpoint-test",
        "status": "paused",
        "optimizer": "random_search",
        "evaluations_done": 8,
        "completed_generations": 2,
        "next_generation": 2,
    }

    optimizer_state = {
        "schema_version": 1,
        "rng_state": (3, (10, 20, 30, 40), None),
    }

    store.save(
        manifest=manifest,
        optimizer_state=optimizer_state,
    )

    loaded = store.load()

    assert loaded.manifest == manifest
    assert loaded.optimizer_state == optimizer_state

    assert (run_dir / "checkpoint" / "manifest.json").is_file()
    assert (run_dir / "checkpoint" / "optimizer_state.bin").is_file()
    assert (run_dir / "checkpoint" / "checkpoint.sha256").is_file()


def test_checkpoint_detects_corrupted_optimizer_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run-corrupted"
    store = CheckpointStore(run_dir)

    store.save(
        manifest={
            "schema_version": 1,
            "run_id": "run-corrupted",
            "status": "paused",
        },
        optimizer_state={
            "schema_version": 1,
            "rng_state": (3, (1, 2, 3), None),
        },
    )

    optimizer_state_path = (
        run_dir
        / "checkpoint"
        / "optimizer_state.bin"
    )
    optimizer_state_path.write_bytes(b"checkpoint-corrupted")

    with pytest.raises(
        CheckpointCorruptionError,
        match="checksum",
    ):
        store.load()