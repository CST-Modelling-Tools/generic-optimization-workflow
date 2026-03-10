from __future__ import annotations

from gow.candidate_ids import (
    derive_run_token,
    format_attempt_id,
    format_candidate_id,
    format_candidate_local_id,
    parse_attempt_id,
    parse_candidate_id,
)


def test_format_candidate_local_id_zero_padded() -> None:
    assert format_candidate_local_id(2, 14) == "g000002_c000014"


def test_format_candidate_id_is_run_aware_and_deterministic() -> None:
    run_id = "7c3f3a2a-7c40-4c7b-b9c6-5b02f3b6c6d0"
    expected = f"r{derive_run_token(run_id)}_g000002_c000014"

    assert format_candidate_id(2, 14, run_id=run_id) == expected
    assert format_candidate_id(2, 14, run_id=run_id) == expected


def test_format_candidate_id_varies_across_runs() -> None:
    candidate_a = format_candidate_id(0, 0, run_id="run-a")
    candidate_b = format_candidate_id(0, 0, run_id="run-b")

    assert candidate_a != candidate_b


def test_format_attempt_id_uses_zero_padded_suffix() -> None:
    candidate_id = "r7c3f3a2a_g000002_c000014"

    assert format_attempt_id(candidate_id, 0) == "r7c3f3a2a_g000002_c000014_a000"
    assert format_attempt_id(candidate_id, 12) == "r7c3f3a2a_g000002_c000014_a012"


def test_parse_candidate_id_supports_run_aware_and_legacy_formats() -> None:
    run_aware = parse_candidate_id("r7c3f3a2a_g000002_c000014")
    legacy = parse_candidate_id("g000002_c000014")

    assert run_aware is not None
    assert run_aware.run_token == "7c3f3a2a"
    assert run_aware.generation_id == 2
    assert run_aware.candidate_index == 14
    assert run_aware.candidate_local_id == "g000002_c000014"

    assert legacy is not None
    assert legacy.run_token is None
    assert legacy.generation_id == 2
    assert legacy.candidate_index == 14
    assert legacy.candidate_local_id == "g000002_c000014"


def test_parse_candidate_id_is_flexible_about_width() -> None:
    run_aware = parse_candidate_id("r7c3f3a2a_g2_c14")
    legacy = parse_candidate_id("g2_c14")

    assert run_aware is not None
    assert run_aware.run_token == "7c3f3a2a"
    assert run_aware.generation_id == 2
    assert run_aware.candidate_index == 14
    assert run_aware.candidate_local_id == "g000002_c000014"

    assert legacy is not None
    assert legacy.generation_id == 2
    assert legacy.candidate_index == 14
    assert legacy.candidate_local_id == "g000002_c000014"


def test_parse_attempt_id_supports_fixed_width_attempts() -> None:
    parts = parse_attempt_id("r7c3f3a2a_g000002_c000014_a012")

    assert parts is not None
    assert parts.candidate_id == "r7c3f3a2a_g000002_c000014"
    assert parts.attempt_index == 12


def test_parse_attempt_id_is_flexible_about_attempt_width() -> None:
    parts = parse_attempt_id("r7c3f3a2a_g2_c14_a3")

    assert parts is not None
    assert parts.candidate_id == "r7c3f3a2a_g2_c14"
    assert parts.attempt_index == 3
