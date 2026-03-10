"""Candidate and attempt identifier formatting.

GOW uses stable, human-readable identifiers across backends (local and FireWorks)
to make logs, paths, and results easy to correlate.

The generation/candidate/attempt components are intentionally fixed-width to preserve
lexicographic ordering. The canonical candidate id is run-aware, while the legacy
local candidate label remains available for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

_GENERATION_WIDTH = 6
_CANDIDATE_WIDTH = 6
_ATTEMPT_WIDTH = 3
_RUN_TOKEN_LENGTH = 8

_LOCAL_CANDIDATE_RE = re.compile(r"^g(?P<g>\d+)_c(?P<c>\d+)$")
_RUN_AWARE_CANDIDATE_RE = re.compile(r"^r(?P<run>[0-9a-f]+)_(?P<local>g\d+_c\d+)$")
_ATTEMPT_ID_RE = re.compile(r"^(?P<candidate>.+)_a(?P<attempt>\d+)$")


@dataclass(frozen=True)
class CandidateIdParts:
    run_token: str | None
    generation_id: int
    candidate_index: int

    @property
    def candidate_local_id(self) -> str:
        return format_candidate_local_id(
            generation_id=self.generation_id,
            candidate_index=self.candidate_index,
        )


@dataclass(frozen=True)
class AttemptIdParts:
    candidate_id: str
    attempt_index: int


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def derive_run_token(run_id: str, *, length: int = _RUN_TOKEN_LENGTH) -> str:
    """Return a short stable token derived from a run id."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    token = hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:length]
    if len(token) < length:
        raise ValueError(f"Unable to derive run token of length {length} from run_id={run_id!r}")
    return token


def format_candidate_local_id(generation_id: int, candidate_index: int) -> str:
    """Return the legacy local candidate label."""
    _validate_non_negative("generation_id", generation_id)
    _validate_non_negative("candidate_index", candidate_index)
    return f"g{generation_id:0{_GENERATION_WIDTH}d}_c{candidate_index:0{_CANDIDATE_WIDTH}d}"


def format_candidate_id(
    generation_id: int,
    candidate_index: int,
    *,
    run_id: str | None = None,
    run_token: str | None = None,
) -> str:
    """Return a candidate id.

    If ``run_id`` or ``run_token`` is provided, returns the canonical run-aware id:
    ``r<run>_g<generation>_c<candidate>``.

    If neither is provided, falls back to the legacy local id:
    ``g<generation>_c<candidate>``.
    """
    local_id = format_candidate_local_id(generation_id, candidate_index)
    if run_token is None and run_id is None:
        return local_id
    token = run_token or derive_run_token(str(run_id))
    return f"r{token}_{local_id}"


def format_attempt_id(candidate_id: str, attempt_index: int) -> str:
    """Return the canonical attempt id for a candidate evaluation attempt."""
    _validate_non_negative("attempt_index", attempt_index)
    return f"{candidate_id}_a{attempt_index:0{_ATTEMPT_WIDTH}d}"


def parse_candidate_id(candidate_id: str) -> CandidateIdParts | None:
    """Parse a legacy or run-aware candidate id."""
    candidate_id = str(candidate_id)
    run_token: str | None = None
    local_candidate_id = candidate_id

    match = _RUN_AWARE_CANDIDATE_RE.match(candidate_id)
    if match:
        run_token = match.group("run")
        local_candidate_id = match.group("local")

    local_match = _LOCAL_CANDIDATE_RE.match(local_candidate_id)
    if not local_match:
        return None

    try:
        generation_id = int(local_match.group("g"))
        candidate_index = int(local_match.group("c"))
    except Exception:
        return None

    return CandidateIdParts(
        run_token=run_token,
        generation_id=generation_id,
        candidate_index=candidate_index,
    )


def parse_attempt_id(attempt_id: str) -> AttemptIdParts | None:
    """Parse an attempt id if its candidate component is valid."""
    match = _ATTEMPT_ID_RE.match(str(attempt_id))
    if not match:
        return None

    candidate_id = match.group("candidate")
    if parse_candidate_id(candidate_id) is None:
        return None

    try:
        attempt_index = int(match.group("attempt"))
    except Exception:
        return None

    return AttemptIdParts(candidate_id=candidate_id, attempt_index=attempt_index)
