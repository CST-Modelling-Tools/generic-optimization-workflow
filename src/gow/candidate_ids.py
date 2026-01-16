"""Candidate identifier formatting.

GOW uses stable, human-readable candidate IDs across backends (local and FireWorks)
to make logs, paths, and results easy to correlate.

The ID format is intentionally fixed-width to preserve lexicographic ordering.
"""

from __future__ import annotations


def format_candidate_id(generation_id: int, candidate_index: int) -> str:
    """Return the canonical candidate id.

    Args:
        generation_id: Zero-based generation number.
        candidate_index: Zero-based index within the generation.

    Returns:
        Candidate id string, e.g. ``g000000_c000000``.
    """
    if generation_id < 0:
        raise ValueError(f"generation_id must be >= 0, got {generation_id}")
    if candidate_index < 0:
        raise ValueError(f"candidate_index must be >= 0, got {candidate_index}")
    return f"g{generation_id:06d}_c{candidate_index:06d}"
