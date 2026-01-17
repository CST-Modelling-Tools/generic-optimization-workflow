"""JSON Lines (JSONL) utilities.

These helpers provide deterministic serialization for results files shared across
backends (local and FireWorks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def jsonl_dumps(obj: Mapping[str, Any]) -> str:
    """Serialize an object to deterministic JSON for JSONL files."""
    # Keep deterministic ordering for stable diffs and reproducibility.
    # keep separators compact for performance and file size.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def append_jsonl_line(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one record as a JSONL line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" enforces consistent LF newlines across platforms.
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(jsonl_dumps(record))
        f.write("\n")