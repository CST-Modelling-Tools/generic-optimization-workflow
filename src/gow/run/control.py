from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PAUSE_REQUEST_FILENAME = "pause.request.json"
PAUSE_ACK_FILENAME = "pause.ack.json"


class RunControlError(RuntimeError):
    """Base error for the filesystem run-control protocol."""


class InvalidPauseRequestError(RunControlError):
    """Raised when a pause request exists but is malformed."""


def control_dir(run_root: str | Path) -> Path:
    """Return the filesystem control directory for one GOW run."""

    return Path(run_root) / "control"


def pause_request_path(run_root: str | Path) -> Path:
    """Return the canonical pause-request path for one run."""

    return control_dir(run_root) / PAUSE_REQUEST_FILENAME


def pause_ack_path(run_root: str | Path) -> Path:
    """Return the canonical pause acknowledgement path for one run."""

    return control_dir(run_root) / PAUSE_ACK_FILENAME


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    """Atomically replace one JSON control artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_name: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name

            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(
                    missing_ok=True,
                )
            except OSError:
                pass

        raise


def read_pause_request(
    run_root: str | Path,
) -> Optional[Dict[str, Any]]:
    """Read and validate a pending cooperative pause request.

    The request is intentionally NOT removed here. It remains pending until
    GOW has successfully persisted a safe optimizer checkpoint.

    External clients must write:

        control/pause.request.json

    with at least:

        {
            "schema_version": 1,
            "action": "pause",
            "request_id": "<unique id>"
        }

    Additional metadata is preserved.
    """

    path = pause_request_path(run_root)

    if not path.is_file():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise InvalidPauseRequestError(
            f"Invalid pause request JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise InvalidPauseRequestError(
            "Pause request must contain a JSON object"
        )

    if payload.get("schema_version") != 1:
        raise InvalidPauseRequestError(
            "Unsupported pause request schema_version"
        )

    if payload.get("action") != "pause":
        raise InvalidPauseRequestError(
            "Pause request action must be 'pause'"
        )

    request_id = payload.get("request_id")

    if (
        not isinstance(request_id, str)
        or not request_id.strip()
    ):
        raise InvalidPauseRequestError(
            "Pause request requires a non-empty request_id"
        )

    return dict(payload)


def acknowledge_pause_request(
    run_root: str | Path,
    request: Dict[str, Any],
    *,
    evaluations_done: int,
    completed_generations: int,
) -> Path:
    """Acknowledge a pause only after the checkpoint is safely persisted."""

    request_id = request.get("request_id")

    if (
        not isinstance(request_id, str)
        or not request_id.strip()
    ):
        raise InvalidPauseRequestError(
            "Cannot acknowledge pause without request_id"
        )

    ack_payload = dict(request)

    ack_payload.update(
        {
            "schema_version": 1,
            "action": "pause",
            "request_id": request_id,
            "status": "paused",
            "acknowledged_at": _utc_now_iso(),
            "evaluations_done": evaluations_done,
            "completed_generations": completed_generations,
        }
    )

    ack_path = pause_ack_path(run_root)

    # Important ordering:
    #
    # 1. Persist acknowledgement atomically.
    # 2. Only then remove the pending request.
    #
    # The request therefore survives if checkpointing/acknowledgement fails.
    _atomic_write_json(
        ack_path,
        ack_payload,
    )

    pause_request_path(
        run_root
    ).unlink(
        missing_ok=True,
    )

    return ack_path
