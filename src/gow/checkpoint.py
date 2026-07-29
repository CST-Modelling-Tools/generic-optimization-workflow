from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


class CheckpointError(RuntimeError):
    """Base exception for checkpoint operations."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint is incomplete or does not exist."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when checkpoint integrity validation fails."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Checkpoint data restored from disk."""

    manifest: Dict[str, Any]
    optimizer_state: Dict[str, Any]


class CheckpointStore:
    """Store and restore a campaign checkpoint atomically.

    Optimizer state uses pickle because random-generator and optimizer
    internals may contain tuples and implementation-specific Python objects.

    Checkpoints must therefore only be loaded from trusted local GOW runs.
    """

    MANIFEST_FILENAME = "manifest.json"
    OPTIMIZER_STATE_FILENAME = "optimizer_state.bin"
    CHECKSUM_FILENAME = "checkpoint.sha256"

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.checkpoint_dir = self.run_dir / "checkpoint"

        self.manifest_path = (
            self.checkpoint_dir
            / self.MANIFEST_FILENAME
        )
        self.optimizer_state_path = (
            self.checkpoint_dir
            / self.OPTIMIZER_STATE_FILENAME
        )
        self.checksum_path = (
            self.checkpoint_dir
            / self.CHECKSUM_FILENAME
        )

    @staticmethod
    def _calculate_checksum(
        manifest_bytes: bytes,
        optimizer_state_bytes: bytes,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(manifest_bytes)
        digest.update(b"\0")
        digest.update(optimizer_state_bytes)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(
        target: Path,
        data: bytes,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, target)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def save(
        self,
        *,
        manifest: Dict[str, Any],
        optimizer_state: Dict[str, Any],
    ) -> None:
        """Write a complete checkpoint to disk."""

        if not isinstance(manifest, dict):
            raise TypeError(
                "Checkpoint manifest must be a dictionary"
            )

        if not isinstance(optimizer_state, dict):
            raise TypeError(
                "Optimizer checkpoint state must be a dictionary"
            )

        try:
            manifest_bytes = json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "Checkpoint manifest is not JSON serializable"
            ) from exc

        try:
            optimizer_state_bytes = pickle.dumps(
                optimizer_state,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (TypeError, ValueError, pickle.PickleError) as exc:
            raise CheckpointError(
                "Optimizer state could not be serialized"
            ) from exc

        checksum = self._calculate_checksum(
            manifest_bytes,
            optimizer_state_bytes,
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._atomic_write(
            self.manifest_path,
            manifest_bytes,
        )
        self._atomic_write(
            self.optimizer_state_path,
            optimizer_state_bytes,
        )

        # The checksum is written last. Its presence indicates that the
        # checkpoint write reached its final stage.
        self._atomic_write(
            self.checksum_path,
            f"{checksum}\n".encode("ascii"),
        )

    def load(self) -> LoadedCheckpoint:
        """Load and validate the latest complete checkpoint."""

        required_paths = (
            self.manifest_path,
            self.optimizer_state_path,
            self.checksum_path,
        )

        missing = [
            path.name
            for path in required_paths
            if not path.is_file()
        ]

        if missing:
            raise CheckpointNotFoundError(
                "Checkpoint is incomplete; missing file(s): "
                + ", ".join(missing)
            )

        try:
            manifest_bytes = self.manifest_path.read_bytes()
            optimizer_state_bytes = (
                self.optimizer_state_path.read_bytes()
            )
            expected_checksum = (
                self.checksum_path
                .read_text(encoding="ascii")
                .strip()
            )
        except OSError as exc:
            raise CheckpointError(
                "Checkpoint files could not be read"
            ) from exc

        actual_checksum = self._calculate_checksum(
            manifest_bytes,
            optimizer_state_bytes,
        )

        if not hmac.compare_digest(
            expected_checksum,
            actual_checksum,
        ):
            raise CheckpointCorruptionError(
                "Checkpoint checksum validation failed"
            )

        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise CheckpointCorruptionError(
                "Checkpoint manifest is corrupted"
            ) from exc

        try:
            optimizer_state = pickle.loads(
                optimizer_state_bytes
            )
        except (
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
            TypeError,
            ValueError,
            pickle.PickleError,
        ) as exc:
            raise CheckpointCorruptionError(
                "Optimizer checkpoint state is corrupted"
            ) from exc

        if not isinstance(manifest, dict):
            raise CheckpointCorruptionError(
                "Checkpoint manifest must contain a dictionary"
            )

        if not isinstance(optimizer_state, dict):
            raise CheckpointCorruptionError(
                "Optimizer checkpoint state must contain a dictionary"
            )

        return LoadedCheckpoint(
            manifest=manifest,
            optimizer_state=optimizer_state,
        )