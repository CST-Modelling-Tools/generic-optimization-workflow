from __future__ import annotations

from pathlib import Path
from typing import Optional


def _ensure_fireworks_imports():
    try:
        from fireworks import LaunchPad  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


_ensure_fireworks_imports()
from fireworks import LaunchPad  # type: ignore  # noqa: E402


def load_launchpad(launchpad_file: Optional[str | Path] = None) -> LaunchPad:
    """
    Load a FireWorks LaunchPad.

    If launchpad_file is provided, it must be a FireWorks-compatible YAML file
    (e.g. my_launchpad.yaml).

    If not provided, FireWorks will try its default LaunchPad configuration
    (environment variables or ~/.fireworks/my_launchpad.yaml).
    """
    if launchpad_file is None:
        return LaunchPad()

    lp_path = Path(launchpad_file).expanduser().resolve()
    if not lp_path.exists():
        raise FileNotFoundError(f"LaunchPad file not found: {lp_path}")

    return LaunchPad.from_file(str(lp_path))