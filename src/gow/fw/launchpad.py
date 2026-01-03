from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def _ensure_fireworks_imports():
    try:
        from fireworks import LaunchPad  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


_ensure_fireworks_imports()
from fireworks import LaunchPad  # type: ignore  # noqa: E402


LaunchPadPath = Union[str, Path]


def _resolve_launchpad_path(p: LaunchPadPath) -> Path:
    path = Path(p).expanduser()

    # If user passed a directory, assume it contains my_launchpad.yaml
    if path.exists() and path.is_dir():
        path = path / "my_launchpad.yaml"

    return path.resolve()


def load_launchpad(launchpad_file: Optional[LaunchPadPath] = None) -> LaunchPad:
    """
    Load a FireWorks LaunchPad.

    - If launchpad_file is a file path: use LaunchPad.from_file(...)
    - If launchpad_file is a directory: use <dir>/my_launchpad.yaml
    - If launchpad_file is None: try FireWorks auto-discovery (if available),
      otherwise fall back to LaunchPad().

    Notes:
      - This code is OS-independent; any OS-specific part is the user's MongoDB install.
    """
    if launchpad_file is None:
        # Different FireWorks versions expose different helpers; keep it robust.
        auto = getattr(LaunchPad, "auto_load", None)
        if callable(auto):
            return auto()
        return LaunchPad()

    lp_path = _resolve_launchpad_path(launchpad_file)
    if not lp_path.exists():
        raise FileNotFoundError(f"LaunchPad file not found: {lp_path}")

    return LaunchPad.from_file(str(lp_path))
