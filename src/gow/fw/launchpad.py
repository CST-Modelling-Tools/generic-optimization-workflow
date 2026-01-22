# gow/fw/launchpad.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def _ensure_fireworks_imports() -> None:
    try:
        from fireworks import LaunchPad  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


_ensure_fireworks_imports()
from fireworks import LaunchPad  # type: ignore  # noqa: E402


LaunchPadPath = Union[str, Path]

ENV_LAUNCHPAD_FILE = "FW_LAUNCHPAD_FILE"  # file path to my_launchpad.yaml
ENV_FW_CONFIG_DIR = "FW_CONFIG_DIR"       # dir containing my_launchpad.yaml


def _resolve_launchpad_path(p: LaunchPadPath) -> Path:
    path = Path(p).expanduser()

    # If user passed a directory, assume it contains my_launchpad.yaml
    if path.exists() and path.is_dir():
        path = path / "my_launchpad.yaml"

    return path.resolve()


def _find_upwards(start: Path, filename: str = "my_launchpad.yaml") -> Optional[Path]:
    """
    Look for `filename` in start and its parents.
    Useful when running from inside a project folder without providing --launchpad.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    for d in [cur, *cur.parents]:
        candidate = d / filename
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_launchpad(launchpad_file: Optional[LaunchPadPath] = None) -> LaunchPad:
    """
    Load a FireWorks LaunchPad deterministically.

    Resolution order:

    If launchpad_file is provided:
      - file path -> LaunchPad.from_file(file)
      - directory -> LaunchPad.from_file(<dir>/my_launchpad.yaml)

    If launchpad_file is None:
      1) $FW_LAUNCHPAD_FILE (file path)
      2) $FW_CONFIG_DIR/my_launchpad.yaml (directory path)
      3) search upwards from CWD for my_launchpad.yaml
      4) LaunchPad.auto_load() if available
      5) raise a clear error (do NOT silently fall back to LaunchPad()).
    """
    # Explicit argument wins
    if launchpad_file is not None:
        lp_path = _resolve_launchpad_path(launchpad_file)
        if not lp_path.exists():
            raise FileNotFoundError(f"LaunchPad file not found: {lp_path}")
        return LaunchPad.from_file(str(lp_path))

    # Env var: file path
    env_file = os.environ.get(ENV_LAUNCHPAD_FILE)
    if env_file:
        lp_path = _resolve_launchpad_path(env_file)
        if not lp_path.exists():
            raise FileNotFoundError(
                f"${ENV_LAUNCHPAD_FILE} is set but file not found: {lp_path}"
            )
        return LaunchPad.from_file(str(lp_path))

    # Env var: config dir
    env_dir = os.environ.get(ENV_FW_CONFIG_DIR)
    if env_dir:
        lp_path = _resolve_launchpad_path(Path(env_dir))
        if not lp_path.exists():
            raise FileNotFoundError(
                f"${ENV_FW_CONFIG_DIR} is set but my_launchpad.yaml not found at: {lp_path}"
            )
        return LaunchPad.from_file(str(lp_path))

    # Search upwards from current working directory
    found = _find_upwards(Path.cwd(), "my_launchpad.yaml")
    if found is not None:
        return LaunchPad.from_file(str(found))

    # FireWorks auto-load (last resort)
    auto = getattr(LaunchPad, "auto_load", None)
    if callable(auto):
        try:
            return auto()
        except Exception:
            # fall through to a clearer error
            pass

    raise RuntimeError(
        "Could not locate a FireWorks LaunchPad configuration.\n\n"
        "Provide one of:\n"
        "  - --launchpad /path/to/my_launchpad.yaml\n"
        "  - --launchpad /path/to/config_dir  (containing my_launchpad.yaml)\n"
        "  - set env var FW_LAUNCHPAD_FILE=/path/to/my_launchpad.yaml\n"
        "  - set env var FW_CONFIG_DIR=/path/to/config_dir\n"
        "  - place my_launchpad.yaml in the current directory or a parent directory\n"
    )