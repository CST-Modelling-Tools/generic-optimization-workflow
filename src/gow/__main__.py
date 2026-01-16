from __future__ import annotations

from .cli import app  # or whatever your Typer/CLI entry object is called

def main() -> None:
    app()

if __name__ == "__main__":
    main()