from pathlib import Path
Path("output.json").write_text("{not valid json", encoding="utf-8")
raise SystemExit(0)
