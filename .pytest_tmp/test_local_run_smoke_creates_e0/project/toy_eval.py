import json
from pathlib import Path

inp = json.loads(Path("input.json").read_text(encoding="utf-8"))
params = inp.get("params", {})
x = float(params.get("x", 0.0))
objective = x * x

out = {
    "status": "ok",
    "metrics": {"objective": objective},
    "objective": objective,
    "constraints": {},
    "artifacts": {},
    "error": None,
}

Path("output.json").write_text(json.dumps(out), encoding="utf-8")
