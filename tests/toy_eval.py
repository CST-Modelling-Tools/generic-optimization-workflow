#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    params = data["params"]
    x = float(params.get("x", 0.0))
    y = float(params.get("y", 0.0))

    # simple objective: sphere function
    obj = x * x + y * y

    out = {
        "status": "ok",
        "metrics": {"sphere": obj},
        "objective": obj,
        "artifacts": {},
    }
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()