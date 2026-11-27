"""Compact progress report for one run directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import lane as lane_mod


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def report(run_dir: Path) -> dict:
    plan = _load(run_dir / "plan.json")
    states = {}
    total_rows = 0
    for cell in plan["cells"]:
        rows = []
        path = run_dir / "cells" / cell["id"] / "rows.jsonl"
        if path.is_file():
            rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        total_rows += len(rows)
        state = lane_mod.read_state(run_dir, cell)
        states[cell["id"]] = {"status": state.get("status") or ("partial" if rows else "pending"),
                              "rows": len(rows),
                              "error": state.get("error")}
    counts = {}
    for value in states.values():
        counts[value["status"]] = counts.get(value["status"], 0) + 1
    lanes = []
    for path in sorted((run_dir / "lanes").glob("lane_*.result.json")):
        lanes.append(_load(path))
    telemetry = []
    for path in sorted((run_dir / "telemetry").glob("telemetry_gpu_*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        telemetry.append({"file": path.name, "last": lines[-1] if len(lines) > 1 else None})
    return {"run_dir": str(run_dir), "wave": plan.get("wave"),
            "cells": plan.get("cell_count"), "rows": total_rows,
            "states": counts, "lanes": lanes, "telemetry": telemetry,
            "pending": [cid for cid, value in states.items()
                        if value["status"] in ("pending", "partial", "running")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    data = report(Path(args.run_dir))
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print("run:   %s" % data["run_dir"])
    print("wave:  %s   cells: %s   rows: %s" % (data["wave"], data["cells"], data["rows"]))
    for status, count in sorted(data["states"].items()):
        print("  %-10s %d" % (status, count))
    for lane in data["lanes"]:
        print("lane %-4s results: %s" % (lane.get("lane"),
              ", ".join("%s/%s" % (r.get("arm"), r.get("stage")) for r in lane.get("results", []))))
    for item in data["telemetry"]:
        print("telemetry %s: %s" % (item["file"], item["last"]))
    if data["pending"]:
        print("pending cells (first %d): %s" % (
            args.limit, ", ".join(data["pending"][:args.limit])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
