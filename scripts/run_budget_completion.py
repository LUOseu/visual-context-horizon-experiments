#!/usr/bin/env python3
"""Run one visual budget for a stage whose other budgets are already archived.

This is the documented completion path for the pilot's final 1280 budget. It
narrows only the in-memory budget grid; sample selection, repeats, timing and
every other frozen parameter stay identical to a full pilot sweep.
"""
import argparse
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip())
    raise SystemExit(0)

vendor = os.environ.get("VELADRAFT_DIR")
if not vendor:
    raise SystemExit("VELADRAFT_DIR must point at the assembled VelaDraft checkout")
sys.path.insert(0, vendor)

from research.visual_context_horizon import run_visual_sweep as sweep  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--budget", type=int, required=True)
known, rest = parser.parse_known_args()
if known.budget not in sweep.CONFIG["budgets"]:
    raise SystemExit(f"--budget must be one of {sweep.CONFIG['budgets']}")

if "--budgets" not in rest:
    rest = ["--budgets", str(known.budget), *rest]
sweep.CONFIG["budgets"] = [known.budget]
sys.argv = ["run_visual_sweep", *rest]
raise SystemExit(sweep.main())
