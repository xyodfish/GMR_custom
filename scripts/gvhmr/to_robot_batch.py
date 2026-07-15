#!/usr/bin/env python3
"""GVHMR .pt offline batch TO (wrapper; prefer scripts/retarget/to_robot_batch.py)."""

from __future__ import annotations

import pathlib
import runpy
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    argv = sys.argv[1:]
    mapped = []
    i = 0
    while i < len(argv):
        if argv[i] == "--gvhmr_pred_file" and i + 1 < len(argv):
            mapped += ["--input_file", argv[i + 1], "--input_type", "gvhmr_pt"]
            i += 2
            continue
        mapped.append(argv[i])
        i += 1
    sys.argv = [str(REPO_ROOT / "scripts" / "retarget" / "to_robot_batch.py")] + mapped
    runpy.run_path(str(REPO_ROOT / "scripts" / "retarget" / "to_robot_batch.py"), run_name="__main__")
