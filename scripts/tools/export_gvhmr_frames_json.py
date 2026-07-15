#!/usr/bin/env python3
"""GVHMR .pt -> human_frame_json (wrapper; prefer export_human_frames_json.py)."""

from __future__ import annotations

import pathlib
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if __name__ == "__main__":
    argv = sys.argv[1:]
    mapped = []
    i = 0
    while i < len(argv):
        if argv[i] == "--pt_file" and i + 1 < len(argv):
            mapped += ["--input_file", argv[i + 1], "--input_type", "gvhmr_pt"]
            i += 2
            continue
        mapped.append(argv[i])
        i += 1
    sys.argv = [str(REPO / "scripts" / "tools" / "export_human_frames_json.py")] + mapped
    runpy.run_path(str(REPO / "scripts" / "tools" / "export_human_frames_json.py"), run_name="__main__")
