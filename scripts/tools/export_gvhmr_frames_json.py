#!/usr/bin/env python3
"""Export GVHMR .pt human frames to C++ human_frame_json for batch TO benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast


def frame_to_json(frame: dict) -> dict:
    out = {}
    for name, (pos, quat_wxyz) in frame.items():
        out[name] = {
            "position": [float(x) for x in pos],
            "orientation": [float(x) for x in quat_wxyz],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_frames", type=int, default=120)
    args = parser.parse_args()

    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pathlib.Path(args.pt_file).expanduser(), REPO / "assets" / "body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    frames = frames[: args.max_frames]

    payload = {
        "fps": float(fps),
        "actual_human_height": float(height),
        "frames": [frame_to_json(f) for f in frames],
    }
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"Wrote {len(frames)} frames @ {fps:.1f} fps -> {out}")


if __name__ == "__main__":
    main()
