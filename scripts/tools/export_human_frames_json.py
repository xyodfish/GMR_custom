#!/usr/bin/env python3
"""Export GVHMR .pt / SMPL-X / BVH human frames to C++ human_frame_json."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.human_frame_loaders import (
    frame_to_json_dict,
    load_human_motion_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="Unified input (.pt/.npz/.pkl/.bvh).")
    parser.add_argument("--input_type", type=str, default="auto",
                        choices=["auto", "gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov"])
    parser.add_argument("--pt_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str(REPO / "assets" / "body_models"),
    )
    args = parser.parse_args()

    input_file = args.input_file or args.pt_file
    if not input_file:
        raise SystemExit("Missing --input_file (or legacy --pt_file).")

    frames, fps, height, src_human = load_human_motion_frames(
        input_file,
        input_type=args.input_type,
        body_model_dir=args.body_model_dir,
        bvh_format=args.format,
        tgt_fps=args.motion_fps,
        max_frames=args.max_frames,
    )

    payload = {
        "fps": float(fps),
        "src_human": src_human,
        "actual_human_height": float(height),
        "input_file": str(pathlib.Path(input_file).resolve()),
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"Wrote {len(frames)} frames @ {fps:.1f} fps ({src_human}) -> {out}")


if __name__ == "__main__":
    main()
