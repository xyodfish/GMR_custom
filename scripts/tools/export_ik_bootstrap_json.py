#!/usr/bin/env python3
"""Export per-frame GMR IK qpos bootstrap for C++ batch TO (--q_init_json)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--input_type", default="auto")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--contact_ground", action="store_true")
    parser.add_argument("--body_model_dir", default=str(REPO / "assets" / "body_models"))
    args = parser.parse_args()

    frames, fps, height, src_human = load_human_motion_frames(
        args.input_file,
        input_type=args.input_type,
        body_model_dir=args.body_model_dir,
        max_frames=args.max_frames,
    )

    gmr = GMR(
        actual_human_height=height,
        src_human=src_human,
        tgt_robot=args.robot,
        verbose=False,
        contact_ground=args.contact_ground,
        motion_fps=fps,
    )
    q_frames = [gmr.retarget(f).tolist() for f in frames]
    payload = {
        "robot": args.robot,
        "src_human": src_human,
        "fps": float(fps),
        "actual_human_height": float(height),
        "method": "gmr_ik_bootstrap",
        "num_frames": len(q_frames),
        "qpos_frames": q_frames,
    }
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(q_frames)} IK bootstrap frames -> {out}")


if __name__ == "__main__":
    main()
