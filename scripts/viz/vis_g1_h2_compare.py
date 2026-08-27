#!/usr/bin/env python3
"""Side-by-side G1 / H2 playback in one MuJoCo window.

G1 is the left (grey) robot; H2 is shifted on +Y and tinted blue.

Example
-------
python scripts/viz/vis_g1_h2_compare.py \\
  --g1_motion ~/Workspace/puppet/output/gmr_references/source/unitree_g1/lafan1/walk1_subject2.qpos.json \\
  --h2_motion ~/Workspace/puppet/output/gmr_references/robot_b/unitree_h2/lafan1/walk1_subject2.qpos.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.dual_robot_viewer import TwoRobotMotionViewer


def _qpos_fps(path: str) -> tuple[np.ndarray, float]:
    _meta, fps, _rp, _rr, _dof, *_rest, qpos = load_robot_motion(path)
    if qpos is None:
        raise ValueError(f"No qpos in {path}")

    return np.asarray(qpos, dtype=float), float(fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1_motion", required=True, help="G1 motion (.qpos.json / .pkl)")
    parser.add_argument("--h2_motion", required=True, help="H2 motion (.qpos.json / .pkl)")
    parser.add_argument(
        "--offset_y",
        type=float,
        default=1.2,
        help="World +Y shift applied to H2 root (meters)",
    )
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/g1_h2_compare.mp4")
    parser.add_argument(
        "--no-tint",
        action="store_true",
        help="Keep H2 original grey instead of blue tint",
    )
    args = parser.parse_args()

    q_g1, fps_g1 = _qpos_fps(args.g1_motion)
    q_h2, fps_h2 = _qpos_fps(args.h2_motion)
    fps = fps_g1
    if abs(fps_g1 - fps_h2) > 1e-3:
        print(f"[g1-h2] warning: fps mismatch g1={fps_g1} h2={fps_h2}, using g1")

    n = min(len(q_g1), len(q_h2))
    q_g1, q_h2 = q_g1[:n], q_h2[:n]
    print(f"[g1-h2] frames={n} @ {fps:.0f}Hz | G1 grey | H2 +Y={args.offset_y} (blue tint)")
    print("[g1-h2] Close the window to exit.")

    tint = None if args.no_tint else (0.35, 0.55, 0.95, 1.0)
    viewer = TwoRobotMotionViewer(
        "unitree_g1",
        "unitree_h2",
        motion_fps=fps,
        offset_b=(0.0, float(args.offset_y), 0.0),
        tint_b=tint,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    idx = 0
    try:
        while viewer.viewer.is_running():
            if idx >= n:
                if args.loop:
                    idx = 0
                else:
                    break

            viewer.step(q_g1[idx], q_h2[idx], rate_limit=True, follow_camera=True)
            idx += 1
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
