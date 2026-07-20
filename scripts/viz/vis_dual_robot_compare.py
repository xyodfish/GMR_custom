#!/usr/bin/env python3
"""Overlay two robot motions in one MuJoCo window (solid + translucent ghost).

Like whole_body_tracking sim2sim ``--ref_viz mesh``: ghost is a second attached
robot with low-alpha purple mesh.

Examples
--------
# Compare torque_limit vs baseline on a GVHMR clip (solid=with limit, ghost=baseline):
python scripts/viz/vis_dual_robot_compare.py \\
  --input_file data/gvhmr_test_videos/tennis/hmr4d_results.pt \\
  --compare_torque_limit \\
  --torque_limit_weight 10

# Compare two saved motions:
python scripts/viz/vis_dual_robot_compare.py \\
  --solid_motion path/to/with_tq.pkl \\
  --ghost_motion path/to/baseline.pkl
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.dual_robot_viewer import DualRobotMotionViewer
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames
from general_motion_retargeting.online_qp_retarget import OnlineQpConfig, OnlineQpRetargeter


def _qpos_from_motion(path: str) -> tuple[np.ndarray, float]:
    _meta, fps, _rp, _rr, _dof, *_rest, qpos = load_robot_motion(path)
    if qpos is None:
        raise ValueError(f"No qpos in {path}")
    return np.asarray(qpos, dtype=float), float(fps)


def _retarget_pair(
    input_file: str,
    *,
    robot: str,
    preset: str,
    max_frames: int | None,
    weight: float,
    margin: float,
    scope: str,
    gate_mode: str,
    contact_ground: bool,
    ground_align=False,
) -> tuple[np.ndarray, np.ndarray, float]:
    frames, fps, height, src = load_human_motion_frames(
        input_file,
        input_type="auto",
        max_frames=max_frames,
        ground_align=ground_align or False,
        ground_align_verbose=bool(ground_align),
    )
    kwargs = dict(
        src_human=src,
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        actual_human_height=height,
        motion_fps=fps,
    )

    def run(enable_tq: bool) -> np.ndarray:
        cfg = OnlineQpConfig.from_preset(preset)
        if enable_tq:
            cfg.torque_limit_constraint = True
            cfg.torque_limit_weight = weight
            cfg.torque_limit_margin = margin
            cfg.torque_limit_scope = scope
            cfg.torque_limit_gate_mode = gate_mode
        r = OnlineQpRetargeter(GMR(**kwargs), cfg)
        r.set_motion_fps(fps)
        print(f"[dual-viz] retargeting {'torque_limit' if enable_tq else 'baseline'} ...")
        return r.retarget_sequence(frames)

    q_base = run(False)
    q_tq = run(True)
    return q_tq, q_base, float(fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--solid_motion", default=None, help="Solid robot motion (.pkl/.json)")
    parser.add_argument("--ghost_motion", default=None, help="Ghost (translucent) motion")
    parser.add_argument(
        "--compare_torque_limit",
        action="store_true",
        help="Retarget input twice: solid=torque_limit, ghost=baseline",
    )
    parser.add_argument("--input_file", default=None, help="Human motion for --compare_torque_limit")
    parser.add_argument("--preset", choices=["default", "smooth", "anti_slip"], default="anti_slip")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--torque_limit_weight", type=float, default=10.0)
    parser.add_argument("--torque_limit_margin", type=float, default=0.1)
    parser.add_argument("--torque_limit_scope", choices=["upper", "all"], default="upper")
    parser.add_argument("--torque_limit_gate_mode", choices=["off", "soft", "hard"], default="soft")
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument(
        "--ground_align",
        nargs="?",
        const="lower_envelope",
        default=None,
        choices=["lower_envelope", "support_hold"],
        help="Offline Z ground-align human frames before retarget (default mode: lower_envelope)",
    )
    parser.add_argument(
        "--ghost_alpha",
        type=float,
        default=0.28,
        help="Ghost opacity (like --ref_alpha in whole_body_tracking)",
    )
    parser.add_argument(
        "--ghost_offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Optional XYZ offset for ghost (default: overlapped)",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/dual_robot_compare.mp4")
    parser.add_argument("--save_pair", default=None, help="Optional dir to save solid/ghost pkl")
    args = parser.parse_args()

    if args.compare_torque_limit:
        if not args.input_file:
            parser.error("--compare_torque_limit requires --input_file")
        q_solid, q_ghost, fps = _retarget_pair(
            args.input_file,
            robot=args.robot,
            preset=args.preset,
            max_frames=args.max_frames,
            weight=args.torque_limit_weight,
            margin=args.torque_limit_margin,
            scope=args.torque_limit_scope,
            gate_mode=args.torque_limit_gate_mode,
            contact_ground=args.contact_ground,
            ground_align=args.ground_align or False,
        )
        solid_label, ghost_label = "torque_limit (solid)", "baseline (ghost)"
    else:
        if not args.solid_motion or not args.ghost_motion:
            parser.error("Provide --solid_motion and --ghost_motion, or --compare_torque_limit")
        q_solid, fps_s = _qpos_from_motion(args.solid_motion)
        q_ghost, fps_g = _qpos_from_motion(args.ghost_motion)
        fps = fps_s
        if abs(fps_s - fps_g) > 1e-3:
            print(f"[dual-viz] warning: fps mismatch solid={fps_s} ghost={fps_g}, using solid")
        solid_label = pathlib.Path(args.solid_motion).name
        ghost_label = pathlib.Path(args.ghost_motion).name

    n = min(len(q_solid), len(q_ghost))
    q_solid, q_ghost = q_solid[:n], q_ghost[:n]
    dq = np.linalg.norm(q_solid[:, 7:] - q_ghost[:, 7:], axis=1)
    print(
        f"[dual-viz] frames={n} @ {fps:.0f}Hz | solid={solid_label} | ghost={ghost_label}"
    )
    print(
        f"[dual-viz] ||Δq_joint|| mean={dq.mean():.4f} max={dq.max():.4f} "
        f"({np.degrees(dq.max()):.1f} deg-eq) @ frame {int(np.argmax(dq))}"
    )

    if args.save_pair:
        out = pathlib.Path(args.save_pair)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "solid.pkl", "wb") as f:
            pickle.dump({"fps": fps, "qpos": q_solid, "label": solid_label}, f)
        with open(out / "ghost.pkl", "wb") as f:
            pickle.dump({"fps": fps, "qpos": q_ghost, "label": ghost_label}, f)
        print(f"[dual-viz] saved pair to {out}")

    rgba = (0.65, 0.25, 1.0, float(args.ghost_alpha))
    viewer = DualRobotMotionViewer(
        args.robot,
        motion_fps=fps,
        ghost_rgba=rgba,
        ghost_offset=tuple(args.ghost_offset),
        record_video=args.record_video,
        video_path=args.video_path,
    )
    print(
        "[dual-viz] solid=opaque robot, ghost=purple translucent "
        f"(alpha={args.ghost_alpha}). Close window to exit."
    )
    idx = 0
    try:
        while viewer.viewer.is_running():
            if idx >= n:
                if args.loop:
                    idx = 0
                else:
                    break
            viewer.step(q_solid[idx], q_ghost[idx], rate_limit=True, follow_camera=True)
            idx += 1
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
