#!/usr/bin/env python3
"""Offline batch trajectory optimization from GVHMR .pt, SMPL-X, or BVH."""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.batch_trajectory_retarget import (
    BatchTrajectoryConfig,
    BatchTrajectoryRetargeter,
)
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def hinge_acc_metric(qpos_seq: np.ndarray, fps: float) -> float:
    if len(qpos_seq) < 3:
        return 0.0
    dt = 1.0 / fps
    acc = (qpos_seq[2:] - 2.0 * qpos_seq[1:-1] + qpos_seq[:-2]) / (dt * dt)
    if acc.shape[1] <= 7:
        return float(np.mean(np.linalg.norm(acc, axis=1)))
    return float(np.mean(np.linalg.norm(acc[:, 7:], axis=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline batch TO for GVHMR .pt, SMPL-X (.npz/.pkl), or BVH (LAFAN1/Nokov)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Motion file (.pt / .npz / .pkl / .bvh). Preferred unified input.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="auto",
        choices=["auto", "gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov"],
        help="Input format (default: infer from extension).",
    )
    parser.add_argument("--gvhmr_pred_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smplx_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bvh_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1", help="BVH skeleton format.")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str((REPO_ROOT / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--motion_fps", type=int, default=30, help="Target FPS for SMPL-X / BVH resampling.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "full"],
    )
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--solver", type=str, default="gn", choices=["gn", "lbfgs"])
    parser.add_argument("--gn_steps", type=int, default=3)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument("--w_foot_height", type=float, default=50.0)
    parser.add_argument("--w_foot_slip", type=float, default=2000.0)
    parser.add_argument("--w_foot_ik_anchor", type=float, default=200.0)
    parser.add_argument("--w_root_xy_contact", type=float, default=100.0)
    parser.add_argument("--w_contact_joint_anchor", type=float, default=400.0)
    parser.add_argument("--foot_contact_margin", type=float, default=0.02)
    parser.add_argument("--foot_contact_from_ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth_root_xyz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_foot_penalties", action="store_true", default=False)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--max_opt_iter", type=int, default=40)
    parser.add_argument("--compare_ik", action="store_true", default=False)
    parser.add_argument("--no_progress", action="store_true", default=False)
    parser.add_argument("--view", action="store_true", default=False)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", dest="rate_limit", action="store_true")
    parser.add_argument("--no-rate-limit", dest="rate_limit", action="store_false")
    parser.set_defaults(rate_limit=True)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/batch_to_output.mp4")
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")
    return parser.parse_args()


def resolve_input_file(args: argparse.Namespace) -> tuple[str, str]:
    if args.input_file:
        return args.input_file, args.input_type
    if args.gvhmr_pred_file:
        return args.gvhmr_pred_file, "gvhmr_pt"
    if args.smplx_file:
        return args.smplx_file, "smplx"
    if args.bvh_file:
        input_type = "bvh_nokov" if args.format == "nokov" else "bvh_lafan1"
        return args.bvh_file, input_type
    raise SystemExit("Missing input: pass --input_file <path> (or legacy --gvhmr_pred_file / --smplx_file / --bvh_file).")


def main() -> None:
    args = parse_args()
    input_file, input_type = resolve_input_file(args)

    human_frames, fps, actual_human_height, src_human = load_human_motion_frames(
        input_file,
        input_type=input_type,
        body_model_dir=args.body_model_dir,
        bvh_format=args.format,
        tgt_fps=args.motion_fps,
        max_frames=args.max_frames,
    )

    print(
        f"[batch-to] {pathlib.Path(input_file).name}: {len(human_frames)} frames @ {fps:.1f} fps "
        f"({len(human_frames) / fps:.1f}s), src_human={src_human}"
    )

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=args.robot,
        contact_ground=args.contact_ground,
        foot_ground_limit=args.foot_ground_limit,
        fix_robot_penetration=args.fix_robot_penetration,
        motion_fps=fps,
    )
    gmr = GMR(**gmr_kwargs)
    window_size = 16 if args.fast else args.window_size
    window_stride = 8 if args.fast else args.window_stride
    gn_steps = 2 if args.fast else args.gn_steps
    gn_alphas = (1.0,) if args.fast else (1.0, 0.5, 0.25, 0.125)
    batch = BatchTrajectoryRetargeter(
        gmr,
        BatchTrajectoryConfig(
            strategy=args.strategy,
            window_size=window_size,
            window_stride=window_stride,
            solver=args.solver,
            gn_steps=gn_steps,
            gn_line_search_alphas=gn_alphas,
            enable_foot_penalties=not args.no_foot_penalties,
            w_foot_height=args.w_foot_height,
            w_foot_slip=args.w_foot_slip,
            w_foot_ik_anchor=args.w_foot_ik_anchor,
            w_root_xy_contact=args.w_root_xy_contact,
            w_contact_joint_anchor=args.w_contact_joint_anchor,
            foot_contact_margin=args.foot_contact_margin,
            foot_contact_from_ref=args.foot_contact_from_ref,
            smooth_root_xyz=args.smooth_root_xyz,
            w_velocity=args.w_velocity,
            w_acceleration=args.w_acceleration,
            max_opt_iter=args.max_opt_iter,
            show_progress=not args.no_progress,
        ),
    )
    batch.set_motion_fps(fps)

    t0 = time.perf_counter()
    q_batch = batch.retarget_batch(human_frames)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[batch-to] total {batch_ms:.0f} ms for {len(human_frames)} frames")

    if args.compare_ik:
        ik = GMR(**gmr_kwargs)
        ik_q = []
        t1 = time.perf_counter()
        frame_iter = human_frames
        if not args.no_progress:
            frame_iter = tqdm(human_frames, desc="[per-frame IK]", unit="frame")
        for frame in frame_iter:
            ik_q.append(ik.retarget(frame).copy())
        ik_ms = (time.perf_counter() - t1) * 1000.0
        ik_arr = np.asarray(ik_q)
        print(f"[per-frame IK] total {ik_ms:.0f} ms")
        print(f"[metrics] IK   hinge_acc_mean={hinge_acc_metric(ik_arr, fps):.4f}")
        print(f"[metrics] batch hinge_acc_mean={hinge_acc_metric(q_batch, fps):.4f}")
        rmse = float(np.sqrt(np.mean((ik_arr - q_batch) ** 2)))
        print(f"[metrics] qpos RMSE vs IK = {rmse:.5f}")

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        motion_data = {
            "fps": fps,
            "root_pos": q_batch[:, :3],
            "root_rot": q_batch[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": q_batch[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
            "qpos": q_batch,
            "method": f"batch_trajectory_optimization_{args.solver}",
            "src_human": src_human,
            "input_file": str(pathlib.Path(input_file).resolve()),
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved {args.save_path}")

    if args.view:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            record_video=args.record_video,
            video_path=args.video_path,
        )
        frame_idx = 0
        n_frames = len(q_batch)
        while True:
            viewer.step(qpos=q_batch[frame_idx], rate_limit=args.rate_limit, follow_camera=True)
            frame_idx += 1
            if frame_idx >= n_frames:
                if args.loop:
                    frame_idx = 0
                else:
                    break
        viewer.close()


if __name__ == "__main__":
    main()
