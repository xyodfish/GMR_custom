"""SMPL-X retargeting with independent causal trajectory optimization."""

import argparse
import os
import pathlib
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, PLANAR_BASE_ROBOTS
from general_motion_retargeting.trajectory_optimization_retarget import (
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationRetargeter,
)
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


if __name__ == "__main__":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="SMPL-X retargeting with independent causal TO (FK + smoothness).",
    )
    parser.add_argument("--smplx_file", type=str, required=True)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str((REPO_ROOT / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--compare_ik", action="store_true", default=False)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--w_anchor", type=float, default=20.0)
    parser.add_argument("--max_opt_iter", type=int, default=25)
    parser.add_argument("--fast_opt_iter", type=int, default=5)
    parser.add_argument("--to_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--use_gmr_init", action="store_true", default=True)
    parser.add_argument("--no-use_gmr_init", dest="use_gmr_init", action="store_false")
    parser.add_argument("--fix_window_prefix", action="store_true", default=False)
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")

    args = parser.parse_args()

    smplx_folder = pathlib.Path(args.body_model_dir)
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, smplx_folder
    )
    human_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    gmr = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        contact_ground=args.contact_ground,
        foot_ground_limit=args.foot_ground_limit,
        fix_robot_penetration=args.fix_robot_penetration,
        motion_fps=aligned_fps,
    )
    to_retarget = TrajectoryOptimizationRetargeter(
        gmr,
        TrajectoryOptimizationConfig(
            window_size=args.window_size,
            mode="fast" if args.fix_window_prefix or args.to_mode == "fast" else "full",
            w_velocity=args.w_velocity,
            w_acceleration=args.w_acceleration,
            w_anchor=args.w_anchor,
            max_opt_iter=args.max_opt_iter,
            fast_opt_iter=args.fast_opt_iter,
            use_gmr_init=args.use_gmr_init,
        ),
    )

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_to_{pathlib.Path(args.smplx_file).stem}.mp4",
    )

    frame_idx = 0
    while True:
        if args.loop:
            frame_idx = (frame_idx + 1) % len(human_frames)
        else:
            if frame_idx >= len(human_frames):
                break

        qpos = to_retarget.retarget(human_frames[frame_idx])
        if args.robot in PLANAR_BASE_ROBOTS:
            viewer.step(
                qpos=qpos,
                human_motion_data=gmr.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
        else:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=gmr.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
        if not args.loop:
            frame_idx += 1

    viewer.close()
