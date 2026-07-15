"""BVH retargeting with independent causal trajectory optimization."""

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
from general_motion_retargeting.utils.lafan1 import load_bvh_file

from rich import print


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def joint_velocity_metric(qpos_seq: np.ndarray) -> float:
    if len(qpos_seq) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(qpos_seq, axis=0), axis=1)))


def joint_acceleration_metric(qpos_seq: np.ndarray) -> float:
    if len(qpos_seq) < 3:
        return 0.0
    acc = qpos_seq[2:] - 2.0 * qpos_seq[1:-1] + qpos_seq[:-2]
    return float(np.mean(np.linalg.norm(acc, axis=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVH retargeting with independent causal TO (FK + smoothness).",
    )
    parser.add_argument("--bvh_file", type=str, required=True, help="BVH motion file.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "unitree_h2",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument(
        "--compare_ik",
        action="store_true",
        help="Also run per-frame GMR IK for end-of-run metrics.",
    )
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--w_anchor", type=float, default=20.0)
    parser.add_argument("--max_opt_iter", type=int, default=25)
    parser.add_argument("--fast_opt_iter", type=int, default=5)
    parser.add_argument(
        "--to_mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: single-frame TO (~real-time). full: joint window (offline).",
    )
    parser.add_argument("--use_gmr_init", action="store_true", default=True)
    parser.add_argument("--no-use_gmr_init", dest="use_gmr_init", action="store_false")
    parser.add_argument("--fix_window_prefix", action="store_true", default=False)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Process only the first N BVH frames (default: all).",
    )
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")

    args = parser.parse_args()

    human_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames is not None:
        human_frames = human_frames[: args.max_frames]
        print(f"Using first {len(human_frames)} BVH frames")

    motion_fps = float(args.motion_fps)
    src_human = f"bvh_{args.format}"

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=args.robot,
        contact_ground=args.contact_ground,
        foot_ground_limit=args.foot_ground_limit,
        fix_robot_penetration=args.fix_robot_penetration,
        motion_fps=motion_fps,
    )
    gmr = GMR(**gmr_kwargs)
    gmr.set_motion_fps(motion_fps)

    to_cfg = TrajectoryOptimizationConfig(
        window_size=args.window_size,
        mode="fast" if args.fix_window_prefix or args.to_mode == "fast" else "full",
        w_velocity=args.w_velocity,
        w_acceleration=args.w_acceleration,
        w_anchor=args.w_anchor,
        max_opt_iter=args.max_opt_iter,
        fast_opt_iter=args.fast_opt_iter,
        use_gmr_init=args.use_gmr_init,
    )
    to_retarget = TrajectoryOptimizationRetargeter(gmr, to_cfg)
    to_retarget.set_motion_fps(motion_fps)
    compare_gmr = GMR(**gmr_kwargs) if args.compare_ik else None
    if compare_gmr is not None:
        compare_gmr.set_motion_fps(motion_fps)

    stem = pathlib.Path(args.bvh_file).stem
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        record_video=args.record_video,
        video_path=args.video_path or f"videos/{args.robot}_to_bvh_{stem}.mp4",
    )

    qpos_list = []
    ik_qpos_list = [] if args.compare_ik else None
    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()

    print(f"mocap_frame_rate: {motion_fps}, frames: {len(human_frames)}")

    while True:
        if args.loop:
            frame_idx = (frame_idx + 1) % len(human_frames)
        else:
            if frame_idx >= len(human_frames):
                break

        t0 = time.perf_counter()
        human_frame = human_frames[frame_idx]
        qpos = to_retarget.retarget(human_frame)
        qpos_list.append(qpos.copy())

        if compare_gmr is not None:
            ik_qpos_list.append(compare_gmr.retarget(human_frame).copy())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fps_counter += 1
        now = time.time()
        if now - fps_start >= 2.0:
            print(
                f"[trajectory-opt] FPS: {fps_counter / (now - fps_start):.1f}, "
                f"last frame: {elapsed_ms:.1f} ms, window={args.window_size}"
            )
            fps_counter = 0
            fps_start = now

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

    arr = np.asarray(qpos_list)
    print(
        f"[metrics] trajectory-opt  vel={joint_velocity_metric(arr):.5f}, "
        f"acc={joint_acceleration_metric(arr):.5f}"
    )
    if ik_qpos_list is not None:
        ik_arr = np.asarray(ik_qpos_list)
        print(
            f"[metrics] per-frame IK      vel={joint_velocity_metric(ik_arr):.5f}, "
            f"acc={joint_acceleration_metric(ik_arr):.5f}"
        )

    if args.save_path is not None:
        import pickle

        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        motion_data = {
            "fps": motion_fps,
            "root_pos": arr[:, :3],
            "root_rot": arr[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": arr[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
            "method": "trajectory_optimization",
            "window_size": args.window_size,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    viewer.close()
