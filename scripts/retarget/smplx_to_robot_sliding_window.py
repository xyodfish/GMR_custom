"""Stream SMPL-X motion with sliding-window retargeting (no real robot required).

Kinematic feedback comes from the previous optimized qpos stored in MuJoCo,
not from hardware encoders.
"""

import argparse
import os
import pathlib
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, PLANAR_BASE_ROBOTS
from general_motion_retargeting.sliding_window_retarget import (
    SlidingWindowConfig,
    SlidingWindowRetargeter,
)
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def joint_velocity_metric(qpos_seq: np.ndarray, dt: float = 1.0 / 30.0) -> float:
    if len(qpos_seq) < 2:
        return 0.0
    diffs = np.diff(qpos_seq, axis=0) / max(dt, 1e-12)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def joint_acceleration_metric(qpos_seq: np.ndarray, dt: float = 1.0 / 30.0) -> float:
    if len(qpos_seq) < 3:
        return 0.0
    acc = (qpos_seq[2:] - 2.0 * qpos_seq[1:-1] + qpos_seq[:-2]) / max(dt * dt, 1e-12)
    return float(np.mean(np.linalg.norm(acc, axis=1)))


if __name__ == "__main__":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="SMPL-X retargeting with causal sliding-window optimization.",
    )
    parser.add_argument("--smplx_file", type=str, required=True)
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "unitree_h2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "galbot_one_golf",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str((REPO_ROOT / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument(
        "--compare_ik",
        action="store_true",
        help="Also run per-frame IK on the same stream and print smoothness metrics.",
    )
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: refine current frame only (default). full: joint L-BFGS-B over window (slow).",
    )
    parser.add_argument(
        "--solver",
        choices=["gn", "lbfgs"],
        default="gn",
        help="fast-mode solver: gn=Jacobian Gauss-Newton (default, real-time); lbfgs=legacy scipy.",
    )
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--w_anchor", type=float, default=50.0)
    parser.add_argument("--ik_warmstart_iters", type=int, default=3)
    parser.add_argument("--gn_steps", type=int, default=3)
    parser.add_argument("--gn_damping", type=float, default=0.05)
    parser.add_argument("--gn_max_step", type=float, default=0.08)
    parser.add_argument("--dq_max", type=float, default=8.0, help="Hard |dq| limit (rad/s or m/s).")
    parser.add_argument("--ddq_max", type=float, default=80.0, help="Hard |ddq| limit (rad/s^2).")
    add_optional_bool_arg(
        parser,
        "enforce_dq_ddq",
        "Project each frame onto q/dq/ddq box limits (default: on).",
    )
    parser.add_argument("--fast_opt_iter", type=int, default=5)
    parser.add_argument("--max_opt_iter", type=int, default=25)
    add_optional_bool_arg(
        parser,
        "contact_ground",
        "Enable streaming contact/ground fix (default: IK config).",
    )
    add_optional_bool_arg(
        parser,
        "foot_ground_limit",
        "Enable QP foot-ground inequality limit (default: IK config).",
    )
    add_optional_bool_arg(
        parser,
        "fix_robot_penetration",
        "Enable post-IK robot root lift penetration repair (default: IK config).",
    )

    args = parser.parse_args()

    smplx_folder = pathlib.Path(args.body_model_dir)
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, smplx_folder
    )
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
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

    sw_cfg = SlidingWindowConfig(
        window_size=args.window_size,
        mode=args.mode,
        solver=args.solver,
        w_velocity=args.w_velocity,
        w_acceleration=args.w_acceleration,
        w_anchor=args.w_anchor,
        ik_warmstart_iters=args.ik_warmstart_iters,
        gn_steps=args.gn_steps,
        gn_damping=args.gn_damping,
        gn_max_step=args.gn_max_step,
        dq_max=args.dq_max,
        ddq_max=args.ddq_max,
        enforce_dq_ddq=True if args.enforce_dq_ddq is None else args.enforce_dq_ddq,
        fast_opt_iter=args.fast_opt_iter,
        max_opt_iter=args.max_opt_iter,
        dt=1.0 / aligned_fps,
    )
    sw_retarget = SlidingWindowRetargeter(gmr, sw_cfg)

    compare_gmr = None
    if args.compare_ik:
        compare_gmr = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
            contact_ground=args.contact_ground,
            foot_ground_limit=args.foot_ground_limit,
            fix_robot_penetration=args.fix_robot_penetration,
            motion_fps=aligned_fps,
        )

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        camera_follow=True,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=(
            f"videos/{args.robot}_sw_{pathlib.Path(args.smplx_file).stem}.mp4"
        ),
    )

    sw_qpos_list = []
    ik_qpos_list = [] if args.compare_ik else None
    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()

    while True:
        if args.loop:
            frame_idx = (frame_idx + 1) % len(smplx_data_frames)
        else:
            if frame_idx >= len(smplx_data_frames):
                break

        t0 = time.perf_counter()
        human_frame = smplx_data_frames[frame_idx]
        qpos = sw_retarget.retarget(human_frame)
        sw_qpos_list.append(qpos.copy())

        if compare_gmr is not None:
            q_ik = compare_gmr.retarget(human_frame)
            ik_qpos_list.append(q_ik.copy())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fps_counter += 1
        now = time.time()
        if now - fps_start >= 2.0:
            print(
                f"[sliding-window] retarget FPS: {fps_counter / (now - fps_start):.1f}, "
                f"last frame: {elapsed_ms:.1f} ms, mode={args.mode}, solver={args.solver}, "
                f"window={args.window_size}"
            )
            fps_counter = 0
            fps_start = now

        if args.robot in PLANAR_BASE_ROBOTS:
            viewer.step(
                qpos=qpos,
                human_motion_data=gmr.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
        else:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=gmr.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

        frame_idx += 1

    sw_arr = np.asarray(sw_qpos_list)
    dt = 1.0 / aligned_fps
    print(
        f"[metrics] sliding-window  vel={joint_velocity_metric(sw_arr, dt):.5f}, "
        f"acc={joint_acceleration_metric(sw_arr, dt):.5f}"
    )
    if ik_qpos_list is not None:
        ik_arr = np.asarray(ik_qpos_list)
        print(
            f"[metrics] per-frame IK      vel={joint_velocity_metric(ik_arr, dt):.5f}, "
            f"acc={joint_acceleration_metric(ik_arr, dt):.5f}"
        )

    if args.save_path is not None:
        import pickle

        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if args.robot in PLANAR_BASE_ROBOTS:
            motion_data = {
                "fps": aligned_fps,
                "qpos": sw_arr,
                "root_pos": sw_arr[:, :3],
                "root_rot": np.zeros((len(sw_arr), 4), dtype=np.float64),
                "dof_pos": sw_arr[:, 3:],
                "local_body_pos": None,
                "link_body_list": None,
                "method": "sliding_window",
                "window_size": args.window_size,
            }
        else:
            motion_data = {
                "fps": aligned_fps,
                "root_pos": sw_arr[:, :3],
                "root_rot": sw_arr[:, 3:7][:, [1, 2, 3, 0]],
                "dof_pos": sw_arr[:, 7:],
                "local_body_pos": None,
                "link_body_list": None,
                "method": "sliding_window",
                "window_size": args.window_size,
            }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    viewer.close()
