#!/usr/bin/env python3
"""GVHMR .pt retargeting with online batch-lite TO (causal multi-frame GN)."""

import argparse
import pathlib
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.online_batch_retarget import (
    OnlineBatchConfig,
    OnlineBatchRetargeter,
)
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)

from rich import print


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


if __name__ == "__main__":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="GVHMR retargeting with online batch-lite trajectory optimization.",
    )
    parser.add_argument(
        "--gvhmr_pred_file",
        type=str,
        required=True,
        help="Path to hmr4d_results.pt from GVHMR.",
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str((REPO_ROOT / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--rate_limit", dest="rate_limit", action="store_true")
    parser.add_argument("--no-rate-limit", dest="rate_limit", action="store_false")
    parser.set_defaults(rate_limit=False)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip MuJoCo viewer; only compute trajectory and print timing.",
    )
    parser.add_argument(
        "--compare_ik",
        action="store_true",
        help="Also run per-frame IK and print timing comparison.",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality", "track", "extrap"],
        default="balanced",
        help="Preset: extrap = GMR bootstrap then history extrapolation (no per-frame IK).",
    )
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--gn_steps", type=int, default=None)
    parser.add_argument("--opt_trailing_frames", type=int, default=None)
    parser.add_argument("--light_ik_iters", type=int, default=None)
    parser.add_argument(
        "--seed_mode",
        choices=["gmr_ik", "extrapolate"],
        default=None,
        help="Warmstart: gmr_ik (light IK) or extrapolate (no GMR IK after bootstrap).",
    )
    parser.add_argument(
        "--gmr_bootstrap_frames",
        type=int,
        default=None,
        help="Use full GMR.retarget() for the first K frames, then seed_mode.",
    )
    parser.add_argument(
        "--extrap_policy",
        choices=["hold", "velocity"],
        default=None,
        help="After bootstrap: hold last q (safer) or velocity extrapolate.",
    )
    add_optional_bool_arg(
        parser,
        "bootstrap_commit_gmr",
        "During bootstrap, commit GMR q and skip window TO (recommended).",
    )
    parser.add_argument(
        "--ik_blend",
        type=float,
        default=None,
        help="Blend TO result toward seed in [0,1]. Higher → closer to warmstart.",
    )
    parser.add_argument(
        "--knee_min_bend_deg",
        type=float,
        default=None,
        help="Enforce a minimum knee bend on near-straight legs (0 disables).",
    )
    parser.add_argument(
        "--joint_limit_margin_deg",
        type=float,
        default=None,
        help="Keep committed hinge joints this many degrees away from hard limits.",
    )
    parser.add_argument("--w_velocity", type=float, default=None)
    parser.add_argument("--w_acceleration", type=float, default=None)
    parser.add_argument("--w_foot_slip", type=float, default=None)
    parser.add_argument("--w_foot_height", type=float, default=None)
    add_optional_bool_arg(
        parser,
        "enable_foot_penalties",
        "Enable / disable foot height+slip penalties in the window GN.",
    )
    add_optional_bool_arg(
        parser,
        "finalize_contact",
        "Enable / disable post-TO contact finalize (root lift).",
    )
    add_optional_bool_arg(
        parser,
        "contact_ground",
        "Enable streaming contact/ground fix (recommended for GVHMR).",
    )
    add_optional_bool_arg(
        parser,
        "fix_robot_penetration",
        "Enable post-IK robot root lift penetration repair.",
    )

    args = parser.parse_args()

    smplx_folder = pathlib.Path(args.body_model_dir)
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, smplx_folder
    )
    tgt_fps = 30
    human_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        contact_ground=args.contact_ground,
        fix_robot_penetration=args.fix_robot_penetration,
        motion_fps=aligned_fps,
        verbose=False,
    )
    gmr = GMR(**gmr_kwargs)

    ob_cfg = OnlineBatchConfig.from_preset(args.preset)
    for name in (
        "window_size",
        "gn_steps",
        "opt_trailing_frames",
        "light_ik_iters",
        "ik_blend",
        "seed_mode",
        "gmr_bootstrap_frames",
        "extrap_policy",
        "bootstrap_commit_gmr",
        "knee_min_bend_deg",
        "joint_limit_margin_deg",
        "w_velocity",
        "w_acceleration",
        "w_foot_slip",
        "w_foot_height",
        "enable_foot_penalties",
        "finalize_contact",
    ):
        val = getattr(args, name)
        if val is not None:
            setattr(ob_cfg, name, val)
    online = OnlineBatchRetargeter(gmr, ob_cfg)
    online.set_motion_fps(aligned_fps)

    compare_gmr = GMR(**gmr_kwargs) if args.compare_ik else None

    if args.headless:
        qpos_list = []
        frame_ms = []
        for frame in human_frames:
            qpos_list.append(online.retarget(frame))
            frame_ms.append(online.last_frame_ms)
        qpos_arr = np.stack(qpos_list)
        ms_arr = np.asarray(frame_ms)
    else:
        stem = pathlib.Path(args.gvhmr_pred_file).parent.name
        default_video = f"videos/{args.robot}_online_batch_gvhmr_{stem}.mp4"
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path or default_video,
        )

        qpos_list = []
        frame_ms = []
        frame_idx = 0

        while True:
            if args.loop:
                frame_idx = (frame_idx + 1) % len(human_frames)
            else:
                if frame_idx >= len(human_frames):
                    break

            human_frame = human_frames[frame_idx]
            qpos = online.retarget(human_frame)
            frame_ms.append(online.last_frame_ms)
            qpos_list.append(qpos.copy())

            # Must use qpos= keyword: positional arg maps to root_pos.
            viewer.step(
                qpos=qpos,
                human_motion_data=online.gmr.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            frame_idx += 1

        viewer.close()
        qpos_arr = np.stack(qpos_list)
        ms_arr = np.asarray(frame_ms)

    warmup = min(3, len(ms_arr))
    steady_ms = float(np.mean(ms_arr[warmup:])) if len(ms_arr) > warmup else float(np.mean(ms_arr))

    print(
        f"\n[online-batch] preset={ob_cfg.preset} seed={ob_cfg.seed_mode} "
        f"bootstrap={ob_cfg.gmr_bootstrap_frames} window={ob_cfg.window_size} "
        f"gn={ob_cfg.gn_steps} ik_blend={ob_cfg.ik_blend}"
    )
    print(f"  frames={len(qpos_arr)}  mean={np.mean(ms_arr):.2f} ms/f  steady={steady_ms:.2f} ms/f")
    print(f"  max={np.max(ms_arr):.2f} ms/f  realtime@30fps={steady_ms <= 1000/30}")

    if compare_gmr is not None:
        t0 = time.perf_counter()
        ik_q = np.stack([compare_gmr.retarget(f).copy() for f in human_frames])
        ik_ms = (time.perf_counter() - t0) * 1000.0 / len(human_frames)
        rmse = float(np.sqrt(np.mean((ik_q - qpos_arr) ** 2)))
        print(f"  vs IK: rmse={rmse:.4f}  ik_ms={ik_ms:.2f} ms/f  ratio={steady_ms/max(ik_ms,1e-9):.1f}x")

    if args.save_path:
        import pickle

        with open(args.save_path, "wb") as f:
            pickle.dump(
                {
                    "fps": aligned_fps,
                    "qpos": qpos_arr,
                    "method": "online_batch",
                    "preset": args.preset,
                    "ms_per_frame": steady_ms,
                },
                f,
            )
        print(f"  saved {args.save_path}")
