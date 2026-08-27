#!/usr/bin/env python3
"""MPC-like short-horizon Online QP retargeting + visualization.

Interactive default: **C++ streaming** via ``gmr_retarget_viewer --method online_qp``
(compute one frame → render immediately, ~8 ms/f).

Alternatives:
  --playback   C++ CLI solve-all then Python MuJoCo playback
  --python     Python QP stream (slow ~40 ms/f)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from rich import print

from general_motion_retargeting.human_frame_loaders import (
    frame_to_json_dict,
    load_human_motion_frames,
)


def env_with_ld() -> dict:
    env = os.environ.copy()
    devel = "/opt/robot/devel/lib"
    if pathlib.Path(devel).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def launch_cpp_streaming_viewer(
    *,
    repo: pathlib.Path,
    viewer: pathlib.Path,
    human_json: pathlib.Path,
    robot: str,
    src_human: str,
    height: float,
    preset: str,
    mode: str,
    contact_ground: bool | None,
    loop: bool,
    max_frames: int | None,
    show_human: bool,
) -> None:
    cmd = [
        str(viewer),
        "--gmr_root",
        str(repo),
        "--robot",
        robot,
        "--backend",
        "mujoco_se3",
        "--human_frame_json",
        str(human_json),
        "--src_human",
        src_human,
        "--actual_human_height",
        str(height),
        "--method",
        "online_qp",
        "--online_qp_preset",
        preset,
        "--online_qp_mode",
        mode,
        "--realtime",
    ]
    if show_human:
        cmd.append("--show_human_overlay")
    else:
        cmd.append("--hide_human_overlay")
    if max_frames:
        cmd += ["--max_frames", str(max_frames)]
    if contact_ground:
        cmd.append("--contact_ground")
    if loop:
        cmd.append("--loop")

    print("[online-qp] C++ streaming viewer (frame QP → render) ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env_with_ld(), cwd=repo)


if __name__ == "__main__":
    REPO = pathlib.Path(__file__).resolve().parents[2]
    VIEWER = REPO / "cpp" / "build" / "gmr_retarget_viewer"
    CPP_CLI = REPO / "cpp" / "build" / "gmr_online_qp_cli"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--gvhmr_pred_file", type=str, default=None)
    parser.add_argument(
        "--input_type",
        choices=["auto", "gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov"],
        default="auto",
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument(
        "--body_model_dir",
        default=str((REPO / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--preset", choices=["default", "smooth", "anti_slip"], default="anti_slip")
    parser.add_argument("--mode", choices=["lookahead", "causal"], default="lookahead")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--show_human", action="store_true")
    parser.add_argument(
        "--stream",
        dest="viz_mode",
        action="store_const",
        const="stream",
        help="C++ stream compute+render (default when viewer exists).",
    )
    parser.add_argument(
        "--playback",
        dest="viz_mode",
        action="store_const",
        const="playback",
        help="C++ solve-all then Python MuJoCo playback.",
    )
    parser.add_argument(
        "--python",
        dest="viz_mode",
        action="store_const",
        const="python",
        help="Python QP stream (slow).",
    )
    parser.set_defaults(viz_mode=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--rate_limit", dest="rate_limit", action="store_true")
    parser.add_argument("--no-rate-limit", dest="rate_limit", action="store_false")
    parser.set_defaults(rate_limit=True)
    parser.add_argument("--save_path", default=None)
    parser.add_argument(
        "--torque_limit_weight",
        type=float,
        default=0.0,
        help=">0 enables ID torque-limit soft constraint (playback/python only).",
    )
    parser.add_argument("--torque_limit_margin", type=float, default=0.1)
    parser.add_argument("--torque_limit_scope", choices=["upper", "all"], default="upper")
    parser.add_argument(
        "--torque_limit_gate_mode",
        choices=["off", "soft", "hard"],
        default="soft",
    )
    add_optional_bool_arg(parser, "contact_ground", "Enable contact/ground pipeline.")
    parser.add_argument(
        "--ground_align",
        nargs="?",
        const="lower_envelope",
        default=None,
        choices=["lower_envelope", "support_hold"],
        help="Offline Z ground-align human frames before retarget (fixes GVHMR float).",
    )
    args = parser.parse_args()

    input_path = args.input_file or args.gvhmr_pred_file
    if not input_path:
        parser.error("Provide --input_file or --gvhmr_pred_file")

    if args.viz_mode is None:
        if args.headless:
            args.viz_mode = "playback"
        elif VIEWER.is_file():
            args.viz_mode = "stream"
        elif CPP_CLI.is_file():
            args.viz_mode = "playback"
        else:
            args.viz_mode = "python"

    # Streaming viewer does not yet forward torque_limit; fall back to playback.
    if args.torque_limit_weight > 0 and args.viz_mode == "stream":
        print(
            "[online-qp] torque_limit_weight>0: switching stream→playback "
            "(viewer has no torque_limit flags yet)"
        )
        args.viz_mode = "playback"

    frames, fps, height, src_human = load_human_motion_frames(
        input_path,
        input_type=args.input_type,
        body_model_dir=args.body_model_dir,
        max_frames=args.max_frames,
        ground_align=args.ground_align or False,
        ground_align_verbose=bool(args.ground_align),
    )

    stem = pathlib.Path(input_path).stem
    if pathlib.Path(input_path).name == "hmr4d_results.pt":
        stem = pathlib.Path(input_path).parent.name

    # ---- C++ streaming viewer (true online) ----
    if args.viz_mode == "stream" and not args.headless:
        if not VIEWER.is_file():
            raise FileNotFoundError(f"Missing {VIEWER}; build gmr_retarget_viewer or use --playback")
        with tempfile.TemporaryDirectory(prefix="gmr_online_qp_viz_") as td:
            hj = pathlib.Path(td) / "human.json"
            hj.write_text(
                json.dumps(
                    {
                        "fps": float(fps),
                        "src_human": src_human,
                        "actual_human_height": float(height),
                        "frames": [frame_to_json_dict(f) for f in frames],
                    }
                )
            )
            launch_cpp_streaming_viewer(
                repo=REPO,
                viewer=VIEWER,
                human_json=hj,
                robot=args.robot,
                src_human=src_human,
                height=height,
                preset=args.preset,
                mode=args.mode,
                contact_ground=args.contact_ground,
                loop=args.loop,
                max_frames=args.max_frames,
                show_human=args.show_human,
            )
        sys.exit(0)

    # ---- playback / python / headless paths ----
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    from general_motion_retargeting.online_qp_retarget import OnlineQpConfig, OnlineQpRetargeter

    qpos_arr: np.ndarray
    mean_ms: float
    backend: str

    if args.viz_mode == "python":
        backend = "python"
        gmr = GMR(
            actual_human_height=height,
            src_human=src_human,
            tgt_robot=args.robot,
            contact_ground=args.contact_ground,
            motion_fps=fps,
            verbose=False,
        )
        cfg = OnlineQpConfig.from_preset(args.preset)
        cfg.use_lookahead = args.mode == "lookahead"
        if args.torque_limit_weight > 0:
            cfg.torque_limit_constraint = True
            cfg.torque_limit_weight = args.torque_limit_weight
            cfg.torque_limit_margin = args.torque_limit_margin
            cfg.torque_limit_scope = args.torque_limit_scope
            cfg.torque_limit_gate_mode = args.torque_limit_gate_mode
        online = OnlineQpRetargeter(gmr, cfg)
        online.set_motion_fps(fps)
        print(f"[online-qp] Python stream {len(frames)} frames (~40ms/f) ...")
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path or f"videos/{args.robot}_online_qp_{stem}.mp4",
        )
        qpos_list, ms_list = [], []
        for qpos in online.iter_retarget_sequence(frames):
            qpos_list.append(qpos.copy())
            ms_list.append(online.last_frame_ms)
            viewer.step(
                qpos=qpos,
                human_motion_data=(online.gmr.scaled_human_data if args.show_human else None),
                rate_limit=args.rate_limit and (not args.headless),
                follow_camera=True,
            )
            if not args.headless and not viewer.viewer.is_running():
                break
        viewer.close()
        qpos_arr = np.stack(qpos_list) if qpos_list else np.zeros((0, 1))
        mean_ms = float(np.mean(ms_list)) if ms_list else 0.0
    else:
        backend = "cpp_playback"
        if not CPP_CLI.is_file():
            raise FileNotFoundError(f"Missing {CPP_CLI}")
        with tempfile.TemporaryDirectory(prefix="gmr_online_qp_") as td:
            hj = pathlib.Path(td) / "h.json"
            oj = pathlib.Path(td) / "o.json"
            hj.write_text(
                json.dumps(
                    {
                        "fps": float(fps),
                        "src_human": src_human,
                        "actual_human_height": float(height),
                        "frames": [frame_to_json_dict(f) for f in frames],
                    }
                )
            )
            cmd = [
                str(CPP_CLI),
                "--gmr_root",
                str(REPO),
                "--robot",
                args.robot,
                "--human_frame_json",
                str(hj),
                "--out_json",
                str(oj),
                "--src_human",
                src_human,
                "--actual_human_height",
                str(height),
                "--preset",
                args.preset,
                "--mode",
                args.mode,
                "--max_frames",
                str(len(frames)),
                "--benchmark",
            ]
            if args.contact_ground:
                cmd.append("--contact_ground")
            if args.torque_limit_weight > 0:
                cmd += [
                    "--torque_limit_weight",
                    str(args.torque_limit_weight),
                    "--torque_limit_margin",
                    str(args.torque_limit_margin),
                    "--torque_limit_scope",
                    args.torque_limit_scope,
                    "--torque_limit_gate_mode",
                    args.torque_limit_gate_mode,
                ]
            print("[online-qp] C++ solve-all then playback ...")
            subprocess.run(cmd, check=True, env=env_with_ld(), cwd=REPO)
            data = json.loads(oj.read_text())
            qpos_arr = np.asarray(data["qpos_frames"], dtype=float)
            mean_ms = float(data["profile"]["ms_per_frame"])

        if (not args.headless) or args.record_video:
            print(f"[online-qp] playing {len(qpos_arr)} frames @ {fps:.0f} FPS ...")
            viewer = RobotMotionViewer(
                robot_type=args.robot,
                motion_fps=fps,
                transparent_robot=0,
                record_video=args.record_video,
                video_path=args.video_path or f"videos/{args.robot}_online_qp_{stem}.mp4",
            )
            idx = 0
            while True:
                if args.loop and not args.headless:
                    if not viewer.viewer.is_running():
                        break
                    q = qpos_arr[idx % len(qpos_arr)]
                else:
                    if idx >= len(qpos_arr) or (
                        not args.headless and not viewer.viewer.is_running()
                    ):
                        break
                    q = qpos_arr[idx]
                viewer.step(
                    qpos=q,
                    rate_limit=args.rate_limit and (not args.headless),
                    follow_camera=True,
                )
                idx += 1
            viewer.close()

    print(f"\n[online-qp] backend={backend} preset={args.preset} mode={args.mode} src={src_human}")
    if args.torque_limit_weight > 0:
        print(
            f"  torque_limit: w={args.torque_limit_weight} "
            f"margin={args.torque_limit_margin} scope={args.torque_limit_scope} "
            f"gate={args.torque_limit_gate_mode}"
        )
    print(f"  frames={len(qpos_arr)}  mean={mean_ms:.2f} ms/f  realtime@30={mean_ms <= 1000/30}")

    if args.save_path:
        import pickle

        with open(args.save_path, "wb") as f:
            pickle.dump(
                {"fps": fps, "qpos": qpos_arr, "method": f"online_qp_{backend}", "preset": args.preset},
                f,
            )
        print(f"  saved {args.save_path}")
