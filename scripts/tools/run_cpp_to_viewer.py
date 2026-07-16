#!/usr/bin/env python3
"""One-shot C++ batch TO / online QP: load motion file → temp human JSON → gmr_retarget_viewer."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from general_motion_retargeting.human_frame_loaders import frame_to_json_dict, load_human_motion_frames

DEFAULT_VIEWER = REPO / "cpp" / "build" / "gmr_retarget_viewer"


def _tri_bool(value: str | None, flag: str, cmd: list[str]) -> None:
    if value == "开启":
        cmd.append(f"--{flag}")
    elif value == "关闭":
        cmd.append(f"--no_{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C++ batch TO / online QP viewer from .pt/.npz/.bvh.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--input_type",
        default="auto",
        choices=["auto", "gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov"],
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--method", required=True, choices=["batch_to", "online_qp"])
    parser.add_argument("--gmr_root", default=str(REPO))
    parser.add_argument("--viewer", default=str(DEFAULT_VIEWER))
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--body_model_dir", default=str(REPO / "assets" / "body_models"))
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--gn_steps", type=int, default=3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--quality", action="store_true", help="batch: best line search + dense GN")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no_loop", action="store_true")
    parser.add_argument("--contact_ground", default=None, help="IK 默认|开启|关闭")
    parser.add_argument("--fix_robot_penetration", default=None)
    parser.add_argument("--foot_ground_limit", default=None)
    parser.add_argument("--out_json", default=None, help="batch_to only: save qpos JSON after optimize")
    parser.add_argument("--online_qp_preset", default="anti_slip", choices=["default", "smooth", "anti_slip"])
    parser.add_argument("--online_qp_mode", default="lookahead", choices=["lookahead", "causal"])
    args = parser.parse_args()

    if args.fast and args.quality:
        parser.error("Use either --fast or --quality, not both.")

    frames, fps, height, src_human = load_human_motion_frames(
        args.input_file,
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
        "input_file": str(pathlib.Path(args.input_file).resolve()),
        "frames": [frame_to_json_dict(f) for f in frames],
    }

    print(f"[run_cpp_to_viewer] loaded {len(frames)} frames @ {fps:.1f} fps src_human={src_human}", flush=True)
    if len(frames) > 300 and args.method == "batch_to":
        print(
            "[run_cpp_to_viewer] Large clip: batch GN runs in the terminal first; "
            "MuJoCo window opens when optimization finishes.",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="gmr_cpp_to_") as tmp:
        human_json = pathlib.Path(tmp) / "human_frames.json"
        human_json.write_text(json.dumps(payload))

        cmd = [
            str(args.viewer),
            "--gmr_root",
            args.gmr_root,
            "--robot",
            args.robot,
            "--backend",
            "mujoco_se3",
            "--human_frame_json",
            str(human_json),
            "--src_human",
            src_human,
            "--actual_human_height",
            str(height),
            "--method",
            args.method,
        ]

        if args.max_frames:
            cmd += ["--max_frames", str(args.max_frames)]

        _tri_bool(args.contact_ground, "contact_ground", cmd)
        _tri_bool(args.fix_robot_penetration, "fix_robot_penetration", cmd)
        _tri_bool(args.foot_ground_limit, "foot_ground_limit", cmd)

        if args.method == "batch_to":
            cmd += [
                "--precompute",
                "--window_size",
                str(args.window_size),
                "--window_stride",
                str(args.window_stride),
                "--gn_steps",
                str(args.gn_steps),
            ]
            if args.fast:
                cmd.append("--fast")
            if args.out_json:
                cmd += ["--out_json", args.out_json]
        else:
            cmd += [
                "--realtime",
                "--online_qp_preset",
                args.online_qp_preset,
                "--online_qp_mode",
                args.online_qp_mode,
            ]

        if args.loop:
            cmd.append("--loop")
        elif args.no_loop:
            cmd.append("--no_loop")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        devel_lib = "/opt/robot/devel/lib"
        if pathlib.Path(devel_lib).is_dir():
            env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        print(f"[run_cpp_to_viewer] launching viewer (human_json temp dir kept alive)...", flush=True)
        subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
