#!/usr/bin/env python3
"""Run C++ online QP-MPC from GVHMR .pt / SMPL-X / BVH."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.human_frame_loaders import frame_to_json_dict, load_human_motion_frames

DEFAULT_CPP_CLI = REPO / "cpp" / "build" / "gmr_online_qp_cli"


def env_with_ld() -> dict:
    env = os.environ.copy()
    devel = "/opt/robot/devel/lib"
    if pathlib.Path(devel).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help=".pt / .npz / .pkl / .bvh")
    parser.add_argument(
        "--input_type",
        default="auto",
        choices=["auto", "gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov"],
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--gmr_root", default=str(REPO))
    parser.add_argument("--cpp_cli", default=str(DEFAULT_CPP_CLI))
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--body_model_dir", default=str(REPO / "assets" / "body_models"))
    parser.add_argument("--preset", choices=["default", "smooth", "anti_slip"], default="anti_slip")
    parser.add_argument("--mode", choices=["lookahead", "causal"], default="lookahead")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--sqp_iters", type=int, default=None)
    parser.add_argument("--w_foot_slip", type=float, default=None)
    parser.add_argument("--w_gmr", type=float, default=None)
    parser.add_argument("--contact_ground", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--human_json", default=None)
    args = parser.parse_args()

    frames, fps, height, src_human = load_human_motion_frames(
        args.input_file,
        input_type=args.input_type,
        body_model_dir=args.body_model_dir,
        tgt_fps=args.motion_fps,
        max_frames=args.max_frames,
    )

    human_json_path: pathlib.Path
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.human_json:
        human_json_path = pathlib.Path(args.human_json)
        human_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="gmr_online_qp_human_")
        human_json_path = pathlib.Path(tmp_dir.name) / "human_frames.json"

    payload = {
        "fps": float(fps),
        "src_human": src_human,
        "actual_human_height": float(height),
        "input_file": str(pathlib.Path(args.input_file).resolve()),
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    human_json_path.write_text(json.dumps(payload))

    cmd = [
        str(args.cpp_cli),
        "--gmr_root",
        args.gmr_root,
        "--robot",
        args.robot,
        "--human_frame_json",
        str(human_json_path),
        "--out_json",
        args.out_json,
        "--src_human",
        src_human,
        "--actual_human_height",
        str(height),
        "--preset",
        args.preset,
        "--mode",
        args.mode,
    ]
    if args.max_frames:
        cmd += ["--max_frames", str(args.max_frames)]
    if args.horizon is not None:
        cmd += ["--horizon", str(args.horizon)]
    if args.sqp_iters is not None:
        cmd += ["--sqp_iters", str(args.sqp_iters)]
    if args.w_foot_slip is not None:
        cmd += ["--w_foot_slip", str(args.w_foot_slip)]
    if args.w_gmr is not None:
        cmd += ["--w_gmr", str(args.w_gmr)]
    if args.contact_ground:
        cmd.append("--contact_ground")
    if args.benchmark:
        cmd.append("--benchmark")

    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env_with_ld(), cwd=REPO)
    if tmp_dir is not None:
        tmp_dir.cleanup()


if __name__ == "__main__":
    main()
