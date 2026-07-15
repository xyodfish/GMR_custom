#!/usr/bin/env python3
"""Run GVHMR on a monocular video, then retarget the result to a robot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.gvhmr_env import DEFAULT_GVHMR_ROOT, resolve_gvhmr_python

HERE = Path(__file__).resolve().parent


def gvhmr_pred_path(gvhmr_root: Path, video_path: Path) -> Path:
    return gvhmr_root / "outputs" / "demo" / video_path.stem / "hmr4d_results.pt"


def run_gvhmr_demo(
    video_path: Path,
    gvhmr_root: Path,
    gvhmr_python: str,
    static_cam: bool,
) -> Path:
    demo_script = gvhmr_root / "tools" / "demo" / "demo.py"
    if not demo_script.is_file():
        raise FileNotFoundError(f"GVHMR demo script not found: {demo_script}")

    pred_path = gvhmr_pred_path(gvhmr_root, video_path)
    if pred_path.is_file():
        print(f"[GVHMR] Reusing existing prediction: {pred_path}")
        return pred_path

    cmd = [gvhmr_python, str(demo_script), f"--video={video_path}"]
    if static_cam:
        cmd.append("-s")

    print("[GVHMR] Running:", " ".join(cmd))
    print(f"[GVHMR] Working directory: {gvhmr_root}")
    subprocess.run(cmd, cwd=str(gvhmr_root), check=True)

    if not pred_path.is_file():
        raise FileNotFoundError(
            f"GVHMR finished but prediction file not found: {pred_path}"
        )
    return pred_path


def build_gvhmr_retarget_cmd(args: argparse.Namespace, pred_path: Path) -> list[str]:
    retarget_algo = getattr(args, "retarget_algo", "ik")
    script_name = "to_robot_trajectory_opt.py" if retarget_algo == "to" else "to_robot.py"
    cmd = [
        sys.executable,
        str(HERE / script_name),
        "--gvhmr_pred_file",
        str(pred_path),
        "--robot",
        args.robot,
    ]
    if args.body_model_dir:
        cmd += ["--body_model_dir", args.body_model_dir]
    if args.loop:
        cmd.append("--loop")
    if args.record_video:
        cmd.append("--record_video")
        cmd += ["--video_path", args.video_path]
    if args.rate_limit:
        cmd.append("--rate_limit")
    else:
        cmd.append("--no-rate-limit")
    if args.save_path:
        cmd += ["--save_path", args.save_path]
    for name in ("contact_ground", "foot_ground_limit", "fix_robot_penetration"):
        value = getattr(args, name)
        if value is True:
            cmd.append(f"--{name}")
        elif value is False:
            cmd.append(f"--no-{name}")
    if retarget_algo == "to":
        cmd += ["--to_mode", getattr(args, "to_mode", "fast")]
        cmd += ["--window_size", str(int(getattr(args, "window_size", 8)))]
        cmd += ["--w_velocity", str(float(getattr(args, "w_velocity", 2.0)))]
        cmd += ["--w_acceleration", str(float(getattr(args, "w_acceleration", 10.0)))]
        if not getattr(args, "use_gmr_init", True):
            cmd.append("--no-use_gmr_init")
    return cmd


def add_optional_bool_arg(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=str, help="Input monocular video path.")
    parser.add_argument(
        "--gvhmr_root",
        type=str,
        default=str(DEFAULT_GVHMR_ROOT),
        help="GVHMR repository root.",
    )
    parser.add_argument(
        "--gvhmr_python",
        type=str,
        default="",
        help="Python executable for GVHMR (default: auto-detect gvhmr conda env).",
    )
    parser.add_argument(
        "--static_cam",
        default=True,
        action="store_true",
        help="Use static camera mode (skip visual odometry).",
    )
    parser.add_argument(
        "--no-static_cam",
        dest="static_cam",
        action="store_false",
        help="Enable moving-camera visual odometry in GVHMR.",
    )
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
        ],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion PKL.")
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str((REPO_ROOT / "assets" / "body_models").resolve()),
    )
    parser.add_argument("--loop", default=False, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/gmr_gui_output.mp4",
        help="Output MP4 path for robot motion recording.",
    )
    parser.add_argument(
        "--rate_limit",
        dest="rate_limit",
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    parser.add_argument(
        "--no-rate-limit",
        dest="rate_limit",
        action="store_false",
        help="Disable realtime playback limiting and render as fast as possible.",
    )
    parser.set_defaults(rate_limit=True)
    parser.add_argument(
        "--retarget_algo",
        choices=["ik", "to"],
        default="ik",
        help="Retargeting algorithm after GVHMR: per-frame IK or trajectory optimization.",
    )
    parser.add_argument("--to_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--use_gmr_init", action="store_true", default=True)
    parser.add_argument("--no-use_gmr_init", dest="use_gmr_init", action="store_false")
    add_optional_bool_arg(parser, "contact_ground", "Enable streaming contact/ground fix.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable QP foot-ground inequality limit.")
    add_optional_bool_arg(
        parser,
        "fix_robot_penetration",
        "Enable post-IK robot root lift penetration repair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    gvhmr_root = Path(args.gvhmr_root).expanduser().resolve()
    if not gvhmr_root.is_dir():
        raise FileNotFoundError(f"GVHMR root not found: {gvhmr_root}")

    gvhmr_python = resolve_gvhmr_python(gvhmr_root, args.gvhmr_python)
    print(f"[GVHMR] Using Python: {gvhmr_python}")

    pred_path = run_gvhmr_demo(video_path, gvhmr_root, gvhmr_python, args.static_cam)
    retarget_cmd = build_gvhmr_retarget_cmd(args, pred_path)
    print("[GMR] Running:", " ".join(retarget_cmd))
    subprocess.run(retarget_cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
