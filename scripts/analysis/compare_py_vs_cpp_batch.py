#!/usr/bin/env python3
"""One-shot Python vs C++ batch TO parity benchmark on the same GVHMR input."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CPP_CLI = REPO_ROOT / "cpp" / "build" / "gmr_batch_to_cli"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def qpos_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_python_batch(
    pt_path: pathlib.Path,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    window_size: int,
    window_stride: int,
    gn_steps: int,
    fast: bool,
) -> tuple[np.ndarray, dict, float]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.batch_trajectory_retarget import (
        BatchTrajectoryConfig,
        BatchTrajectoryRetargeter,
    )
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )

    body_model_dir = REPO_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt_path, body_model_dir
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    frames = frames[:max_frames]

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        motion_fps=fps,
    )
    gn_alphas = (1.0,) if fast else (1.0, 0.5, 0.25, 0.125)
    batch_cfg = BatchTrajectoryConfig(
        window_size=window_size,
        window_stride=window_stride,
        gn_steps=gn_steps,
        gn_line_search_alphas=gn_alphas,
        verbose=False,
        show_progress=False,
        profile=True,
    )
    batch = BatchTrajectoryRetargeter(GMR(**kwargs), batch_cfg)
    batch.set_motion_fps(fps)

    t0 = time.perf_counter()
    q_py = batch.retarget_batch(frames)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    profile = dict(batch.last_profile)
    profile["wall_ms"] = wall_ms
    profile["fps"] = fps
    profile["actual_human_height"] = height
    return q_py, profile, fps


def export_human_json(
    pt_path: pathlib.Path,
    out_json: pathlib.Path,
    max_frames: int,
) -> float:
    from scripts.tools.export_gvhmr_frames_json import frame_to_json
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )

    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt_path, REPO_ROOT / "assets" / "body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    frames = frames[:max_frames]
    payload = {
        "fps": float(fps),
        "actual_human_height": float(height),
        "frames": [frame_to_json(f) for f in frames],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return height


def run_cpp_batch(
    human_json: pathlib.Path,
    robot: str,
    out_json: pathlib.Path,
    max_frames: int,
    actual_human_height: float,
    contact_ground: bool,
    window_size: int,
    window_stride: int,
    gn_steps: int,
    fast: bool,
    cpp_cli: pathlib.Path,
) -> tuple[np.ndarray, dict]:
    cmd = [
        str(cpp_cli),
        "--gmr_root",
        str(REPO_ROOT),
        "--robot",
        robot,
        "--human_frame_json",
        str(human_json),
        "--out_json",
        str(out_json),
        "--actual_human_height",
        str(actual_human_height),
        "--max_frames",
        str(max_frames),
        "--window_size",
        str(window_size),
        "--window_stride",
        str(window_stride),
        "--gn_steps",
        str(gn_steps),
        "--benchmark",
    ]
    if contact_ground:
        cmd.append("--contact_ground")
    if fast:
        cmd.append("--fast")

    env = os.environ.copy()
    devel_lib = "/opt/robot/devel/lib"
    if pathlib.Path(devel_lib).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
    payload = json.loads(out_json.read_text())
    q_cpp = np.asarray(payload["qpos_frames"], dtype=float)
    profile = payload.get("profile", {})
    profile["config"] = payload.get("config", {})
    return q_cpp, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--gn_steps", type=int, default=3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument("--cpp_cli", type=str, default=str(DEFAULT_CPP_CLI))
    parser.add_argument("--output_json", default="output/py_vs_cpp_batch.json")
    parser.add_argument("--keep_human_json", type=str, default="")
    args = parser.parse_args()

    pt_path = pathlib.Path(args.pt_file).expanduser()
    cpp_cli = pathlib.Path(args.cpp_cli).expanduser()
    if not cpp_cli.is_file():
        raise FileNotFoundError(f"C++ CLI not found: {cpp_cli} (build with cmake --build cpp/build)")

    with tempfile.TemporaryDirectory(prefix="batch_to_cmp_") as tmp:
        human_json = (
            pathlib.Path(args.keep_human_json).expanduser()
            if args.keep_human_json
            else pathlib.Path(tmp) / "human_frames.json"
        )
        cpp_out = pathlib.Path(tmp) / "cpp_batch.json"

        height = export_human_json(pt_path, human_json, args.max_frames)

        print(f"[py]  running batch TO on {pt_path.name} ({args.max_frames} frames)...")
        q_py, py_prof, fps = run_python_batch(
            pt_path,
            args.robot,
            args.max_frames,
            args.contact_ground,
            args.window_size,
            args.window_stride,
            args.gn_steps,
            args.fast,
        )

        print(f"[cpp] running gmr_batch_to_cli...")
        q_cpp, cpp_prof = run_cpp_batch(
            human_json,
            args.robot,
            cpp_out,
            args.max_frames,
            height,
            args.contact_ground,
            args.window_size,
            args.window_stride,
            args.gn_steps,
            args.fast,
            cpp_cli,
        )

    n = min(len(q_py), len(q_cpp))
    rmse = qpos_rmse(q_py[:n], q_cpp[:n])
    max_abs = float(np.max(np.abs(q_py[:n] - q_cpp[:n])))

    speedup = py_prof.get("total_ms", 0.0) / max(cpp_prof.get("total_ms", 1e-9), 1e-9)

    result = {
        "pt_file": str(pt_path),
        "robot": args.robot,
        "n_frames": n,
        "fps": fps,
        "contact_ground": args.contact_ground,
        "config": {
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "gn_steps": args.gn_steps,
            "fast": args.fast,
        },
        "python_profile": py_prof,
        "cpp_profile": cpp_prof,
        "qpos_rmse_py_vs_cpp": rmse,
        "qpos_max_abs_py_vs_cpp": max_abs,
        "cpp_speedup_vs_py": speedup,
    }

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(result), indent=2))
    print(f"\nWrote {out}")
    print(
        f"RMSE={rmse:.5f} max_abs={max_abs:.5f} | "
        f"py={py_prof.get('ms_per_frame', 0):.2f} ms/f "
        f"cpp={cpp_prof.get('ms_per_frame', 0):.2f} ms/f "
        f"speedup={speedup:.1f}x"
    )


if __name__ == "__main__":
    main()
