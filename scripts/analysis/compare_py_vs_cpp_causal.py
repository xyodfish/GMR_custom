#!/usr/bin/env python3
"""Python (L-BFGS causal TO) vs C++ (light IK causal TO) parity benchmark."""

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

DEFAULT_CPP_CLI = REPO_ROOT / "cpp" / "build" / "gmr_causal_to_cli"


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


def per_dim_rmse(a: np.ndarray, b: np.ndarray) -> list[float]:
    n = min(len(a), len(b))
    if n == 0:
        return []
    return [float(np.sqrt(np.mean((a[:n, i] - b[:n, i]) ** 2))) for i in range(a.shape[1])]


def joint_velocity_metric(qpos_seq: np.ndarray) -> float:
    if len(qpos_seq) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(qpos_seq, axis=0), axis=1)))


def joint_acceleration_metric(qpos_seq: np.ndarray) -> float:
    if len(qpos_seq) < 3:
        return 0.0
    acc = qpos_seq[2:] - 2.0 * qpos_seq[1:-1] + qpos_seq[:-2]
    return float(np.mean(np.linalg.norm(acc, axis=1)))


def root_z_stats(qpos_seq: np.ndarray) -> dict:
    if qpos_seq.size == 0 or qpos_seq.shape[1] < 3:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "below_0.05": 0}
    z = qpos_seq[:, 2]
    return {
        "mean": float(np.mean(z)),
        "min": float(np.min(z)),
        "max": float(np.max(z)),
        "below_0.05": int(np.sum(z < 0.05)),
    }


def summarize_vs_reference(q: np.ndarray, ref: np.ndarray, label: str) -> dict:
    n = min(len(q), len(ref))
    diff = q[:n] - ref[:n]
    return {
        "label": label,
        "rmse": qpos_rmse(q[:n], ref[:n]),
        "max_abs": float(np.max(np.abs(diff))),
        "root_z_rmse": float(np.sqrt(np.mean((q[:n, 2] - ref[:n, 2]) ** 2))) if n > 0 else 0.0,
        "per_dim_rmse": per_dim_rmse(q, ref),
    }


def summarize_temporal(q: np.ndarray, fps: float) -> dict:
    return {
        "mean_dq_norm": joint_velocity_metric(q),
        "mean_ddq_norm": joint_acceleration_metric(q),
        "mean_dq_norm_per_sec": joint_velocity_metric(q) * fps,
        "mean_ddq_norm_per_sec2": joint_acceleration_metric(q) * (fps ** 2),
        "root_z": root_z_stats(q),
    }


def run_python_causal(
    pt_path: pathlib.Path,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    solver: str,
    gn_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict, float]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.trajectory_optimization_retarget import (
        TrajectoryOptimizationConfig,
        TrajectoryOptimizationRetargeter,
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

    gmr_kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        motion_fps=fps,
    )
    gmr = GMR(**gmr_kwargs)
    ik_gmr = GMR(**gmr_kwargs)

    to_cfg = TrajectoryOptimizationConfig(
        mode="fast",
        solver=solver,
        gn_steps=gn_steps,
        use_gmr_init=True,
    )
    to = TrajectoryOptimizationRetargeter(gmr, to_cfg)
    to.set_motion_fps(fps)

    q_to: list[np.ndarray] = []
    q_ik: list[np.ndarray] = []
    frame_ms: list[float] = []

    t0 = time.perf_counter()
    for frame in frames:
        t_frame = time.perf_counter()
        q_to.append(to.retarget(frame))
        frame_ms.append((time.perf_counter() - t_frame) * 1000.0)
        q_ik.append(ik_gmr.retarget(frame))
    wall_ms = (time.perf_counter() - t0) * 1000.0

    profile = {
        "solver": solver,
        "gn_steps": gn_steps,
        "mode": "fast",
        "use_gmr_init": True,
        "wall_ms": wall_ms,
        "total_ms": float(np.sum(frame_ms)),
        "ms_per_frame": float(np.mean(frame_ms)) if frame_ms else 0.0,
        "max_frame_ms": float(np.max(frame_ms)) if frame_ms else 0.0,
        "effective_fps": 1000.0 / float(np.mean(frame_ms)) if frame_ms else 0.0,
        "fps": fps,
        "actual_human_height": height,
    }
    return np.asarray(q_to), np.asarray(q_ik), profile, fps


def export_human_json(pt_path: pathlib.Path, out_json: pathlib.Path, max_frames: int) -> float:
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


def run_cpp_causal(
    human_json: pathlib.Path,
    robot: str,
    out_json: pathlib.Path,
    max_frames: int,
    actual_human_height: float,
    contact_ground: bool,
    solver: str,
    gn_steps: int,
    light_ik_iters: int,
    fast_opt_iter: int,
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
        "--solver",
        solver,
        "--gn_steps",
        str(gn_steps),
        "--light_ik_iters",
        str(light_ik_iters),
        "--fast_opt_iter",
        str(fast_opt_iter),
        "--benchmark",
    ]
    if contact_ground:
        cmd.append("--contact_ground")

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


def print_summary(result: dict) -> None:
    py_cpp = result["py_vs_cpp"]
    py_ik = result["python_vs_ik"]
    cpp_ik = result["cpp_vs_ik"]
    print("\n=== Causal TO: Python (L-BFGS) vs C++ (L-BFGS) ===")
    print(f"Frames: {result['n_frames']} @ {result['fps']:.1f} fps")
    print(
        f"Py vs C++  RMSE={py_cpp['rmse']:.5f}  max_abs={py_cpp['max_abs']:.5f}  "
        f"root_z_rmse={py_cpp['root_z_rmse']:.5f}"
    )
    print(f"Py vs IK   RMSE={py_ik['rmse']:.5f}  max_abs={py_ik['max_abs']:.5f}")
    print(f"C++ vs IK  RMSE={cpp_ik['rmse']:.5f}  max_abs={cpp_ik['max_abs']:.5f}")
    print("\nTemporal smoothness (lower = smoother):")
    for key in ("python", "cpp", "ik"):
        t = result["temporal"][key]
        print(
            f"  {key:6s}  dq={t['mean_dq_norm']:.5f}  ddq={t['mean_ddq_norm']:.5f}  "
            f"root_z=[{t['root_z']['min']:.3f}, {t['root_z']['max']:.3f}]"
        )
    print(
        f"\nTiming: py={result['python_profile']['ms_per_frame']:.2f} ms/f  "
        f"cpp={result['cpp_profile']['ms_per_frame']:.2f} ms/f  "
        f"speedup={result['cpp_speedup_vs_py']:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt_file", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument(
        "--py_solver",
        choices=["lbfgs", "gn"],
        default="lbfgs",
        help="Python causal TO solver (default lbfgs).",
    )
    parser.add_argument("--py_gn_steps", type=int, default=3)
    parser.add_argument("--cpp_solver", choices=["lbfgs", "gn", "none"], default="lbfgs")
    parser.add_argument("--cpp_gn_steps", type=int, default=3)
    parser.add_argument("--cpp_light_ik_iters", type=int, default=5)
    parser.add_argument("--cpp_fast_opt_iter", type=int, default=5)
    parser.add_argument("--cpp_cli", type=str, default=str(DEFAULT_CPP_CLI))
    parser.add_argument("--output_json", default="output/py_vs_cpp_causal.json")
    parser.add_argument("--keep_human_json", type=str, default="")
    args = parser.parse_args()

    pt_path = pathlib.Path(args.pt_file).expanduser()
    cpp_cli = pathlib.Path(args.cpp_cli).expanduser()
    if not cpp_cli.is_file():
        raise FileNotFoundError(
            f"C++ CLI not found: {cpp_cli} (cmake --build cpp/build --target gmr_causal_to_cli)"
        )

    with tempfile.TemporaryDirectory(prefix="causal_to_cmp_") as tmp:
        human_json = (
            pathlib.Path(args.keep_human_json).expanduser()
            if args.keep_human_json
            else pathlib.Path(tmp) / "human_frames.json"
        )
        cpp_out = pathlib.Path(tmp) / "cpp_causal.json"

        height = export_human_json(pt_path, human_json, args.max_frames)

        print(f"[py]  causal TO ({args.py_solver}) on {pt_path.name} ({args.max_frames} frames)...")
        q_py, q_ik, py_prof, fps = run_python_causal(
            pt_path,
            args.robot,
            args.max_frames,
            args.contact_ground,
            args.py_solver,
            args.py_gn_steps,
        )

        print("[cpp] gmr_causal_to_cli...")
        q_cpp, cpp_prof = run_cpp_causal(
            human_json,
            args.robot,
            cpp_out,
            args.max_frames,
            height,
            args.contact_ground,
            args.cpp_solver,
            args.cpp_gn_steps,
            args.cpp_light_ik_iters,
            args.cpp_fast_opt_iter,
            cpp_cli,
        )

    n = min(len(q_py), len(q_cpp), len(q_ik))
    q_py = q_py[:n]
    q_cpp = q_cpp[:n]
    q_ik = q_ik[:n]

    py_cpp_rmse = qpos_rmse(q_py, q_cpp)
    cpp_ik_rmse = qpos_rmse(q_cpp, q_ik)
    py_ik_rmse = qpos_rmse(q_py, q_ik)

    result = {
        "pt_file": str(pt_path),
        "robot": args.robot,
        "n_frames": n,
        "fps": fps,
        "contact_ground": args.contact_ground,
        "python_config": {
            "solver": args.py_solver,
            "gn_steps": args.py_gn_steps,
            "mode": "fast",
            "use_gmr_init": True,
        },
        "cpp_config": cpp_prof.get("config", {}),
        "python_profile": py_prof,
        "cpp_profile": cpp_prof,
        "py_vs_cpp": summarize_vs_reference(q_py, q_cpp, "python_vs_cpp"),
        "python_vs_ik": summarize_vs_reference(q_py, q_ik, "python_vs_ik"),
        "cpp_vs_ik": summarize_vs_reference(q_cpp, q_ik, "cpp_vs_ik"),
        "temporal": {
            "python": summarize_temporal(q_py, fps),
            "cpp": summarize_temporal(q_cpp, fps),
            "ik": summarize_temporal(q_ik, fps),
        },
        "cpp_speedup_vs_py": py_prof.get("total_ms", 0.0) / max(cpp_prof.get("total_ms", 1e-9), 1e-9),
        "verdict": {
            "agreement_acceptable_py_cpp": py_cpp_rmse < 0.05,
            "cpp_close_to_ik": cpp_ik_rmse < 0.03,
            "python_closer_to_ik_than_cpp": py_ik_rmse < cpp_ik_rmse,
            "cpp_root_on_ground": summarize_temporal(q_cpp, fps)["root_z"]["below_0.05"] == 0,
            "notes": (
                "Python default: L-BFGS temporal refine. C++ default: L-BFGS (solver=lbfgs). "
                "py_vs_cpp = cross-impl; *_vs_ik = tracking fidelity; temporal = smoothness."
            ),
        },
    }

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(result), indent=2))
    print(f"\nWrote {out}")
    print_summary(result)


if __name__ == "__main__":
    main()
