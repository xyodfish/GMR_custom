#!/usr/bin/env python3
"""Compare retarget quality: IK vs Python batch vs optimized C++ batch."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import mujoco as mj
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.batch_trajectory_retarget import (
    BatchTrajectoryConfig,
    BatchTrajectoryRetargeter,
)
from general_motion_retargeting.human_frame_loaders import frame_to_json_dict
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)
from scripts.analysis.analyze_saved_motion_metrics import (
    DEFAULT_FOOT_BODIES,
    ROBOT_XML_PATHS,
    foot_slip_metrics,
    scalar_q_indices,
    smoothness_metrics,
)


def pct(before: float, after: float) -> float:
    return 100.0 * (after - before) / max(abs(before), 1e-9)


def run_cpp_batch(
    frames,
    fps: float,
    height: float,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    gn_line_search: str,
    q_init_json: str | None = None,
    fast: bool = False,
) -> np.ndarray:
    payload = {
        "fps": float(fps),
        "actual_human_height": float(height),
        "src_human": "smplx",
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    with tempfile.TemporaryDirectory(prefix="batch_qual_") as tmp:
        human_json = pathlib.Path(tmp) / "human.json"
        out_json = pathlib.Path(tmp) / "cpp.json"
        human_json.write_text(json.dumps(payload))
        cmd = [
            str(REPO_ROOT / "cpp/build/gmr_batch_to_cli"),
            "--gmr_root",
            str(REPO_ROOT),
            "--robot",
            robot,
            "--human_frame_json",
            str(human_json),
            "--out_json",
            str(out_json),
            "--max_frames",
            str(max_frames),
        ]
        if contact_ground:
            cmd.append("--contact_ground")
        if fast:
            cmd.append("--fast")
        if gn_line_search == "best":
            cmd.extend(["--gn_line_search", "best"])
        elif gn_line_search == "armijo":
            cmd.extend(["--gn_line_search", "armijo"])
        elif gn_line_search == "dense":
            cmd.append("--no_banded_solver")
        elif gn_line_search == "banded":
            cmd.append("--banded_solver")
        if q_init_json:
            cmd.extend(["--q_init_json", q_init_json])
        env = os.environ.copy()
        devel_lib = "/opt/robot/devel/lib"
        if pathlib.Path(devel_lib).is_dir():
            env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
        return np.asarray(json.loads(out_json.read_text())["qpos_frames"], dtype=float)


def summarize(name: str, q: np.ndarray, fps: float, robot: str, batch: BatchTrajectoryRetargeter, targets, ik_q: np.ndarray, py_q: np.ndarray):
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qidx = scalar_q_indices(model)
    foot = DEFAULT_FOOT_BODIES[robot]
    fk = float(np.mean([batch._fk_tracking_cost(qi, t) for qi, t in zip(q, targets)]))
    sm = smoothness_metrics(q, fps, qidx)
    slip = foot_slip_metrics(model, q, foot, 0.02)
    return {
        "name": name,
        "fk_mean": fk,
        "jerk_mean": sm["jerk"]["mean"],
        "foot_slip_total": slip["total_slip"],
        "rmse_vs_ik": float(np.sqrt(np.mean((ik_q - q) ** 2))),
        "rmse_vs_py": float(np.sqrt(np.mean((py_q - q) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--contact_ground", action="store_true")
    parser.add_argument("--output_json", default="output/batch_opt_quality.json")
    parser.add_argument("--py_cpp_rmse_max", type=float, default=1e-5)
    parser.add_argument("--banded_dense_rmse_max", type=float, default=1e-6)
    parser.add_argument("--fail_on_threshold", action="store_true")
    args = parser.parse_args()

    pt_path = pathlib.Path(args.pt_file).expanduser()
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt_path, REPO_ROOT / "assets/body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    frames = frames[: args.max_frames]

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=False,
        contact_ground=args.contact_ground,
        motion_fps=fps,
    )

    ik = GMR(**kwargs)
    ik_q = np.stack([ik.retarget(f).copy() for f in frames])

    py_batch = BatchTrajectoryRetargeter(
        GMR(**kwargs),
        BatchTrajectoryConfig(window_size=16, window_stride=8, gn_steps=3, verbose=False, show_progress=False),
    )
    py_batch.set_motion_fps(fps)
    py_q = py_batch.retarget_batch(frames)

    cpp_armijo = run_cpp_batch(frames, fps, height, args.robot, args.max_frames, args.contact_ground, "armijo")
    cpp_best = run_cpp_batch(frames, fps, height, args.robot, args.max_frames, args.contact_ground, "best")
    cpp_dense = run_cpp_batch(frames, fps, height, args.robot, args.max_frames, args.contact_ground, "dense")
    cpp_banded = run_cpp_batch(frames, fps, height, args.robot, args.max_frames, args.contact_ground, "banded")
    cpp_fast = run_cpp_batch(
        frames, fps, height, args.robot, args.max_frames, args.contact_ground, "best", fast=True
    )

    qinit_path = REPO_ROOT / "output" / "_py_ik_bootstrap.json"
    qinit_path.write_text(
        json.dumps({"qpos_frames": [row.tolist() for row in ik_q]})
    )
    cpp_pyinit = run_cpp_batch(
        frames, fps, height, args.robot, args.max_frames, args.contact_ground, "best", str(qinit_path)
    )

    prepared = [py_batch.gmr._prepare_scaled_human_data(f) for f in frames]
    targets = [py_batch._targets_for_prepared(p) for p in prepared]

    rows = [
        summarize("IK", ik_q, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Py_batch_quality", py_q, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_armijo", cpp_armijo, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_best_LS", cpp_best, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_dense_solver", cpp_dense, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_banded_solver", cpp_banded, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_fast", cpp_fast, fps, args.robot, py_batch, targets, ik_q, py_q),
        summarize("Cpp_py_ik_init", cpp_pyinit, fps, args.robot, py_batch, targets, ik_q, py_q),
    ]

    banded_vs_dense_rmse = float(np.sqrt(np.mean((cpp_banded - cpp_dense) ** 2)))
    fast_vs_dense_rmse = float(np.sqrt(np.mean((cpp_fast - cpp_dense) ** 2)))
    ik_row = rows[0]
    result = {
        "pt_file": str(pt_path),
        "contact_ground": args.contact_ground,
        "methods": rows,
        "cpp_armijo_vs_best_rmse": float(np.sqrt(np.mean((cpp_armijo - cpp_best) ** 2))),
        "cpp_armijo_vs_dense_rmse": float(np.sqrt(np.mean((cpp_armijo - cpp_dense) ** 2))),
        "cpp_banded_vs_dense_rmse": banded_vs_dense_rmse,
        "cpp_fast_vs_dense_rmse": fast_vs_dense_rmse,
        "py_vs_cpp_armijo_rmse": rows[2]["rmse_vs_py"],
        "py_vs_cpp_dense_rmse": rows[4]["rmse_vs_py"],
        "py_vs_cpp_banded_rmse": rows[5]["rmse_vs_py"],
        "py_vs_cpp_fast_rmse": rows[6]["rmse_vs_py"],
        "py_vs_cpp_pyinit_rmse": rows[7]["rmse_vs_py"],
    }

    print(f"\n=== Quality ({pt_path.name}, contact_ground={args.contact_ground}) ===")
    print(f"{'method':18s} {'FK':>7s} {'jerk':>9s} {'foot_slip':>10s} {'rmse_ik':>9s} {'rmse_py':>9s}")
    for r in rows:
        print(
            f"{r['name']:18s} {r['fk_mean']:7.3f} {r['jerk_mean']:9.1f} "
            f"{r['foot_slip_total']:10.3f} {r['rmse_vs_ik']:9.4f} {r['rmse_vs_py']:9.4f}"
        )

    for r in rows[1:]:
        print(
            f"\n{r['name']} vs IK: FK {pct(ik_row['fk_mean'], r['fk_mean']):+.1f}%  "
            f"jerk {pct(ik_row['jerk_mean'], r['jerk_mean']):+.1f}%  "
            f"foot_slip {pct(ik_row['foot_slip_total'], r['foot_slip_total']):+.1f}%"
        )

    print(f"\n优化一致性: armijo vs best RMSE={result['cpp_armijo_vs_best_rmse']:.6f}")
    print(f"带状求解器: banded vs dense RMSE={result['cpp_banded_vs_dense_rmse']:.6f}")
    print(f"fast 档: fast vs dense RMSE={result['cpp_fast_vs_dense_rmse']:.6f}")
    print(f"Py vs C++ armijo RMSE={result['py_vs_cpp_armijo_rmse']:.5f}")
    print(f"Py vs C++ dense  RMSE={result['py_vs_cpp_dense_rmse']:.5f}")
    print(f"Py vs C++ banded RMSE={result['py_vs_cpp_banded_rmse']:.5f}")
    print(f"Py vs C++ fast   RMSE={result['py_vs_cpp_fast_rmse']:.5f}")
    print(f"Py vs C++ py_ik_init RMSE={result['py_vs_cpp_pyinit_rmse']:.5f}")

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

    failed = []
    if result["py_vs_cpp_dense_rmse"] > args.py_cpp_rmse_max:
        failed.append(
            f"Py vs C++ dense RMSE {result['py_vs_cpp_dense_rmse']:.2e} > {args.py_cpp_rmse_max:.2e}"
        )
    if banded_vs_dense_rmse > args.banded_dense_rmse_max:
        failed.append(
            f"banded vs dense RMSE {banded_vs_dense_rmse:.2e} > {args.banded_dense_rmse_max:.2e}"
        )
    if failed:
        print("\nPARITY CHECK FAILED:")
        for msg in failed:
            print(f"  - {msg}")
        if args.fail_on_threshold:
            sys.exit(1)
    else:
        print("\nPARITY CHECK PASSED")


if __name__ == "__main__":
    main()
