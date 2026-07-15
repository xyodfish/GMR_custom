"""Batch compare per-frame IK vs sliding-window on LAFAN1 BVH files.

Example:
    python scripts/analysis/batch_lafan1_retarget_compare.py \\
        --bvh_dir ~/Workspace/data/lafan1 \\
        --robot unitree_g1 \\
        --contact_ground \\
        --out_dir output/lafan1_ik_vs_sw
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.sliding_window_retarget import (
    SlidingWindowConfig,
    SlidingWindowRetargeter,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scripts.analysis.analyze_saved_motion_metrics import (
    DEFAULT_FOOT_BODIES,
    foot_slip_metrics,
    scalar_q_indices,
    smoothness_metrics,
)


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def list_bvh_files(root: pathlib.Path, pattern: str, max_files: int, selected_files):
    if selected_files:
        files = []
        for name in selected_files:
            path = pathlib.Path(name).expanduser()
            if not path.is_absolute():
                path = root / path
            files.append(path)
    else:
        files = sorted(root.glob(pattern))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No BVH files matched {root}/{pattern}")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BVH files: {missing}")
    return files


def summarize_qpos(model, qpos, fps, foot_body_names, contact_height_margin):
    q_indices = scalar_q_indices(model)
    sm = smoothness_metrics(qpos, fps, q_indices)
    slip = foot_slip_metrics(model, qpos, foot_body_names, contact_height_margin)
    return {
        "dq_mean": sm["dq"]["mean"],
        "dq_p95": sm["dq"]["p95"],
        "ddq_mean": sm["ddq"]["mean"],
        "ddq_p95": sm["ddq"]["p95"],
        "jerk_mean": sm["jerk"]["mean"],
        "jerk_p95": sm["jerk"]["p95"],
        "foot_slip_mean_step": slip["mean_slip_per_contact_step"],
        "foot_slip_total": slip["total_slip"],
        "contact_step_count": slip["contact_step_count"],
    }


def prefixed(prefix: str, row: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in row.items()}


def retarget_ik(frames, gmr_kwargs, src_human, actual_human_height, motion_fps):
    gmr = GMR(
        src_human=src_human,
        actual_human_height=actual_human_height,
        motion_fps=motion_fps,
        verbose=False,
        **gmr_kwargs,
    )
    gmr.set_motion_fps(motion_fps)
    q_list = []
    times_ms = []
    for frame in frames:
        t0 = time.perf_counter()
        q_list.append(gmr.retarget(frame).copy())
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(q_list), gmr.model, times_ms


def retarget_sw(frames, gmr_kwargs, src_human, actual_human_height, motion_fps, sw_cfg):
    gmr = GMR(
        src_human=src_human,
        actual_human_height=actual_human_height,
        motion_fps=motion_fps,
        verbose=False,
        **gmr_kwargs,
    )
    gmr.set_motion_fps(motion_fps)
    sw = SlidingWindowRetargeter(gmr, sw_cfg)
    q_list = []
    times_ms = []
    for frame in frames:
        t0 = time.perf_counter()
        q_list.append(sw.retarget(frame).copy())
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(q_list), times_ms


def delta_pct(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / max(abs(baseline), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compare per-frame IK vs sliding-window on LAFAN1 BVH.",
    )
    parser.add_argument(
        "--bvh_dir",
        type=pathlib.Path,
        default=pathlib.Path("~/Workspace/data/lafan1").expanduser(),
    )
    parser.add_argument("--pattern", type=str, default="*.bvh")
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--max_files", type=int, default=0, help="0 = all files.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all frames.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("output/lafan1_ik_vs_sw"),
    )
    parser.add_argument("--contact_height_margin", type=float, default=0.035)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--solver", choices=["gn", "lbfgs"], default="gn")
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--ik_warmstart_iters", type=int, default=3)
    parser.add_argument("--gn_steps", type=int, default=3)
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")
    args = parser.parse_args()

    bvh_files = list_bvh_files(args.bvh_dir, args.pattern, args.max_files, args.files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    foot_body_names = DEFAULT_FOOT_BODIES.get(args.robot, [])
    src_human = f"bvh_{args.format}"
    motion_fps = float(args.motion_fps)
    dt = 1.0 / motion_fps

    gmr_kwargs = dict(
        tgt_robot=args.robot,
        contact_ground=args.contact_ground,
        foot_ground_limit=args.foot_ground_limit,
        fix_robot_penetration=args.fix_robot_penetration,
    )
    sw_cfg = SlidingWindowConfig(
        window_size=args.window_size,
        solver=args.solver,
        w_velocity=args.w_velocity,
        w_acceleration=args.w_acceleration,
        ik_warmstart_iters=args.ik_warmstart_iters,
        gn_steps=args.gn_steps,
        dt=dt,
    )

    rows = []
    for bvh_path in tqdm(bvh_files, desc="LAFAN1 BVH"):
        t0 = time.time()
        frames, actual_human_height = load_bvh_file(str(bvh_path), format=args.format)
        end = len(frames) if args.max_frames <= 0 else min(len(frames), args.start_frame + args.max_frames)
        frames = frames[args.start_frame:end]
        if len(frames) < 4:
            continue

        q_ik, model, ik_ms = retarget_ik(
            frames, gmr_kwargs, src_human, actual_human_height, motion_fps
        )
        q_sw, sw_ms = retarget_sw(
            frames, gmr_kwargs, src_human, actual_human_height, motion_fps, sw_cfg
        )

        ik_stats = summarize_qpos(
            model, q_ik, motion_fps, foot_body_names, args.contact_height_margin
        )
        sw_stats = summarize_qpos(
            model, q_sw, motion_fps, foot_body_names, args.contact_height_margin
        )

        qpos_rmse = float(np.sqrt(np.mean((q_ik - q_sw) ** 2)))
        qpos_mae = float(np.mean(np.linalg.norm(q_ik - q_sw, axis=1)))

        row = {
            "file": bvh_path.name,
            "frames": len(frames),
            "seconds": len(frames) / motion_fps,
            "ik_mean_ms": float(np.mean(ik_ms)),
            "ik_p95_ms": float(np.percentile(ik_ms, 95)),
            "sw_mean_ms": float(np.mean(sw_ms)),
            "sw_p95_ms": float(np.percentile(sw_ms, 95)),
            "sw_over_ik_speed_ratio": float(np.mean(sw_ms) / max(np.mean(ik_ms), 1e-9)),
            "qpos_rmse": qpos_rmse,
            "qpos_mae": qpos_mae,
            "ddq_improve_pct": delta_pct(sw_stats["ddq_mean"], ik_stats["ddq_mean"]),
            "jerk_improve_pct": delta_pct(sw_stats["jerk_mean"], ik_stats["jerk_mean"]),
            "elapsed_sec": time.time() - t0,
        }
        row.update(prefixed("ik", ik_stats))
        row.update(prefixed("sw", sw_stats))
        rows.append(row)

    csv_path = args.out_dir / "summary.csv"
    json_path = args.out_dir / "summary.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Compared {len(rows)} BVH files")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    if rows:
        print(
            f"IK mean latency: {np.mean([r['ik_mean_ms'] for r in rows]):.2f} ms/frame"
        )
        print(
            f"SW mean latency: {np.mean([r['sw_mean_ms'] for r in rows]):.2f} ms/frame"
        )
        print(
            f"SW ddq mean improve: {np.mean([r['ddq_improve_pct'] for r in rows]):+.1f}%"
        )
        print(
            f"SW jerk mean improve: {np.mean([r['jerk_improve_pct'] for r in rows]):+.1f}%"
        )


if __name__ == "__main__":
    main()
