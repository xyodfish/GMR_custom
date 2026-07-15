#!/usr/bin/env python3
"""Benchmark per-frame IK vs offline batch TO on GVHMR .pt files."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def hinge_acc_mean(qpos: np.ndarray, fps: float) -> float:
    if len(qpos) < 3:
        return 0.0
    dt = 1.0 / fps
    acc = (qpos[2:] - 2.0 * qpos[1:-1] + qpos[:-2]) / (dt * dt)
    start = 7 if qpos.shape[1] > 7 else 0
    return float(np.mean(np.linalg.norm(acc[:, start:], axis=1)))


def run_one_pt(
    pt_path: pathlib.Path,
    robot: str,
    contact_ground,
    foot_ground_limit,
    fix_robot_penetration,
    max_frames: int | None,
    w_velocity: float,
    w_acceleration: float,
    max_opt_iter: int,
) -> dict:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.batch_trajectory_retarget import (
        BatchTrajectoryConfig,
        BatchTrajectoryRetargeter,
    )
    from general_motion_retargeting.utils.smpl import (
        load_gvhmr_pred_file,
        get_gvhmr_data_offline_fast,
    )

    body_model_dir = REPO_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt_path, body_model_dir
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    if max_frames is not None:
        frames = frames[:max_frames]

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )

    ik = GMR(**kwargs)
    t0 = time.perf_counter()
    ik_q = np.asarray([ik.retarget(f).copy() for f in frames])
    ik_ms = (time.perf_counter() - t0) * 1000.0

    batch = BatchTrajectoryRetargeter(
        GMR(**kwargs),
        BatchTrajectoryConfig(
            w_velocity=w_velocity,
            w_acceleration=w_acceleration,
            max_opt_iter=max_opt_iter,
            verbose=True,
        ),
    )
    batch.set_motion_fps(fps)
    t1 = time.perf_counter()
    batch_q = batch.retarget_batch(frames)
    batch_ms = (time.perf_counter() - t1) * 1000.0

    ik_acc = hinge_acc_mean(ik_q, fps)
    batch_acc = hinge_acc_mean(batch_q, fps)
    qpos_rmse = float(np.sqrt(np.mean((ik_q - batch_q) ** 2)))

    return {
        "pt": str(pt_path),
        "n_frames": len(frames),
        "fps": fps,
        "ik_ms_total": ik_ms,
        "ik_ms_per_frame": ik_ms / max(len(frames), 1),
        "batch_ms_total": batch_ms,
        "batch_ms_per_frame": batch_ms / max(len(frames), 1),
        "ik_hinge_acc_mean": ik_acc,
        "batch_hinge_acc_mean": batch_acc,
        "acc_improve_pct": 100.0 * (batch_acc - ik_acc) / max(ik_acc, 1e-9),
        "qpos_rmse_vs_ik": qpos_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_files",
        nargs="+",
        required=True,
        help="GVHMR hmr4d_results.pt paths",
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    parser.add_argument("--max_opt_iter", type=int, default=150)
    parser.add_argument("--output_json", type=str, default="output/gvhmr_batch_to_benchmark.json")
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument("--foot_ground_limit", action="store_true", default=True)
    parser.add_argument("--no-foot_ground_limit", dest="foot_ground_limit", action="store_false")
    parser.add_argument("--fix_robot_penetration", action="store_true", default=True)
    parser.add_argument("--no-fix_robot_penetration", dest="fix_robot_penetration", action="store_false")
    args = parser.parse_args()

    results = []
    for pt in args.pt_files:
        print(f"\n=== {pt} ===")
        results.append(
            run_one_pt(
                pathlib.Path(pt).expanduser(),
                args.robot,
                args.contact_ground,
                args.foot_ground_limit,
                args.fix_robot_penetration,
                args.max_frames,
                args.w_velocity,
                args.w_acceleration,
                args.max_opt_iter,
            )
        )

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nWrote {out}")

    for r in results:
        print(
            f"{pathlib.Path(r['pt']).name}: "
            f"frames={r['n_frames']} "
            f"acc {r['ik_hinge_acc_mean']:.3f}->{r['batch_hinge_acc_mean']:.3f} "
            f"({r['acc_improve_pct']:+.1f}%) "
            f"rmse={r['qpos_rmse_vs_ik']:.4f} "
            f"batch={r['batch_ms_per_frame']:.1f}ms/f"
        )


if __name__ == "__main__":
    main()
