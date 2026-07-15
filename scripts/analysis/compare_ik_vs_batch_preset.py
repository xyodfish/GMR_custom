#!/usr/bin/env python3
"""Compare GMR IK vs batch GN quality/perf presets (quality vs fast)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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

PRESETS = {
    "quality": dict(
        window_size=16,
        window_stride=8,
        gn_steps=3,
        gn_line_search_alphas=(1.0, 0.5, 0.25, 0.125),
    ),
    "fast": dict(
        window_size=16,
        window_stride=8,
        gn_steps=2,
        gn_line_search_alphas=(1.0,),
    ),
}


def pct(before: float, after: float) -> float:
    return 100.0 * (after - before) / max(abs(before), 1e-9)


def run_one(pt_path: pathlib.Path, robot: str, max_frames: int, preset: str) -> dict:
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt_path, REPO_ROOT / "assets" / "body_models"
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
        motion_fps=fps,
    )

    ik = GMR(**kwargs)
    t0 = time.perf_counter()
    ik_q = np.stack([ik.retarget(f).copy() for f in frames])
    ik_ms = (time.perf_counter() - t0) * 1000.0

    cfg = BatchTrajectoryConfig(
        verbose=False,
        show_progress=False,
        **PRESETS[preset],
    )
    batch = BatchTrajectoryRetargeter(GMR(**kwargs), cfg)
    batch.set_motion_fps(fps)
    t1 = time.perf_counter()
    batch_q = batch.retarget_batch(frames)
    batch_ms = (time.perf_counter() - t1) * 1000.0

    prepared = [batch.gmr._prepare_scaled_human_data(f) for f in frames]
    targets = [batch._targets_for_prepared(p) for p in prepared]
    ik_fk = float(np.mean([batch._fk_tracking_cost(q, t) for q, t in zip(ik_q, targets)]))
    batch_fk = float(np.mean([batch._fk_tracking_cost(q, t) for q, t in zip(batch_q, targets)]))

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qidx = scalar_q_indices(model)
    foot = DEFAULT_FOOT_BODIES[robot]
    ik_j = smoothness_metrics(ik_q, fps, qidx)["jerk"]["mean"]
    batch_j = smoothness_metrics(batch_q, fps, qidx)["jerk"]["mean"]
    ik_slip = foot_slip_metrics(model, ik_q, foot, 0.02)["total_slip"]
    batch_slip = foot_slip_metrics(model, batch_q, foot, 0.02)["total_slip"]

    n = len(frames)
    return {
        "preset": preset,
        "pt": str(pt_path),
        "name": pt_path.parent.name if pt_path.parent.name != "gvhmr_pt" else pt_path.stem,
        "n_frames": n,
        "ms_per_frame": {"ik": ik_ms / n, "batch": batch_ms / n},
        "qpos_rmse_vs_ik": float(np.sqrt(np.mean((ik_q - batch_q) ** 2))),
        "fk_mean": {"ik": ik_fk, "batch": batch_fk, "delta_pct": pct(ik_fk, batch_fk)},
        "jerk_mean": {"ik": ik_j, "batch": batch_j, "delta_pct": pct(ik_j, batch_j)},
        "foot_slip": {"ik": ik_slip, "batch": batch_slip, "delta_pct": pct(ik_slip, batch_slip)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_files", nargs="+", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--preset", default="fast", choices=list(PRESETS))
    parser.add_argument(
        "--output_json",
        default=str(REPO_ROOT / "output" / "gvhmr_ik_vs_batch_fast_metrics.json"),
    )
    args = parser.parse_args()

    results = []
    for pt in args.pt_files:
        row = run_one(pathlib.Path(pt).expanduser(), args.robot, args.max_frames, args.preset)
        results.append(row)
        print(f"\n=== {row['name']} preset={args.preset} ===")
        print(
            f"time IK {row['ms_per_frame']['ik']:.2f} ms/f | "
            f"batch {row['ms_per_frame']['batch']:.2f} ms/f | "
            f"rmse={row['qpos_rmse_vs_ik']:.4f}"
        )
        print(
            f"FK {row['fk_mean']['ik']:.2f}->{row['fk_mean']['batch']:.2f} "
            f"({row['fk_mean']['delta_pct']:+.1f}%)"
        )
        print(
            f"jerk {row['jerk_mean']['ik']:.0f}->{row['jerk_mean']['batch']:.0f} "
            f"({row['jerk_mean']['delta_pct']:+.1f}%)"
        )
        print(
            f"foot_slip {row['foot_slip']['ik']:.3f}->{row['foot_slip']['batch']:.3f} "
            f"({row['foot_slip']['delta_pct']:+.1f}%)"
        )

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "robot": args.robot,
                "max_frames": args.max_frames,
                "preset": args.preset,
                "baseline": "gmr_per_frame_ik",
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
