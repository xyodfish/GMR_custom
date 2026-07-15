#!/usr/bin/env python3
"""Compare traditional GMR IK vs batch GN TO on GVHMR .pt (smoothness + FK + foot slip)."""

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
    load_gvhmr_pred_file,
    get_gvhmr_data_offline_fast,
)
from scripts.analysis.analyze_saved_motion_metrics import (
    DEFAULT_FOOT_BODIES,
    ROBOT_XML_PATHS,
    foot_slip_metrics,
    scalar_q_indices,
    smoothness_metrics,
)


def pct_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / max(abs(before), 1e-9)


def run_one(
    pt_path: pathlib.Path,
    robot: str,
    max_frames: int | None,
    window_size: int,
    window_stride: int,
    gn_steps: int,
) -> dict:
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
        motion_fps=fps,
    )

    t0 = time.perf_counter()
    ik = GMR(**kwargs)
    ik_q = np.stack([ik.retarget(f).copy() for f in frames])
    ik_ms = (time.perf_counter() - t0) * 1000.0

    batch = BatchTrajectoryRetargeter(
        GMR(**kwargs),
        BatchTrajectoryConfig(
            window_size=window_size,
            window_stride=window_stride,
            gn_steps=gn_steps,
            verbose=False,
            show_progress=False,
        ),
    )
    batch.set_motion_fps(fps)
    t1 = time.perf_counter()
    batch_q = batch.retarget_batch(frames)
    batch_ms = (time.perf_counter() - t1) * 1000.0

    prepared = [batch.gmr._prepare_scaled_human_data(f) for f in frames]
    targets = [batch._targets_for_prepared(p) for p in prepared]
    ik_fk = np.asarray([batch._fk_tracking_cost(q, t) for q, t in zip(ik_q, targets)])
    batch_fk = np.asarray([batch._fk_tracking_cost(q, t) for q, t in zip(batch_q, targets)])

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qidx = scalar_q_indices(model)
    foot_names = DEFAULT_FOOT_BODIES[robot]
    ik_sm = smoothness_metrics(ik_q, fps, qidx)
    batch_sm = smoothness_metrics(batch_q, fps, qidx)
    ik_slip = foot_slip_metrics(model, ik_q, foot_names, 0.02)
    batch_slip = foot_slip_metrics(model, batch_q, foot_names, 0.02)

    return {
        "pt": str(pt_path),
        "name": pt_path.parent.name if pt_path.parent.name != "gvhmr_pt" else pt_path.stem,
        "n_frames": len(frames),
        "fps": fps,
        "timing_ms": {
            "ik_total": ik_ms,
            "batch_gn_total": batch_ms,
            "ik_per_frame": ik_ms / max(len(frames), 1),
            "batch_gn_per_frame": batch_ms / max(len(frames), 1),
        },
        "qpos_rmse_vs_ik": float(np.sqrt(np.mean((ik_q - batch_q) ** 2))),
        "fk_tracking_cost_mean": {
            "ik": float(np.mean(ik_fk)),
            "batch_gn": float(np.mean(batch_fk)),
            "delta_pct": pct_change(float(np.mean(ik_fk)), float(np.mean(batch_fk))),
        },
        "smoothness": {
            "ddq_mean": {
                "ik": ik_sm["ddq"]["mean"],
                "batch_gn": batch_sm["ddq"]["mean"],
                "delta_pct": pct_change(ik_sm["ddq"]["mean"], batch_sm["ddq"]["mean"]),
            },
            "jerk_mean": {
                "ik": ik_sm["jerk"]["mean"],
                "batch_gn": batch_sm["jerk"]["mean"],
                "delta_pct": pct_change(ik_sm["jerk"]["mean"], batch_sm["jerk"]["mean"]),
            },
        },
        "foot_slip_total": {
            "ik": ik_slip["total_slip"],
            "batch_gn": batch_slip["total_slip"],
            "delta_pct": pct_change(ik_slip["total_slip"], batch_slip["total_slip"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_files", nargs="+", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--gn_steps", type=int, default=3)
    parser.add_argument(
        "--output_json",
        default=str(REPO_ROOT / "output" / "gvhmr_ik_vs_batch_gn_metrics.json"),
    )
    args = parser.parse_args()

    results = []
    for pt in args.pt_files:
        print(f"\n=== {pt} ===")
        row = run_one(
            pathlib.Path(pt).expanduser(),
            args.robot,
            args.max_frames,
            args.window_size,
            args.window_stride,
            args.gn_steps,
        )
        results.append(row)
        print(
            f"FK {row['fk_tracking_cost_mean']['ik']:.2f} -> "
            f"{row['fk_tracking_cost_mean']['batch_gn']:.2f} "
            f"({row['fk_tracking_cost_mean']['delta_pct']:+.1f}%)"
        )
        print(
            f"jerk {row['smoothness']['jerk_mean']['ik']:.1f} -> "
            f"{row['smoothness']['jerk_mean']['batch_gn']:.1f} "
            f"({row['smoothness']['jerk_mean']['delta_pct']:+.1f}%)"
        )
        print(
            f"foot_slip {row['foot_slip_total']['ik']:.3f} -> "
            f"{row['foot_slip_total']['batch_gn']:.3f} "
            f"({row['foot_slip_total']['delta_pct']:+.1f}%)"
        )

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot": args.robot,
        "max_frames": args.max_frames,
        "solver": "batch_gn",
        "baseline": "gmr_per_frame_ik",
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
