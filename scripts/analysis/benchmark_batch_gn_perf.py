#!/usr/bin/env python3
"""Profile batch GN TO throughput: phase breakdown + fast-mode ceiling sweeps."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRESETS = {
    "quality": {
        "window_size": 16,
        "window_stride": 8,
        "gn_steps": 3,
        "gn_line_search_alphas": (1.0, 0.5, 0.25, 0.125),
        "enable_foot_penalties": True,
    },
    "fast": {
        "window_size": 16,
        "window_stride": 8,
        "gn_steps": 2,
        "gn_line_search_alphas": (1.0,),
        "enable_foot_penalties": True,
    },
    "ceiling": {
        "window_size": 16,
        "window_stride": 16,
        "gn_steps": 1,
        "gn_line_search_alphas": (1.0,),
        "enable_foot_penalties": False,
        "finalize_contact": False,
    },
}


def run_preset(
    pt_path: pathlib.Path,
    robot: str,
    max_frames: int,
    preset_name: str,
    preset: dict,
) -> dict:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.batch_trajectory_retarget import (
        BatchTrajectoryConfig,
        BatchTrajectoryRetargeter,
    )
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

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        motion_fps=fps,
    )

    ik = GMR(**kwargs)
    t0 = time.perf_counter()
    ik_q = [ik.retarget(f).copy() for f in frames]
    ik_ms = (time.perf_counter() - t0) * 1000.0

    cfg = BatchTrajectoryConfig(
        verbose=False,
        show_progress=False,
        profile=True,
        **preset,
    )
    batch = BatchTrajectoryRetargeter(GMR(**kwargs), cfg)
    batch.set_motion_fps(fps)
    t1 = time.perf_counter()
    batch_q = batch.retarget_batch(frames)
    batch_ms = (time.perf_counter() - t1) * 1000.0

    import numpy as np

    rmse = float(np.sqrt(np.mean((np.asarray(ik_q) - batch_q) ** 2)))
    prof = dict(batch.last_profile)
    n = len(frames)
    return {
        "preset": preset_name,
        "pt": str(pt_path),
        "n_frames": n,
        "ik_ms_per_frame": ik_ms / max(n, 1),
        "ik_effective_fps": 1000.0 * n / max(ik_ms, 1e-9),
        "batch_ms_per_frame": batch_ms / max(n, 1),
        "batch_effective_fps": 1000.0 * n / max(batch_ms, 1e-9),
        "speedup_vs_ik": ik_ms / max(batch_ms, 1e-9),
        "qpos_rmse_vs_ik": rmse,
        "phases_ms": prof,
        "config": preset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["quality", "fast", "ceiling"],
        choices=list(PRESETS),
    )
    parser.add_argument(
        "--output_json",
        default=str(REPO_ROOT / "output" / "batch_gn_perf_profile.json"),
    )
    args = parser.parse_args()

    pt = pathlib.Path(args.pt_file).expanduser()
    results = []
    for name in args.presets:
        print(f"\n=== preset={name} ===")
        row = run_preset(pt, args.robot, args.max_frames, name, PRESETS[name])
        results.append(row)
        p = row["phases_ms"]
        print(
            f"IK {row['ik_ms_per_frame']:.2f} ms/f ({row['ik_effective_fps']:.0f} FPS)"
        )
        print(
            f"batch {row['batch_ms_per_frame']:.2f} ms/f ({row['batch_effective_fps']:.0f} FPS) "
            f"rmse={row['qpos_rmse_vs_ik']:.4f}"
        )
        if p:
            print(
                f"  phases: prepare={p.get('prepare_ms', 0):.0f}ms "
                f"bootstrap={p.get('bootstrap_ms', 0):.0f}ms "
                f"optimize={p.get('optimize_ms', 0):.0f}ms "
                f"finalize={p.get('finalize_ms', 0):.0f}ms"
            )

    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
