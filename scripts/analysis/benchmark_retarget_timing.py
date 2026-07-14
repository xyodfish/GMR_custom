"""Benchmark retargeting latency: per-frame IK vs sliding-window.

Example:
    python scripts/analysis/benchmark_retarget_timing.py \\
        --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \\
        --robot unitree_g1 \\
        --contact_ground
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.sliding_window_retarget import (
    SlidingWindowConfig,
    SlidingWindowRetargeter,
)


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {}
    arr = np.asarray(samples_ms, dtype=float)
    return {
        "n": float(len(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "fps_mean": float(1000.0 / np.mean(arr)),
    }


def print_summary(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label:16s}  "
        f"mean={stats['mean_ms']:7.2f} ms  "
        f"median={stats['median_ms']:7.2f} ms  "
        f"p95={stats['p95_ms']:7.2f} ms  "
        f"min={stats['min_ms']:7.2f} ms  "
        f"max={stats['max_ms']:7.2f} ms  "
        f"fps={stats['fps_mean']:6.1f}"
    )


def load_human_frames(args) -> tuple[list, float, float | None]:
    body_model_dir = ROOT / "assets" / "body_models"
    if args.gvhmr_pred_file:
        from general_motion_retargeting.utils.smpl import (
            load_gvhmr_pred_file,
            get_gvhmr_data_offline_fast,
        )

        path = pathlib.Path(args.gvhmr_pred_file)
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            path, body_model_dir
        )
        frames, fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        return frames, fps, actual_human_height

    if args.smplx_file:
        from general_motion_retargeting.utils.smpl import (
            load_smplx_file,
            get_smplx_data_offline_fast,
        )

        path = pathlib.Path(args.smplx_file)
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            path, body_model_dir
        )
        frames, fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        return frames, fps, actual_human_height

    raise ValueError("Provide --gvhmr_pred_file or --smplx_file")


def benchmark_ik(
    frames: list,
    robot: str,
    actual_human_height: float | None,
    contact_ground,
    foot_ground_limit,
    fix_robot_penetration,
    fps: float,
    warmup: int,
    max_frames: int | None,
) -> list[float]:
    gmr = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )
    subset = frames[:max_frames] if max_frames else frames
    times_ms: list[float] = []

    for i, frame in enumerate(subset):
        t0 = time.perf_counter()
        gmr.retarget(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times_ms.append(elapsed_ms)
    return times_ms


def benchmark_sliding_window(
    frames: list,
    robot: str,
    actual_human_height: float | None,
    contact_ground,
    foot_ground_limit,
    fix_robot_penetration,
    fps: float,
    warmup: int,
    max_frames: int | None,
    window_size: int,
    mode: str,
    ik_warmstart_iters: int,
    fast_opt_iter: int,
    w_velocity: float,
    w_acceleration: float,
) -> list[float]:
    gmr = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )
    sw = SlidingWindowRetargeter(
        gmr,
        SlidingWindowConfig(
            window_size=window_size,
            mode=mode,
            w_velocity=w_velocity,
            w_acceleration=w_acceleration,
            ik_warmstart_iters=ik_warmstart_iters,
            fast_opt_iter=fast_opt_iter,
        ),
    )
    subset = frames[:max_frames] if max_frames else frames
    times_ms: list[float] = []

    for i, frame in enumerate(subset):
        t0 = time.perf_counter()
        sw.retarget(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times_ms.append(elapsed_ms)
    return times_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retargeting timing.")
    parser.add_argument("--gvhmr_pred_file", type=str, default=None)
    parser.add_argument("--smplx_file", type=str, default=None)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--ik_warmstart_iters", type=int, default=3)
    parser.add_argument("--fast_opt_iter", type=int, default=5)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")
    args = parser.parse_args()

    frames, fps, actual_human_height = load_human_frames(args)
    n_use = args.max_frames if args.max_frames else len(frames)
    n_use = min(n_use, len(frames))

    print(f"Input frames: {len(frames)} (benchmark first {n_use}), fps={fps:.1f}")
    print(f"Robot: {args.robot}, sliding-window mode={args.mode}, window={args.window_size}")
    print(f"Warmup frames excluded: {args.warmup}")
    print()

    ik_ms = benchmark_ik(
        frames,
        args.robot,
        actual_human_height,
        args.contact_ground,
        args.foot_ground_limit,
        args.fix_robot_penetration,
        fps,
        args.warmup,
        args.max_frames,
    )
    sw_ms = benchmark_sliding_window(
        frames,
        args.robot,
        actual_human_height,
        args.contact_ground,
        args.foot_ground_limit,
        args.fix_robot_penetration,
        fps,
        args.warmup,
        args.max_frames,
        args.window_size,
        args.mode,
        args.ik_warmstart_iters,
        args.fast_opt_iter,
        args.w_velocity,
        args.w_acceleration,
    )

    ik_stats = summarize_ms(ik_ms)
    sw_stats = summarize_ms(sw_ms)

    print("Per-frame retarget time (viewer excluded):")
    print_summary("per-frame IK", ik_stats)
    print_summary("sliding-window", sw_stats)
    print()

    ratio = sw_stats["mean_ms"] / ik_stats["mean_ms"]
    realtime_ik = ik_stats["mean_ms"] <= 1000.0 / fps
    realtime_sw = sw_stats["mean_ms"] <= 1000.0 / fps
    print(f"Speed ratio (SW / IK): {ratio:.2f}x")
    print(f"Real-time @ {fps:.0f} fps needs <= {1000.0 / fps:.1f} ms/frame")
    print(f"  per-frame IK:     {'OK' if realtime_ik else 'NO'}")
    print(f"  sliding-window:   {'OK' if realtime_sw else 'NO'}")
    print()

    total_ik = sum(ik_ms) / 1000.0
    total_sw = sum(sw_ms) / 1000.0
    print(f"Total measured time ({int(ik_stats['n'])} frames after warmup):")
    print(f"  per-frame IK:     {total_ik:.2f} s")
    print(f"  sliding-window:   {total_sw:.2f} s")


if __name__ == "__main__":
    main()
