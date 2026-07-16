#!/usr/bin/env python3
"""Benchmark retarget methods on GVHMR .pt clips.

Methods:
  - ik                Python per-frame GMR (online baseline)
  - online_batch      Online batch-lite multi-frame GN (online)
  - py_batch_to       Python BatchTrajectoryRetargeter quality (offline)
  - cpp_batch_to      C++ gmr_batch_to_cli quality (offline)

Quality metrics (vs IK baseline where applicable):
  - dq / ddq / jerk   joint-space smoothness (mean L2 norms)
  - fk                weighted FK tracking cost (same as Batch TO)
  - foot_slip         contact-frame XY foot slip
  - rmse_vs_ik        qpos deviation from per-frame IK
  - ms/frame          realtime feasibility @ 30 FPS

Example:
  python scripts/analysis/benchmark_gvhmr_retarget_methods.py \\
    --pt_glob 'data/gvhmr_test_videos/*/hmr4d_results.pt' \\
    --robot unitree_g1 --contact_ground --max_frames 200
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
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
from general_motion_retargeting.online_batch_retarget import (
    OnlineBatchConfig,
    OnlineBatchRetargeter,
)
from general_motion_retargeting.human_frame_loaders import frame_to_json_dict, load_human_motion_frames
from scripts.analysis.analyze_saved_motion_metrics import (
    DEFAULT_FOOT_BODIES,
    ROBOT_XML_PATHS,
    foot_slip_metrics,
    scalar_q_indices,
    smoothness_metrics,
)

CPP_BATCH = REPO_ROOT / "cpp" / "build" / "gmr_batch_to_cli"

ONLINE_METHODS = ("ik", "online_batch")
OFFLINE_METHODS = ("py_batch_to", "cpp_batch_to")
ALL_METHODS = ONLINE_METHODS + OFFLINE_METHODS


def pct(before: float, after: float) -> float:
    return 100.0 * (after - before) / max(abs(before), 1e-9)


def env_with_ld() -> dict:
    env = os.environ.copy()
    devel = "/opt/robot/devel/lib"
    if pathlib.Path(devel).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def make_fk_evaluator(gmr_kwargs: dict, fps: float) -> BatchTrajectoryRetargeter:
    """Reuse Batch TO helpers for weighted FK tracking cost vs human targets."""
    gmr = GMR(**gmr_kwargs)
    batch = BatchTrajectoryRetargeter(
        gmr,
        BatchTrajectoryConfig(verbose=False, show_progress=False),
    )
    batch.set_motion_fps(fps)
    return batch


def prepare_fk_targets(fk_eval: BatchTrajectoryRetargeter, frames: list) -> list[dict]:
    prepared = [fk_eval.gmr._prepare_scaled_human_data(f) for f in frames]
    return [fk_eval._targets_for_prepared(p) for p in prepared]


def mean_fk_cost(
    fk_eval: BatchTrajectoryRetargeter,
    q: np.ndarray,
    targets: list[dict],
) -> float:
    costs = [fk_eval._fk_tracking_cost(qi, t) for qi, t in zip(q, targets)]
    return float(np.mean(costs)) if costs else 0.0


def quality_metrics(
    q: np.ndarray,
    ik_q: np.ndarray,
    fps: float,
    robot: str,
    fk_eval: BatchTrajectoryRetargeter | None = None,
    targets: list[dict] | None = None,
    ik_fk_mean: float | None = None,
) -> dict:
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qidx = scalar_q_indices(model)
    foot = DEFAULT_FOOT_BODIES[robot]
    sm = smoothness_metrics(q, fps, qidx)
    slip = foot_slip_metrics(model, q, foot, 0.02)
    ik_sm = smoothness_metrics(ik_q, fps, qidx)
    ik_slip = foot_slip_metrics(model, ik_q, foot, 0.02)

    out = {
        "rmse_vs_ik": float(np.sqrt(np.mean((ik_q - q) ** 2))),
        "dq_mean": sm["dq"]["mean"],
        "ddq_mean": sm["ddq"]["mean"],
        "jerk_mean": sm["jerk"]["mean"],
        "foot_slip_total": slip["total_slip"],
        "dq_change_pct": pct(ik_sm["dq"]["mean"], sm["dq"]["mean"]),
        "ddq_change_pct": pct(ik_sm["ddq"]["mean"], sm["ddq"]["mean"]),
        "jerk_change_pct": pct(ik_sm["jerk"]["mean"], sm["jerk"]["mean"]),
        "foot_slip_change_pct": pct(ik_slip["total_slip"], slip["total_slip"]),
    }

    if fk_eval is not None and targets is not None:
        fk_mean = mean_fk_cost(fk_eval, q, targets)
        if ik_fk_mean is None:
            ik_fk_mean = mean_fk_cost(fk_eval, ik_q, targets)
        out["fk_mean"] = fk_mean
        out["fk_change_pct"] = pct(float(ik_fk_mean), fk_mean)
    else:
        out["fk_mean"] = float("nan")
        out["fk_change_pct"] = float("nan")

    return out


def run_ik(frames, gmr_kwargs, fps) -> tuple[np.ndarray, float]:
    gmr = GMR(**gmr_kwargs)
    t0 = time.perf_counter()
    q = np.stack([gmr.retarget(f).copy() for f in frames])
    ms = (time.perf_counter() - t0) * 1000.0
    return q, ms / len(frames)


def run_online_batch(frames, gmr_kwargs, fps, preset: str = "balanced") -> tuple[np.ndarray, float]:
    gmr = GMR(**gmr_kwargs)
    online = OnlineBatchRetargeter(gmr, OnlineBatchConfig.from_preset(preset))
    online.set_motion_fps(fps)
    t0 = time.perf_counter()
    q = online.retarget_sequence(frames)
    ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(q), ms / len(frames)


def run_py_batch(frames, gmr_kwargs, fps) -> tuple[np.ndarray, float]:
    gmr = GMR(**gmr_kwargs)
    batch = BatchTrajectoryRetargeter(
        gmr,
        BatchTrajectoryConfig(window_size=16, window_stride=8, gn_steps=3, verbose=False, show_progress=False),
    )
    batch.set_motion_fps(fps)
    t0 = time.perf_counter()
    q = batch.retarget_batch(frames)
    ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(q), ms / len(frames)


def run_cpp_batch(
    frames,
    fps: float,
    height: float,
    src_human: str,
    robot: str,
    contact_ground: bool,
) -> tuple[np.ndarray, float]:
    payload = {
        "fps": float(fps),
        "src_human": src_human,
        "actual_human_height": float(height),
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    with tempfile.TemporaryDirectory(prefix="gvhmr_bench_") as tmp:
        hj = pathlib.Path(tmp) / "human.json"
        oj = pathlib.Path(tmp) / "out.json"
        hj.write_text(json.dumps(payload))
        cmd = [
            str(CPP_BATCH),
            "--gmr_root",
            str(REPO_ROOT),
            "--robot",
            robot,
            "--human_frame_json",
            str(hj),
            "--out_json",
            str(oj),
            "--src_human",
            src_human,
            "--actual_human_height",
            str(height),
            "--max_frames",
            str(len(frames)),
            "--gn_line_search",
            "best",
        ]
        if contact_ground:
            cmd.append("--contact_ground")
        subprocess.run(cmd, check=True, env=env_with_ld(), cwd=REPO_ROOT)
        data = json.loads(oj.read_text())
        q = np.asarray(data["qpos_frames"], dtype=float)
        prof = data.get("profile", {})
        ms_pf = float(prof.get("ms_per_frame", 0.0))
        if ms_pf <= 0 and "total_ms" in prof:
            n = max(len(frames), 1)
            ms_pf = float(prof["total_ms"]) / n
        return q, ms_pf


def benchmark_pt(
    pt_path: pathlib.Path,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    methods: tuple[str, ...],
) -> dict:
    frames, fps, height, src_human = load_human_motion_frames(
        pt_path, input_type="gvhmr_pt", max_frames=max_frames
    )
    gmr_kwargs = dict(
        actual_human_height=height,
        src_human=src_human,
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        motion_fps=fps,
    )

    results: dict[str, dict] = {}
    ik_q = None

    fk_eval = make_fk_evaluator(gmr_kwargs, fps)
    targets = prepare_fk_targets(fk_eval, frames)

    if "ik" in methods or any(m != "ik" for m in methods):
        ik_q, ik_ms = run_ik(frames, gmr_kwargs, fps)
        ik_fk_mean = mean_fk_cost(fk_eval, ik_q, targets)
        if "ik" in methods:
            results["ik"] = {
                "kind": "online",
                "ms_per_frame": ik_ms,
                "realtime_30fps": ik_ms <= 1000.0 / 30.0,
                **quality_metrics(
                    ik_q, ik_q, fps, robot, fk_eval, targets, ik_fk_mean=ik_fk_mean
                ),
            }
            results["ik"]["rmse_vs_ik"] = 0.0
            results["ik"]["dq_change_pct"] = 0.0
            results["ik"]["ddq_change_pct"] = 0.0
            results["ik"]["jerk_change_pct"] = 0.0
            results["ik"]["foot_slip_change_pct"] = 0.0
            results["ik"]["fk_change_pct"] = 0.0
    else:
        ik_fk_mean = None

    if ik_q is None:
        ik_q, _ = run_ik(frames, gmr_kwargs, fps)
        ik_fk_mean = mean_fk_cost(fk_eval, ik_q, targets)
    elif ik_fk_mean is None:
        ik_fk_mean = mean_fk_cost(fk_eval, ik_q, targets)

    def _qm(q: np.ndarray) -> dict:
        return quality_metrics(
            q, ik_q, fps, robot, fk_eval, targets, ik_fk_mean=ik_fk_mean
        )

    if "py_batch_to" in methods:
        q, ms = run_py_batch(frames, gmr_kwargs, fps)
        results["py_batch_to"] = {
            "kind": "offline",
            "ms_per_frame": ms,
            "realtime_30fps": ms <= 1000.0 / 30.0,
            **_qm(q),
        }

    if "online_batch" in methods:
        q, ms = run_online_batch(frames, gmr_kwargs, fps)
        results["online_batch"] = {
            "kind": "online",
            "ms_per_frame": ms,
            "realtime_30fps": ms <= 1000.0 / 30.0,
            **_qm(q),
        }

    if "cpp_batch_to" in methods:
        q, ms = run_cpp_batch(frames, fps, height, src_human, robot, contact_ground)
        results["cpp_batch_to"] = {
            "kind": "offline",
            "ms_per_frame": ms,
            "realtime_30fps": ms <= 1000.0 / 30.0,
            **_qm(q),
        }

    return {
        "pt_file": str(pt_path),
        "clip_name": pt_path.parent.name if pt_path.name == "hmr4d_results.pt" else pt_path.stem,
        "n_frames": len(frames),
        "fps": float(fps),
        "methods": results,
    }


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for method in ALL_METHODS:
        vals = []
        for row in rows:
            m = row.get("methods", {}).get(method)
            if m:
                vals.append(m)
        if not vals:
            continue
        summary[method] = {
            "kind": vals[0]["kind"],
            "mean_ms_per_frame": float(np.mean([v["ms_per_frame"] for v in vals])),
            "max_ms_per_frame": float(np.max([v["ms_per_frame"] for v in vals])),
            "realtime_30fps_count": sum(1 for v in vals if v["realtime_30fps"]),
            "mean_dq_change_pct": float(np.mean([v["dq_change_pct"] for v in vals])),
            "mean_ddq_change_pct": float(np.mean([v["ddq_change_pct"] for v in vals])),
            "mean_jerk_change_pct": float(np.mean([v["jerk_change_pct"] for v in vals])),
            "mean_foot_slip_change_pct": float(np.mean([v["foot_slip_change_pct"] for v in vals])),
            "mean_fk_change_pct": float(np.mean([v["fk_change_pct"] for v in vals])),
            "mean_fk": float(np.mean([v["fk_mean"] for v in vals])),
            "mean_dq": float(np.mean([v["dq_mean"] for v in vals])),
            "mean_ddq": float(np.mean([v["ddq_mean"] for v in vals])),
            "mean_jerk": float(np.mean([v["jerk_mean"] for v in vals])),
            "mean_rmse_vs_ik": float(np.mean([v["rmse_vs_ik"] for v in vals])),
            "n_clips": len(vals),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt_glob", default="data/gvhmr_test_videos/*/hmr4d_results.pt")
    parser.add_argument("--pt_file", action="append", default=[])
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help=f"Comma-separated subset of: {','.join(ALL_METHODS)}",
    )
    parser.add_argument("--output_json", default=str(REPO_ROOT / "output" / "gvhmr_retarget_benchmark.json"))
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}")

    if args.pt_file:
        pts = [pathlib.Path(p).expanduser().resolve() for p in args.pt_file]
    else:
        pts = sorted(REPO_ROOT.glob(args.pt_glob))
    pts = [p for p in pts if p.is_file()]
    if not pts:
        pts = sorted((REPO_ROOT / "output" / "gvhmr_pt").glob("*.pt"))

    if not pts:
        raise FileNotFoundError("No .pt files found. Run batch_video_to_gvhmr.py first.")

    rows = []
    print(f"Benchmarking {len(pts)} GVHMR clips, methods={methods}")
    for pt in pts:
        print(f"\n=== {pt.parent.name} ===", flush=True)
        try:
            row = benchmark_pt(pt, args.robot, args.max_frames, args.contact_ground, methods)
            rows.append(row)
            for name, m in row["methods"].items():
                print(
                    f"  {name:18s} {m['ms_per_frame']:6.1f} ms/f  "
                    f"dq {m['dq_change_pct']:+5.1f}%  ddq {m['ddq_change_pct']:+5.1f}%  "
                    f"jerk {m['jerk_change_pct']:+5.1f}%  "
                    f"fk {m['fk_change_pct']:+5.1f}%  slip {m['foot_slip_change_pct']:+5.1f}%  "
                    f"rmse {m['rmse_vs_ik']:.4f}"
                )
        except Exception as exc:
            print(f"  FAIL: {exc}")
            rows.append({"pt_file": str(pt), "clip_name": pt.stem, "error": str(exc)})

    ok = [r for r in rows if "methods" in r]
    result = {
        "robot": args.robot,
        "max_frames": args.max_frames,
        "contact_ground": args.contact_ground,
        "methods_tested": list(methods),
        "summary": summarize_rows(ok),
        "clips": rows,
    }
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
