#!/usr/bin/env python3
"""Benchmark C++ Causal TO (L-BFGS) vs per-frame IK on multiple motions.

Curated for teleop / realtime replay evaluation across motion types:
  LAFAN1 BVH themes (walk, run, jump, dance, sports, fall, ground, aiming)
  + optional GVHMR .pt clips.

Example:
  python scripts/analysis/benchmark_cpp_causal_suite.py \\
    --robot unitree_g1 --contact_ground --max_frames 200 \\
    --output_json output/causal_to_suite.json
"""

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
from general_motion_retargeting.human_frame_loaders import frame_to_json_dict, load_human_motion_frames
from scripts.analysis.analyze_saved_motion_metrics import (
    DEFAULT_FOOT_BODIES,
    ROBOT_XML_PATHS,
    foot_slip_metrics,
    scalar_q_indices,
    smoothness_metrics,
)

DEFAULT_CPP_CLI = REPO_ROOT / "cpp" / "build" / "gmr_causal_to_cli"
DEFAULT_LAFAN1 = pathlib.Path("~/Workspace/data/lafan1").expanduser()

# Representative LAFAN1 clips (Ubisoft La Forge Animation Dataset).
DEFAULT_SUITE = [
    ("walk1_subject5.bvh", "walk"),
    ("walk4_subject1.bvh", "walk"),
    ("run2_subject4.bvh", "run"),
    ("sprint1_subject4.bvh", "sprint"),
    ("jumps1_subject1.bvh", "jumps"),
    ("dance1_subject1.bvh", "dance"),
    ("fightAndSports1_subject1.bvh", "sports"),
    ("fallAndGetUp1_subject1.bvh", "fall"),
    ("ground1_subject5.bvh", "ground"),
    ("aiming1_subject1.bvh", "aiming"),
]


def pct(before: float, after: float) -> float:
    return 100.0 * (after - before) / max(abs(before), 1e-9)


def run_ik(frames, fps: float, height: float, src_human: str, robot: str, contact_ground: bool) -> np.ndarray:
    gmr = GMR(
        actual_human_height=height,
        src_human=src_human,
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        motion_fps=fps,
    )
    return np.stack([gmr.retarget(f).copy() for f in frames])


def run_cpp_causal(
    frames,
    fps: float,
    height: float,
    src_human: str,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    solver: str,
    cpp_cli: pathlib.Path,
) -> tuple[np.ndarray, dict]:
    payload = {
        "fps": float(fps),
        "src_human": src_human,
        "actual_human_height": float(height),
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    env = os.environ.copy()
    devel_lib = "/opt/robot/devel/lib"
    if pathlib.Path(devel_lib).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    with tempfile.TemporaryDirectory(prefix="causal_suite_") as tmp:
        human_json = pathlib.Path(tmp) / "human.json"
        out_json = pathlib.Path(tmp) / "out.json"
        human_json.write_text(json.dumps(payload))
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
            "--src_human",
            src_human,
            "--actual_human_height",
            str(height),
            "--max_frames",
            str(max_frames),
            "--solver",
            solver,
        ]
        if contact_ground:
            cmd.append("--contact_ground")
        subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)
        data = json.loads(out_json.read_text())
        q = np.asarray(data["qpos_frames"], dtype=float)
        return q, data.get("profile", {})


def evaluate_motion(
    name: str,
    theme: str,
    input_path: pathlib.Path,
    robot: str,
    max_frames: int,
    contact_ground: bool,
    solver: str,
    cpp_cli: pathlib.Path,
) -> dict:
    frames, fps, height, src_human = load_human_motion_frames(
        input_path,
        input_type="auto",
        bvh_format="lafan1",
        max_frames=max_frames,
    )
    n = len(frames)

    ik_q = run_ik(frames, fps, height, src_human, robot, contact_ground)
    causal_q, prof = run_cpp_causal(
        frames, fps, height, src_human, robot, n, contact_ground, solver, cpp_cli
    )

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qidx = scalar_q_indices(model)
    foot = DEFAULT_FOOT_BODIES[robot]

    ik_sm = smoothness_metrics(ik_q, fps, qidx)
    ca_sm = smoothness_metrics(causal_q, fps, qidx)
    ik_slip = foot_slip_metrics(model, ik_q, foot, 0.02)
    ca_slip = foot_slip_metrics(model, causal_q, foot, 0.02)

    rmse = float(np.sqrt(np.mean((ik_q - causal_q) ** 2)))
    ms_per_frame = float(prof.get("ms_per_frame", 0.0))
    max_frame_ms = float(prof.get("max_frame_ms", 0.0))

    return {
        "name": name,
        "theme": theme,
        "input": str(input_path),
        "n_frames": n,
        "fps": float(fps),
        "src_human": src_human,
        "timing_ms_per_frame": ms_per_frame,
        "timing_max_frame_ms": max_frame_ms,
        "realtime_30fps": ms_per_frame <= 1000.0 / 30.0,
        "rmse_vs_ik": rmse,
        "ik": {
            "jerk_mean": ik_sm["jerk"]["mean"],
            "foot_slip_total": ik_slip["total_slip"],
        },
        "causal": {
            "jerk_mean": ca_sm["jerk"]["mean"],
            "foot_slip_total": ca_slip["total_slip"],
        },
        "jerk_change_pct": pct(ik_sm["jerk"]["mean"], ca_sm["jerk"]["mean"]),
        "foot_slip_change_pct": pct(ik_slip["total_slip"], ca_slip["total_slip"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument("--solver", default="lbfgs", choices=["lbfgs", "gn", "none"])
    parser.add_argument("--bvh_dir", type=pathlib.Path, default=DEFAULT_LAFAN1)
    parser.add_argument("--cpp_cli", type=pathlib.Path, default=DEFAULT_CPP_CLI)
    parser.add_argument("--output_json", type=pathlib.Path, default=REPO_ROOT / "output" / "causal_to_suite.json")
    parser.add_argument("--extra_pt", action="append", default=[], help="Extra GVHMR .pt (name:path or path)")
    parser.add_argument("--suite", action="store_true", help="Run default LAFAN1 theme suite")
    parser.add_argument("--bvh", action="append", default=[], help="Extra BVH filename under --bvh_dir")
    args = parser.parse_args()

    if not args.cpp_cli.is_file():
        raise FileNotFoundError(f"Build first: {args.cpp_cli}")

    motions: list[tuple[str, str, pathlib.Path]] = []

    if args.suite or (not args.bvh and not args.extra_pt):
        for fname, theme in DEFAULT_SUITE:
            motions.append((fname, theme, args.bvh_dir / fname))

    for bvh in args.bvh:
        motions.append((pathlib.Path(bvh).name, "custom", args.bvh_dir / bvh))

    for item in args.extra_pt:
        if ":" in item:
            label, path = item.split(":", 1)
        else:
            path = item
            label = pathlib.Path(path).stem
        motions.append((label, "gvhmr", pathlib.Path(path).expanduser()))

    # Always include cxk-ball if present
    cxk = REPO_ROOT / "output" / "gvhmr_pt" / "cxk-ball_hmr4d_results.pt"
    if cxk.is_file() and not any(m[2] == cxk for m in motions):
        motions.append(("cxk-ball", "gvhmr_sports", cxk))

    rows = []
    print(f"Benchmarking {len(motions)} motions (max_frames={args.max_frames}, solver={args.solver})")
    for name, theme, path in motions:
        if not path.is_file():
            print(f"  SKIP {name}: missing {path}")
            continue
        print(f"  RUN  {name} ({theme}) ...", flush=True)
        try:
            row = evaluate_motion(
                name, theme, path, args.robot, args.max_frames, args.contact_ground, args.solver, args.cpp_cli
            )
            rows.append(row)
            print(
                f"       {row['timing_ms_per_frame']:.1f} ms/f  "
                f"jerk {row['jerk_change_pct']:+.1f}%  foot_slip {row['foot_slip_change_pct']:+.1f}%  "
                f"rmse_vs_ik={row['rmse_vs_ik']:.4f}"
            )
        except Exception as exc:
            print(f"       FAIL: {exc}")
            rows.append({"name": name, "theme": theme, "input": str(path), "error": str(exc)})

    ok = [r for r in rows if "error" not in r]
    summary = {}
    if ok:
        summary = {
            "n_ok": len(ok),
            "mean_ms_per_frame": float(np.mean([r["timing_ms_per_frame"] for r in ok])),
            "max_ms_per_frame": float(np.max([r["timing_ms_per_frame"] for r in ok])),
            "mean_jerk_change_pct": float(np.mean([r["jerk_change_pct"] for r in ok])),
            "mean_foot_slip_change_pct": float(np.mean([r["foot_slip_change_pct"] for r in ok])),
            "jerk_improved_count": sum(1 for r in ok if r["jerk_change_pct"] < 0),
            "foot_slip_improved_count": sum(1 for r in ok if r["foot_slip_change_pct"] < 0),
            "realtime_30fps_count": sum(1 for r in ok if r["realtime_30fps"]),
        }

    result = {
        "robot": args.robot,
        "max_frames": args.max_frames,
        "contact_ground": args.contact_ground,
        "solver": args.solver,
        "dataset_note": (
            "LAFAN1 = Ubisoft La Forge mocap BVH (walk/run/jump/dance/sports/...); "
            "gvhmr = monocular video pose (.pt). Non-commercial research use for LAFAN1."
        ),
        "summary": summary,
        "motions": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))

    print("\n=== Suite summary ===")
    if summary:
        print(f"Motions OK: {summary['n_ok']}")
        print(f"Timing: mean={summary['mean_ms_per_frame']:.1f} ms/f  max={summary['max_ms_per_frame']:.1f} ms/f")
        print(f"Realtime@30fps: {summary['realtime_30fps_count']}/{summary['n_ok']}")
        print(
            f"Jerk improved: {summary['jerk_improved_count']}/{summary['n_ok']} "
            f"(mean {summary['mean_jerk_change_pct']:+.1f}%)"
        )
        print(
            f"Foot slip improved: {summary['foot_slip_improved_count']}/{summary['n_ok']} "
            f"(mean {summary['mean_foot_slip_change_pct']:+.1f}%)"
        )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
