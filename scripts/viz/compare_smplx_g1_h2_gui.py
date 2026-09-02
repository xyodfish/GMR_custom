#!/usr/bin/env python3
"""Compare SMPL-X retarget: G1 vs H2 bridge, and two independent H2 entry paths.

Pipeline:
  NPZ -> run_cpp_batch_to.py -> G1 json + H2 json (H2 uses the internal G1 bridge)

Outputs under --out_dir:
  metrics.json
  g1_h2_compare.mp4          (side-by-side G1 grey | H2 blue)
  h2_vs_gui_compare.mp4      (H2 bridge grey | GUI H2 orange), if benchmark exists
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.dual_robot_viewer import TwoRobotMotionViewer

GUI_H2_DIR = REPO / "output" / "robot_to_robot_gui" / "robot_b" / "unitree_h2" / "gui"
RUN_CPP = REPO / "scripts" / "tools" / "run_cpp_batch_to.py"
DEFAULT_PY = Path(os.environ.get("GMR_PYTHON", Path.home() / "miniconda3/envs/gmr/bin/python"))


def _run_batch(python: Path, npz: Path, robot: str, out_json: Path, max_frames: int | None) -> None:
    cmd = [
        str(python),
        str(RUN_CPP),
        "--input_file",
        str(npz),
        "--input_type",
        "smplx",
        "--robot",
        robot,
        "--out_json",
        str(out_json),
        "--fast",
        "--contact_ground",
    ]
    if max_frames is not None:
        cmd += ["--max_frames", str(max_frames)]

    subprocess.run(cmd, cwd=REPO, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def _load_qpos(path: Path) -> tuple[np.ndarray, float, dict]:
    meta, fps, *_rest, qpos = load_robot_motion(str(path))
    if qpos is None:
        raise ValueError(f"No qpos in {path}")

    extra = meta if isinstance(meta, dict) else {}
    return np.asarray(qpos, dtype=float), float(fps), extra


def _resolve_gui_h2(npz: Path, amass_root: Path) -> Path | None:
    rel = npz.resolve().relative_to(amass_root.resolve())
    stem = str(rel.with_suffix("")).replace("/", "_")
    for suffix in (
        "_gmr_realtime_canonical",
        "_gmr_realtime",
        "_gmr",
    ):
        candidate = GUI_H2_DIR / f"{stem}{suffix}.qpos.json"
        if candidate.is_file():
            return candidate

    return None


def _joint_metrics(q_a: np.ndarray, q_b: np.ndarray) -> dict:
    n = min(len(q_a), len(q_b))
    qa, qb = q_a[:n], q_b[:n]
    dof_diff = np.abs(qa[:, 7:] - qb[:, 7:])
    return {
        "frames": n,
        "mean_abs_dof_rad": float(dof_diff.mean()),
        "max_abs_dof_rad": float(dof_diff.max()),
        "root_z_mean_a": float(qa[:, 2].mean()),
        "root_z_mean_b": float(qb[:, 2].mean()),
        "root_z_mean_abs_err_m": float(np.abs(qa[:, 2] - qb[:, 2]).mean()),
        "hip_roll_mean_abs_a": float(np.abs(qa[:, 10]).mean()),
        "hip_roll_mean_abs_b": float(np.abs(qb[:, 10]).mean()),
        "per_joint_mean_abs_rad": dof_diff.mean(axis=0).tolist(),
    }


def _record_g1_h2(g1_json: Path, h2_json: Path, out_mp4: Path, offset_y: float, max_frames: int | None) -> int:
    q_g1, fps, _ = _load_qpos(g1_json)
    q_h2, fps_h2, _ = _load_qpos(h2_json)
    if max_frames is not None:
        q_g1 = q_g1[:max_frames]
        q_h2 = q_h2[:max_frames]

    n = min(len(q_g1), len(q_h2))
    viewer = TwoRobotMotionViewer(
        "unitree_g1",
        "unitree_h2",
        motion_fps=fps,
        offset_b=(0.0, offset_y, 0.0),
        tint_b=(0.35, 0.55, 0.95, 1.0),
        record_video=True,
        video_path=str(out_mp4),
    )
    try:
        for i in range(n):
            viewer.step(q_g1[i], q_h2[i], rate_limit=False, follow_camera=True)
    finally:
        viewer.close()

    return n


def _record_h2_gui(h2_json: Path, gui_json: Path, out_mp4: Path, offset_y: float, max_frames: int | None) -> int:
    q_bridge, fps, _ = _load_qpos(h2_json)
    q_gui, fps_gui, _ = _load_qpos(gui_json)
    if max_frames is not None:
        q_bridge = q_bridge[:max_frames]
        q_gui = q_gui[:max_frames]

    n = min(len(q_bridge), len(q_gui))
    viewer = TwoRobotMotionViewer(
        "unitree_h2",
        "unitree_h2",
        motion_fps=fps,
        offset_b=(0.0, offset_y, 0.0),
        tint_b=(0.95, 0.45, 0.15, 1.0),
        record_video=True,
        video_path=str(out_mp4),
    )
    try:
        for i in range(n):
            viewer.step(q_bridge[i], q_gui[i], rate_limit=False, follow_camera=True)
    finally:
        viewer.close()

    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="SMPL-X .npz")
    parser.add_argument("--amass_root", default=str(Path.home() / "Workspace/data"))
    parser.add_argument("--out_dir", default=str(REPO / "videos" / "compare_reports"))
    parser.add_argument("--python", default=str(DEFAULT_PY))
    parser.add_argument("--max_frames", type=int, default=180)
    parser.add_argument("--offset_y", type=float, default=1.4)
    parser.add_argument("--skip_retarget", action="store_true", help="Reuse existing json in out_dir")
    args = parser.parse_args()

    npz = Path(args.input_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_name = npz.stem
    g1_json = out_dir / f"{clip_name}_g1.json"
    h2_json = out_dir / f"{clip_name}_h2_g1_bridge.json"

    if not args.skip_retarget:
        print(f"[compare] retarget G1: {npz}")
        _run_batch(Path(args.python), npz, "unitree_g1", g1_json, args.max_frames)
        print(f"[compare] retarget H2 (internal G1 bridge): {npz}")
        _run_batch(Path(args.python), npz, "unitree_h2", h2_json, args.max_frames)

    _, _, g1_meta = _load_qpos(g1_json)
    _, _, h2_meta = _load_qpos(h2_json)
    gui_json = _resolve_gui_h2(npz, Path(args.amass_root))

    q_g1, _, _ = _load_qpos(g1_json)
    q_h2, _, _ = _load_qpos(h2_json)
    metrics = {
        "input_npz": str(npz),
        "g1_method": g1_meta.get("method"),
        "h2_method": h2_meta.get("method"),
        "h2_g1_bridge": h2_meta.get("g1_bridge"),
        "g1_vs_h2": _joint_metrics(q_g1, q_h2),
        "gui_h2_benchmark": str(gui_json) if gui_json else None,
    }

    if gui_json is not None:
        q_gui, _, _ = _load_qpos(gui_json)
        metrics["h2_bridge_vs_gui"] = _joint_metrics(q_h2, q_gui)

    metrics_path = out_dir / f"{clip_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    g1_h2_mp4 = out_dir / f"{clip_name}_g1_h2_compare.mp4"
    n_g1_h2 = _record_g1_h2(g1_json, h2_json, g1_h2_mp4, args.offset_y, args.max_frames)
    print(f"[compare] wrote {g1_h2_mp4} ({n_g1_h2} frames)")

    if gui_json is not None:
        h2_gui_mp4 = out_dir / f"{clip_name}_h2_vs_gui_compare.mp4"
        n_h2_gui = _record_h2_gui(h2_json, gui_json, h2_gui_mp4, args.offset_y, args.max_frames)
        print(f"[compare] wrote {h2_gui_mp4} ({n_h2_gui} frames)")
        m = metrics["h2_bridge_vs_gui"]
        print(
            f"[compare] H2 bridge vs GUI: mean_dof={m['mean_abs_dof_rad']:.4f} rad, "
            f"root_z {m['root_z_mean_a']:.3f} vs {m['root_z_mean_b']:.3f}, "
            f"|hip_roll| {m['hip_roll_mean_abs_a']:.3f} vs {m['hip_roll_mean_abs_b']:.3f}"
        )
    else:
        print(f"[compare] no GUI H2 benchmark for {npz.name} (skipped h2_vs_gui video)")

    m = metrics["g1_vs_h2"]
    print(
        f"[compare] G1 vs H2: mean_dof={m['mean_abs_dof_rad']:.4f} rad, "
        f"root_z {m['root_z_mean_a']:.3f} vs {m['root_z_mean_b']:.3f}"
    )
    print(f"[compare] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
