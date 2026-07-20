#!/usr/bin/env python3
"""Plot per-frame torque-ratio curves: online_qp baseline vs torque_limit."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames
from general_motion_retargeting.online_qp_retarget import OnlineQpConfig, OnlineQpRetargeter
from general_motion_retargeting.params import ROBOT_XML_DICT
from scripts.analysis.benchmark_control_feasibility import (
    UPPER_KEYS,
    group_stats,
    torque_ratios,
)

UPPER_ONLY = UPPER_KEYS


def frame_upper_peak(ratios: np.ndarray, jt: list) -> np.ndarray:
  cols = [i for i, (n, _, _) in enumerate(jt) if any(k in n for k in UPPER_ONLY)]
  if not cols:
    return np.zeros(ratios.shape[0])
  return ratios[:, cols].max(axis=1)


def run_clip(clip: str, max_frames: int, preset: str, weight: float, gate_mode: str):
  pt = REPO / f"data/gvhmr_test_videos/{clip}/hmr4d_results.pt"
  frames, fps, h, src = load_human_motion_frames(
    pt, input_type="gvhmr_pt", max_frames=max_frames
  )
  kwargs = dict(
    src_human=src,
    tgt_robot="unitree_g1",
    verbose=False,
    contact_ground=True,
    actual_human_height=h,
    motion_fps=fps,
  )
  model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))

  def retarget(enable_tq: bool) -> np.ndarray:
    cfg = OnlineQpConfig.from_preset(preset)
    if enable_tq:
      cfg.torque_limit_constraint = True
      cfg.torque_limit_weight = weight
      cfg.torque_limit_scope = "upper"
      cfg.torque_limit_margin = 0.1
      cfg.torque_limit_gate_mode = gate_mode
    r = OnlineQpRetargeter(GMR(**kwargs), cfg)
    r.set_motion_fps(fps)
    return r.retarget_sequence(frames)

  Q0 = retarget(False)
  Q1 = retarget(True)
  r0, jt = torque_ratios(model, Q0, fps)
  r1, _ = torque_ratios(model, Q1, fps)
  peak0 = frame_upper_peak(r0, jt)
  peak1 = frame_upper_peak(r1, jt)
  return {
    "clip": clip,
    "fps": fps,
    "peak0": peak0,
    "peak1": peak1,
    "stats0": group_stats(r0, jt, UPPER_ONLY),
    "stats1": group_stats(r1, jt, UPPER_ONLY),
  }


def plot_one(ax, t: np.ndarray, peak0: np.ndarray, peak1: np.ndarray, title: str, kappa: float):
  ax.plot(t, peak0, label="baseline (no torque_limit)", color="#4C72B0", lw=1.2, alpha=0.9)
  ax.plot(t, peak1, label="torque_limit (soft gate)", color="#C44E52", lw=1.2, alpha=0.9)
  ax.axhline(kappa, color="gray", ls="--", lw=0.9, alpha=0.7, label=f"margin line ({kappa:.2f})")
  ax.set_title(title)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("upper-body max |τ|/τ_max")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="upper right", fontsize=8)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--clips", nargs="+", default=["walking", "tennis"])
  parser.add_argument("--max_frames", type=int, default=200)
  parser.add_argument("--preset", default="anti_slip")
  parser.add_argument("--weight", type=float, default=10.0)
  parser.add_argument("--gate_mode", default="soft")
  parser.add_argument("--out_dir", default=str(REPO / "output" / "torque_limit_curves"))
  args = parser.parse_args()
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  kappa = 1.0 - 0.1  # torque_limit_margin default
  results = []
  for clip in args.clips:
    print(f"Running {clip}...")
    results.append(
      run_clip(clip, args.max_frames, args.preset, args.weight, args.gate_mode)
    )

  n = len(results)
  fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=False)
  if n == 1:
    axes = [axes]
  for ax, res in zip(axes, results):
    t = np.arange(len(res["peak0"])) / res["fps"]
    s0, s1 = res["stats0"], res["stats1"]
    title = (
      f"{res['clip']}: peak {s0['peak']:.2f} → {s1['peak']:.2f} "
      f"({100 * (s1['peak'] - s0['peak']) / max(s0['peak'], 1e-9):+.0f}%)"
    )
    plot_one(ax, t, res["peak0"], res["peak1"], title, kappa)

  fig.suptitle(
    f"online_qp preset={args.preset} | torque_limit w={args.weight} gate={args.gate_mode}",
    fontsize=11,
    y=1.01,
  )
  fig.tight_layout()
  combined = out_dir / "torque_limit_curve_compare.png"
  fig.savefig(combined, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Wrote {combined}")

  for res in results:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = np.arange(len(res["peak0"])) / res["fps"]
    s0, s1 = res["stats0"], res["stats1"]
    title = (
      f"{res['clip']}: peak {s0['peak']:.2f} → {s1['peak']:.2f}"
    )
    plot_one(ax, t, res["peak0"], res["peak1"], title, kappa)
    path = out_dir / f"{res['clip']}_torque_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
  main()
