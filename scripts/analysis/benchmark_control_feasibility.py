#!/usr/bin/env python3
"""Control-feasibility benchmark for native GMR.

Compares three per-frame retargeting objectives on the same clips:

  * ``baseline``  – native GMR IK (kinematic only)
  * ``uniform``   – GMR + uniform joint-acceleration cap (kinematic smoothing)
  * ``torque``    – GMR + per-joint torque-budget cap (control feasibility)

and reports, for each, the actuator-torque headroom of the committed trajectory
(computed by inverse dynamics) alongside the FK tracking fidelity. The point of the
study is to show that the torque-aware objective moves further down the
FK-error-vs-torque-saturation trade-off than plain smoothing.

Torque model: tau = M(q) qddot + C(q,qddot) qddot + g(q) via ``mj_rne`` with
finite-difference velocity/acceleration. Upper-body joints (waist/arms) are
contact-independent and are the trustworthy headroom signal; leg torques during
stance require ground-reaction forces and are reported with that caveat.

Usage:
    python scripts/analysis/benchmark_control_feasibility.py \
        --clips walking tennis --robot unitree_g1 --max_frames 200
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco as mj
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.batch_trajectory_retarget import (
    BatchTrajectoryConfig,
    BatchTrajectoryRetargeter,
)
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames
from general_motion_retargeting.params import ROBOT_XML_DICT
from scripts.analysis.benchmark_gvhmr_retarget_methods import (
    make_fk_evaluator,
    mean_fk_cost,
    prepare_fk_targets,
)

UPPER_KEYS = ("waist", "shoulder", "elbow", "wrist")
LOWER_KEYS = ("hip", "knee", "ankle")


def actuated_joint_table(model: mj.MjModel):
    rows = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = model.jnt_actfrcrange[j]
        tmax = float(max(abs(lo), abs(hi)))
        if tmax <= 0:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
        rows.append((name, int(model.jnt_dofadr[j]), tmax))
    return rows


def torque_ratios(model: mj.MjModel, Q: np.ndarray, fps: float) -> tuple[np.ndarray, list]:
    """Return |tau|/taumax array [T, nJoint] and the joint table."""
    data = mj.MjData(model)
    dt = 1.0 / fps
    jt = actuated_joint_table(model)
    T = Q.shape[0]

    V = np.zeros((T, model.nv))
    for t in range(1, T):
        mj.mj_differentiatePos(model, V[t], dt, Q[t - 1], Q[t])
    A = np.zeros((T, model.nv))
    A[1:-1] = (V[2:] - V[1:-1]) / dt

    ratios = np.zeros((T, len(jt)))
    tau = np.zeros(model.nv)
    for t in range(T):
        data.qpos[:] = Q[t]
        data.qvel[:] = V[t]
        mj.mj_forward(model, data)
        data.qacc[:] = A[t]
        mj.mj_rne(model, data, 1, tau)
        for i, (_, dof, tmax) in enumerate(jt):
            ratios[t, i] = abs(tau[dof]) / tmax
    return ratios, jt


def group_stats(ratios: np.ndarray, jt: list, keys: tuple) -> dict:
    cols = [i for i, (n, _, _) in enumerate(jt) if any(k in n for k in keys)]
    if not cols:
        return {"peak": float("nan"), "rms": float("nan"), "sat": 0}
    R = ratios[:, cols]
    return {
        "peak": float(R.max()),
        "rms": float(np.sqrt(np.mean(R ** 2))),
        "sat": int(np.sum(np.any(R > 0.8, axis=1))),
    }


def velocity_stats(model: mj.MjModel, Q: np.ndarray, fps: float, vel_ref: float) -> dict:
    """Peak joint speed and saturation frames (|qdot| > vel_ref) over actuated hinges."""
    dof = [int(model.jnt_dofadr[j]) for j in range(model.njnt)
           if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
    dt = 1.0 / fps
    V = np.zeros((Q.shape[0], model.nv))
    for t in range(1, Q.shape[0]):
        mj.mj_differentiatePos(model, V[t], dt, Q[t - 1], Q[t])
    Vd = np.abs(V[:, dof])
    return {"vpeak": float(Vd.max()), "vsat": int(np.sum(np.any(Vd > vel_ref, axis=1)))}


def jerk_mean(model: mj.MjModel, Q: np.ndarray, fps: float) -> float:
    qidx = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
            if model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)]
    Jq = Q[:, qidx]
    jerk = np.diff(Jq, axis=0, n=3) * (fps ** 3)
    return float(np.mean(np.abs(jerk)))


def run(frames, gmr_kwargs, fps) -> tuple[np.ndarray, float]:
    gmr = GMR(**gmr_kwargs)
    gmr.set_motion_fps(fps)
    t0 = time.perf_counter()
    Q = np.stack([gmr.retarget(f).copy() for f in frames])
    ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
    return Q, ms


def run_batch(frames, gmr_kwargs, fps, batch_cfg: BatchTrajectoryConfig) -> tuple[np.ndarray, float]:
    gmr = GMR(**gmr_kwargs)
    batch = BatchTrajectoryRetargeter(gmr, batch_cfg)
    batch.set_motion_fps(fps)
    t0 = time.perf_counter()
    Q = np.asarray(batch.retarget_batch(frames))
    ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
    return Q, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["walking", "tennis"])
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--data_dir", default=str(REPO / "data/gvhmr_test_videos"))
    ap.add_argument("--include_batch", action="store_true",
                    help="also compare batch TO uniform vs torque-weighted smoothing")
    ap.add_argument("--vel_ref", type=float, default=9.42,
                    help="reference joint speed limit (rad/s) for velocity saturation count")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))

    base_kwargs = dict(src_human="smplx", tgt_robot=args.robot, verbose=False,
                       contact_ground=True)
    configs = [
        ("baseline", dict()),
        ("uniform@a15", dict(control_feasibility=True, cf_mode="uniform", cf_uniform_accel_cap=15.0)),
        ("uniform@a8", dict(control_feasibility=True, cf_mode="uniform", cf_uniform_accel_cap=8.0)),
        ("torque@m0.2", dict(control_feasibility=True, cf_mode="torque", cf_margin=0.2)),
        ("torque@m0.4", dict(control_feasibility=True, cf_mode="torque", cf_margin=0.4)),
    ]

    for clip in args.clips:
        pt = pathlib.Path(args.data_dir) / clip / "hmr4d_results.pt"
        frames, fps, height, src = load_human_motion_frames(
            pt, input_type="gvhmr_pt", max_frames=args.max_frames)
        kwargs = dict(base_kwargs, actual_human_height=height, src_human=src, motion_fps=fps)

        fk_eval = make_fk_evaluator(kwargs, fps)
        targets = prepare_fk_targets(fk_eval, frames)

        print(f"\n=== {clip}  ({len(frames)}f @ {fps:.0f}fps, robot={args.robot}) ===")
        print(f"{'config':14s} {'ms/f':>6s} {'FKcost':>8s} {'UBpeak':>7s} {'UBsat':>6s} "
              f"{'ALLpeak':>8s} {'vpeak':>6s} {'vsat':>5s} {'jerk':>9s}")

        def report(name, Q, ms):
            ratios, jt = torque_ratios(model, Q, fps)
            ub = group_stats(ratios, jt, UPPER_KEYS)
            allg = group_stats(ratios, jt, UPPER_KEYS + LOWER_KEYS)
            vs = velocity_stats(model, Q, fps, args.vel_ref)
            fk = mean_fk_cost(fk_eval, Q, targets)
            jm = jerk_mean(model, Q, fps)
            print(f"{name:14s} {ms:6.2f} {fk:8.4f} {ub['peak']:7.2f} {ub['sat']:6d} "
                  f"{allg['peak']:8.2f} {vs['vpeak']:6.1f} {vs['vsat']:5d} {jm:9.1f}")

        for name, extra in configs:
            Q, ms = run(frames, dict(kwargs, **extra), fps)
            report(name, Q, ms)

        if args.include_batch:
            common = dict(verbose=False, show_progress=False)
            batch_configs = [
                ("batch/uniform", BatchTrajectoryConfig(w_acceleration=40.0, **common)),
                ("batch/velAware", BatchTrajectoryConfig(w_acceleration=40.0, vel_aware_smoothing=True, vel_ref=args.vel_ref, **common)),
                ("batch/tqLimUB", BatchTrajectoryConfig(w_acceleration=10.0, torque_limit_constraint=True, torque_limit_scope="upper", torque_limit_margin=0.1, torque_limit_weight=20.0, **common)),
                ("batch/tqLimAll", BatchTrajectoryConfig(w_acceleration=10.0, torque_limit_constraint=True, torque_limit_scope="all", torque_limit_margin=0.1, torque_limit_weight=20.0, **common)),
            ]
            for name, bcfg in batch_configs:
                Q, ms = run_batch(frames, kwargs, fps, bcfg)
                report(name, Q, ms)


if __name__ == "__main__":
    main()
