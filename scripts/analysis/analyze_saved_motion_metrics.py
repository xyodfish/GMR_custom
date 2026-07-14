import argparse
import json
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FOOT_BODIES = {
    "unitree_g1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1_with_hands": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_h2": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "booster_t1_29dof": ["left_foot_link", "right_foot_link"],
    "fourier_n1": ["left_foot_roll_link", "right_foot_roll_link"],
    "engineai_pm01": ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
    "stanford_toddy": ["ank_roll_link", "ank_roll_link_2"],
    "pal_talos": ["leg_left_6_link", "leg_right_6_link"],
}

ROBOT_XML_PATHS = {
    "unitree_g1": ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_g1_with_hands": ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof_with_hands.xml",
    "unitree_h2": ROOT / "assets" / "unitree_h2" / "h2.xml",
    "booster_t1_29dof": ROOT / "assets" / "booster_t1_29dof" / "t1_mocap.xml",
    "stanford_toddy": ROOT / "assets" / "stanford_toddy" / "toddy_mocap.xml",
    "fourier_n1": ROOT / "assets" / "fourier_n1" / "n1_mocap.xml",
    "engineai_pm01": ROOT / "assets" / "engineai_pm01" / "pm_v2.xml",
    "pal_talos": ROOT / "assets" / "pal_talos" / "talos.xml",
}


def load_qpos(path):
    with open(path, "rb") as f:
        motion = pickle.load(f)
    fps = float(motion.get("fps", 30))
    qpos = motion.get("qpos")
    if qpos is not None:
        return np.asarray(qpos), float(fps)
    root_pos = motion["root_pos"]
    root_rot = motion["root_rot"]
    dof_pos = motion["dof_pos"]
    root_pos = np.asarray(root_pos)
    root_rot = np.asarray(root_rot)
    dof_pos = np.asarray(dof_pos)
    if root_rot.shape[1] == 4:
        qpos = np.hstack([root_pos, root_rot[:, [3, 0, 1, 2]], dof_pos])
    else:
        qpos = np.hstack([root_pos, dof_pos])
    return qpos, float(fps)


def scalar_q_indices(model):
    indices = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            indices.append(int(model.jnt_qposadr[jid]))
    return np.asarray(indices, dtype=int)


def percentile(values, p):
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, p))


def norm_stats(x):
    x = np.asarray(x)
    if x.size == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    vals = np.linalg.norm(x, axis=1)
    return {
        "mean": float(np.mean(vals)),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "max": float(np.max(vals)),
    }


def smoothness_metrics(qpos, fps, q_indices):
    dt = 1.0 / fps
    q = qpos[:, q_indices]
    v = np.diff(q, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return {
        "dq": norm_stats(v),
        "ddq": norm_stats(a),
        "jerk": norm_stats(j),
    }


def foot_positions(model, qpos, foot_body_names):
    data = mj.MjData(model)
    body_ids = [model.body(name).id for name in foot_body_names]
    out = []
    for q in qpos:
        data.qpos[:] = q
        mj.mj_forward(model, data)
        out.append(data.xpos[body_ids].copy())
    return np.asarray(out)


def foot_slip_metrics(model, qpos, foot_body_names, contact_height_margin):
    pos = foot_positions(model, qpos, foot_body_names)
    z_min = np.min(pos[:, :, 2], axis=0)
    contact = pos[:, :, 2] <= (z_min[None, :] + contact_height_margin)
    xy_step = np.linalg.norm(np.diff(pos[:, :, :2], axis=0), axis=2)
    contact_step = contact[1:] & contact[:-1]
    slip_steps = xy_step[contact_step]
    total_by_foot = np.sum(xy_step * contact_step, axis=0)
    frames_by_foot = np.sum(contact, axis=0)
    steps_by_foot = np.sum(contact_step, axis=0)
    return {
        "total_slip": float(np.sum(slip_steps)) if slip_steps.size else 0.0,
        "mean_slip_per_contact_step": float(np.mean(slip_steps)) if slip_steps.size else 0.0,
        "p95_slip_step": percentile(slip_steps, 95),
        "max_slip_step": float(np.max(slip_steps)) if slip_steps.size else 0.0,
        "contact_step_count": int(np.sum(contact_step)),
        "contact_frame_count": int(np.sum(contact)),
        "foot_bodies": foot_body_names,
        "per_foot_total_slip": {name: float(v) for name, v in zip(foot_body_names, total_by_foot)},
        "per_foot_contact_frames": {name: int(v) for name, v in zip(foot_body_names, frames_by_foot)},
        "per_foot_contact_steps": {name: int(v) for name, v in zip(foot_body_names, steps_by_foot)},
    }


def analyze_motion(robot, path, foot_body_names, contact_height_margin):
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_PATHS[robot]))
    qpos, fps = load_qpos(path)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{path}: qpos width {qpos.shape[1]} does not match model nq {model.nq}")
    q_indices = scalar_q_indices(model)
    return {
        "path": str(path),
        "frames": int(len(qpos)),
        "fps": fps,
        "smoothness": smoothness_metrics(qpos, fps, q_indices),
        "foot_slip": foot_slip_metrics(model, qpos, foot_body_names, contact_height_margin),
    }


def print_summary(label, metrics):
    slip = metrics["foot_slip"]
    sm = metrics["smoothness"]
    print(f"\n[{label}] {metrics['path']}")
    print(f"  frames={metrics['frames']} fps={metrics['fps']:.1f}")
    print(
        "  foot slip: "
        f"total={slip['total_slip']:.4f} m, "
        f"mean/contact-step={slip['mean_slip_per_contact_step']:.6f} m, "
        f"p95-step={slip['p95_slip_step']:.6f} m, "
        f"max-step={slip['max_slip_step']:.6f} m, "
        f"contact_steps={slip['contact_step_count']}"
    )
    for key in ("dq", "ddq", "jerk"):
        s = sm[key]
        print(
            f"  {key}: mean={s['mean']:.4f}, p95={s['p95']:.4f}, "
            f"p99={s['p99']:.4f}, max={s['max']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze saved retargeted motion smoothness and foot slip.")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument("--compare", type=pathlib.Path, default=None)
    parser.add_argument("--labels", nargs=2, default=["motion", "compare"])
    parser.add_argument("--foot_bodies", nargs="+", default=None)
    parser.add_argument("--contact_height_margin", type=float, default=0.035)
    parser.add_argument("--json_out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    foot_body_names = args.foot_bodies or DEFAULT_FOOT_BODIES.get(args.robot)
    if not foot_body_names:
        raise ValueError(f"No default foot bodies for robot={args.robot}. Pass --foot_bodies.")

    primary = analyze_motion(args.robot, args.motion, foot_body_names, args.contact_height_margin)
    print_summary(args.labels[0], primary)

    out = {"primary": primary}
    if args.compare is not None:
        compare = analyze_motion(args.robot, args.compare, foot_body_names, args.contact_height_margin)
        print_summary(args.labels[1], compare)
        out["compare"] = compare

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {args.json_out}")
