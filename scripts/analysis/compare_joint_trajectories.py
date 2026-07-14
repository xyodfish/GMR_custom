"""Compare retargeting methods via joint trajectory curves.

Example (two saved motions):
    python scripts/analysis/compare_joint_trajectories.py \\
        --robot unitree_g1 \\
        --baseline output/walking_ik.pkl \\
        --candidate output/walking_sw.pkl \\
        --labels "per-frame IK" "sliding-window" \\
        --output output/walking_joint_compare.png

Example (run GVHMR offline, then plot):
    python scripts/analysis/compare_joint_trajectories.py \\
        --robot unitree_g1 \\
        --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \\
        --contact_ground \\
        --output output/walking_joint_compare.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.params import ROBOT_XML_DICT


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for trajectory plots. Install with: pip install matplotlib"
        ) from exc
    return plt


def get_dof_names(robot: str) -> list[str]:
    import mujoco as mj

    xml_path = ROBOT_XML_DICT[robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    names: list[str] = []
    for i in range(model.nv):
        jnt_id = model.dof_jntid[i]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        names.append(name if name is not None else f"dof_{i}")
    return names


def load_dof_trajectory(motion_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, float]:
    _, fps, root_pos, root_rot, dof_pos, _, _, qpos = load_robot_motion(str(motion_path))
    root_pos = np.asarray(root_pos)
    root_rot = np.asarray(root_rot)
    dof_pos = np.asarray(dof_pos)
    if qpos is not None:
        traj = np.asarray(qpos)
    elif root_rot.shape[1] == 4:
        traj = np.hstack([root_pos, root_rot, dof_pos])
    else:
        traj = np.hstack([root_pos, dof_pos])
    n = len(traj)
    t = np.arange(n, dtype=float) / float(fps)
    return t, traj, float(fps)


def finite_diff(x: np.ndarray, fps: float, order: int) -> np.ndarray:
    if order == 0:
        return x
    dt = 1.0 / fps
    out = np.gradient(x, dt, axis=0, edge_order=1)
    if order == 1:
        return out
    return np.gradient(out, dt, axis=0, edge_order=1)


def per_joint_metrics(a: np.ndarray, b: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    vel_a = finite_diff(a, fps, 1)
    vel_b = finite_diff(b, fps, 1)
    acc_a = finite_diff(a, fps, 2)
    acc_b = finite_diff(b, fps, 2)
    return {
        "vel_rmse": np.sqrt(np.mean((vel_a - vel_b) ** 2, axis=0)),
        "acc_rmse": np.sqrt(np.mean((acc_a - acc_b) ** 2, axis=0)),
        "vel_std_a": np.std(vel_a, axis=0),
        "vel_std_b": np.std(vel_b, axis=0),
        "acc_std_a": np.std(acc_a, axis=0),
        "acc_std_b": np.std(acc_b, axis=0),
    }


def run_gvhmr_compare(
    gvhmr_pred_file: pathlib.Path,
    robot: str,
    contact_ground: bool | None,
    foot_ground_limit: bool | None,
    fix_robot_penetration: bool | None,
    window_size: int,
    w_velocity: float,
    w_acceleration: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.sliding_window_retarget import (
        SlidingWindowConfig,
        SlidingWindowRetargeter,
    )
    from general_motion_retargeting.utils.smpl import (
        load_gvhmr_pred_file,
        get_gvhmr_data_offline_fast,
    )

    body_model_dir = ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        gvhmr_pred_file, body_model_dir
    )
    human_frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )
    ik = GMR(**gmr_kwargs)
    sw = SlidingWindowRetargeter(
        GMR(**gmr_kwargs),
        SlidingWindowConfig(
            window_size=window_size,
            w_velocity=w_velocity,
            w_acceleration=w_acceleration,
        ),
    )

    ik_qpos, sw_qpos = [], []
    for frame in human_frames:
        ik_qpos.append(ik.retarget(frame).copy())
        sw_qpos.append(sw.retarget(frame).copy())

    dof_names = get_dof_names(robot)
    return np.asarray(ik_qpos), np.asarray(sw_qpos), float(fps), dof_names


def run_smplx_compare(
    smplx_file: pathlib.Path,
    robot: str,
    contact_ground: bool | None,
    foot_ground_limit: bool | None,
    fix_robot_penetration: bool | None,
    window_size: int,
    w_velocity: float,
    w_acceleration: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.sliding_window_retarget import (
        SlidingWindowConfig,
        SlidingWindowRetargeter,
    )
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    body_model_dir = ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, body_model_dir
    )
    human_frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )
    ik = GMR(**gmr_kwargs)
    sw = SlidingWindowRetargeter(
        GMR(**gmr_kwargs),
        SlidingWindowConfig(
            window_size=window_size,
            w_velocity=w_velocity,
            w_acceleration=w_acceleration,
        ),
    )

    ik_qpos, sw_qpos = [], []
    for frame in human_frames:
        ik_qpos.append(ik.retarget(frame).copy())
        sw_qpos.append(sw.retarget(frame).copy())

    dof_names = get_dof_names(robot)
    return np.asarray(ik_qpos), np.asarray(sw_qpos), float(fps), dof_names


def select_joint_indices(
    dof_names: list[str],
    joints: list[str] | None,
    top_k: int,
    metrics: dict[str, np.ndarray],
) -> list[int]:
    if joints:
        wanted = {j.lower() for j in joints}
        idx = [i for i, n in enumerate(dof_names) if n.lower() in wanted]
        if idx:
            return idx
    if top_k > 0:
        score = metrics["vel_rmse"] + metrics["acc_rmse"]
        return np.argsort(score)[::-1][:top_k].tolist()
    return list(range(len(dof_names)))


def plot_joint_curves(
    t: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    dof_names: list[str],
    joint_indices: list[int],
    labels: tuple[str, str],
    output: pathlib.Path,
    derivative: int,
) -> None:
    plt = _require_matplotlib()

    n = len(joint_indices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 2.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    fps = 1.0 / np.mean(np.diff(t)) if len(t) > 1 else 30.0
    y_a = finite_diff(baseline, fps, derivative)
    y_b = finite_diff(candidate, fps, derivative)

    ylabel = {0: "position", 1: "velocity", 2: "acceleration"}[derivative]

    for ax, j in zip(axes, joint_indices):
        name = dof_names[j] if j < len(dof_names) else f"dof_{j}"
        ax.plot(t, y_a[:, j], label=labels[0], linewidth=1.2, alpha=0.9)
        ax.plot(t, y_b[:, j], label=labels[1], linewidth=1.2, alpha=0.9, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("time (s)")
    fig.supylabel(ylabel)
    fig.suptitle(f"Joint {ylabel}: {labels[0]} vs {labels[1]}", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output}")


def print_metric_table(
    dof_names: list[str],
    metrics: dict[str, np.ndarray],
    joint_indices: list[int],
) -> None:
    print("\nPer-joint smoothness (std) and method difference (RMSE):")
    print(f"{'joint':32s}  vel_std_IK  vel_std_SW  acc_std_IK  acc_std_SW  vel_rmse  acc_rmse")
    for j in joint_indices:
        name = dof_names[j] if j < len(dof_names) else f"dof_{j}"
        print(
            f"{name:32s}  "
            f"{metrics['vel_std_a'][j]:9.5f}  {metrics['vel_std_b'][j]:9.5f}  "
            f"{metrics['acc_std_a'][j]:9.5f}  {metrics['acc_std_b'][j]:9.5f}  "
            f"{metrics['vel_rmse'][j]:9.5f}  {metrics['acc_rmse'][j]:9.5f}"
        )


def add_optional_bool_arg(parser, name, help_text):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: None})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot joint trajectory comparison curves.")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline motion .pkl")
    parser.add_argument("--candidate", type=str, default=None, help="Candidate motion .pkl")
    parser.add_argument("--gvhmr_pred_file", type=str, default=None)
    parser.add_argument("--smplx_file", type=str, default=None)
    parser.add_argument("--labels", nargs=2, default=["per-frame IK", "sliding-window"])
    parser.add_argument("--output", type=str, default="output/joint_trajectory_compare.png")
    parser.add_argument(
        "--plot",
        choices=["position", "velocity", "acceleration", "all"],
        default="all",
    )
    parser.add_argument(
        "--joints",
        nargs="*",
        default=None,
        help="Optional joint name substrings to plot.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Plot top-K joints with largest vel/acc RMSE between methods (default 12).",
    )
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--w_velocity", type=float, default=2.0)
    parser.add_argument("--w_acceleration", type=float, default=10.0)
    add_optional_bool_arg(parser, "contact_ground", "Enable contact_ground during offline run.")
    add_optional_bool_arg(parser, "foot_ground_limit", "Enable foot_ground_limit.")
    add_optional_bool_arg(parser, "fix_robot_penetration", "Enable fix_robot_penetration.")
    args = parser.parse_args()

    dof_names: list[str] | None = None

    if args.baseline and args.candidate:
        t_a, traj_a, fps_a = load_dof_trajectory(pathlib.Path(args.baseline))
        t_b, traj_b, fps_b = load_dof_trajectory(pathlib.Path(args.candidate))
        n = min(len(traj_a), len(traj_b))
        traj_a, traj_b = traj_a[:n], traj_b[:n]
        t = t_a[:n]
        fps = fps_a
        dof_names = get_dof_names(args.robot)
    elif args.gvhmr_pred_file:
        print(f"Running offline compare on GVHMR: {args.gvhmr_pred_file}")
        traj_a, traj_b, fps, dof_names = run_gvhmr_compare(
            pathlib.Path(args.gvhmr_pred_file),
            args.robot,
            args.contact_ground,
            args.foot_ground_limit,
            args.fix_robot_penetration,
            args.window_size,
            args.w_velocity,
            args.w_acceleration,
        )
        n = len(traj_a)
        t = np.arange(n, dtype=float) / fps
    elif args.smplx_file:
        print(f"Running offline compare on SMPL-X: {args.smplx_file}")
        traj_a, traj_b, fps, dof_names = run_smplx_compare(
            pathlib.Path(args.smplx_file),
            args.robot,
            args.contact_ground,
            args.foot_ground_limit,
            args.fix_robot_penetration,
            args.window_size,
            args.w_velocity,
            args.w_acceleration,
        )
        n = len(traj_a)
        t = np.arange(n, dtype=float) / fps
    else:
        parser.error("Provide --baseline/--candidate, or --gvhmr_pred_file, or --smplx_file.")

    width = min(traj_a.shape[1], traj_b.shape[1], len(dof_names))
    traj_a = traj_a[:, :width]
    traj_b = traj_b[:, :width]
    dof_names = dof_names[:width]
    if traj_a.shape[1] != len(dof_names):
        dof_names = [f"qpos_{i}" for i in range(traj_a.shape[1])]

    metrics = per_joint_metrics(traj_a, traj_b, fps)
    joint_indices = select_joint_indices(dof_names, args.joints, args.top_k, metrics)
    print_metric_table(dof_names, metrics, joint_indices)

    out = pathlib.Path(args.output)
    stem = out.stem
    suffix = out.suffix or ".png"
    parent = out.parent

    plot_map = {
        "position": 0,
        "velocity": 1,
        "acceleration": 2,
    }
    to_plot = list(plot_map.keys()) if args.plot == "all" else [args.plot]
    for name in to_plot:
        plot_joint_curves(
            t,
            traj_a,
            traj_b,
            dof_names,
            joint_indices,
            tuple(args.labels),
            parent / f"{stem}_{name}{suffix}",
            plot_map[name],
        )


if __name__ == "__main__":
    main()
