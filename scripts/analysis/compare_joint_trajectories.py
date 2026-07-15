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
    out = x
    for _ in range(order):
        out = np.gradient(out, dt, axis=0, edge_order=1)
    return out


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
    candidate_method: str = "sw",
    max_frames: int | None = None,
    max_opt_iter: int = 25,
    fast_opt_iter: int = 5,
    to_mode: str = "fast",
    fix_window_prefix: bool = False,
    use_gmr_init: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.sliding_window_retarget import (
        SlidingWindowConfig,
        SlidingWindowRetargeter,
    )
    from general_motion_retargeting.trajectory_optimization_retarget import (
        TrajectoryOptimizationConfig,
        TrajectoryOptimizationRetargeter,
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
    if max_frames is not None:
        human_frames = human_frames[:max_frames]

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
    if candidate_method == "sw":
        candidate = SlidingWindowRetargeter(
            GMR(**gmr_kwargs),
            SlidingWindowConfig(
                window_size=window_size,
                solver="gn",
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                dt=1.0 / fps,
            ),
        )
    elif candidate_method == "to":
        mode = "fast" if fix_window_prefix or to_mode == "fast" else "full"
        candidate = TrajectoryOptimizationRetargeter(
            GMR(**gmr_kwargs),
            TrajectoryOptimizationConfig(
                window_size=window_size,
                mode=mode,
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                max_opt_iter=max_opt_iter,
                fast_opt_iter=fast_opt_iter,
                use_gmr_init=use_gmr_init,
            ),
        )
    else:
        raise ValueError(f"Unknown candidate_method: {candidate_method}")

    ik_qpos, cand_qpos = [], []
    for i, frame in enumerate(human_frames):
        ik_qpos.append(ik.retarget(frame).copy())
        cand_qpos.append(candidate.retarget(frame).copy())
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(human_frames)} frames")

    dof_names = get_dof_names(robot)
    return np.asarray(ik_qpos), np.asarray(cand_qpos), float(fps), dof_names


def run_bvh_compare(
    bvh_file: pathlib.Path,
    robot: str,
    bvh_format: str,
    contact_ground: bool | None,
    foot_ground_limit: bool | None,
    fix_robot_penetration: bool | None,
    motion_fps: int,
    window_size: int,
    w_velocity: float,
    w_acceleration: float,
    candidate_method: str = "sw",
    solver: str = "gn",
    gn_steps: int = 3,
    max_frames: int | None = None,
    start_frame: int = 0,
    max_opt_iter: int = 25,
    fast_opt_iter: int = 5,
    to_mode: str = "fast",
    fix_window_prefix: bool = False,
    use_gmr_init: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.sliding_window_retarget import (
        SlidingWindowConfig,
        SlidingWindowRetargeter,
    )
    from general_motion_retargeting.trajectory_optimization_retarget import (
        TrajectoryOptimizationConfig,
        TrajectoryOptimizationRetargeter,
    )
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    human_frames, actual_human_height = load_bvh_file(str(bvh_file), format=bvh_format)
    fps = float(motion_fps)
    dt = 1.0 / fps
    end = len(human_frames) if max_frames is None else min(len(human_frames), start_frame + max_frames)
    human_frames = human_frames[start_frame:end]
    src_human = f"bvh_{bvh_format}"

    gmr_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        foot_ground_limit=foot_ground_limit,
        fix_robot_penetration=fix_robot_penetration,
        motion_fps=fps,
    )
    ik = GMR(**gmr_kwargs)
    ik.set_motion_fps(fps)
    if candidate_method == "sw":
        candidate = SlidingWindowRetargeter(
            GMR(**gmr_kwargs),
            SlidingWindowConfig(
                window_size=window_size,
                solver=solver,
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                gn_steps=gn_steps,
                dt=dt,
            ),
        )
    elif candidate_method == "to":
        mode = "fast" if fix_window_prefix or to_mode == "fast" else "full"
        candidate = TrajectoryOptimizationRetargeter(
            GMR(**gmr_kwargs),
            TrajectoryOptimizationConfig(
                window_size=window_size,
                mode=mode,
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                max_opt_iter=max_opt_iter,
                fast_opt_iter=fast_opt_iter,
                use_gmr_init=use_gmr_init,
            ),
        )
        candidate.set_motion_fps(fps)
    else:
        raise ValueError(f"Unknown candidate_method: {candidate_method}")

    ik_qpos, cand_qpos = [], []
    for i, frame in enumerate(human_frames):
        ik_qpos.append(ik.retarget(frame).copy())
        cand_qpos.append(candidate.retarget(frame).copy())
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(human_frames)} frames")

    dof_names = get_dof_names(robot)
    return np.asarray(ik_qpos), np.asarray(cand_qpos), fps, dof_names


def run_smplx_compare(
    smplx_file: pathlib.Path,
    robot: str,
    contact_ground: bool | None,
    foot_ground_limit: bool | None,
    fix_robot_penetration: bool | None,
    window_size: int,
    w_velocity: float,
    w_acceleration: float,
    candidate_method: str = "sw",
    max_frames: int | None = None,
    max_opt_iter: int = 25,
    fast_opt_iter: int = 5,
    to_mode: str = "fast",
    fix_window_prefix: bool = False,
    use_gmr_init: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.sliding_window_retarget import (
        SlidingWindowConfig,
        SlidingWindowRetargeter,
    )
    from general_motion_retargeting.trajectory_optimization_retarget import (
        TrajectoryOptimizationConfig,
        TrajectoryOptimizationRetargeter,
    )
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    body_model_dir = ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, body_model_dir
    )
    human_frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    if max_frames is not None:
        human_frames = human_frames[:max_frames]

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
    if candidate_method == "sw":
        candidate = SlidingWindowRetargeter(
            GMR(**gmr_kwargs),
            SlidingWindowConfig(
                window_size=window_size,
                solver="gn",
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                dt=1.0 / fps,
            ),
        )
    elif candidate_method == "to":
        mode = "fast" if fix_window_prefix or to_mode == "fast" else "full"
        candidate = TrajectoryOptimizationRetargeter(
            GMR(**gmr_kwargs),
            TrajectoryOptimizationConfig(
                window_size=window_size,
                mode=mode,
                w_velocity=w_velocity,
                w_acceleration=w_acceleration,
                max_opt_iter=max_opt_iter,
                fast_opt_iter=fast_opt_iter,
                use_gmr_init=use_gmr_init,
            ),
        )
    else:
        raise ValueError(f"Unknown candidate_method: {candidate_method}")

    ik_qpos, cand_qpos = [], []
    for i, frame in enumerate(human_frames):
        ik_qpos.append(ik.retarget(frame).copy())
        cand_qpos.append(candidate.retarget(frame).copy())
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(human_frames)} frames")

    dof_names = get_dof_names(robot)
    return np.asarray(ik_qpos), np.asarray(cand_qpos), float(fps), dof_names


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

    ylabel = {0: "position", 1: "velocity", 2: "acceleration", 3: "jerk"}[derivative]

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


def global_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    fps: float,
    labels: tuple[str, str],
) -> dict:
    dt = 1.0 / fps
    vel_a = finite_diff(baseline, fps, 1)
    vel_b = finite_diff(candidate, fps, 1)
    acc_a = finite_diff(baseline, fps, 2)
    acc_b = finite_diff(candidate, fps, 2)
    jerk_a = finite_diff(baseline, fps, 3)
    jerk_b = finite_diff(candidate, fps, 3)

    def _norm_mean(x: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        return float(np.mean(np.linalg.norm(x, axis=1)))

    qpos_rmse = float(np.sqrt(np.mean((baseline - candidate) ** 2)))
    qpos_mae = float(np.mean(np.linalg.norm(baseline - candidate, axis=1)))

    return {
        "fps": fps,
        "n_frames": int(len(baseline)),
        "labels": list(labels),
        labels[0]: {
            "vel_mean": _norm_mean(vel_a),
            "acc_mean": _norm_mean(acc_a),
            "jerk_mean": _norm_mean(jerk_a),
        },
        labels[1]: {
            "vel_mean": _norm_mean(vel_b),
            "acc_mean": _norm_mean(acc_b),
            "jerk_mean": _norm_mean(jerk_b),
        },
        "delta_pct": {
            "vel": 100.0 * (_norm_mean(vel_b) - _norm_mean(vel_a)) / max(_norm_mean(vel_a), 1e-9),
            "acc": 100.0 * (_norm_mean(acc_b) - _norm_mean(acc_a)) / max(_norm_mean(acc_a), 1e-9),
            "jerk": 100.0 * (_norm_mean(jerk_b) - _norm_mean(jerk_a)) / max(_norm_mean(jerk_a), 1e-9),
        },
        "qpos_rmse": qpos_rmse,
        "qpos_mae": qpos_mae,
        "per_joint_vel_rmse_mean": float(np.mean(per_joint_metrics(baseline, candidate, fps)["vel_rmse"])),
        "per_joint_acc_rmse_mean": float(np.mean(per_joint_metrics(baseline, candidate, fps)["acc_rmse"])),
    }


def print_global_summary(summary: dict) -> None:
    la, lb = summary["labels"]
    print("\nGlobal trajectory metrics:")
    print(f"  frames={summary['n_frames']}  fps={summary['fps']:.1f}")
    print(f"  qpos RMSE={summary['qpos_rmse']:.4f}  MAE={summary['qpos_mae']:.4f}")
    print(f"  {la:18s}  vel={summary[la]['vel_mean']:.5f}  acc={summary[la]['acc_mean']:.5f}  jerk={summary[la]['jerk_mean']:.5f}")
    print(f"  {lb:18s}  vel={summary[lb]['vel_mean']:.5f}  acc={summary[lb]['acc_mean']:.5f}  jerk={summary[lb]['jerk_mean']:.5f}")
    print(
        f"  {lb} vs {la}:  "
        f"vel {summary['delta_pct']['vel']:+.1f}%  "
        f"acc {summary['delta_pct']['acc']:+.1f}%  "
        f"jerk {summary['delta_pct']['jerk']:+.1f}%"
    )


def save_motion_pkl(path: pathlib.Path, qpos: np.ndarray, fps: float, method: str) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "fps": fps,
                "root_pos": qpos[:, :3],
                "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
                "dof_pos": qpos[:, 7:],
                "local_body_pos": None,
                "link_body_list": None,
                "qpos": qpos,
                "method": method,
            },
            f,
        )
    print(f"Saved motion: {path}")


def print_metric_table(
    dof_names: list[str],
    metrics: dict[str, np.ndarray],
    joint_indices: list[int],
    labels: tuple[str, str] = ("baseline", "candidate"),
) -> None:
    la, lb = labels
    print("\nPer-joint smoothness (std) and method difference (RMSE):")
    print(
        f"{'joint':32s}  "
        f"{'vel_'+la[:6]:>10s}  {'vel_'+lb[:6]:>10s}  "
        f"{'acc_'+la[:6]:>10s}  {'acc_'+lb[:6]:>10s}  "
        f"vel_rmse  acc_rmse"
    )
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
    parser.add_argument("--bvh_file", type=str, default=None)
    parser.add_argument("--bvh_format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--labels", nargs=2, default=["per-frame IK", "sliding-window"])
    parser.add_argument("--output", type=str, default="output/joint_trajectory_compare.png")
    parser.add_argument(
        "--plot",
        choices=["position", "velocity", "acceleration", "jerk", "all"],
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
    parser.add_argument(
        "--candidate_method",
        choices=["sw", "to"],
        default="sw",
        help="Offline candidate retargeter when using --gvhmr_pred_file / --smplx_file / --bvh_file.",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_opt_iter", type=int, default=25)
    parser.add_argument("--fast_opt_iter", type=int, default=5)
    parser.add_argument("--to_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--fix_window_prefix", action="store_true", default=False)
    parser.add_argument("--use_gmr_init", action="store_true", default=True)
    parser.add_argument("--no-use_gmr_init", dest="use_gmr_init", action="store_false")
    parser.add_argument("--save_motions", action="store_true", default=False)
    parser.add_argument("--metrics_json", type=str, default=None)
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
            args.candidate_method,
            args.max_frames,
            args.max_opt_iter,
            args.fast_opt_iter,
            args.to_mode,
            args.fix_window_prefix,
            args.use_gmr_init,
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
            args.candidate_method,
            args.max_frames,
            args.max_opt_iter,
            args.fast_opt_iter,
            args.to_mode,
            args.fix_window_prefix,
            args.use_gmr_init,
        )
        n = len(traj_a)
        t = np.arange(n, dtype=float) / fps
    elif args.bvh_file:
        print(f"Running offline compare on BVH: {args.bvh_file}")
        traj_a, traj_b, fps, dof_names = run_bvh_compare(
            pathlib.Path(args.bvh_file),
            args.robot,
            args.bvh_format,
            args.contact_ground,
            args.foot_ground_limit,
            args.fix_robot_penetration,
            args.motion_fps,
            args.window_size,
            args.w_velocity,
            args.w_acceleration,
            args.candidate_method,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            max_opt_iter=args.max_opt_iter,
            fast_opt_iter=args.fast_opt_iter,
            to_mode=args.to_mode,
            fix_window_prefix=args.fix_window_prefix,
            use_gmr_init=args.use_gmr_init,
        )
        n = len(traj_a)
        t = np.arange(n, dtype=float) / fps
    else:
        parser.error(
            "Provide --baseline/--candidate, or --gvhmr_pred_file, or --smplx_file, or --bvh_file."
        )

    width = min(traj_a.shape[1], traj_b.shape[1], len(dof_names))
    traj_a = traj_a[:, :width]
    traj_b = traj_b[:, :width]
    dof_names = dof_names[:width]
    if traj_a.shape[1] != len(dof_names):
        dof_names = [f"qpos_{i}" for i in range(traj_a.shape[1])]

    labels = tuple(args.labels)
    metrics = per_joint_metrics(traj_a, traj_b, fps)
    summary = global_metrics(traj_a, traj_b, fps, labels)
    joint_indices = select_joint_indices(dof_names, args.joints, args.top_k, metrics)
    print_global_summary(summary)
    print_metric_table(dof_names, metrics, joint_indices, labels)

    out = pathlib.Path(args.output)
    stem = out.stem
    suffix = out.suffix or ".png"
    parent = out.parent

    if args.metrics_json:
        import json

        metrics_path = pathlib.Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"global": summary, "per_joint_indices": joint_indices}, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    if args.save_motions:
        save_motion_pkl(parent / f"{stem}_ik.pkl", traj_a, fps, labels[0])
        save_motion_pkl(parent / f"{stem}_to.pkl", traj_b, fps, labels[1])

    plot_map = {
        "position": 0,
        "velocity": 1,
        "acceleration": 2,
        "jerk": 3,
    }
    to_plot = list(plot_map.keys()) if args.plot == "all" else [args.plot]
    for name in to_plot:
        plot_joint_curves(
            t,
            traj_a,
            traj_b,
            dof_names,
            joint_indices,
            labels,
            parent / f"{stem}_{name}{suffix}",
            plot_map[name],
        )


if __name__ == "__main__":
    main()
