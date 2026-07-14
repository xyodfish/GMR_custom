import argparse
import csv
import json
import pathlib
import sys
from itertools import product

import mujoco as mj
import numpy as np
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file


MODE_BITS = ("contact_ground", "foot_ground_limit", "fix_robot_penetration")
GROUPS = ("foot", "leg", "trunk", "arm")


def mode_name(mode):
    return "_".join(f"{key}{int(mode[key])}" for key in MODE_BITS)


def geom_name(model, geom_id):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"geom_{geom_id}"


def body_name(model, body_id):
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    return name if name is not None else f"body_{body_id}"


def split_left_right_geoms(model, geom_ids):
    sides = {"left": [], "right": []}
    for gid in geom_ids:
        bname = body_name(model, int(model.geom_bodyid[gid])).lower()
        gname = geom_name(model, gid).lower()
        text = f"{bname} {gname}"
        if "left" in text or "_l" in text or text.endswith(" l"):
            sides["left"].append(gid)
        elif "right" in text or "_r" in text or text.endswith(" r"):
            sides["right"].append(gid)
    return sides


def contact_side(contact_name):
    lower = contact_name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return None


def side_xy(data, geom_ids):
    if not geom_ids:
        return None
    xy = [data.geom_xpos[gid, :2].copy() for gid in geom_ids]
    return np.mean(xy, axis=0)


def geom_signed_distance(model, data, geom_id, floor_id):
    if floor_id < 0:
        return float(data.geom_xpos[geom_id, 2])
    fromto = np.zeros(6, dtype=np.float64)
    return float(mj.mj_geomDistance(model, data, geom_id, floor_id, 10.0, fromto))


def group_penetration_stats(model, data, floor_id, geom_ids, margin):
    if not geom_ids:
        return {
            "min_dist": np.nan,
            "max_pen": 0.0,
            "mean_pen": 0.0,
            "penetrating": False,
        }
    dists = np.array(
        [geom_signed_distance(model, data, gid, floor_id) for gid in geom_ids],
        dtype=np.float64,
    )
    depths = np.maximum(0.0, margin - dists)
    return {
        "min_dist": float(np.min(dists)),
        "max_pen": float(np.max(depths)),
        "mean_pen": float(np.mean(depths)),
        "penetrating": bool(np.any(depths > 1e-9)),
    }


def percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def mean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def max_or_zero(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def load_frames(args):
    if args.format not in ("lafan1", "nokov"):
        raise ValueError(f"Unsupported BVH format: {args.format}")
    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    return frames, actual_human_height


def discover_bvh_files(args):
    if args.bvh_files:
        return [pathlib.Path(path) for path in args.bvh_files]
    if args.bvh_dir is None:
        return [pathlib.Path(args.bvh_file)]
    bvh_files = sorted(pathlib.Path(args.bvh_dir).glob(args.bvh_pattern))
    if args.max_files is not None:
        bvh_files = bvh_files[: args.max_files]
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files matched {args.bvh_dir}/{args.bvh_pattern}")
    return bvh_files


def run_mode(args, frames, actual_human_height, mode):
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        verbose=False,
        contact_ground=mode["contact_ground"],
        foot_ground_limit=mode["foot_ground_limit"],
        fix_robot_penetration=mode["fix_robot_penetration"],
        motion_fps=args.motion_fps,
    )
    retargeter.set_motion_fps(args.motion_fps)

    model = retargeter.model
    data = retargeter.configuration.data
    pipeline = retargeter.contact_ground
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, pipeline.floor_geom_name)
    geom_groups = {
        "foot": pipeline.foot_geom_ids,
        "leg": pipeline.leg_geom_ids,
        "trunk": pipeline.trunk_geom_ids,
        "arm": pipeline.arm_geom_ids,
    }
    foot_sides = split_left_right_geoms(model, pipeline.foot_geom_ids)
    last_contact_xy = {"left": None, "right": None}
    slip_stats = {
        side: {
            "total": 0.0,
            "max_step": 0.0,
            "step_count": 0,
            "contact_frames": 0,
        }
        for side in ("left", "right")
    }
    frame_rows = []

    for frame_idx, frame in enumerate(tqdm(frames, desc=mode_name(mode), leave=False)):
        retargeter.retarget(frame)
        mj.mj_forward(model, data)

        low_pose = pipeline._is_low_pose()
        margin = pipeline.lying_penetration_margin if low_pose else pipeline.penetration_margin
        group_stats = {
            group: group_penetration_stats(model, data, floor_id, geom_ids, margin)
            for group, geom_ids in geom_groups.items()
        }
        all_depths = [group_stats[group]["max_pen"] for group in GROUPS]
        all_min_dists = [
            group_stats[group]["min_dist"]
            for group in GROUPS
            if np.isfinite(group_stats[group]["min_dist"])
        ]

        frame_slip = 0.0
        frame_slip_by_side = {"left": 0.0, "right": 0.0}
        frame_contact_sides = set()
        for contact_name, in_contact in pipeline.last_contacts.items():
            side = contact_side(contact_name)
            if side is None:
                continue
            if not in_contact:
                last_contact_xy[side] = None
                continue
            xy = side_xy(data, foot_sides[side])
            if xy is None:
                continue
            frame_contact_sides.add(side)
            slip_stats[side]["contact_frames"] += 1
            if last_contact_xy[side] is not None:
                step = float(np.linalg.norm(xy - last_contact_xy[side]))
                frame_slip_by_side[side] += step
                frame_slip += step
                slip_stats[side]["total"] += step
                slip_stats[side]["max_step"] = max(slip_stats[side]["max_step"], step)
                slip_stats[side]["step_count"] += 1
            last_contact_xy[side] = xy

        error1 = retargeter.error1() if retargeter.use_ik_match_table1 else 0.0
        error2 = retargeter.error2() if retargeter.use_ik_match_table2 else 0.0

        row = {
            "frame": frame_idx,
            "low_pose": low_pose,
            "margin": margin,
            "root_lift": float(pipeline.last_root_lift),
            "frame_xy_slip": frame_slip,
            "left_frame_xy_slip": frame_slip_by_side["left"],
            "right_frame_xy_slip": frame_slip_by_side["right"],
            "contact_sides": ",".join(sorted(frame_contact_sides)),
            "error1": float(error1),
            "error2": float(error2),
            "all_min_signed_distance": min(all_min_dists) if all_min_dists else np.nan,
            "all_max_penetration": max(all_depths) if all_depths else 0.0,
            "all_penetrating": any(group_stats[group]["penetrating"] for group in GROUPS),
        }
        for group in GROUPS:
            row[f"{group}_min_signed_distance"] = group_stats[group]["min_dist"]
            row[f"{group}_max_penetration"] = group_stats[group]["max_pen"]
            row[f"{group}_mean_penetration"] = group_stats[group]["mean_pen"]
            row[f"{group}_penetrating"] = group_stats[group]["penetrating"]
        frame_rows.append(row)

    n = len(frame_rows)
    all_max_pen = [row["all_max_penetration"] for row in frame_rows]
    root_lifts = [row["root_lift"] for row in frame_rows]
    positive_lifts = [v for v in root_lifts if v > 1e-9]
    total_slip = sum(slip_stats[side]["total"] for side in ("left", "right"))
    total_slip_steps = sum(slip_stats[side]["step_count"] for side in ("left", "right"))
    total_contact_foot_frames = sum(slip_stats[side]["contact_frames"] for side in ("left", "right"))
    max_slip_step = max(slip_stats[side]["max_step"] for side in ("left", "right"))

    summary = {
        **mode,
        "mode": mode_name(mode),
        "frames": n,
        "all_max_penetration": max_or_zero(all_max_pen),
        "all_penetration_frame_ratio": mean([row["all_penetrating"] for row in frame_rows]),
        "all_mean_penetration": mean(all_max_pen),
        "all_p95_penetration": percentile(all_max_pen, 95),
        "total_root_lift": float(np.sum(root_lifts)),
        "max_root_lift": max_or_zero(root_lifts),
        "frames_with_lift": int(sum(v > 1e-9 for v in root_lifts)),
        "mean_lift_when_active": mean(positive_lifts),
        "total_foot_slip": total_slip,
        "mean_xy_slip_per_contact_step": total_slip / max(1, total_slip_steps),
        "mean_xy_slip_per_contact_foot_frame": total_slip / max(1, total_contact_foot_frames),
        "left_total_foot_slip": slip_stats["left"]["total"],
        "right_total_foot_slip": slip_stats["right"]["total"],
        "left_mean_xy_slip_per_contact_step": slip_stats["left"]["total"] / max(1, slip_stats["left"]["step_count"]),
        "right_mean_xy_slip_per_contact_step": slip_stats["right"]["total"] / max(1, slip_stats["right"]["step_count"]),
        "left_contact_step_count": slip_stats["left"]["step_count"],
        "right_contact_step_count": slip_stats["right"]["step_count"],
        "left_contact_frame_count": slip_stats["left"]["contact_frames"],
        "right_contact_frame_count": slip_stats["right"]["contact_frames"],
        "max_xy_slip_step": max_slip_step,
        "contact_step_count": total_slip_steps,
        "contact_foot_frame_count": total_contact_foot_frames,
        "mean_error1": mean([row["error1"] for row in frame_rows]),
        "max_error1": max_or_zero([row["error1"] for row in frame_rows]),
        "mean_error2": mean([row["error2"] for row in frame_rows]),
        "max_error2": max_or_zero([row["error2"] for row in frame_rows]),
    }
    for group in GROUPS:
        values = [row[f"{group}_max_penetration"] for row in frame_rows]
        summary[f"{group}_max_penetration"] = max_or_zero(values)
        summary[f"{group}_mean_penetration"] = mean(values)
        summary[f"{group}_p95_penetration"] = percentile(values, 95)
        summary[f"{group}_penetration_frame_ratio"] = mean(
            [row[f"{group}_penetrating"] for row in frame_rows]
        )
    return summary, frame_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(rows):
    mode_rows = []
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        out = {
            key: subset[0][key]
            for key in ("contact_ground", "foot_ground_limit", "fix_robot_penetration", "mode")
        }
        out["bvh_count"] = len(subset)
        out["frames"] = sum(int(row["frames"]) for row in subset)

        weighted_mean_keys = (
            "all_penetration_frame_ratio",
            "all_mean_penetration",
            "all_p95_penetration",
            "mean_xy_slip_per_contact_step",
            "mean_xy_slip_per_contact_foot_frame",
            "mean_error1",
            "mean_error2",
            "foot_mean_penetration",
            "foot_p95_penetration",
            "foot_penetration_frame_ratio",
            "leg_mean_penetration",
            "leg_p95_penetration",
            "leg_penetration_frame_ratio",
            "trunk_mean_penetration",
            "trunk_p95_penetration",
            "trunk_penetration_frame_ratio",
            "arm_mean_penetration",
            "arm_p95_penetration",
            "arm_penetration_frame_ratio",
        )
        max_keys = (
            "all_max_penetration",
            "max_root_lift",
            "max_xy_slip_step",
            "max_error1",
            "max_error2",
            "foot_max_penetration",
            "leg_max_penetration",
            "trunk_max_penetration",
            "arm_max_penetration",
        )
        sum_keys = (
            "total_root_lift",
            "frames_with_lift",
            "total_foot_slip",
            "left_total_foot_slip",
            "right_total_foot_slip",
            "left_contact_step_count",
            "right_contact_step_count",
            "left_contact_frame_count",
            "right_contact_frame_count",
            "contact_step_count",
            "contact_foot_frame_count",
        )

        total_frames = max(1, out["frames"])
        for key in weighted_mean_keys:
            out[key] = sum(float(row[key]) * int(row["frames"]) for row in subset) / total_frames
        for key in max_keys:
            out[key] = max(float(row[key]) for row in subset)
        for key in sum_keys:
            out[key] = sum(float(row[key]) for row in subset)

        out["mean_lift_when_active"] = (
            out["total_root_lift"] / out["frames_with_lift"] if out["frames_with_lift"] > 0 else 0.0
        )
        out["left_mean_xy_slip_per_contact_step"] = (
            out["left_total_foot_slip"] / out["left_contact_step_count"]
            if out["left_contact_step_count"] > 0
            else 0.0
        )
        out["right_mean_xy_slip_per_contact_step"] = (
            out["right_total_foot_slip"] / out["right_contact_step_count"]
            if out["right_contact_step_count"] > 0
            else 0.0
        )
        mode_rows.append(out)
    return mode_rows


def print_summary(rows):
    ranked = sorted(rows, key=lambda r: (r["all_max_penetration"], r["total_foot_slip"]))
    header = (
        "mode",
        "all_max_pen",
        "all_pen_ratio",
        "leg_max",
        "trunk_max",
        "arm_max",
        "root_lift",
        "foot_slip",
        "mean_slip_step",
        "mean_e1",
        "mean_e2",
    )
    print(",".join(header))
    for row in ranked:
        print(
            ",".join(
                [
                    row["mode"],
                    f"{row['all_max_penetration']:.5f}",
                    f"{row['all_penetration_frame_ratio']:.3f}",
                    f"{row['leg_max_penetration']:.5f}",
                    f"{row['trunk_max_penetration']:.5f}",
                    f"{row['arm_max_penetration']:.5f}",
                    f"{row['total_root_lift']:.5f}",
                    f"{row['total_foot_slip']:.5f}",
                    f"{row['mean_xy_slip_per_contact_step']:.6f}",
                    f"{row['mean_error1']:.5f}",
                    f"{row['mean_error2']:.5f}",
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze contact/ground retargeting mode combinations.")
    parser.add_argument(
        "--bvh_file",
        default="/data2/Documents/lafan1/fallAndGetUp1_subject4.bvh",
    )
    parser.add_argument("--bvh_dir", default=None)
    parser.add_argument("--bvh_files", nargs="*", default=None)
    parser.add_argument("--bvh_pattern", default="*.bvh")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=float, default=30.0)
    parser.add_argument("--out_dir", default="analysis/contact_modes")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--write_frames", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bvh_files = discover_bvh_files(args)
    summary_rows = []
    summary_json = []
    modes = [
        {
            "contact_ground": contact_ground,
            "foot_ground_limit": foot_ground_limit,
            "fix_robot_penetration": fix_robot_penetration,
        }
        for contact_ground, foot_ground_limit, fix_robot_penetration in product([False, True], repeat=3)
    ]
    for bvh_file in tqdm(bvh_files, desc="bvh"):
        args.bvh_file = str(bvh_file)
        frames, actual_human_height = load_frames(args)
        bvh_name = pathlib.Path(args.bvh_file).stem
        for mode in modes:
            summary, frame_rows = run_mode(args, frames, actual_human_height, mode)
            summary = {"bvh_file": str(bvh_file), "bvh_name": bvh_name, **summary}
            summary_rows.append(summary)
            item = {"summary": summary}
            if args.write_frames:
                frames_csv = f"frames_{bvh_name}_{summary['mode']}.csv"
                write_csv(out_dir / frames_csv, frame_rows)
                item["frames_csv"] = frames_csv
            summary_json.append(item)

    aggregate_rows = aggregate_summaries(summary_rows)
    write_csv(out_dir / "per_bvh_summary.csv", summary_rows)
    write_csv(out_dir / "summary.csv", aggregate_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bvh_files": [str(path) for path in bvh_files],
                "aggregate": aggregate_rows,
                "per_bvh": summary_json,
            },
            f,
            indent=2,
        )
    print_summary(aggregate_rows)
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
