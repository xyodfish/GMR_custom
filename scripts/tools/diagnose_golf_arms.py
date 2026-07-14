#!/usr/bin/env python3
"""Diagnose galbot_one_golf arm retargeting errors per link."""
from __future__ import annotations

import numpy as np
import mujoco as mj
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

REPO = Path(__file__).resolve().parents[2]


def pe(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def re(hq, mat):
    return float(
        np.degrees(
            (R.from_quat([hq[1], hq[2], hq[3], hq[0]]).inv() * R.from_matrix(mat.reshape(3, 3))).magnitude()
        )
    )


def angle(v1, v2):
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))))


def diagnose(npz: str, label: str, frame_idx: int | None = None):
    sd, bm, out, h = load_smplx_file(npz, REPO / "assets/body_models")
    frames, _ = get_smplx_data_offline_fast(sd, bm, out, tgt_fps=30)
    f = frames[0] if frame_idx is None else frames[frame_idx]

    g = GeneralMotionRetargeting("smplx", "galbot_one_golf", actual_human_height=h, verbose=False)
    g.retarget(f)
    mj.mj_forward(g.model, g.configuration.data)
    g.update_targets(f)
    sd3 = g.scaled_human_data
    d2 = g.configuration.data
    bn = g.robot_body_names

    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    for side in ["left", "right"]:
        sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        print(f"\n  [{side} arm]")
        rows = [
            (sh, f"{side}_arm_link1", "肩支架 link1"),
            (sh, f"{side}_arm_link3", "上臂末端 link3"),
            (el, f"{side}_arm_link4", "肘旋转 link4"),
            (el, f"{side}_arm_link5", "肘位置 link5"),
            (wr, f"{side}_arm_link7", "腕 link7"),
        ]
        for hb, rb, desc in rows:
            print(
                f"    {desc}: pos={pe(sd3[hb][0], d2.xpos[bn[rb]]):.2f}m  "
                f"rot={re(sd3[hb][1], d2.xmat[bn[rb]]):.0f}°"
            )

        upper_h = np.array(sd3[el][0]) - np.array(sd3[sh][0])
        fore_h = np.array(sd3[wr][0]) - np.array(sd3[el][0])
        upper_r = d2.xpos[bn[f"{side}_arm_link3"]] - d2.xpos[bn[f"{side}_arm_link1"]]
        fore_r = d2.xpos[bn[f"{side}_arm_link7"]] - d2.xpos[bn[f"{side}_arm_link5"]]

        print(f"    上臂方向误差(肩→肘): {angle(upper_r, upper_h):.0f}°")
        print(f"    前臂方向误差(肘→腕): {angle(fore_r, fore_h):.0f}°")

    qi = 8
    print(f"\n  qpos L={np.round(d2.qpos[qi:qi+7], 2)}")
    print(f"  qpos R={np.round(d2.qpos[qi+7:qi+14], 2)}")


if __name__ == "__main__":
    diagnose(
        "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        "Stand (frame 0)",
    )
    diagnose(
        "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz",
        "Swing (mid frame)",
        frame_idx=None,
    )
    # also frame 0 swing - arms may be T-pose like in screenshot
    sd, bm, out, h = load_smplx_file(
        "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz",
        REPO / "assets/body_models",
    )
    frames, _ = get_smplx_data_offline_fast(sd, bm, out, tgt_fps=30)
    diagnose(
        "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz",
        "Swing (frame 0)",
        frame_idx=0,
    )
