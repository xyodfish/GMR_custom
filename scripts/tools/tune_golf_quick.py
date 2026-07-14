#!/usr/bin/env python3
"""Quick grid search for galbot_one_golf arm IK."""
from __future__ import annotations

import itertools
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

REPO = Path(__file__).resolve().parents[2]
NPZ = Path(
    "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/"
    "General_A3_-_Swing_Arms_While_Stand_stageii.npz"
)


def pe(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def ang(v1, v2):
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))))


def off(hq, Rr):
    Rh = R.from_quat([hq[1], hq[2], hq[3], hq[0]])
    q = (Rh.inv() * R.from_matrix(Rr)).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def main():
    sd, bm, out, h = load_smplx_file(str(NPZ), REPO / "assets/body_models")
    frames, _ = get_smplx_data_offline_fast(sd, bm, out, tgt_fps=30)
    frame = frames[0]
    m = mj.MjModel.from_xml_path(str(REPO / "assets/galbot_one_golf/galbot_one_golf.xml"))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    idq = [1.0, 0.0, 0.0, 0.0]
    qp = [0.5, -0.5, -0.5, -0.5]
    offs = {}
    for side in ["left", "right"]:
        for hb, lk in [("shoulder", "1"), ("shoulder", "3"), ("elbow", "4")]:
            hb_name = f"{side}_{hb}"
            rb = f"{side}_arm_link{lk}"
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb)
            offs[f"{side}_L{lk}"] = off(frame[hb_name][1], d.xmat[bid].reshape(3, 3))

    def score(r1, r3, r4, shp, ep, wp, arm_scale, leg_p):
        g = GeneralMotionRetargeting("smplx", "galbot_one_golf", actual_human_height=h, verbose=False)
        g.human_scale_table = {
            "pelvis": 1.1,
            "left_foot": 1.1,
            "left_hip": 1.1,
            "left_knee": 1.1,
            "left_shoulder": arm_scale,
            "right_shoulder": arm_scale,
            "left_elbow": arm_scale,
            "right_elbow": arm_scale,
            "left_wrist": arm_scale,
            "right_wrist": arm_scale,
            "head": arm_scale,
        }
        g.ik_match_table1 = {
            "base_link": ["pelvis", 200, 100, [0, 0, 0], [1, 0, 0, 0]],
            "leg_link5": ["pelvis", leg_p, 60, [0, 0, 0], qp],
            "left_arm_link1": ["left_shoulder", 0, r1, [0, 0, 0], offs["left_L1"]],
            "left_arm_link3": ["left_shoulder", 0, r3, [0, 0, 0], offs["left_L3"]],
            "left_arm_link4": ["left_elbow", 0, r4, [0, 0, 0], offs["left_L4"]],
            "right_arm_link1": ["right_shoulder", 0, r1, [0, 0, 0], offs["right_L1"]],
            "right_arm_link3": ["right_shoulder", 0, r3, [0, 0, 0], offs["right_L3"]],
            "right_arm_link4": ["right_elbow", 0, r4, [0, 0, 0], offs["right_L4"]],
        }
        g.ik_match_table2 = {
            "left_arm_link1": ["left_shoulder", shp, 0, [0, 0, 0], idq],
            "right_arm_link1": ["right_shoulder", shp, 0, [0, 0, 0], idq],
            "left_arm_link4": ["left_elbow", ep, 0, [0, 0, 0], idq],
            "right_arm_link4": ["right_elbow", ep, 0, [0, 0, 0], idq],
            "left_arm_link5": ["left_wrist", wp, 0, [0, 0, 0], idq],
            "right_arm_link5": ["right_wrist", wp, 0, [0, 0, 0], idq],
        }
        g.setup_retarget_configuration()
        g.retarget(frame)
        mj.mj_forward(g.model, g.configuration.data)
        g.update_targets(frame)
        sd3 = g.scaled_human_data
        d2 = g.configuration.data
        bn = g.robot_body_names
        metrics = []
        for side in ["left", "right"]:
            sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            upper = ang(
                d2.xpos[bn[f"{side}_arm_link3"]] - d2.xpos[bn[f"{side}_arm_link1"]],
                np.array(sd3[el][0]) - np.array(sd3[sh][0]),
            )
            fore = ang(
                d2.xpos[bn[f"{side}_arm_link5"]] - d2.xpos[bn[f"{side}_arm_link4"]],
                np.array(sd3[wr][0]) - np.array(sd3[el][0]),
            )
            metrics.append(
                (
                    upper,
                    fore,
                    pe(sd3[el][0], d2.xpos[bn[f"{side}_arm_link4"]]),
                    pe(sd3[wr][0], d2.xpos[bn[f"{side}_arm_link5"]]),
                    pe(sd3[sh][0], d2.xpos[bn[f"{side}_arm_link1"]]),
                )
            )
        upper = float(np.nanmean([m[0] for m in metrics]))
        fore = float(np.nanmean([m[1] for m in metrics]))
        pos = float(np.mean([m[2] + m[3] + m[4] for m in metrics]) / 3)
        sh_z = float(d2.xpos[bn["left_arm_link1"]][2])
        tgt_z = float(sd3["left_shoulder"][0][2])
        return upper, fore, pos, sh_z, tgt_z, g.configuration.data.qpos[8:15].copy()

    best = []
    for arm_scale in [0.72, 0.78, 0.85, 1.0]:
        for r1, r3, r4 in [(15, 15, 15), (20, 20, 20), (25, 10, 25)]:
            for shp, ep, wp in [(40, 100, 150), (50, 120, 180)]:
                for leg_p in [80, 120, 150]:
                    upper, fore, pos, sh_z, tgt_z, q = score(
                        r1, r3, r4, shp, ep, wp, arm_scale, leg_p
                    )
                    cost = upper + pos * 40 + abs(sh_z - tgt_z) * 30
                    best.append((cost, arm_scale, r1, r3, r4, shp, ep, wp, leg_p, upper, fore, pos, sh_z, tgt_z, q))
    best.sort(key=lambda x: x[0])
    print("Top configs:")
    for row in best[:10]:
        print(
            f"scale={row[1]} r=({row[2]},{row[3]},{row[4]}) pos=({row[5]},{row[6]},{row[7]}) leg={row[8]} | "
            f"upper={row[9]:.0f} fore={row[10]:.0f} pos={row[11]:.2f} sh_z={row[12]:.2f}/{row[13]:.2f} qL={np.round(row[14],2)}"
        )


if __name__ == "__main__":
    main()
