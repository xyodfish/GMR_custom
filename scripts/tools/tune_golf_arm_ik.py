#!/usr/bin/env python3
"""Grid search Golf arm IK mappings and report errors."""
from __future__ import annotations

import itertools
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

REPO = Path(__file__).resolve().parents[2]
NPZ = Path("/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz")
SWING = Path("/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz")


def off(hq, Rr):
    Rh = R.from_quat([hq[1], hq[2], hq[3], hq[0]])
    q = (Rh.inv() * R.from_matrix(Rr)).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def rot_err(hq, mat):
    return float(
        np.degrees(
            (R.from_quat([hq[1], hq[2], hq[3], hq[0]]).inv() * R.from_matrix(mat.reshape(3, 3))).magnitude()
        )
    )


def build_table(frame, sh, el, wr, pos_w=0):
    m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["galbot_one_golf"]))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    quat_p = [0.5, -0.5, -0.5, -0.5]
    table = {
        "base_link": ["left_foot", 100, 10, [0.0, 0.0, 0.0], quat_p],
        "leg_link5": ["pelvis", 30, 10, [0.0, 0.0, -0.20], quat_p],
    }
    for hb, idx, pw in [
        ("left_shoulder", sh, pos_w),
        ("left_elbow", el, pos_w),
        ("left_wrist", wr, pos_w),
        ("right_shoulder", sh, pos_w),
        ("right_elbow", el, pos_w),
        ("right_wrist", wr, pos_w),
    ]:
        side = "left" if hb.startswith("left") else "right"
        rb = f"{side}_arm_link{idx}"
        Rr = d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb)].reshape(3, 3)
        table[rb] = [hb, pw, 10, [0.0, 0.0, 0.0], off(frame[hb][1], Rr)]
    return table


def score(table, frame, h):
    gmr = GeneralMotionRetargeting("smplx", "galbot_one_golf", actual_human_height=h, verbose=False)
    gmr.ik_match_table1 = table
    gmr.human_scale_table = {
        k: 1.1
        for k in [
            "pelvis",
            "left_foot",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "head",
        ]
    }
    gmr.setup_retarget_configuration()
    q = gmr.retarget(frame)
    mj.mj_forward(gmr.model, gmr.configuration.data)
    gmr.update_targets(frame)
    d = gmr.configuration.data
    bn = gmr.robot_body_names
    sd = gmr.scaled_human_data
    sh_rb = [k for k, v in table.items() if v[0] == "left_shoulder"][0]
    el_rb = [k for k, v in table.items() if v[0] == "left_elbow"][0]
    wr_rb = [k for k, v in table.items() if v[0] == "left_wrist"][0]
    rot = []
    pos = []
    for hb, rb in [("left_shoulder", sh_rb), ("left_elbow", el_rb), ("left_wrist", wr_rb)]:
        rot.append(rot_err(sd[hb][1], d.xmat[bn[rb]]))
        pos.append(float(np.linalg.norm(sd[hb][0] - d.xpos[bn[rb]])))
    l12 = float(np.linalg.norm(d.xpos[bn["left_arm_link1"]] - d.xpos[bn["left_arm_link2"]]))
    l34 = float(np.linalg.norm(d.xpos[bn["left_arm_link3"]] - d.xpos[bn["left_arm_link4"]]))
    return {
        "rot": rot,
        "pos": pos,
        "rot_mean": float(np.mean(rot)),
        "pos_mean": float(np.mean(pos)),
        "l12": l12,
        "l34": l34,
        "q_arm": [float(q[i]) for i in range(8, 15)],
    }


def main():
    smplx_data, body_model, smplx_output, h = load_smplx_file(str(NPZ), REPO / "assets/body_models")
    frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    stand = frames[0]

    smplx_data2, body_model2, smplx_output2, h2 = load_smplx_file(str(SWING), REPO / "assets/body_models")
    frames2, _ = get_smplx_data_offline_fast(smplx_data2, body_model2, smplx_output2, tgt_fps=30)
    swing = frames2[len(frames2) // 2]

    # R1pro reference
    gmr = GeneralMotionRetargeting("smplx", "galaxea_r1pro", actual_human_height=h, verbose=False)
    gmr.retarget(stand)
    mj.mj_forward(gmr.model, gmr.configuration.data)
    gmr.update_targets(stand)
    print("R1pro reference rot errs (link2/4/7):")
    for hb, rb in [("left_shoulder", "left_arm_link2"), ("left_elbow", "left_arm_link4"), ("left_wrist", "left_arm_link7")]:
        print(f"  {rb}: {rot_err(gmr.scaled_human_data[hb][1], gmr.configuration.data.xmat[gmr.robot_body_names[rb]]):.1f}deg")

    candidates = []
    for sh, el, wr in itertools.product([1, 2, 3], [3, 4, 5], [6, 7]):
        if el <= sh or wr <= el:
            continue
        for pw in [0, 5, 10]:
            name = f"sh{sh}-el{el}-wr{wr}-pw{pw}"
            tbl = build_table(stand, sh, el, wr, pw)
            s1 = score(tbl, stand, h)
            s2 = score(tbl, swing, h2)
            candidates.append((s1["rot_mean"] + s2["rot_mean"], name, sh, el, wr, pw, s1, s2))

    candidates.sort(key=lambda x: x[0])
    print("\nTop 8 configs (mean rot err stand+swing):")
    for item in candidates[:8]:
        _, name, sh, el, wr, pw, s1, s2 = item
        print(
            f"  {name}: stand rot={np.round(s1['rot'],1)} swing rot={np.round(s2['rot'],1)} "
            f"stand pos={np.round(s1['pos'],2)} l12={s1['l12']:.3f}"
        )

    best = candidates[0]
    print(f"\nBEST: {best[1]}")
    tbl = build_table(stand, best[2], best[3], best[4], best[5])
    print("offsets:")
    for k, v in sorted(tbl.items()):
        if "arm" in k:
            print(f"  {k}: {np.round(v[4], 4)}")


if __name__ == "__main__":
    main()
