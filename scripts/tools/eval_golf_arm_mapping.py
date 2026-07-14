#!/usr/bin/env python3
"""Evaluate Golf arm IK link mapping candidates."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

REPO = Path(__file__).resolve().parents[2]
NPZ = Path("/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz")


def rot_err_deg(hq_wxyz, body_mat):
    t = R.from_quat([hq_wxyz[1], hq_wxyz[2], hq_wxyz[3], hq_wxyz[0]])
    a = R.from_matrix(body_mat.reshape(3, 3))
    return np.degrees((t.inv() * a).magnitude())


def offset_from_q0(human_quat_wxyz, robot_body_mat):
    Rh = R.from_quat([human_quat_wxyz[1], human_quat_wxyz[2], human_quat_wxyz[3], human_quat_wxyz[0]])
    Rr = R.from_matrix(robot_body_mat.reshape(3, 3))
    q = (Rh.inv() * Rr).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def eval_mapping(shoulder_link, elbow_link, wrist_link, offsets=None):
    cfg = json.loads((REPO / "general_motion_retargeting/ik_configs/smplx_to_galbot_one_golf.json").read_text())
    quat_p = [0.5, -0.5, -0.5, -0.5]
    table = {
        "base_link": ["left_foot", 100, 10, [0.0, 0.0, 0.0], quat_p],
        "leg_link5": ["pelvis", 30, 10, [0.0, 0.0, -0.20], quat_p],
        f"left_arm_link{shoulder_link}": ["left_shoulder", 0, 10, [0.0, 0.0, 0.0], offsets["sh"] if offsets else quat_p],
        f"left_arm_link{elbow_link}": ["left_elbow", 0, 10, [0.0, 0.0, 0.0], offsets["el"] if offsets else quat_p],
        f"left_arm_link{wrist_link}": ["left_wrist", 0, 10, [0.0, 0.0, 0.0], offsets["wr"] if offsets else quat_p],
        f"right_arm_link{shoulder_link}": ["right_shoulder", 0, 10, [0.0, 0.0, 0.0], offsets["sh_r"] if offsets else quat_p],
        f"right_arm_link{elbow_link}": ["right_elbow", 0, 10, [0.0, 0.0, 0.0], offsets["el_r"] if offsets else quat_p],
        f"right_arm_link{wrist_link}": ["right_wrist", 0, 10, [0.0, 0.0, 0.0], offsets["wr_r"] if offsets else quat_p],
    }
    gmr = GeneralMotionRetargeting("smplx", "galbot_one_golf", actual_human_height=h, verbose=False)
    gmr.ik_match_table1 = table
    gmr.human_scale_table = cfg["human_scale_table"]
    gmr.setup_retarget_configuration()
    q = gmr.retarget(frame0)
    mj.mj_forward(gmr.model, gmr.configuration.data)
    gmr.update_targets(frame0)
    d = gmr.configuration.data
    bn = gmr.robot_body_names
    sd = gmr.scaled_human_data
    out = {}
    for hb, rb in [
        ("left_shoulder", f"left_arm_link{shoulder_link}"),
        ("left_elbow", f"left_arm_link{elbow_link}"),
        ("left_wrist", f"left_arm_link{wrist_link}"),
    ]:
        out[rb] = {
            "pos": float(np.linalg.norm(sd[hb][0] - d.xpos[bn[rb]])),
            "rot": float(rot_err_deg(sd[hb][1], d.xmat[bn[rb]])),
        }
    return out, q


if __name__ == "__main__":
    smplx_data, body_model, smplx_output, h = load_smplx_file(str(NPZ), REPO / "assets/body_models")
    frame0, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    frame0 = frame0[0]

    m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["galbot_one_golf"]))
    d = mj.MjData(m)
    mj.mj_forward(m, d)

    print("Q0 offsets inv(human)*robot:")
    left = {}
    right = {}
    for hb, rb in [
        ("left_shoulder", "left_arm_link1"),
        ("left_shoulder", "left_arm_link3"),
        ("left_elbow", "left_arm_link4"),
        ("left_elbow", "left_arm_link5"),
        ("left_wrist", "left_arm_link7"),
    ]:
        off = offset_from_q0(frame0[hb][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb)])
        print(f"  {rb} <- {hb}: {np.round(off, 4)}")

    mappings = [
        ("current", 2, 4, 7),
        ("link1-4-7", 1, 4, 7),
        ("link3-5-7", 3, 5, 7),
        ("link1-5-7", 1, 5, 7),
        ("link3-4-7", 3, 4, 7),
    ]
    print("\nMapping evaluation:")
    for name, sh, el, wr in mappings:
        offsets = {
            "sh": offset_from_q0(frame0["left_shoulder"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"left_arm_link{sh}")]),
            "el": offset_from_q0(frame0["left_elbow"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"left_arm_link{el}")]),
            "wr": offset_from_q0(frame0["left_wrist"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"left_arm_link{wr}")]),
            "sh_r": offset_from_q0(frame0["right_shoulder"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"right_arm_link{sh}")]),
            "el_r": offset_from_q0(frame0["right_elbow"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"right_arm_link{el}")]),
            "wr_r": offset_from_q0(frame0["right_wrist"][1], d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"right_arm_link{wr}")]),
        }
        res, _ = eval_mapping(sh, el, wr, offsets)
        print(f"  {name} ({sh},{el},{wr}):")
        for k, v in res.items():
            print(f"    {k}: pos={v['pos']:.3f} rot={v['rot']:.1f}deg")
