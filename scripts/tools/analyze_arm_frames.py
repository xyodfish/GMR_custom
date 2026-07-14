#!/usr/bin/env python3
"""Compare arm body/joint coordinate frames between robots."""
from __future__ import annotations

import numpy as np
import mujoco as mj
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def fmt(v):
    return f"[{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}]"


def body_frame(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    mat = data.xmat[bid].reshape(3, 3)
    return mat[:, 0], mat[:, 1], mat[:, 2], data.xpos[bid]


def joint_on_body(model, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    if jadr < 0:
        return None
    axis = model.jnt_axis[jadr].copy()
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jadr)
    return name, axis


def fixed_offset(model, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    return model.body_pos[bid].copy(), model.body_quat[bid].copy()


def report_chain(robot, links):
    m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    print(f"\n{'=' * 72}")
    print(f"{robot}")
    print("=" * 72)
    for link in links:
        X, Y, Z, pos = body_frame(m, d, link)
        pos_off, quat = fixed_offset(m, link)
        j = joint_on_body(m, link)
        print(f"\n{link} @ world pos {fmt(pos)}")
        print(f"  fixed offset pos={fmt(pos_off)} quat(wxyz)={np.round(quat, 4)}")
        if j:
            jn, axis_local = j
            axis_world = np.column_stack([X, Y, Z]) @ axis_local
            print(f"  joint {jn}: local Z-hinge axis={fmt(axis_local)} world={fmt(axis_world)}")
        print(f"  body +X={fmt(X)}")
        print(f"  body +Y={fmt(Y)}")
        print(f"  body +Z={fmt(Z)}")


def rot_offset_quat(robot_from, body_from, robot_to, body_to):
    """Constant offset mapping body_from frame to body_to at q=0."""
    m1 = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_from]))
    d1 = mj.MjData(m1)
    mj.mj_forward(m1, d1)
    m2 = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_to]))
    d2 = mj.MjData(m2)
    mj.mj_forward(m2, d2)
    R1 = d1.xmat[mj.mj_name2id(m1, mj.mjtObj.mjOBJ_BODY, body_from)].reshape(3, 3)
    R2 = d2.xmat[mj.mj_name2id(m2, mj.mjtObj.mjOBJ_BODY, body_to)].reshape(3, 3)
    off = R1.T @ R2
    q = R.from_matrix(off).as_quat()  # xyzw
    return [q[3], q[0], q[1], q[2]]


def human_frame_at_tpose(body_name, human_frame):
    pos, quat_wxyz = human_frame[body_name]
    mat = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    return mat[:, 0], mat[:, 1], mat[:, 2]


def main():
    g1_links = [
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ]
    golf_links = [f"left_arm_link{i}" for i in range(1, 8)]
    report_chain("unitree_g1", g1_links)
    report_chain("galbot_one_golf", golf_links)

    print(f"\n{'=' * 72}")
    print("IK target link frame comparison @ q=0")
    print("=" * 72)
    targets = [
        ("unitree_g1", "left_shoulder_yaw_link", "galbot_one_golf", "left_arm_link2", "shoulder"),
        ("unitree_g1", "left_elbow_link", "galbot_one_golf", "left_arm_link4", "elbow"),
        ("unitree_g1", "left_wrist_yaw_link", "galbot_one_golf", "left_arm_link7", "wrist"),
    ]
    for r1, b1, r2, b2, label in targets:
        for robot, body in [(r1, b1), (r2, b2)]:
            m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
            d = mj.MjData(m)
            mj.mj_forward(m, d)
            X, Y, Z, _ = body_frame(m, d, body)
            print(f"\n{label} | {robot} :: {body}")
            print(f"  +X={fmt(X)}  +Y={fmt(Y)}  +Z={fmt(Z)}")
        q = rot_offset_quat(r1, b1, r2, b2)
        print(f"  Golf->G1 frame offset quat(wxyz) for IK: {np.round(q, 6)}")

    npz = Path("/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz")
    if npz.exists():
        smplx_data, body_model, smplx_output, _ = load_smplx_file(
            str(npz), Path("assets/body_models")
        )
        frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
        hf = frames[0]
        print(f"\n{'=' * 72}")
        print("SMPL-X human frames @ stand pose")
        print("=" * 72)
        for hb in ["left_shoulder", "left_elbow", "left_wrist"]:
            X, Y, Z = human_frame_at_tpose(hb, hf)
            print(f"\n{hb}:")
            print(f"  +X={fmt(X)}  +Y={fmt(Y)}  +Z={fmt(Z)}")

        print(f"\n{'=' * 72}")
        print("Recommended rot_offset = inv(human) @ robot_ik_link @ q=0")
        print("=" * 72)
        m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["galbot_one_golf"]))
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        for hb, rb in [
            ("left_shoulder", "left_arm_link2"),
            ("left_elbow", "left_arm_link4"),
            ("left_wrist", "left_arm_link7"),
        ]:
            hq = hf[hb][1]
            Rh = R.from_quat([hq[1], hq[2], hq[3], hq[0]])
            Rr = d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb)].reshape(3, 3)
            off = Rh.inv() * R.from_matrix(Rr)
            q = off.as_quat()
            print(f"{rb} <- {hb}: {np.round([q[3], q[0], q[1], q[2]], 6)}")


if __name__ == "__main__":
    main()
