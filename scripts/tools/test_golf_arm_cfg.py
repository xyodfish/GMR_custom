#!/usr/bin/env python3
import numpy as np
import mujoco as mj
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def off(hq, Rr):
    Rh = R.from_quat([hq[1], hq[2], hq[3], hq[0]])
    q = (Rh.inv() * R.from_matrix(Rr)).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def re(hq, mat):
    return float(
        np.degrees(
            (R.from_quat([hq[1], hq[2], hq[3], hq[0]]).inv() * R.from_matrix(mat.reshape(3, 3))).magnitude()
        )
    )


def pe(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def main():
    sd, bm, out, h = load_smplx_file(
        "/home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        Path("assets/body_models"),
    )
    frames, _ = get_smplx_data_offline_fast(sd, bm, out, tgt_fps=30)
    stand = frames[0]
    m = mj.MjModel.from_xml_path("assets/galbot_one_golf/galbot_one_golf.xml")
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    qp = [0.5, -0.5, -0.5, -0.5]
    scale = {
        k: 1.1
        for k in [
            "pelvis",
            "left_foot",
            "left_hip",
            "left_knee",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "head",
        ]
    }
    leg = {
        "base_link": ["left_foot", 100, 10, [0, 0, 0], qp],
        "leg_link5": ["pelvis", 200, 10, [0, 0, 0], qp],
        "leg_link2": ["left_hip", 0, 10, [0, 0, 0], qp],
        "leg_link3": ["left_knee", 0, 10, [0, 0, 0], qp],
    }

    def rb(rb_name, hb, f):
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, rb_name)
        return off(f[hb][1], d.xmat[bid].reshape(3, 3))

    def test(name, t1, t2):
        g = GeneralMotionRetargeting("smplx", "galbot_one_golf", actual_human_height=h, verbose=False)
        g.ik_match_table1 = {**leg, **t1}
        g.use_ik_match_table2 = True
        g.ik_match_table2 = {**t2}
        g.human_scale_table = scale
        g.setup_retarget_configuration()
        g.retarget(stand)
        mj.mj_forward(g.model, g.configuration.data)
        g.update_targets(stand)
        sd3 = g.scaled_human_data
        d2 = g.configuration.data
        bn = g.robot_body_names
        print(
            f"{name}: l1r={re(sd3['left_shoulder'][1], d2.xmat[bn['left_arm_link1']]):.0f} "
            f"wr_p={pe(sd3['left_wrist'][0], d2.xpos[bn['left_arm_link7']]):.2f} "
            f"wrr={re(sd3['left_wrist'][1], d2.xmat[bn['left_arm_link7']]):.0f} "
            f"q={np.round(d2.qpos[8:15], 2)}"
        )

    t1c, t2c = {}, {}
    for s in ["left", "right"]:
        t1c[f"{s}_arm_link5"] = [f"{s}_elbow", 100, 0, [0, 0, 0], qp]
        t1c[f"{s}_arm_link7"] = [f"{s}_wrist", 200, 0, [0, 0, 0], qp]
        t2c[f"{s}_arm_link1"] = [f"{s}_shoulder", 0, 5, [0, 0, 0], rb(f"{s}_arm_link1", f"{s}_shoulder", stand)]
        t2c[f"{s}_arm_link4"] = [f"{s}_elbow", 0, 5, [0, 0, 0], rb(f"{s}_arm_link4", f"{s}_elbow", stand)]
        t2c[f"{s}_arm_link7"] = [f"{s}_wrist", 0, 5, [0, 0, 0], rb(f"{s}_arm_link7", f"{s}_wrist", stand)]
    test("pos t1 + light rot t2", t1c, t2c)

    t1b, t2b = {}, {}
    for s in ["left", "right"]:
        t1b[f"{s}_arm_link1"] = [f"{s}_shoulder", 0, 10, [0, 0, 0], rb(f"{s}_arm_link1", f"{s}_shoulder", stand)]
        t1b[f"{s}_arm_link4"] = [f"{s}_elbow", 0, 10, [0, 0, 0], rb(f"{s}_arm_link4", f"{s}_elbow", stand)]
        t1b[f"{s}_arm_link7"] = [f"{s}_wrist", 0, 10, [0, 0, 0], rb(f"{s}_arm_link7", f"{s}_wrist", stand)]
        t2b[f"{s}_arm_link5"] = [f"{s}_elbow", 100, 0, [0, 0, 0], qp]
        t2b[f"{s}_arm_link7"] = [f"{s}_wrist", 200, 0, [0, 0, 0], qp]
    test("rot t1 + pos t2", t1b, t2b)


if __name__ == "__main__":
    main()
