import unittest

import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting


def _standing_human_frame():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "pelvis": [np.array([0.0, 0.0, 1.0]), identity],
        "spine3": [np.array([0.0, 0.0, 1.35]), identity],
        "head": [np.array([0.0, 0.0, 1.65]), identity],
        "left_shoulder": [np.array([0.0, 0.2, 1.42]), identity],
        "left_elbow": [np.array([0.0, 0.2, 1.08]), identity],
        "left_wrist": [np.array([0.08, 0.2, 0.75]), identity],
        "right_shoulder": [np.array([0.0, -0.2, 1.42]), identity],
        "right_elbow": [np.array([0.0, -0.2, 1.08]), identity],
        "right_wrist": [np.array([0.08, -0.2, 0.75]), identity],
    }


def _direction_error_deg(actual, target):
    actual = actual / np.linalg.norm(actual)
    target = target / np.linalg.norm(target)
    return np.degrees(np.arccos(np.clip(np.dot(actual, target), -1.0, 1.0)))


class GalbotOneGolfRetargetTest(unittest.TestCase):
    def test_tracks_arm_directions_with_joint_margin(self):
        frame = _standing_human_frame()
        retarget = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot="galbot_one_golf",
            actual_human_height=1.8,
            verbose=False,
        )

        retarget.retarget(frame)
        qpos = retarget.retarget(frame)

        self.assertEqual(qpos.shape, (retarget.model.nq,))
        self.assertTrue(np.isfinite(qpos).all())
        margin = np.deg2rad(2.0) - 1e-8
        for joint_id in range(3, retarget.model.njnt):
            if not retarget.model.jnt_limited[joint_id]:
                continue

            qpos_id = retarget.model.jnt_qposadr[joint_id]
            lower, upper = retarget.model.jnt_range[joint_id]
            self.assertGreaterEqual(qpos[qpos_id], lower + margin)
            self.assertLessEqual(qpos[qpos_id], upper - margin)

        data = retarget.configuration.data
        for side in ("left", "right"):
            shoulder_id = mj.mj_name2id(
                retarget.model, mj.mjtObj.mjOBJ_BODY, f"{side}_arm_link1"
            )
            elbow_id = mj.mj_name2id(
                retarget.model, mj.mjtObj.mjOBJ_BODY, f"{side}_arm_link4"
            )
            wrist_id = mj.mj_name2id(
                retarget.model, mj.mjtObj.mjOBJ_BODY, f"{side}_arm_link5"
            )
            robot_upper = data.xpos[elbow_id] - data.xpos[shoulder_id]
            robot_forearm = data.xpos[wrist_id] - data.xpos[elbow_id]
            human_upper = frame[f"{side}_elbow"][0] - frame[f"{side}_shoulder"][0]
            human_forearm = frame[f"{side}_wrist"][0] - frame[f"{side}_elbow"][0]
            self.assertLess(_direction_error_deg(robot_upper, human_upper), 10.0)
            self.assertLess(_direction_error_deg(robot_forearm, human_forearm), 10.0)

    def test_rejects_zero_length_human_arm(self):
        frame = _standing_human_frame()
        frame["left_elbow"][0] = frame["left_shoulder"][0].copy()
        retarget = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot="galbot_one_golf",
            verbose=False,
        )

        with self.assertRaisesRegex(ValueError, "left_shoulder->left_elbow"):
            retarget.retarget(frame)


if __name__ == "__main__":
    unittest.main()
