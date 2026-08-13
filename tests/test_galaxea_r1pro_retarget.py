import unittest

import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from tests.test_galbot_one_golf_retarget import (
    _direction_error_deg,
    _standing_human_frame,
)


class GalaxeaR1ProRetargetTest(unittest.TestCase):
    def test_tracks_arm_directions_without_moving_wheel_joints(self):
        frame = _standing_human_frame()
        retarget = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot="galaxea_r1pro",
            actual_human_height=1.8,
            verbose=False,
        )

        retarget.retarget(frame)
        qpos = retarget.retarget(frame)

        self.assertEqual(qpos.shape, (27,))
        self.assertTrue(np.isfinite(qpos).all())
        np.testing.assert_allclose(qpos[3:9], 0.0, atol=1e-10)
        self.assertIsNone(retarget.mobile_upper_body_tasks["head"])

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
                retarget.model, mj.mjtObj.mjOBJ_BODY, f"{side}_arm_link7"
            )
            robot_upper = data.xpos[elbow_id] - data.xpos[shoulder_id]
            robot_forearm = data.xpos[wrist_id] - data.xpos[elbow_id]
            human_upper = frame[f"{side}_elbow"][0] - frame[f"{side}_shoulder"][0]
            human_forearm = frame[f"{side}_wrist"][0] - frame[f"{side}_elbow"][0]
            self.assertLess(_direction_error_deg(robot_upper, human_upper), 10.0)
            self.assertLess(_direction_error_deg(robot_forearm, human_forearm), 10.0)


if __name__ == "__main__":
    unittest.main()
