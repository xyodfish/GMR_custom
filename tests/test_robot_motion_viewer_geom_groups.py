import unittest

import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_BASE_DICT, ROBOT_XML_DICT
from general_motion_retargeting.dual_robot_viewer import build_two_robot_model
from general_motion_retargeting.robot_motion_viewer import (
    RobotMotionViewer,
    hide_collision_duplicate_geoms,
)


class RobotMotionViewerGeomGroupTest(unittest.TestCase):
    def test_galbot_keeps_its_group_zero_visual_meshes(self):
        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["galbot_one_golf"]))

        changed = hide_collision_duplicate_geoms(model)

        robot_mesh_groups = [
            int(model.geom_group[gid])
            for gid in range(model.ngeom)
            if int(model.geom_type[gid]) == int(mj.mjtGeom.mjGEOM_MESH)
        ]
        self.assertFalse(changed)
        self.assertEqual(len(robot_mesh_groups), 22)
        self.assertEqual(set(robot_mesh_groups), {0})

        for name in (
            "left_upper_arm_connector",
            "left_forearm_connector",
            "right_upper_arm_connector",
            "right_forearm_connector",
        ):
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(int(model.geom_group[geom_id]), 0)

    def test_galbot_tint_preserves_head_and_chassis_contrast(self):
        model, _, _, _ = build_two_robot_model(
            ROBOT_XML_DICT["unitree_g1"],
            ROBOT_XML_DICT["galbot_one_golf"],
        )

        def brightness(name):
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"b_{name}")
            self.assertGreaterEqual(geom_id, 0)
            return float(np.mean(model.geom_rgba[geom_id, :3]))

        self.assertLess(brightness("head_neck_visual"), brightness("head_shell_visual"))
        self.assertLess(brightness("chassis_inner_core"), brightness("chassis_visual"))

    def test_g1_still_hides_group_zero_collision_duplicates(self):
        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
        floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")

        changed = hide_collision_duplicate_geoms(model)

        self.assertTrue(changed)
        for gid in range(model.ngeom):
            if gid == floor_id or int(model.geom_group[gid]) == 1:
                continue

            self.assertNotEqual(int(model.geom_group[gid]), 0)


class RobotMotionViewerCameraTest(unittest.TestCase):
    def setUp(self):
        self.viewer = RobotMotionViewer.__new__(RobotMotionViewer)
        self.viewer.model = mj.MjModel.from_xml_path(
            str(ROBOT_XML_DICT["galbot_one_golf"])
        )
        self.viewer.data = mj.MjData(self.viewer.model)
        self.viewer.robot_base = ROBOT_BASE_DICT["galbot_one_golf"]
        self.viewer.viewer_cam_distance = 3.0
        self.viewer._record_cam_azimuth = 135.0
        self.viewer._record_cam_elevation = -15.0
        self.viewer.data.qpos[:2] = [1.0, 2.0]
        mj.mj_forward(self.viewer.model, self.viewer.data)

    def test_interactive_camera_preserves_user_view(self):
        camera = mj.MjvCamera()
        camera.azimuth = 42.0
        camera.elevation = -33.0
        camera.distance = 5.0

        self.viewer._sync_camera(camera, follow=True)

        np.testing.assert_allclose(
            camera.lookat,
            self.viewer.data.xpos[
                self.viewer.model.body(self.viewer.robot_base).id
            ],
        )
        self.assertEqual(camera.azimuth, 42.0)
        self.assertEqual(camera.elevation, -33.0)
        self.assertEqual(camera.distance, 5.0)

    def test_record_camera_resets_to_deterministic_view(self):
        camera = mj.MjvCamera()
        camera.azimuth = 42.0
        camera.elevation = -33.0
        camera.distance = 5.0

        self.viewer._sync_camera(camera, follow=True, reset_view=True)

        self.assertEqual(camera.azimuth, 135.0)
        self.assertEqual(camera.elevation, -15.0)
        self.assertEqual(camera.distance, 3.0)


if __name__ == "__main__":
    unittest.main()
