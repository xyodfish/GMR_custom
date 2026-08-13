import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT, PLANAR_BASE_ROBOTS
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print

def quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz)
    return quat_wxyz[[1, 2, 3, 0]]


def hide_collision_duplicate_geoms(model):
    """Hide group-0 robot geoms only when group 1 provides visual meshes."""
    has_group1_visual_mesh = any(
        int(model.geom_group[gid]) == 1
        and int(model.geom_type[gid]) == int(mj.mjtGeom.mjGEOM_MESH)
        for gid in range(model.ngeom)
    )
    if not has_group1_visual_mesh:
        return False

    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) != 0:
            continue
        if floor_id >= 0 and gid == floor_id:
            continue
        model.geom_group[gid] = 3

    return True


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        self._record_cam_azimuth = 135.0
        self._record_cam_elevation = -15.0

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = bool(transparent_robot)
        # Some models duplicate robot meshes in group 0 (collision) and group 1
        # (visual). Only hide group 0 when that visual-mesh convention is present;
        # models such as Galbot keep their only visual meshes in group 0.
        hide_collision_duplicate_geoms(self.model)

        if hasattr(self.viewer.opt, "geomgroup"):
            self.viewer.opt.geomgroup[0] = 1  # floor
            self.viewer.opt.geomgroup[1] = 1  # visual meshes
            self.viewer.opt.geomgroup[3] = 0  # hidden collision duplicates

        self._sync_camera(self.viewer.cam, self.camera_follow, reset_view=True)

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
            self.record_cam = mj.MjvCamera()
            self.record_cam.type = mj.mjtCamera.mjCAMERA_FREE
            self.record_cam.azimuth = self._record_cam_azimuth
            self.record_cam.elevation = self._record_cam_elevation
            self.record_cam.distance = self.viewer_cam_distance
            self.record_cam.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
        
    def _sync_camera(self, cam, follow: bool, reset_view: bool = False) -> None:
        if follow:
            cam.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
        if reset_view:
            cam.distance = self.viewer_cam_distance
            cam.azimuth = self._record_cam_azimuth
            cam.elevation = self._record_cam_elevation

    def step(self, 
            # robot data
            root_pos=None, root_rot=None, dof_pos=None, qpos=None,
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        if qpos is not None:
            self.data.qpos[:] = qpos
        elif self.robot_type in PLANAR_BASE_ROBOTS:
            self.data.qpos[:3] = root_pos
            self.data.qpos[3:] = dof_pos
        else:
            self.data.qpos[:3] = root_pos
            self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
            self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        
        self._sync_camera(self.viewer.cam, follow_camera)
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(quat_wxyz_to_xyzw(rot)).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use a deterministic record camera; do not read interactive viewer.cam.
            self._sync_camera(self.record_cam, follow_camera, reset_view=True)
            self.renderer.update_scene(self.data, camera=self.record_cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
