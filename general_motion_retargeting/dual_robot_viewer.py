"""Dual-robot MuJoCo viewer: solid robot + semi-transparent ghost overlay.

Mirrors whole_body_tracking sim2sim ``ref_viz=mesh``: attach a second robot via
``MjSpec.attach`` and tint its geoms with low alpha.
"""

from __future__ import annotations

import os
import time

import imageio
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from rich import print

from general_motion_retargeting import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)

GHOST_PREFIX = "ghost_"


def build_dual_robot_model(
    xml_path: str | os.PathLike,
    *,
    ghost_rgba: tuple[float, float, float, float] = (0.65, 0.25, 1.0, 0.28),
) -> tuple[mj.MjModel, int]:
    """Compile a model with a second ghost copy of the robot.

    Returns ``(model, nq_single)`` where solid qpos is ``[0:nq_single)`` and
    ghost qpos is ``[nq_single:2*nq_single)``.
    """
    xml_path = str(xml_path)
    spec = mj.MjSpec.from_file(xml_path)
    ghost_spec = mj.MjSpec.from_file(xml_path)
    floor = ghost_spec.geom("floor")
    if floor is not None:
        ghost_spec.delete(floor)

    frame = spec.worldbody.add_frame(name="ghost_robot_frame")
    spec.attach(ghost_spec, prefix=GHOST_PREFIX, frame=frame)
    model = spec.compile()

    solid_pelvis = model.joint("pelvis").id
    ghost_pelvis = model.joint(f"{GHOST_PREFIX}pelvis").id
    solid_adr = int(model.jnt_qposadr[solid_pelvis])
    ghost_adr = int(model.jnt_qposadr[ghost_pelvis])
    nq_single = ghost_adr - solid_adr
    if solid_adr != 0 or model.nq != 2 * nq_single:
        raise RuntimeError(
            f"Unexpected dual qpos layout: solid@{solid_adr} ghost@{ghost_adr} nq={model.nq}"
        )

    rgba = np.asarray(ghost_rgba, dtype=np.float32)
    for gid in range(model.ngeom):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if body_name and body_name.startswith(GHOST_PREFIX):
            model.geom_rgba[gid] = rgba
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0

    return model, nq_single


class DualRobotMotionViewer:
    """Playback two qpos trajectories in one scene (solid + translucent ghost)."""

    def __init__(
        self,
        robot_type: str,
        *,
        motion_fps: float = 30.0,
        ghost_rgba: tuple[float, float, float, float] = (0.65, 0.25, 1.0, 0.28),
        ghost_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        record_video: bool = False,
        video_path: str | None = None,
        video_width: int = 960,
        video_height: int = 540,
        keyboard_callback=None,
    ):
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model, self.nq_single = build_dual_robot_model(
            self.xml_path, ghost_rgba=ghost_rgba
        )
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        self.ghost_offset = np.asarray(ghost_offset, dtype=float)
        self.motion_fps = float(motion_fps)
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.record_video = record_video
        self._record_cam_azimuth = 135.0
        self._record_cam_elevation = -15.0

        mj.mj_forward(self.model, self.data)

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )

        # Hide collision-group duplicates (same trick as RobotMotionViewer).
        floor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        for gid in range(self.model.ngeom):
            if int(self.model.geom_group[gid]) != 0:
                continue
            if floor_id >= 0 and gid == floor_id:
                continue
            self.model.geom_group[gid] = 3
        if hasattr(self.viewer.opt, "geomgroup"):
            self.viewer.opt.geomgroup[0] = 1
            self.viewer.opt.geomgroup[1] = 1
            self.viewer.opt.geomgroup[3] = 0

        base_id = self.model.body(self.robot_base).id
        self.viewer.cam.trackbodyid = base_id
        self.viewer.cam.distance = self.viewer_cam_distance
        self.viewer.cam.azimuth = self._record_cam_azimuth
        self.viewer.cam.elevation = self._record_cam_elevation

        if self.record_video:
            assert video_path is not None, "Provide video_path when record_video=True"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
            self.record_cam = mj.MjvCamera()
            self.record_cam.type = mj.mjtCamera.mjCAMERA_FREE
            self.record_cam.azimuth = self._record_cam_azimuth
            self.record_cam.elevation = self._record_cam_elevation
            self.record_cam.distance = self.viewer_cam_distance

    def _sync_camera(self, cam, follow: bool) -> None:
        if follow:
            cam.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
        cam.distance = self.viewer_cam_distance
        cam.azimuth = self._record_cam_azimuth
        cam.elevation = self._record_cam_elevation

    def step(
        self,
        qpos_solid: np.ndarray,
        qpos_ghost: np.ndarray,
        *,
        rate_limit: bool = True,
        follow_camera: bool = True,
    ) -> None:
        q_s = np.asarray(qpos_solid, dtype=float).reshape(-1)
        q_g = np.asarray(qpos_ghost, dtype=float).reshape(-1)
        if q_s.shape[0] != self.nq_single or q_g.shape[0] != self.nq_single:
            raise ValueError(
                f"Expected qpos length {self.nq_single}, got solid={q_s.shape[0]} ghost={q_g.shape[0]}"
            )

        self.data.qpos[: self.nq_single] = q_s
        self.data.qpos[self.nq_single : 2 * self.nq_single] = q_g
        self.data.qpos[self.nq_single : self.nq_single + 3] += self.ghost_offset
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

        self._sync_camera(self.viewer.cam, follow_camera)
        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            self._sync_camera(self.record_cam, follow_camera)
            self.renderer.update_scene(self.data, camera=self.record_cam)
            self.mp4_writer.append_data(self.renderer.render())

    def close(self) -> None:
        self.viewer.close()
        time.sleep(0.3)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
