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


def _delete_named_geom(spec: mj.MjSpec, name: str) -> None:
    geom = spec.geom(name)
    if geom is not None:
        spec.delete(geom)


def build_two_robot_model(
    xml_a: str | os.PathLike,
    xml_b: str | os.PathLike,
    *,
    prefix_b: str = "b_",
    offset_b: tuple[float, float, float] = (0.0, 1.2, 0.0),
    tint_b: tuple[float, float, float, float] | None = (0.35, 0.55, 0.95, 1.0),
) -> tuple[mj.MjModel, int, int, int | None]:
    """Attach two (possibly different) robots into one model.

    Returns ``(model, nq_a, nq_b, free_root_qadr_b)``. Robot-B bodies/joints are
    prefixed with ``prefix_b``. Robot-B floor geom is removed so only robot-A's
    ground plane remains.
    """
    xml_a = str(xml_a)
    xml_b = str(xml_b)
    model_a = mj.MjModel.from_xml_path(xml_a)
    model_b = mj.MjModel.from_xml_path(xml_b)
    nq_a = int(model_a.nq)
    nq_b = int(model_b.nq)
    free_roots_b = [
        int(model_b.jnt_qposadr[j])
        for j in range(model_b.njnt)
        if model_b.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
    ]
    if len(free_roots_b) > 1:
        raise RuntimeError(f"Robot B has multiple free joints: {free_roots_b}")

    free_root_qadr_b = free_roots_b[0] if free_roots_b else None
    spec_a = mj.MjSpec.from_file(xml_a)
    spec_b = mj.MjSpec.from_file(xml_b)
    _delete_named_geom(spec_b, "floor")
    frame = spec_a.worldbody.add_frame(name="robot_b_frame")
    if free_root_qadr_b is None:
        frame.pos = np.asarray(offset_b, dtype=float)

    spec_a.attach(spec_b, prefix=prefix_b, frame=frame)
    model = spec_a.compile()

    if model.nq != nq_a + nq_b:
        raise RuntimeError(
            f"Unexpected attached qpos layout: A={nq_a}, B={nq_b}, combined={model.nq}"
        )

    if tint_b is not None:
        tint = np.asarray(tint_b, dtype=np.float32)
        for gid in range(model.ngeom):
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
            if body_name and body_name.startswith(prefix_b):
                source = model.geom_rgba[gid].copy()
                if float(source[3]) > 1e-3:
                    shade = np.clip(float(np.mean(source[:3])) / 0.84, 0.16, 1.15)
                    model.geom_rgba[gid, :3] = np.minimum(tint[:3] * shade, 1.0)
                    model.geom_rgba[gid, 3] = tint[3] * source[3]

    return model, nq_a, nq_b, free_root_qadr_b


class TwoRobotMotionViewer:
    """Playback two different robots side-by-side in one MuJoCo window."""

    def __init__(
        self,
        robot_a: str,
        robot_b: str,
        *,
        motion_fps: float = 30.0,
        offset_b: tuple[float, float, float] = (0.0, 1.2, 0.0),
        tint_b: tuple[float, float, float, float] | None = (0.35, 0.55, 0.95, 1.0),
        prefix_b: str = "b_",
        record_video: bool = False,
        video_path: str | None = None,
        video_width: int = 1280,
        video_height: int = 720,
        keyboard_callback=None,
    ):
        self.robot_a = robot_a
        self.robot_b = robot_b
        self.prefix_b = prefix_b
        self.offset_b = np.asarray(offset_b, dtype=float)
        self.motion_fps = float(motion_fps)
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.record_video = record_video
        self._record_cam_azimuth = 135.0
        self._record_cam_elevation = -15.0

        xml_a = ROBOT_XML_DICT[robot_a]
        xml_b = ROBOT_XML_DICT[robot_b]
        self.model, self.nq_a, self.nq_b, self.free_root_qadr_b = build_two_robot_model(
            xml_a,
            xml_b,
            prefix_b=prefix_b,
            offset_b=offset_b,
            tint_b=tint_b,
        )
        self.data = mj.MjData(self.model)
        self.base_a = ROBOT_BASE_DICT[robot_a]
        self.base_b = f"{prefix_b}{ROBOT_BASE_DICT[robot_b]}"
        cam_a = VIEWER_CAM_DISTANCE_DICT.get(robot_a, 3.0)
        cam_b = VIEWER_CAM_DISTANCE_DICT.get(robot_b, 3.0)
        self.viewer_cam_distance = max(float(cam_a), float(cam_b)) + 1.0

        mj.mj_forward(self.model, self.data)
        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )

        # Hide each robot's group-0 collision geoms only when that same robot has
        # dedicated visual geoms. Some robots, including Galbot, render from group 0.
        floor_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        geom_is_b = []
        has_visual_group = {False: False, True: False}
        for gid in range(self.model.ngeom):
            body_name = mj.mj_id2name(
                self.model,
                mj.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[gid],
            ) or ""
            is_b = body_name.startswith(prefix_b)
            geom_is_b.append(is_b)
            if int(self.model.geom_group[gid]) in (1, 2):
                has_visual_group[is_b] = True

        for gid in range(self.model.ngeom):
            if int(self.model.geom_group[gid]) != 0:
                continue

            if floor_gid >= 0 and gid == floor_gid:
                continue

            if has_visual_group[geom_is_b[gid]]:
                self.model.geom_group[gid] = 3

        if hasattr(self.viewer.opt, "geomgroup"):
            self.viewer.opt.geomgroup[0] = 1
            self.viewer.opt.geomgroup[1] = 1
            self.viewer.opt.geomgroup[2] = 1  # H2 visual class uses group 2
            self.viewer.opt.geomgroup[3] = 0

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

    def _lookat_midpoint(self) -> np.ndarray:
        pa = self.data.xpos[self.model.body(self.base_a).id]
        pb = self.data.xpos[self.model.body(self.base_b).id]
        return 0.5 * (pa + pb)

    def step(
        self,
        qpos_a: np.ndarray,
        qpos_b: np.ndarray,
        *,
        rate_limit: bool = True,
        follow_camera: bool = True,
    ) -> None:
        qa = np.asarray(qpos_a, dtype=float).reshape(-1)
        qb = np.asarray(qpos_b, dtype=float).reshape(-1)
        if qa.shape[0] != self.nq_a or qb.shape[0] != self.nq_b:
            raise ValueError(
                f"Expected qpos lengths ({self.nq_a}, {self.nq_b}), "
                f"got ({qa.shape[0]}, {qb.shape[0]})"
            )

        self.data.qpos[: self.nq_a] = qa
        self.data.qpos[self.nq_a : self.nq_a + self.nq_b] = qb
        if self.free_root_qadr_b is not None:
            begin = self.nq_a + self.free_root_qadr_b
            self.data.qpos[begin : begin + 3] += self.offset_b

        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

        if follow_camera:
            self.viewer.cam.lookat[:] = self._lookat_midpoint()

        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            if follow_camera:
                self.record_cam.lookat[:] = self._lookat_midpoint()

            self.record_cam.distance = self.viewer_cam_distance
            self.record_cam.azimuth = self._record_cam_azimuth
            self.record_cam.elevation = self._record_cam_elevation
            self.renderer.update_scene(self.data, camera=self.record_cam)
            self.mp4_writer.append_data(self.renderer.render())

    def close(self) -> None:
        self.viewer.close()
        time.sleep(0.3)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
