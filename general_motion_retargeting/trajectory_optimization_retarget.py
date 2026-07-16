"""Shared FK/TO helpers for BatchTrajectoryRetargeter (not a standalone retarget algorithm)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from .motion_retarget import GeneralMotionRetargeting
from .params import PLANAR_BASE_ROBOTS


@dataclass
class TrajectoryOptimizationConfig:
    """Config fields consumed by Batch TO via ``BatchTrajectoryConfig.to_to_config``."""

    window_size: int = 8
    mode: Literal["fast", "full"] = "full"
    solver: Literal["gn", "lbfgs"] = "gn"
    w_velocity: float = 2.0
    w_acceleration: float = 10.0
    w_anchor: float = 20.0
    max_opt_iter: int = 25
    fast_opt_iter: int = 5
    gn_steps: int = 3
    gn_steps_no_init: int = 25
    light_ik_warmstart_iters: int = 5
    gn_damping: float = 0.05
    gn_max_step: float = 0.08
    gn_max_step_no_init: float = 0.05
    opt_tol: float = 1e-4
    min_frames: int = 1
    use_gmr_init: bool = True
    fix_window_prefix: bool = False
    dt: Optional[float] = None
    dq_max: float = 8.0
    ddq_max: float = 80.0
    enforce_dq_ddq: bool = True
    smooth_root_rot: bool = False


@dataclass
class _TrackEntry:
    robot_frame: str
    body_id: int
    pos_weight: float
    rot_weight: float
    human_body: str
    pos_offset: np.ndarray
    rot_offset: R


class TrajectoryOptimizationRetargeter:
    """Shared MuJoCo FK tracking base used by Batch TO (not a public algorithm)."""

    def __init__(
        self,
        retargeter: GeneralMotionRetargeting,
        config: Optional[TrajectoryOptimizationConfig] = None,
    ) -> None:
        self.gmr = retargeter
        self.config = config or TrajectoryOptimizationConfig()
        if self.config.window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.model = self.gmr.model
        self.data = mj.MjData(self.model)
        self._track_entries = self._build_track_entries()
        self._qpos_lower, self._qpos_upper = self._build_qpos_bounds()
        self._opt_vidx, self._smooth_qidx = self._build_opt_indices()
        self._frame_index = 0

    @property
    def dt(self) -> float:
        if self.config.dt is not None and self.config.dt > 0.0:
            return float(self.config.dt)
        return float(self.model.opt.timestep)

    def set_motion_fps(self, fps: float) -> None:
        if fps > 0.0:
            self.config.dt = 1.0 / float(fps)
        self.gmr.set_motion_fps(fps)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def reset(self) -> None:
        self._frame_index = 0

    def _clip_hinge_qpos(self, q: np.ndarray) -> None:
        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            if not self.model.jnt_limited[j]:
                continue
            qadr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            q[qadr] = np.clip(q[qadr], lo, hi)

    def _fk_tracking_cost(self, qpos: np.ndarray, targets: dict) -> float:
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        cost = 0.0
        for entry in self._track_entries:
            target = targets.get(entry.robot_frame)
            if target is None:
                continue
            pos_t, quat_t = target
            pos_e = self.data.xpos[entry.body_id] - pos_t
            if entry.pos_weight > 0.0:
                cost += entry.pos_weight * float(np.dot(pos_e, pos_e))

            if entry.rot_weight > 0.0:
                rot_body = self.data.xmat[entry.body_id].reshape(3, 3)
                rot_t = R.from_quat(self.gmr._quat_wxyz_to_xyzw(quat_t))
                rot_err = (rot_t.inv() * R.from_matrix(rot_body)).as_rotvec()
                cost += entry.rot_weight * float(np.dot(rot_err, rot_err))
        return cost

    def _targets_for_prepared(self, prepared: dict) -> dict:
        targets: dict = {}
        for entry in self.gmr.task_frames1:
            pos, rot = self.gmr._resolve_ik_target(entry, prepared)
            targets[entry["robot_frame"]] = (pos, rot)
        if self.gmr.use_ik_match_table2:
            for entry in self.gmr.task_frames2:
                pos, rot = self.gmr._resolve_ik_target(entry, prepared)
                targets[entry["robot_frame"]] = (pos, rot)
        return targets

    def _build_track_entries(self) -> List[_TrackEntry]:
        entries: dict[str, _TrackEntry] = {}

        def merge_table(table, task_frames) -> None:
            frame_lookup = {e["robot_frame"]: e for e in task_frames}
            for frame_name, row in table.items():
                entry_dict = frame_lookup.get(frame_name)
                if entry_dict is None:
                    continue
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, frame_name)
                if body_id < 0:
                    continue
                _, pos_w, rot_w, _, _ = row
                pos_w, rot_w = float(pos_w), float(rot_w)
                if pos_w == 0.0 and rot_w == 0.0:
                    continue
                prev = entries.get(frame_name)
                if prev is None:
                    entries[frame_name] = _TrackEntry(
                        robot_frame=frame_name,
                        body_id=int(body_id),
                        pos_weight=pos_w,
                        rot_weight=rot_w,
                        human_body=entry_dict["human_body"],
                        pos_offset=entry_dict["pos_offset"],
                        rot_offset=entry_dict["rot_offset"],
                    )
                else:
                    prev.pos_weight = max(prev.pos_weight, pos_w)
                    prev.rot_weight = max(prev.rot_weight, rot_w)

        merge_table(self.gmr.ik_match_table1, self.gmr.task_frames1)
        if self.gmr.use_ik_match_table2:
            merge_table(self.gmr.ik_match_table2, self.gmr.task_frames2)
        return list(entries.values())

    def _finalize_qpos(
        self,
        qpos: np.ndarray,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        del offset_to_ground  # reserved for API parity with Batch finalize
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

        freeze_base = (
            self.gmr.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.gmr.planar_base_cfg)
        )
        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            qpos = self.data.qpos.copy()

        if self.gmr.contact_ground.fix_penetration:
            self.gmr.contact_ground.fix_robot_penetration(self.model, self.data)
            qpos = self.data.qpos.copy()

        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            qpos = self.data.qpos.copy()

        return qpos.copy()

    def _build_qpos_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.model.nq, -np.inf, dtype=float)
        upper = np.full(self.model.nq, np.inf, dtype=float)
        for j in range(self.model.njnt):
            if not self.model.jnt_limited[j]:
                continue
            qadr = self.model.jnt_qposadr[j]
            jtype = self.model.jnt_type[j]
            lo, hi = self.model.jnt_range[j]
            if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                lower[qadr] = lo
                upper[qadr] = hi
        return lower, upper

    def _build_opt_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Velocity indices: free joint (6) + hinges; smoothness on root xyz + hinges."""
        v_indices: List[int] = []
        smooth_q: List[int] = []

        if self.model.njnt > 0 and self.model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            v_indices.extend(range(min(6, self.model.nv)))
            smooth_q.extend(range(min(3, self.model.nq)))
            if self.config.smooth_root_rot:
                smooth_q.extend(range(3, min(7, self.model.nq)))

        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            vadr = int(self.model.jnt_dofadr[j])
            qadr = int(self.model.jnt_qposadr[j])
            v_indices.append(vadr)
            smooth_q.append(qadr)

        return np.asarray(v_indices, dtype=int), np.asarray(smooth_q, dtype=int)

    def _flat_bounds(self, n_frames: int) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for _ in range(n_frames):
            for lo, hi in zip(self._qpos_lower, self._qpos_upper):
                bounds.append((lo, hi))
        return bounds
