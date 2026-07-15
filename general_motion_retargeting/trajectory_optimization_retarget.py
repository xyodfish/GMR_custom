"""Independent causal trajectory optimization for humanoid retargeting.

Unlike ``SlidingWindowRetargeter``, this module does **not** call ``GMR.retarget()``
inside the optimizer loop. It directly minimizes task-space FK tracking error plus
temporal smoothness over a causal human frame buffer.

GMR is used only for:
- human preprocessing (scale / contact_ground)
- IK config / body mapping metadata
- optional per-frame ``q_ref`` warm start (``use_gmr_init=True``, default)

``mode="fast"`` (default): GMR bootstrap on frame 0, then causal chain
(``q_prev`` + light IK + mink GN/L-BFGS temporal refine). Does **not** call
``GMR.retarget()`` every frame.
Without ``use_gmr_init``, runs FK GN bootstrap then temporal smoothing.
``mode="full"``: joint window L-BFGS-B with MuJoCo FK costs (offline).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Sequence, Tuple

import mujoco as mj
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from .motion_retarget import GeneralMotionRetargeting
from .params import PLANAR_BASE_ROBOTS


@dataclass
class TrajectoryOptimizationConfig:
    window_size: int = 8
    mode: Literal["fast", "full"] = "fast"
    solver: Literal["gn", "lbfgs"] = "lbfgs"
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
    """Causal receding-horizon kinematic TO independent of per-frame GMR IK."""

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
        self._prepared_human: Deque[dict] = deque(maxlen=self.config.window_size)
        self._q_window: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._qpos_lower, self._qpos_upper = self._build_qpos_bounds()
        self._opt_vidx, self._smooth_qidx = self._build_opt_indices()
        self._dq_limit_q, self._ddq_limit_q = self._build_dq_ddQ_limits()
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
        self._prepared_human.clear()
        self._q_window.clear()
        self._frame_index = 0

    def retarget(self, human_data, offset_to_ground: bool = False) -> np.ndarray:
        prepared = self.gmr._prepare_scaled_human_data(human_data, offset_to_ground)
        self._prepared_human.append(prepared)
        self._frame_index += 1
        targets = self._targets_for_prepared(prepared)

        if self.config.mode == "full" and len(self._prepared_human) >= self.config.min_frames:
            q_out = self._optimize_causal_window(human_data, prepared, offset_to_ground)
        else:
            q_out = self._retarget_fast(human_data, prepared, offset_to_ground, targets)

        self._q_window.append(q_out.copy())
        self.gmr.configuration.data.qpos[:] = q_out
        mj.mj_forward(self.model, self.gmr.configuration.data)
        self.gmr.scaled_human_data = self.gmr._build_scaled_human_data(prepared)
        return q_out

    def _retarget_fast(
        self,
        human_data,
        prepared: dict,
        offset_to_ground: bool,
        targets: dict,
    ) -> np.ndarray:
        if not self._q_window:
            if self.config.use_gmr_init:
                q_opt = self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)
            else:
                q_init = self._seed_root_translation(
                    self.gmr.configuration.data.qpos.copy(), targets
                )
                q_opt = self._retarget_pure_fk(
                    q_init, targets, prepared, human_data, offset_to_ground
                )
            return self._finalize_qpos(q_opt, prepared, human_data, offset_to_ground)

        q_init = self._q_window[-1].copy()
        q_init = self._light_ik_warmstart(q_init, prepared, human_data, offset_to_ground)
        q_prev = self._q_window[-1]
        q_prev2 = self._q_window[-2] if len(self._q_window) >= 2 else q_prev
        q_opt = self._optimize_single_frame(q_init, prepared, q_prev, q_prev2)
        return self._finalize_qpos(q_opt, prepared, human_data, offset_to_ground)

    def _retarget_pure_fk(self, q_init: np.ndarray, targets: dict, prepared: dict, human_data, offset_to_ground: bool) -> np.ndarray:
        q_init = self._light_ik_warmstart(q_init, prepared, human_data, offset_to_ground)

        if self.config.light_ik_warmstart_iters > 0:
            if not self._q_window:
                return q_init
            q_prev = self._q_window[-1]
            q_prev2 = self._q_window[-2] if len(self._q_window) >= 2 else q_prev
            return self._optimize_single_frame(q_init, prepared, q_prev, q_prev2)

        q_track = self._optimize_current_frame_gn(
            q_init,
            targets,
            q_init,
            q_init,
            n_steps=self.config.gn_steps_no_init,
            w_velocity=0.0,
            w_acceleration=0.0,
            max_step=self.config.gn_max_step_no_init,
            damping=self.config.gn_damping * 0.5,
        )
        if not self._q_window:
            return q_track

        q_prev = self._q_window[-1]
        q_prev2 = self._q_window[-2] if len(self._q_window) >= 2 else q_prev
        return self._optimize_current_frame_gn(
            q_track,
            targets,
            q_prev,
            q_prev2,
            n_steps=self.config.gn_steps,
            w_velocity=self.config.w_velocity,
            w_acceleration=self.config.w_acceleration,
            max_step=self.config.gn_max_step,
        )

    def _seed_root_translation(self, qpos: np.ndarray, targets: dict) -> np.ndarray:
        q = qpos.copy()
        if self.model.nq < 3:
            return q
        target = targets.get(self.gmr.robot_root_name)
        if target is None:
            return q
        q[0:3] = target[0]
        return q

    def _light_ik_warmstart(
        self,
        q_init: np.ndarray,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        """Few mink IK iterations to seed q (not full ``GMR.retarget()``)."""
        if self.config.light_ik_warmstart_iters <= 0:
            return q_init

        import mink

        self.gmr.configuration.data.qpos[:] = q_init
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        self._set_mink_targets(prepared)

        freeze_base = (
            self.gmr.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.gmr.planar_base_cfg)
        )
        base_qpos = None
        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            base_qpos = self.gmr.configuration.data.qpos[:3].copy()

        n_iter = self.config.light_ik_warmstart_iters
        if self.gmr.use_ik_match_table1:
            self.gmr._run_ik_tasks(
                self.gmr.tasks1,
                max_iter=n_iter,
                freeze_base=freeze_base,
                base_qpos=base_qpos,
            )
        if self.gmr.use_ik_match_table2:
            self.gmr._run_ik_tasks(
                self.gmr.tasks2,
                max_iter=n_iter,
                freeze_base=freeze_base,
                base_qpos=base_qpos,
            )
        return self.gmr.configuration.data.qpos.copy()

    def _set_mink_targets(self, prepared: dict) -> None:
        import mink

        if self.gmr.use_ik_match_table1:
            for entry in self.gmr.task_frames1:
                pos, rot = self.gmr._resolve_ik_target(entry, prepared)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )
        if self.gmr.use_ik_match_table2:
            for entry in self.gmr.task_frames2:
                pos, rot = self.gmr._resolve_ik_target(entry, prepared)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def _mink_tracking_cost(self, qpos: np.ndarray, *, targets_set: bool) -> float:
        self.gmr.configuration.data.qpos[:] = qpos
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        if not targets_set:
            raise ValueError("mink targets must be set before evaluating tracking cost")

        errors = []
        if self.gmr.use_ik_match_table1:
            errors.extend(
                task.compute_error(self.gmr.configuration) for task in self.gmr.tasks1
            )
        if self.gmr.use_ik_match_table2:
            errors.extend(
                task.compute_error(self.gmr.configuration) for task in self.gmr.tasks2
            )
        if not errors:
            return 0.0
        err = np.concatenate(errors)
        return float(np.dot(err, err))

    def _optimize_single_frame(
        self,
        q_init: np.ndarray,
        prepared: dict,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        self._set_mink_targets(prepared)
        if self.config.solver == "gn":
            return self._optimize_single_frame_gn(q_init, q_prev, q_prev2)
        return self._optimize_single_frame_lbfgs(q_init, q_prev, q_prev2)

    def _optimize_single_frame_gn(
        self,
        q_init: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        q = q_init.copy()
        vidx = self._opt_vidx
        smooth_qidx = self._smooth_qidx
        nv = self.model.nv
        dq_v = np.zeros(nv)
        m = len(vidx)
        dt = self.dt
        dt2 = dt * dt
        w_v = self.config.w_velocity / max(dt2, 1e-12)
        w_a = self.config.w_acceleration / max(dt2 * dt2, 1e-12)
        damp = self.config.gn_damping
        max_step = self.config.gn_max_step

        for _ in range(self.config.gn_steps):
            self.gmr.configuration.data.qpos[:] = q
            mj.mj_forward(self.model, self.gmr.configuration.data)

            H = np.zeros((m, m), dtype=float)
            g = np.zeros(m, dtype=float)

            if self.gmr.use_ik_match_table1:
                self._accumulate_gn_tasks(self.gmr.tasks1, H, g, vidx)
            if self.gmr.use_ik_match_table2:
                self._accumulate_gn_tasks(self.gmr.tasks2, H, g, vidx)

            self._accumulate_gn_smoothness(
                H, g, q, q_prev, q_prev2, smooth_qidx, vidx, w_v, w_a
            )

            try:
                dq_sub = np.linalg.solve(H + damp * np.eye(m), g)
            except np.linalg.LinAlgError:
                break
            dq_sub = np.clip(dq_sub, -max_step, max_step)
            dq_v[:] = 0.0
            dq_v[vidx] = -dq_sub
            mj.mj_integratePos(self.model, q, dq_v, 1.0)
            self._clip_qpos(q)
            if self.config.enforce_dq_ddq:
                self._project_dq_ddQ(q, q_prev, q_prev2)

        return q

    def _optimize_single_frame_lbfgs(
        self,
        q_init: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        bounds = list(zip(self._qpos_lower, self._qpos_upper))
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration

        def objective(q: np.ndarray) -> float:
            cost = self._mink_tracking_cost(q, targets_set=True)
            if w_v > 0.0:
                delta = q - q_prev
                cost += w_v * float(np.dot(delta, delta))
            if w_a > 0.0:
                acc = q - 2.0 * q_prev + q_prev2
                cost += w_a * float(np.dot(acc, acc))
            return cost

        result = minimize(
            objective,
            q_init.copy(),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.config.fast_opt_iter,
                "ftol": self.config.opt_tol,
            },
        )
        return result.x.copy() if result.success else q_init.copy()

    def _accumulate_gn_tasks(self, tasks, H: np.ndarray, g: np.ndarray, vidx: np.ndarray) -> None:
        for task in tasks:
            err = task.compute_error(self.gmr.configuration)
            jac = task.compute_jacobian(self.gmr.configuration)[:, vidx]
            H += jac.T @ jac
            g += jac.T @ err

    def _project_dq_ddQ(
        self,
        q: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> None:
        dt = self.dt
        dt2 = dt * dt
        qidx = self._smooth_qidx

        q[qidx] = np.clip(q[qidx], self._qpos_lower[qidx], self._qpos_upper[qidx])

        dq_delta = q[qidx] - q_prev[qidx]
        dq_lim = self._dq_limit_q[qidx] * dt
        dq_delta = np.clip(dq_delta, -dq_lim, dq_lim)
        q[qidx] = q_prev[qidx] + dq_delta

        ddq_delta = q[qidx] - 2.0 * q_prev[qidx] + q_prev2[qidx]
        ddq_lim = self._ddq_limit_q[qidx] * dt2
        ddq_delta = np.clip(ddq_delta, -ddq_lim, ddq_lim)
        q[qidx] = 2.0 * q_prev[qidx] - q_prev2[qidx] + ddq_delta

        q[qidx] = np.clip(q[qidx], self._qpos_lower[qidx], self._qpos_upper[qidx])

    def _clip_qpos(self, q: np.ndarray) -> None:
        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            if not self.model.jnt_limited[j]:
                continue
            qadr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            q[qadr] = np.clip(q[qadr], lo, hi)

    def _optimize_current_frame_gn(
        self,
        q_init: np.ndarray,
        targets: dict,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
        *,
        n_steps: Optional[int] = None,
        w_velocity: Optional[float] = None,
        w_acceleration: Optional[float] = None,
        max_step: Optional[float] = None,
        damping: Optional[float] = None,
    ) -> np.ndarray:
        q = q_init.copy()
        vidx = self._opt_vidx
        smooth_qidx = self._smooth_qidx
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        dq_v = np.zeros(nv)
        m = len(vidx)
        n_steps = self.config.gn_steps if n_steps is None else n_steps
        w_velocity = self.config.w_velocity if w_velocity is None else w_velocity
        w_acceleration = self.config.w_acceleration if w_acceleration is None else w_acceleration
        max_step = self.config.gn_max_step if max_step is None else max_step
        damp = self.config.gn_damping if damping is None else damping

        for _ in range(n_steps):
            self.data.qpos[:] = q
            mj.mj_forward(self.model, self.data)

            H = np.zeros((m, m), dtype=float)
            g = np.zeros(m, dtype=float)

            for entry in self._track_entries:
                target = targets.get(entry.robot_frame)
                if target is None:
                    continue
                pos_t, quat_t = target
                body_pos = self.data.xpos[entry.body_id]

                if entry.pos_weight > 0.0:
                    pos_e = body_pos - pos_t
                    mj.mj_jac(
                        self.model,
                        self.data,
                        jacp,
                        None,
                        body_pos,
                        entry.body_id,
                    )
                    J = jacp[:, vidx]
                    w = entry.pos_weight
                    H += w * (J.T @ J)
                    g += w * (J.T @ pos_e)

                if entry.rot_weight > 0.0:
                    rot_body = self.data.xmat[entry.body_id].reshape(3, 3)
                    rot_t = R.from_quat(self.gmr._quat_wxyz_to_xyzw(quat_t))
                    rot_err = (rot_t.inv() * R.from_matrix(rot_body)).as_rotvec()
                    mj.mj_jac(
                        self.model,
                        self.data,
                        None,
                        jacr,
                        body_pos,
                        entry.body_id,
                    )
                    J = jacr[:, vidx]
                    w = entry.rot_weight
                    H += w * (J.T @ J)
                    g += w * (J.T @ rot_err)

            self._accumulate_gn_smoothness(
                H, g, q, q_prev, q_prev2, smooth_qidx, vidx, w_velocity, w_acceleration
            )

            try:
                dq_sub = np.linalg.solve(H + damp * np.eye(m), g)
            except np.linalg.LinAlgError:
                break
            dq_sub = np.clip(dq_sub, -max_step, max_step)
            dq_v[:] = 0.0
            dq_v[vidx] = -dq_sub
            mj.mj_integratePos(self.model, q, dq_v, 1.0)
            self._clip_hinge_qpos(q)

        return q

    def _accumulate_gn_smoothness(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
        smooth_qidx: np.ndarray,
        vidx: np.ndarray,
        w_velocity: float,
        w_acceleration: float,
    ) -> None:
        if len(smooth_qidx) == 0:
            return
        # Map smoothness on selected qpos coords through their velocity columns.
        q_to_v = {int(qadr): i for i, qadr in enumerate(smooth_qidx)}
        smooth_v = []
        smooth_q = []
        for vi, v in enumerate(vidx):
            j = self.model.dof_jntid[v]
            qadr = int(self.model.jnt_qposadr[j])
            if qadr in q_to_v:
                smooth_v.append(vi)
                smooth_q.append(qadr)
        if not smooth_v:
            return
        smooth_v = np.asarray(smooth_v, dtype=int)
        smooth_q = np.asarray(smooth_q, dtype=int)
        n = len(smooth_v)
        if w_velocity > 0.0:
            H[np.ix_(smooth_v, smooth_v)] += w_velocity * np.eye(n)
            g[smooth_v] += w_velocity * (q[smooth_q] - q_prev[smooth_q])
        if w_acceleration > 0.0:
            acc_target = 2.0 * q_prev[smooth_q] - q_prev2[smooth_q]
            H[np.ix_(smooth_v, smooth_v)] += w_acceleration * np.eye(n)
            g[smooth_v] += w_acceleration * (q[smooth_q] - acc_target)

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

    def _optimize_causal_window(
        self,
        human_data,
        prepared: dict,
        offset_to_ground: bool,
    ) -> np.ndarray:
        prepared_list = list(self._prepared_human)
        targets_list = [self._targets_for_prepared(p) for p in prepared_list]
        q_init = self._initial_q_window(prepared_list, human_data, offset_to_ground)
        anchor = q_init[0].copy()
        q_opt = self._optimize_full_window(q_init, targets_list, anchor)
        q_out = q_opt[-1].copy()
        return self._finalize_qpos(q_out, prepared, human_data, offset_to_ground)

    def _initial_q_window(
        self,
        prepared_list: Sequence[dict],
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        n = len(prepared_list)
        q_init = np.zeros((n, self.model.nq), dtype=float)
        prev_qs = list(self._q_window)

        for i in range(n):
            if i < len(prev_qs):
                q_init[i] = prev_qs[i]
            elif self.config.use_gmr_init and i == n - 1:
                q_init[i] = self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)
            elif prev_qs:
                q_init[i] = prev_qs[-1]
            else:
                q_init[i] = self.gmr.configuration.data.qpos.copy()
        return q_init

    def _optimize_full_window(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
    ) -> np.ndarray:
        x0 = q_init.reshape(-1)
        bounds = self._flat_bounds(len(q_init))

        def objective(x: np.ndarray) -> float:
            q_window = x.reshape(q_init.shape)
            return self._window_cost(q_window, targets_list, anchor)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.config.max_opt_iter,
                "ftol": self.config.opt_tol,
            },
        )
        return result.x.reshape(q_init.shape) if result.success else q_init

    def _window_cost(
        self,
        q_window: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
    ) -> float:
        cost = 0.0
        for q, targets in zip(q_window, targets_list):
            cost += self._fk_tracking_cost(q, targets)

        if self.config.w_velocity > 0.0 and q_window.shape[0] >= 2:
            diffs = np.diff(q_window, axis=0)
            cost += self.config.w_velocity * float(np.sum(diffs * diffs))

        if self.config.w_acceleration > 0.0 and q_window.shape[0] >= 3:
            acc = q_window[2:] - 2.0 * q_window[1:-1] + q_window[:-2]
            cost += self.config.w_acceleration * float(np.sum(acc * acc))

        if self.config.w_anchor > 0.0:
            delta = q_window[0] - anchor
            cost += self.config.w_anchor * float(np.dot(delta, delta))

        return cost

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

        def merge_table(table, task_frames, table_name: str) -> None:
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

        merge_table(self.gmr.ik_match_table1, self.gmr.task_frames1, "table1")
        if self.gmr.use_ik_match_table2:
            merge_table(self.gmr.ik_match_table2, self.gmr.task_frames2, "table2")
        return list(entries.values())

    def _finalize_qpos(
        self,
        qpos: np.ndarray,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
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

    def _build_dq_ddQ_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        nq = self.model.nq
        dq_lim = np.full(nq, self.config.dq_max, dtype=float)
        ddq_lim = np.full(nq, self.config.ddq_max, dtype=float)
        if self.model.njnt > 0 and self.model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            dq_lim[0:3] = max(self.config.dq_max, 2.0)
            ddq_lim[0:3] = max(self.config.ddq_max, 20.0)
            if self.config.smooth_root_rot:
                dq_lim[3:7] = self.config.dq_max
                ddq_lim[3:7] = self.config.ddq_max
        return dq_lim, ddq_lim

    def _flat_bounds(self, n_frames: int) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for _ in range(n_frames):
            for lo, hi in zip(self._qpos_lower, self._qpos_upper):
                bounds.append((lo, hi))
        return bounds
