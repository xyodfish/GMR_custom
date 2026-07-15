"""Sliding-window kinematic trajectory optimization for real-time retargeting.

Pipeline: GMR preprocessing -> light IK warm start -> causal temporal refinement.

``mode="fast"`` (default) refines the current frame with a closed-form
Gauss-Newton step using mink task Jacobians (no scipy L-BFGS per frame).
Temporal terms use explicit ``dt`` scaling so velocity / acceleration
penalties and hard limits apply to ``dq`` and ``ddq``, not raw ``q`` diffs.

``mode="full"`` joint-optimizes every frame in the window via L-BFGS-B and
is much slower (offline only).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Sequence, Tuple

import mujoco as mj
import numpy as np
from scipy.optimize import minimize

from .motion_retarget import GeneralMotionRetargeting
from .params import PLANAR_BASE_ROBOTS


@dataclass
class SlidingWindowConfig:
    window_size: int = 8
    mode: Literal["fast", "full"] = "fast"
    solver: Literal["gn", "lbfgs"] = "gn"
    w_velocity: float = 2.0
    w_acceleration: float = 10.0
    w_anchor: float = 50.0
    ik_warmstart_iters: int = 3
    fast_opt_iter: int = 5
    max_opt_iter: int = 25
    gn_steps: int = 3
    gn_damping: float = 0.05
    gn_max_step: float = 0.08
    opt_tol: float = 1e-4
    min_frames: int = 2
    dt: Optional[float] = None
    dq_max: float = 8.0
    ddq_max: float = 80.0
    enforce_dq_ddq: bool = True
    optimize_root: bool = False
    use_full_ik_init: bool = False
    smooth_legs: bool = True
    lock_grounded_from_ik: bool = False
    smooth_joints: Literal["all", "upper_body", "arms_only", "none"] = "upper_body"
    w_ik_anchor: float = 0.0
    arm_ema_alpha: float = 0.0


class SlidingWindowRetargeter:
    """Causal sliding-window retargeting: GMR IK + kinematic TO smoothing."""

    def __init__(
        self,
        retargeter: GeneralMotionRetargeting,
        config: Optional[SlidingWindowConfig] = None,
    ) -> None:
        self.gmr = retargeter
        self.config = config or SlidingWindowConfig()
        if self.config.window_size < self.config.min_frames:
            raise ValueError("window_size must be >= min_frames")

        self.model = self.gmr.model
        self._prepared_human: Deque[dict] = deque(maxlen=self.config.window_size)
        self._q_window: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._qpos_lower, self._qpos_upper = self._build_qpos_bounds()
        self._opt_vidx, self._smooth_qidx = self._build_opt_indices()
        self._leg_qidx = self._build_leg_q_indices()
        self._dq_limit_q, self._ddq_limit_q = self._build_dq_ddQ_limits()
        self._frame_index = 0

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def dt(self) -> float:
        if self.config.dt is not None and self.config.dt > 0.0:
            return float(self.config.dt)
        return float(self.model.opt.timestep)

    def set_motion_fps(self, fps: float) -> None:
        if fps > 0.0:
            self.config.dt = 1.0 / float(fps)
        self.gmr.set_motion_fps(fps)

    def reset(self) -> None:
        self._prepared_human.clear()
        self._q_window.clear()
        self._frame_index = 0

    def retarget(self, human_data, offset_to_ground: bool = False) -> np.ndarray:
        """Retarget one streaming human frame and return the latest ``qpos``."""
        prepared = self.gmr._prepare_scaled_human_data(human_data, offset_to_ground)
        self._prepared_human.append(prepared)
        self._frame_index += 1

        if self.config.mode == "full":
            q_out = self._retarget_full(prepared, human_data, offset_to_ground)
        else:
            q_out = self._retarget_fast(prepared, human_data, offset_to_ground)

        self._q_window.append(q_out.copy())
        self.gmr.configuration.data.qpos[:] = q_out
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        self.gmr.scaled_human_data = self.gmr._build_scaled_human_data(prepared)
        return q_out

    def _retarget_fast(
        self,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        q_ik_ref = None
        if self.config.use_full_ik_init:
            q_ik_ref = self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)
            q_curr = q_ik_ref.copy()
        else:
            if self._q_window:
                q_init = self._q_window[-1].copy()
            else:
                q_init = self.gmr.configuration.data.qpos.copy()

            self.gmr.configuration.data.qpos[:] = q_init
            mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
            self._set_targets_from_prepared(prepared)
            self._run_light_ik(prepared, human_data, offset_to_ground)
            q_curr = self.gmr.configuration.data.qpos.copy()

        if len(self._q_window) < 1:
            return self._finalize_qpos(
                q_curr, prepared, human_data, offset_to_ground, grounded_locked=False
            )

        q_prev = self._q_window[-1]
        q_prev2 = self._q_window[-2] if len(self._q_window) >= 2 else q_prev

        if self._should_skip_temporal_opt():
            q_opt = self._apply_arm_ema(q_curr, q_prev)
            grounded_locked = q_ik_ref is not None and self.config.lock_grounded_from_ik
            if grounded_locked and q_ik_ref is not None:
                q_opt = self._lock_grounded_from_ik(q_opt, q_ik_ref)
            return self._finalize_qpos(
                q_opt, prepared, human_data, offset_to_ground, grounded_locked=grounded_locked
            )

        if self.config.solver == "gn":
            q_opt = self._optimize_single_frame_gn(q_curr, q_prev, q_prev2)
        else:
            q_opt = self._optimize_single_frame_lbfgs(q_curr, q_prev, q_prev2)

        grounded_locked = False
        if q_ik_ref is not None and self.config.lock_grounded_from_ik:
            q_opt = self._lock_grounded_from_ik(q_opt, q_ik_ref)
            grounded_locked = True

        return self._finalize_qpos(
            q_opt, prepared, human_data, offset_to_ground, grounded_locked=grounded_locked
        )

    def _retarget_full(
        self,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        q_ik = self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)
        q_list = list(self._q_window) + [q_ik.copy()]
        prepared_list = list(self._prepared_human)

        if len(q_list) < self.config.min_frames:
            return q_ik

        q_init = np.stack(q_list, axis=0)
        anchor = q_init[0].copy()
        x0 = q_init.reshape(-1)
        bounds = self._flat_bounds(len(q_init))

        def objective(x: np.ndarray) -> float:
            q_window = x.reshape(q_init.shape)
            return self._window_cost(q_window, prepared_list, anchor)

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
        q_window = result.x.reshape(q_init.shape) if result.success else q_init
        return q_window[-1].copy()

    def _lock_grounded_from_ik(self, q: np.ndarray, q_ik: np.ndarray) -> np.ndarray:
        """Keep root + non-smoothed joints identical to full IK."""
        out = q.copy()
        out[:7] = q_ik[:7]
        smooth_set = set(int(i) for i in self._smooth_qidx)
        for qadr in range(self.model.nq):
            if qadr in smooth_set:
                continue
            out[qadr] = q_ik[qadr]
        return out

    def _finalize_qpos(
        self,
        qpos: np.ndarray,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
        *,
        grounded_locked: bool = False,
    ) -> np.ndarray:
        self.gmr.configuration.data.qpos[:] = qpos
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)

        freeze_base = (
            self.gmr.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.gmr.planar_base_cfg)
        )
        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            qpos = self.gmr.configuration.data.qpos.copy()

        skip_penetration = grounded_locked and self.config.lock_grounded_from_ik
        if self.gmr.contact_ground.fix_penetration and not skip_penetration:
            self.gmr.contact_ground.fix_robot_penetration(
                self.gmr.model, self.gmr.configuration.data
            )
            qpos = self.gmr.configuration.data.qpos.copy()

        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            qpos = self.gmr.configuration.data.qpos.copy()

        return qpos.copy()

    def _run_light_ik(self, prepared: dict, human_data, offset_to_ground: bool) -> None:
        self._set_targets_from_prepared(prepared)
        freeze_base = (
            self.gmr.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.gmr.planar_base_cfg)
        )
        base_qpos = None
        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            base_qpos = self.gmr.configuration.data.qpos[:3].copy()

        n_iter = self.config.ik_warmstart_iters
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

    def _should_skip_temporal_opt(self) -> bool:
        if self.config.smooth_joints == "none":
            return True
        if len(self._smooth_qidx) == 0:
            return True
        return self.config.w_velocity <= 0.0 and self.config.w_acceleration <= 0.0

    def _arm_q_indices(self) -> np.ndarray:
        arm_q: List[int] = []
        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if self._is_arm_joint(joint_name):
                arm_q.append(int(self.model.jnt_qposadr[j]))
        return np.asarray(arm_q, dtype=int)

    def _apply_arm_ema(self, q_curr: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        alpha = float(self.config.arm_ema_alpha)
        if alpha <= 0.0:
            return q_curr.copy()
        arm_q = self._arm_q_indices()
        if len(arm_q) == 0:
            return q_curr.copy()
        out = q_curr.copy()
        out[arm_q] = alpha * q_curr[arm_q] + (1.0 - alpha) * q_prev[arm_q]
        return out

    def _optimize_single_frame_gn(
        self,
        q_init: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        """Closed-form GN on mink task errors + dt-scaled temporal terms."""
        q = q_init.copy()
        vidx = self._opt_vidx
        smooth_qidx = self._smooth_qidx
        nv = self.model.nv
        dq_v = np.zeros(nv)
        m = len(vidx)
        # Soft smoothness on joint q finite-differences (user weights are in q-space).
        # Hard dq/ddq limits use physical units via _project_dq_ddQ.
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration
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
            self._accumulate_gn_ik_anchor(H, g, q, q_init, smooth_qidx, vidx)

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

    def _accumulate_gn_ik_anchor(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q: np.ndarray,
        q_ik: np.ndarray,
        smooth_qidx: np.ndarray,
        vidx: np.ndarray,
    ) -> None:
        w = self.config.w_ik_anchor
        if w <= 0.0 or len(smooth_qidx) == 0:
            return
        q_to_v = {int(qadr): i for i, qadr in enumerate(smooth_qidx)}
        anchor_v: List[int] = []
        anchor_q: List[int] = []
        for vi, v in enumerate(vidx):
            j = self.model.dof_jntid[v]
            qadr = int(self.model.jnt_qposadr[j])
            if qadr in q_to_v:
                anchor_v.append(vi)
                anchor_q.append(qadr)
        if not anchor_v:
            return
        anchor_v_arr = np.asarray(anchor_v, dtype=int)
        anchor_q_arr = np.asarray(anchor_q, dtype=int)
        n = len(anchor_v_arr)
        H[np.ix_(anchor_v_arr, anchor_v_arr)] += w * np.eye(n)
        g[anchor_v_arr] += w * (q[anchor_q_arr] - q_ik[anchor_q_arr])

    def _accumulate_gn_tasks(self, tasks, H: np.ndarray, g: np.ndarray, vidx: np.ndarray) -> None:
        for task in tasks:
            err = task.compute_error(self.gmr.configuration)
            jac = task.compute_jacobian(self.gmr.configuration)[:, vidx]
            H += jac.T @ jac
            g += jac.T @ err

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
        q_to_v = {int(qadr): i for i, qadr in enumerate(smooth_qidx)}
        smooth_v: List[int] = []
        smooth_q: List[int] = []
        for vi, v in enumerate(vidx):
            j = self.model.dof_jntid[v]
            qadr = int(self.model.jnt_qposadr[j])
            if qadr in q_to_v:
                smooth_v.append(vi)
                smooth_q.append(qadr)
        if not smooth_v:
            return
        smooth_v_arr = np.asarray(smooth_v, dtype=int)
        smooth_q_arr = np.asarray(smooth_q, dtype=int)
        n = len(smooth_v_arr)
        if w_velocity > 0.0:
            H[np.ix_(smooth_v_arr, smooth_v_arr)] += w_velocity * np.eye(n)
            g[smooth_v_arr] += w_velocity * (q[smooth_q_arr] - q_prev[smooth_q_arr])
        if w_acceleration > 0.0:
            acc_target = 2.0 * q_prev[smooth_q_arr] - q_prev2[smooth_q_arr]
            H[np.ix_(smooth_v_arr, smooth_v_arr)] += w_acceleration * np.eye(n)
            g[smooth_v_arr] += w_acceleration * (q[smooth_q_arr] - acc_target)

    def _optimize_single_frame_lbfgs(
        self,
        q_init: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        bounds = list(zip(self._qpos_lower, self._qpos_upper))
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration
        smooth_qidx = self._smooth_qidx

        def objective(q: np.ndarray) -> float:
            cost = self._frame_tracking_cost(q, targets_already_set=True)
            if w_v > 0.0 and len(smooth_qidx) > 0:
                delta = q[smooth_qidx] - q_prev[smooth_qidx]
                cost += w_v * float(np.dot(delta, delta))
            if w_a > 0.0 and len(smooth_qidx) > 0:
                acc = q[smooth_qidx] - 2.0 * q_prev[smooth_qidx] + q_prev2[smooth_qidx]
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
        q_out = result.x.copy() if result.success else q_init.copy()
        if self.config.enforce_dq_ddq:
            self._clip_qpos(q_out)
            self._project_dq_ddQ(q_out, q_prev, q_prev2)
        return q_out

    def _project_dq_ddQ(
        self,
        q: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> None:
        """Hard box limits on q, dq, and ddq (in-place)."""
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

    def _window_cost(
        self,
        q_window: np.ndarray,
        prepared_human: Sequence[dict],
        anchor: np.ndarray,
    ) -> float:
        cost = 0.0
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration
        smooth_qidx = self._smooth_qidx

        for q, human in zip(q_window, prepared_human):
            self._set_targets_from_prepared(human)
            cost += self._frame_tracking_cost(q, targets_already_set=True)

        if w_v > 0.0 and q_window.shape[0] >= 2 and len(smooth_qidx) > 0:
            diffs = np.diff(q_window[:, smooth_qidx], axis=0)
            cost += w_v * float(np.sum(diffs * diffs))

        if w_a > 0.0 and q_window.shape[0] >= 3 and len(smooth_qidx) > 0:
            acc = (
                q_window[2:, smooth_qidx]
                - 2.0 * q_window[1:-1, smooth_qidx]
                + q_window[:-2, smooth_qidx]
            )
            cost += w_a * float(np.sum(acc * acc))

        if self.config.w_anchor > 0.0:
            delta = q_window[0] - anchor
            cost += self.config.w_anchor * float(np.dot(delta, delta))

        return cost

    def _frame_tracking_cost(
        self,
        qpos: np.ndarray,
        prepared_human: Optional[dict] = None,
        *,
        targets_already_set: bool = False,
    ) -> float:
        self.gmr.configuration.data.qpos[:] = qpos
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        if prepared_human is not None:
            self._set_targets_from_prepared(prepared_human)
        elif not targets_already_set:
            raise ValueError("prepared_human required when targets are not cached")

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

    def _set_targets_from_prepared(self, prepared_human: dict) -> None:
        import mink

        if self.gmr.use_ik_match_table1:
            for entry in self.gmr.task_frames1:
                pos, rot = self.gmr._resolve_ik_target(entry, prepared_human)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

        if self.gmr.use_ik_match_table2:
            for entry in self.gmr.task_frames2:
                pos, rot = self.gmr._resolve_ik_target(entry, prepared_human)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def _build_qpos_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        model = self.gmr.model
        lower = np.full(model.nq, -np.inf, dtype=float)
        upper = np.full(model.nq, np.inf, dtype=float)

        for j in range(model.njnt):
            if not model.jnt_limited[j]:
                continue
            qadr = model.jnt_qposadr[j]
            jtype = model.jnt_type[j]
            lo, hi = model.jnt_range[j]
            if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                lower[qadr] = lo
                upper[qadr] = hi

        return lower, upper

    def _is_arm_joint(self, joint_name: str) -> bool:
        name = joint_name.lower()
        return any(kw in name for kw in ("shoulder", "elbow", "wrist"))

    def _is_waist_joint(self, joint_name: str) -> bool:
        return "waist" in joint_name.lower()

    def _include_in_smooth(self, joint_name: str) -> bool:
        if self.config.smooth_joints == "none":
            return False
        if self._is_leg_joint(joint_name):
            return self.config.smooth_legs
        mode = self.config.smooth_joints
        if mode == "all":
            return True
        if mode == "arms_only":
            return self._is_arm_joint(joint_name)
        # upper_body: everything except legs
        return True

    def _is_leg_joint(self, joint_name: str) -> bool:
        name = joint_name.lower()
        return any(kw in name for kw in ("hip", "knee", "ankle"))

    def _build_leg_q_indices(self) -> np.ndarray:
        leg_q: List[int] = []
        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if self._is_leg_joint(joint_name):
                leg_q.append(int(self.model.jnt_qposadr[j]))
        return np.asarray(leg_q, dtype=int)

    def _build_opt_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """GN / smoothness indices: hinge + slide joints only (root from IK)."""
        v_indices: List[int] = []
        smooth_q: List[int] = []

        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype == mj.mjtJoint.mjJNT_FREE:
                if self.config.optimize_root:
                    v_indices.extend(range(min(6, self.model.nv)))
                    smooth_q.extend(range(min(3, self.model.nq)))
                continue
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or ""
            if not self._include_in_smooth(joint_name):
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
        return dq_lim, ddq_lim

    def _flat_bounds(self, n_frames: int) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for _ in range(n_frames):
            for lo, hi in zip(self._qpos_lower, self._qpos_upper):
                bounds.append((lo, hi))
        return bounds
