"""Sliding-window kinematic trajectory optimization for real-time retargeting.

Without a physical robot, the previous optimized ``qpos`` stored in the
MuJoCo configuration acts as kinematic state feedback (open-loop MPC with
model-based warm start).

``mode="fast"`` (default) only refines the current frame against a short
history. ``mode="full"`` joint-optimizes every frame in the window via
L-BFGS-B and is much slower.
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
    w_velocity: float = 2.0
    w_acceleration: float = 10.0
    w_anchor: float = 50.0
    ik_warmstart_iters: int = 3
    fast_opt_iter: int = 5
    max_opt_iter: int = 25
    opt_tol: float = 1e-4
    min_frames: int = 2


class SlidingWindowRetargeter:
    """Causal sliding-window retargeting with kinematic warm start."""

    def __init__(
        self,
        retargeter: GeneralMotionRetargeting,
        config: Optional[SlidingWindowConfig] = None,
    ) -> None:
        self.gmr = retargeter
        self.config = config or SlidingWindowConfig()
        if self.config.window_size < self.config.min_frames:
            raise ValueError("window_size must be >= min_frames")

        self._prepared_human: Deque[dict] = deque(maxlen=self.config.window_size)
        self._q_window: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._qpos_lower, self._qpos_upper = self._build_qpos_bounds()
        self._frame_index = 0

    @property
    def frame_index(self) -> int:
        return self._frame_index

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
            return self._finalize_qpos(q_curr, prepared, human_data, offset_to_ground)

        q_prev = self._q_window[-1]
        q_prev2 = self._q_window[-2] if len(self._q_window) >= 2 else q_prev
        q_opt = self._optimize_single_frame(q_curr, q_prev, q_prev2)
        return self._finalize_qpos(q_opt, prepared, human_data, offset_to_ground)

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

    def _finalize_qpos(
        self,
        qpos: np.ndarray,
        prepared: dict,
        human_data,
        offset_to_ground: bool,
    ) -> np.ndarray:
        self.gmr.configuration.data.qpos[:] = qpos
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)

        freeze_base = (
            self.gmr.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.gmr.planar_base_cfg)
        )
        if freeze_base:
            self.gmr._snap_planar_base_qpos(prepared, raw_human_data=human_data)
            qpos = self.gmr.configuration.data.qpos.copy()

        if self.gmr.contact_ground.fix_penetration:
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

    def _optimize_single_frame(
        self,
        q_init: np.ndarray,
        q_prev: np.ndarray,
        q_prev2: np.ndarray,
    ) -> np.ndarray:
        bounds = list(zip(self._qpos_lower, self._qpos_upper))

        def objective(q: np.ndarray) -> float:
            cost = self._frame_tracking_cost(q, targets_already_set=True)
            if self.config.w_velocity > 0.0:
                delta = q - q_prev
                cost += self.config.w_velocity * float(np.dot(delta, delta))
            if self.config.w_acceleration > 0.0:
                acc = q - 2.0 * q_prev + q_prev2
                cost += self.config.w_acceleration * float(np.dot(acc, acc))
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

    def _window_cost(
        self,
        q_window: np.ndarray,
        prepared_human: Sequence[dict],
        anchor: np.ndarray,
    ) -> float:
        cost = 0.0
        for q, human in zip(q_window, prepared_human):
            self._set_targets_from_prepared(human)
            cost += self._frame_tracking_cost(q, targets_already_set=True)

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

    def _flat_bounds(self, n_frames: int) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for _ in range(n_frames):
            for lo, hi in zip(self._qpos_lower, self._qpos_upper):
                bounds.append((lo, hi))
        return bounds
