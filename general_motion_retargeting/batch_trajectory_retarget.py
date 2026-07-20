"""Offline batch retargeting trajectory optimization.

Optimizes robot joint trajectories jointly per window:
  min  Σ_t FK_tracking(q_t, human_target_t) + w_v||Δq||² + w_a||Δ²q||²

Default solver is multi-frame Gauss-Newton (linearized least squares per step),
not L-BFGS. GMR is used only for preprocessing, FK targets, and optional bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Literal, Sequence, Tuple

import mujoco as mj
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from .trajectory_optimization_retarget import (
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationRetargeter,
)


@dataclass
class BatchTrajectoryConfig:
    """Offline batch multi-frame q TO (paper-style batch retargeting)."""

    strategy: str = "sliding_window"  # sliding_window | full
    window_size: int = 32
    window_stride: int = 16
    w_velocity: float = 2.0
    w_acceleration: float = 10.0
    w_anchor: float = 0.0
    window_anchor_weight: float = 2.0
    solver: Literal["gn", "lbfgs"] = "gn"
    gn_steps: int = 3
    gn_damping: float = 0.1
    gn_max_step: float = 0.05
    gn_line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    profile: bool = False
    enable_foot_penalties: bool = True
    w_foot_height: float = 50.0
    w_foot_slip: float = 2000.0
    w_foot_ik_anchor: float = 200.0
    w_root_xy_contact: float = 100.0
    w_contact_joint_anchor: float = 400.0
    lock_root_xy_on_contact: bool = False
    foot_contact_margin: float = 0.02
    foot_contact_from_ref: bool = True
    smooth_root_xyz: bool = False
    # Control-feasibility: scale the per-joint smoothing penalty by (M_jj/tau_max)^(2*power)
    # so the (fixed) smoothing budget is spent on joints closest to torque saturation
    # instead of uniformly. This upgrades the objective from kinematic smoothness to
    # control feasibility without changing the GN structure or run time.
    torque_weighted_smoothing: bool = False
    torque_weight_power: float = 1.0
    # Control-feasibility: adaptively boost the smoothing penalty on joints whose current
    # trajectory speed approaches the actuator velocity limit. Unlike the static torque
    # weighting, this is measured per GN step from the working trajectory, so the smoothing
    # budget is concentrated exactly on the joints that are actually saturating.
    vel_aware_smoothing: bool = False
    vel_ref: float = 9.42  # rad/s reference joint-speed limit (~3*pi)
    vel_aware_thresh: float = 0.7  # start boosting above this fraction of vel_ref
    vel_aware_gain: float = 6.0  # max extra multiplier at/above the limit
    # Control-feasibility: soft inverse-dynamics torque-limit barrier inside the window GN.
    # Penalises only the torque that exceeds kappa*tau_max (kappa = 1 - torque_limit_margin),
    # using RNE on the finite-difference (qvel, qacc) available across the window. Because
    # leg torques during stance need ground-reaction forces they are excluded by default;
    # scope "upper" constrains contact-independent joints (waist/arms), "all" adds legs.
    torque_limit_constraint: bool = False
    torque_limit_margin: float = 0.1
    torque_limit_weight: float = 20.0
    torque_limit_scope: str = "upper"  # "upper" | "all"
    # Risk-driven gating: scale torque_limit_weight by estimated headroom (see
    # _update_torque_limit_gate_from_window).  "off" = always full weight.
    torque_limit_gate_mode: str = "soft"  # "off" | "soft" | "hard"
    torque_limit_gate_r_on: float = 0.85  # soft: start ramp; hard: unused
    torque_limit_gate_r_full: float = 0.95  # soft: full weight; hard: trigger on
    torque_limit_gate_r_off: float = 0.85  # hard mode hysteresis off threshold
    torque_limit_gate_min_on_frames: int = 5
    torque_limit_gate_min_off_frames: int = 10
    torque_limit_gate_floor: float = 0.0  # min weight fraction when gate > 0
    max_opt_iter: int = 40
    opt_tol: float = 1e-5
    use_gmr_init: bool = True
    finalize_contact: bool = True
    verbose: bool = True
    show_progress: bool = True

    def to_to_config(self) -> TrajectoryOptimizationConfig:
        return TrajectoryOptimizationConfig(
            mode="full",
            w_velocity=self.w_velocity,
            w_acceleration=self.w_acceleration,
            w_anchor=self.w_anchor,
            max_opt_iter=self.max_opt_iter,
            opt_tol=self.opt_tol,
            gn_steps=self.gn_steps,
            gn_damping=self.gn_damping,
            gn_max_step=self.gn_max_step,
            use_gmr_init=self.use_gmr_init,
        )


class BatchTrajectoryRetargeter(TrajectoryOptimizationRetargeter):
    """Offline simultaneous multi-frame q optimizer for retargeting."""

    def __init__(
        self,
        retargeter,
        config: BatchTrajectoryConfig | None = None,
    ) -> None:
        self.batch_config = config or BatchTrajectoryConfig()
        super().__init__(retargeter, self.batch_config.to_to_config())
        if not self.batch_config.smooth_root_xyz:
            self._smooth_qidx = np.asarray(
                [q for q in self._smooth_qidx if q >= 3],
                dtype=int,
            )
        self._torque_w_by_dof = self._build_torque_smoothing_weights()
        # Static per-DoF smoothing weights (torque re-weighting), aligned to self._smooth_qidx.
        qadr_to_dof = {
            int(self.model.jnt_qposadr[j]): int(self.model.jnt_dofadr[j])
            for j in range(self.model.njnt)
        }
        self._smooth_qidx_dof = np.array(
            [qadr_to_dof.get(int(q), -1) for q in self._smooth_qidx], dtype=int
        )
        self._smooth_static_w = np.ones(len(self._smooth_qidx), dtype=float)
        if self._torque_w_by_dof:
            for i, dof in enumerate(self._smooth_qidx_dof):
                if dof >= 0:
                    self._smooth_static_w[i] = self._torque_w_by_dof.get(int(dof), 1.0)
        # Effective weights (static x dynamic velocity-aware) consumed by GN + cost.
        self._eff_smooth_qidx_w = self._smooth_static_w.copy()
        self._eff_smooth_w_by_dof = dict(self._torque_w_by_dof)
        self._torque_limit_joints = self._build_torque_limit_joints()
        self._M_full = np.zeros((self.model.nv, self.model.nv))
        self._torque_limit_gate = 1.0
        self._torque_gate_active = False
        self._torque_gate_on_streak = 0
        self._torque_gate_off_streak = 0
        self.last_torque_gate: float = 1.0
        self.last_torque_peak_ratio: float = 0.0
        self._torque_gate_updates = 0
        self._torque_gate_sum = 0.0
        self._torque_peak_ratio_max = 0.0
        self._foot_body_ids = self._resolve_foot_body_ids()
        self._ground_z = float(self.gmr.contact_ground.ground_aligner.ground_z)
        self._global_ref_contact: np.ndarray | None = None
        self._window_frame_offset: int = 0
        self.last_profile: Dict[str, float] = {}

    def _build_torque_smoothing_weights(self) -> Dict[int, float]:
        """Per-DoF smoothing weight ~ (M_jj/tau_max)^(2*power), normalised to unit mean.

        Effective joint torque scales as ``M_jj * qddot``; penalising ``(M_jj/tau_max)^2 *
        qddot^2`` therefore penalises squared normalised torque. Weights are computed once
        at the neutral pose (inertia varies little over the reachable range) so the GN
        Hessian keeps its constant per-joint structure. Non-actuated / base DoFs get 1.0.
        """
        if not self.batch_config.torque_weighted_smoothing:
            return {}
        data = mj.MjData(self.model)
        mj.mj_resetData(self.model, data)
        mj.mj_forward(self.model, data)
        M = np.zeros((self.model.nv, self.model.nv))
        mj.mj_fullM(self.model, data, M)
        power = float(self.batch_config.torque_weight_power)

        dof_ratio: Dict[int, float] = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                continue
            lo, hi = self.model.jnt_actfrcrange[j]
            taumax = float(max(abs(lo), abs(hi)))
            if taumax <= 0.0:
                continue
            dof = int(self.model.jnt_dofadr[j])
            m_jj = max(float(M[dof, dof]), 1e-4)
            dof_ratio[dof] = (m_jj / taumax) ** (2.0 * power)

        if not dof_ratio:
            return {}
        mean = float(np.mean(list(dof_ratio.values())))
        if mean <= 0.0:
            return {}
        return {dof: r / mean for dof, r in dof_ratio.items()}

    def _build_torque_limit_joints(self) -> List[Tuple[int, int, float]]:
        """(global dof, local index within opt vidx, tau_max) for torque-limited joints."""
        if not self.batch_config.torque_limit_constraint:
            return []
        upper_keys = ("waist", "shoulder", "elbow", "wrist")
        lower_keys = ("hip", "knee", "ankle")
        scope = self.batch_config.torque_limit_scope
        dof_to_local = {int(v): i for i, v in enumerate(self._opt_vidx)}
        out: List[Tuple[int, int, float]] = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                continue
            lo, hi = self.model.jnt_actfrcrange[j]
            taumax = float(max(abs(lo), abs(hi)))
            if taumax <= 0.0:
                continue
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or ""
            is_upper = any(k in name for k in upper_keys)
            is_lower = any(k in name for k in lower_keys)
            if scope == "upper" and not is_upper:
                continue
            if scope != "all" and scope != "upper" and not (is_upper or is_lower):
                continue
            dof = int(self.model.jnt_dofadr[j])
            local = dof_to_local.get(dof)
            if local is None:
                continue
            out.append((dof, local, taumax))
        return out

    def _window_torque_peak_ratio(self, q_win: np.ndarray) -> float:
        """Max |tau|/tau_max over constrained joints and interior window frames."""
        joints = self._torque_limit_joints
        n_frames = len(q_win)
        if not joints or n_frames < 3:
            return 0.0
        dt = self.dt
        model = self.model
        data = self.data
        tau = np.zeros(model.nv)
        v_plus = np.zeros(model.nv)
        v_minus = np.zeros(model.nv)
        peak = 0.0
        for t in range(1, n_frames - 1):
            mj.mj_differentiatePos(model, v_plus, dt, q_win[t], q_win[t + 1])
            mj.mj_differentiatePos(model, v_minus, dt, q_win[t - 1], q_win[t])
            data.qpos[:] = q_win[t]
            data.qvel[:] = 0.5 * (v_plus + v_minus)
            mj.mj_forward(model, data)
            data.qacc[:] = (v_plus - v_minus) / dt
            mj.mj_rne(model, data, 1, tau)
            for dof, _local, taumax in joints:
                peak = max(peak, abs(tau[dof]) / taumax)
        return float(peak)

    def _torque_limit_gate_from_ratio(self, r_peak: float) -> float:
        cfg = self.batch_config
        mode = str(cfg.torque_limit_gate_mode).lower()
        if not cfg.torque_limit_constraint or mode == "off":
            return 1.0
        if mode == "soft":
            r_on = float(cfg.torque_limit_gate_r_on)
            r_full = float(cfg.torque_limit_gate_r_full)
            if r_peak <= r_on:
                gate = 0.0
            elif r_peak >= r_full:
                gate = 1.0
            else:
                gate = (r_peak - r_on) / max(r_full - r_on, 1e-9)
            floor = float(cfg.torque_limit_gate_floor)
            return float(floor + (1.0 - floor) * gate)
        if mode == "hard":
            r_on = float(cfg.torque_limit_gate_r_full)
            r_off = float(cfg.torque_limit_gate_r_off)
            if self._torque_gate_active:
                if r_peak < r_off:
                    self._torque_gate_off_streak += 1
                    self._torque_gate_on_streak = 0
                    if self._torque_gate_off_streak >= int(cfg.torque_limit_gate_min_off_frames):
                        self._torque_gate_active = False
                        self._torque_gate_off_streak = 0
                else:
                    self._torque_gate_off_streak = 0
            elif r_peak > r_on:
                self._torque_gate_on_streak += 1
                self._torque_gate_off_streak = 0
                if self._torque_gate_on_streak >= int(cfg.torque_limit_gate_min_on_frames):
                    self._torque_gate_active = True
                    self._torque_gate_on_streak = 0
            else:
                self._torque_gate_on_streak = 0
            return 1.0 if self._torque_gate_active else 0.0
        return 1.0

    def _update_torque_limit_gate_from_window(self, q_win: np.ndarray) -> float:
        r_peak = self._window_torque_peak_ratio(q_win)
        gate = self._torque_limit_gate_from_ratio(r_peak)
        self._torque_limit_gate = gate
        self.last_torque_gate = gate
        self.last_torque_peak_ratio = r_peak
        self._torque_gate_updates += 1
        self._torque_gate_sum += gate
        self._torque_peak_ratio_max = max(self._torque_peak_ratio_max, r_peak)
        return gate

    def reset_torque_limit_gate(self) -> None:
        self._torque_limit_gate = 1.0
        self._torque_gate_active = False
        self._torque_gate_on_streak = 0
        self._torque_gate_off_streak = 0
        self.last_torque_gate = 1.0
        self.last_torque_peak_ratio = 0.0
        self._torque_gate_updates = 0
        self._torque_gate_sum = 0.0
        self._torque_peak_ratio_max = 0.0

    def torque_limit_gate_stats(self) -> Dict[str, float]:
        n = max(1, int(self._torque_gate_updates))
        return {
            "last_gate": float(self.last_torque_gate),
            "mean_gate": float(self._torque_gate_sum / n) if self._torque_gate_updates > 0 else 1.0,
            "last_peak_ratio": float(self.last_torque_peak_ratio),
            "max_peak_ratio": float(self._torque_peak_ratio_max),
            "updates": float(self._torque_gate_updates),
        }

    def _accumulate_window_torque_limit_gn(
        self, H: np.ndarray, g: np.ndarray, q_win: np.ndarray, m: int
    ) -> None:
        """Soft one-sided barrier on inverse-dynamics torque exceeding kappa*tau_max.

        Residual per active (frame, joint): e = (|tau| - kappa*tau_max) / tau_max, with the
        GN Jacobian approximated by the dominant inertial term ``d tau/d q ~ M_jj * accel
        stencil`` (central second difference). tau itself is the full RNE torque (inertia +
        Coriolis + gravity), so the barrier is placed at the true operating point.
        """
        joints = self._torque_limit_joints
        n_frames = len(q_win)
        if not joints or n_frames < 3:
            return
        dt = self.dt
        kappa = max(0.0, 1.0 - float(self.batch_config.torque_limit_margin))
        w = float(self.batch_config.torque_limit_weight) * float(self._torque_limit_gate)
        if w <= 0.0:
            return
        model = self.model
        data = self.data
        tau = np.zeros(model.nv)
        v_plus = np.zeros(model.nv)
        v_minus = np.zeros(model.nv)

        for t in range(1, n_frames - 1):
            mj.mj_differentiatePos(model, v_plus, dt, q_win[t], q_win[t + 1])
            mj.mj_differentiatePos(model, v_minus, dt, q_win[t - 1], q_win[t])
            qvel = 0.5 * (v_plus + v_minus)
            qacc = (v_plus - v_minus) / dt
            data.qpos[:] = q_win[t]
            data.qvel[:] = qvel
            mj.mj_forward(model, data)
            data.qacc[:] = qacc
            mj.mj_rne(model, data, 1, tau)
            mj.mj_fullM(model, data, self._M_full)

            for dof, local, taumax in joints:
                tj = tau[dof]
                bound = kappa * taumax
                if abs(tj) <= bound:
                    continue
                e0 = (tj - np.sign(tj) * bound) / taumax  # normalised signed excess
                c = (max(self._M_full[dof, dof], 1e-4) / taumax) / (dt * dt)
                i0 = (t - 1) * m + local
                i1 = t * m + local
                i2 = (t + 1) * m + local
                # J = c * [+1, -2, +1] on (t-1, t, t+1)
                H[i2, i2] += w * c * c
                H[i1, i1] += w * 4.0 * c * c
                H[i0, i0] += w * c * c
                H[i2, i1] += -2.0 * w * c * c
                H[i1, i2] += -2.0 * w * c * c
                H[i2, i0] += w * c * c
                H[i0, i2] += w * c * c
                H[i1, i0] += -2.0 * w * c * c
                H[i0, i1] += -2.0 * w * c * c
                ge = w * c * e0
                g[i2] += ge
                g[i1] += -2.0 * ge
                g[i0] += ge

    def _window_torque_cost(self, q_win: np.ndarray) -> float:
        joints = self._torque_limit_joints
        n_frames = len(q_win)
        if not joints or n_frames < 3:
            return 0.0
        dt = self.dt
        kappa = max(0.0, 1.0 - float(self.batch_config.torque_limit_margin))
        w = float(self.batch_config.torque_limit_weight) * float(self._torque_limit_gate)
        if w <= 0.0:
            return 0.0
        model = self.model
        data = self.data
        tau = np.zeros(model.nv)
        v_plus = np.zeros(model.nv)
        v_minus = np.zeros(model.nv)
        cost = 0.0
        for t in range(1, n_frames - 1):
            mj.mj_differentiatePos(model, v_plus, dt, q_win[t], q_win[t + 1])
            mj.mj_differentiatePos(model, v_minus, dt, q_win[t - 1], q_win[t])
            data.qpos[:] = q_win[t]
            data.qvel[:] = 0.5 * (v_plus + v_minus)
            mj.mj_forward(model, data)
            data.qacc[:] = (v_plus - v_minus) / dt
            mj.mj_rne(model, data, 1, tau)
            for dof, _local, taumax in joints:
                tj = tau[dof]
                bound = kappa * taumax
                if abs(tj) > bound:
                    e0 = (tj - np.sign(tj) * bound) / taumax
                    cost += w * e0 * e0
        return cost

    def _update_dynamic_smooth_weights(self, q_win: np.ndarray) -> None:
        """Recompute effective per-joint smoothing weights from the current window speed.

        For joints whose peak speed exceeds ``vel_aware_thresh * vel_ref``, the smoothing
        weight is ramped up (linearly to ``1 + vel_aware_gain`` at the limit) on top of the
        static torque weight. Called once per GN step so the boost tracks the working
        trajectory. When velocity-aware smoothing is off this is a no-op.
        """
        if not self.batch_config.vel_aware_smoothing or len(self._smooth_qidx) == 0:
            return
        if q_win.shape[0] < 2:
            return
        dt = self.dt
        vref = max(float(self.batch_config.vel_ref), 1e-6)
        thresh = float(self.batch_config.vel_aware_thresh)
        gain = float(self.batch_config.vel_aware_gain)

        # peak |speed| per smoothed column over the window
        vmax = np.abs(np.diff(q_win[:, self._smooth_qidx], axis=0)).max(axis=0) / dt
        frac = (vmax / vref - thresh) / max(1.0 - thresh, 1e-6)
        boost = 1.0 + gain * np.clip(frac, 0.0, 1.0)
        self._eff_smooth_qidx_w = self._smooth_static_w * boost
        self._eff_smooth_w_by_dof = {
            int(dof): float(self._eff_smooth_qidx_w[i])
            for i, dof in enumerate(self._smooth_qidx_dof)
            if dof >= 0
        }

    def _resolve_foot_body_ids(self) -> List[int]:
        ids = list(getattr(self.gmr.contact_ground, "foot_body_ids", []) or [])
        if ids:
            return ids
        out: List[int] = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            try:
                out.append(self.model.body(name).id)
            except KeyError:
                continue
        return out

    def _foot_penalties_active(self) -> bool:
        return bool(
            self.batch_config.enable_foot_penalties
            and self._foot_body_ids
            and (
                self.batch_config.w_foot_height > 0.0
                or self.batch_config.w_foot_slip > 0.0
                or self.batch_config.w_foot_ik_anchor > 0.0
                or self.batch_config.w_root_xy_contact > 0.0
                or self.batch_config.w_contact_joint_anchor > 0.0
            )
        )

    def _foot_positions(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)
        return np.stack([self.data.xpos[bid].copy() for bid in self._foot_body_ids])

    def _foot_positions_jac(
        self,
        q: np.ndarray,
        vidx: np.ndarray,
        jacp: np.ndarray,
    ) -> tuple[np.ndarray, List[np.ndarray]]:
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)
        pos_list = []
        jac_list = []
        for bid in self._foot_body_ids:
            body_pos = self.data.xpos[bid].copy()
            mj.mj_jac(self.model, self.data, jacp, None, body_pos, bid)
            pos_list.append(body_pos)
            jac_list.append(jacp[:, vidx].copy())
        return np.stack(pos_list), jac_list

    def _foot_contact_masks(
        self,
        foot_pos_seq: np.ndarray,
        ref_foot_pos_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Contact mask aligned with foot_slip_metrics (per-foot min-z + margin).

        When ``foot_contact_from_ref`` is set, uses the IK/bootstrap reference
        trajectory so the optimizer cannot drop penalties by lifting the foot.
        """
        margin = self.batch_config.foot_contact_margin
        src = (
            ref_foot_pos_seq
            if (
                ref_foot_pos_seq is not None
                and self.batch_config.foot_contact_from_ref
            )
            else foot_pos_seq
        )
        z_min_per_foot = np.min(src[:, :, 2], axis=0)
        return src[:, :, 2] <= (z_min_per_foot[None, :] + margin)

    def _batch_contact_mask(self, q_ref: np.ndarray) -> np.ndarray:
        """Full-sequence contact mask (matches foot_slip_metrics on the reference traj)."""
        foot_pos = np.stack([self._foot_positions(q) for q in q_ref])
        return self._foot_contact_masks(foot_pos, foot_pos)

    def _window_contact_seq(self, n_frames: int) -> np.ndarray | None:
        if self._global_ref_contact is None:
            return None
        start = self._window_frame_offset
        end = start + n_frames
        return self._global_ref_contact[start:end]

    def _window_foot_cost(
        self,
        q_window: np.ndarray,
        q_ref_window: np.ndarray | None = None,
    ) -> float:
        if not self._foot_penalties_active():
            return 0.0

        foot_pos_seq = np.stack([self._foot_positions(q) for q in q_window])
        ref_seq = None
        ref_foot_seq = None
        if q_ref_window is not None:
            ref_foot_seq = np.stack([self._foot_positions(q) for q in q_ref_window])
            if self.batch_config.w_foot_ik_anchor > 0.0:
                ref_seq = ref_foot_seq
        contact_seq = self._window_contact_seq(foot_pos_seq.shape[0])
        if contact_seq is None:
            contact_seq = self._foot_contact_masks(foot_pos_seq, ref_foot_seq)

        cost = 0.0
        w_h = self.batch_config.w_foot_height
        w_s = self.batch_config.w_foot_slip
        w_a = self.batch_config.w_foot_ik_anchor
        w_root = self.batch_config.w_root_xy_contact
        w_j = self.batch_config.w_contact_joint_anchor
        ground = self._ground_z

        if w_h > 0.0:
            for t in range(foot_pos_seq.shape[0]):
                for f_idx in range(foot_pos_seq.shape[1]):
                    if contact_seq[t, f_idx]:
                        dz = foot_pos_seq[t, f_idx, 2] - ground
                        cost += w_h * float(dz * dz)

        if w_s > 0.0 and foot_pos_seq.shape[0] >= 2:
            for t in range(1, foot_pos_seq.shape[0]):
                both = contact_seq[t] & contact_seq[t - 1]
                for f_idx in np.where(both)[0]:
                    dxy = foot_pos_seq[t, f_idx, :2] - foot_pos_seq[t - 1, f_idx, :2]
                    cost += w_s * float(np.dot(dxy, dxy))

        if ref_seq is not None and w_a > 0.0:
            for t in range(foot_pos_seq.shape[0]):
                for f_idx in range(foot_pos_seq.shape[1]):
                    if contact_seq[t, f_idx]:
                        dxy = foot_pos_seq[t, f_idx, :2] - ref_seq[t, f_idx, :2]
                        cost += w_a * float(np.dot(dxy, dxy))

        if (
            q_ref_window is not None
            and w_root > 0.0
            and self.model.nq >= 2
        ):
            for t in range(foot_pos_seq.shape[0]):
                if not np.any(contact_seq[t]):
                    continue
                dxy = q_window[t, :2] - q_ref_window[t, :2]
                cost += w_root * float(np.dot(dxy, dxy))

        if q_ref_window is not None and w_j > 0.0:
            for t in range(foot_pos_seq.shape[0]):
                if not np.any(contact_seq[t]):
                    continue
                for qadr in self._smooth_qidx:
                    e = q_window[t, qadr] - q_ref_window[t, qadr]
                    cost += w_j * float(e * e)
        return cost

    def _accumulate_window_foot_gn(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q_win: np.ndarray,
        vidx: np.ndarray,
        m: int,
        jacp: np.ndarray,
        q_ref_win: np.ndarray | None = None,
    ) -> None:
        if not self._foot_penalties_active():
            return

        w_h = self.batch_config.w_foot_height
        w_s = self.batch_config.w_foot_slip
        w_a = self.batch_config.w_foot_ik_anchor
        w_root = self.batch_config.w_root_xy_contact
        w_j = self.batch_config.w_contact_joint_anchor
        ground = self._ground_z
        n_frames = len(q_win)
        q_to_v = {
            int(self.model.jnt_qposadr[self.model.dof_jntid[v]]): i
            for i, v in enumerate(vidx)
        }
        smooth_v, smooth_q = self._smooth_v_in_frame(vidx)

        foot_cache: List[tuple[np.ndarray, List[np.ndarray]]] = []
        for t in range(n_frames):
            pos, jac_list = self._foot_positions_jac(q_win[t], vidx, jacp)
            foot_cache.append((pos, jac_list))

        foot_pos_seq = np.stack([c[0] for c in foot_cache])
        ref_seq = None
        ref_foot_seq = None
        if q_ref_win is not None:
            ref_foot_seq = np.stack([self._foot_positions(q) for q in q_ref_win])
            if w_a > 0.0:
                ref_seq = ref_foot_seq
        contact_seq = self._window_contact_seq(n_frames)
        if contact_seq is None:
            contact_seq = self._foot_contact_masks(foot_pos_seq, ref_foot_seq)

        for t in range(n_frames):
            pos, jac_list = foot_cache[t]
            contact = contact_seq[t]
            off_t = t * m
            if w_h > 0.0:
                for f_idx in range(pos.shape[0]):
                    if not contact[f_idx]:
                        continue
                    err = pos[f_idx, 2] - ground
                    Jz = jac_list[f_idx][2:3, :]
                    H[off_t : off_t + m, off_t : off_t + m] += w_h * (Jz.T @ Jz)
                    g[off_t : off_t + m] += w_h * (Jz.T @ np.asarray([err]))

            if ref_seq is not None and w_a > 0.0:
                for f_idx in range(pos.shape[0]):
                    if not contact[f_idx]:
                        continue
                    err = pos[f_idx, :2] - ref_seq[t, f_idx, :2]
                    Jxy = jac_list[f_idx][:2, :]
                    H[off_t : off_t + m, off_t : off_t + m] += w_a * (Jxy.T @ Jxy)
                    g[off_t : off_t + m] += w_a * (Jxy.T @ err)

            if w_s <= 0.0 or t == 0:
                continue
            pos_prev, jac_prev = foot_cache[t - 1]
            contact_prev = contact_seq[t - 1]
            off_prev = (t - 1) * m
            both = contact & contact_prev
            for f_idx in np.where(both)[0]:
                err = pos[f_idx, :2] - pos_prev[f_idx, :2]
                Jt = jac_list[f_idx][:2, :]
                Jp = jac_prev[f_idx][:2, :]
                H[off_t : off_t + m, off_t : off_t + m] += w_s * (Jt.T @ Jt)
                H[off_prev : off_prev + m, off_prev : off_prev + m] += w_s * (Jp.T @ Jp)
                H[off_t : off_t + m, off_prev : off_prev + m] -= w_s * (Jt.T @ Jp)
                H[off_prev : off_prev + m, off_t : off_t + m] -= w_s * (Jp.T @ Jt)
                g[off_t : off_t + m] += w_s * (Jt.T @ err)
                g[off_prev : off_prev + m] -= w_s * (Jp.T @ err)

        if q_ref_win is not None and w_root > 0.0:
            for t in range(n_frames):
                if not np.any(contact_seq[t]):
                    continue
                off_t = t * m
                for qadr in (0, 1):
                    if qadr not in q_to_v:
                        continue
                    vi = q_to_v[qadr]
                    err = q_win[t, qadr] - q_ref_win[t, qadr]
                    idx = off_t + vi
                    H[idx, idx] += w_root
                    g[idx] += w_root * err

        if q_ref_win is not None and w_j > 0.0:
            for t in range(n_frames):
                if not np.any(contact_seq[t]):
                    continue
                off_t = t * m
                for vi, qadr in zip(smooth_v, smooth_q):
                    err = q_win[t, qadr] - q_ref_win[t, qadr]
                    idx = off_t + vi
                    H[idx, idx] += w_j
                    g[idx] += w_j * err

    def retarget_batch(
        self,
        human_frames: Sequence[dict],
        offset_to_ground: bool = False,
    ) -> np.ndarray:
        if not human_frames:
            return np.zeros((0, self.model.nq), dtype=float)

        profile = self.batch_config.profile
        t_total = time.perf_counter() if profile else 0.0

        prepared_list: List[dict] = []
        targets_list = []
        frame_iter = human_frames
        if self.batch_config.show_progress:
            frame_iter = tqdm(
                human_frames,
                desc="[batch-to] prepare targets",
                unit="frame",
            )
        t0 = time.perf_counter() if profile else 0.0
        for frame in frame_iter:
            prepared = self.gmr._prepare_scaled_human_data(frame, offset_to_ground)
            prepared_list.append(prepared)
            targets_list.append(self._targets_for_prepared(prepared))
        t_prepare = (time.perf_counter() - t0) * 1000.0 if profile else 0.0

        t0 = time.perf_counter() if profile else 0.0
        q_init = self._bootstrap_q_sequence(human_frames, prepared_list, offset_to_ground)
        t_bootstrap = (time.perf_counter() - t0) * 1000.0 if profile else 0.0
        if self._foot_penalties_active():
            self._global_ref_contact = self._batch_contact_mask(q_init)
        else:
            self._global_ref_contact = None
        if self.batch_config.verbose:
            print(
                f"[batch-to] strategy={self.batch_config.strategy}, "
                f"solver={self.batch_config.solver}, "
                f"frames={len(human_frames)}, nq={self.model.nq}"
            )
            if self.batch_config.solver == "gn":
                print(f"[batch-to] gn_steps={self.batch_config.gn_steps}")
            else:
                print(f"[batch-to] max_iter={self.batch_config.max_opt_iter}")
            if self.batch_config.strategy == "sliding_window":
                print(
                    f"[batch-to] window_size={self.batch_config.window_size}, "
                    f"stride={self.batch_config.window_stride}"
                )
            if self._foot_penalties_active():
                print(
                    f"[batch-to] foot penalties: w_h={self.batch_config.w_foot_height}, "
                    f"w_slip={self.batch_config.w_foot_slip}, "
                    f"w_ik_anchor={self.batch_config.w_foot_ik_anchor}, "
                    f"w_root_xy={self.batch_config.w_root_xy_contact}, "
                    f"w_joint_anchor={self.batch_config.w_contact_joint_anchor}, "
                    f"contact_from_ref={self.batch_config.foot_contact_from_ref}, "
                    f"smooth_root_xyz={self.batch_config.smooth_root_xyz}, "
                    f"feet={len(self._foot_body_ids)}"
                )

        t0 = time.perf_counter() if profile else 0.0
        if self.batch_config.strategy == "full":
            q_opt = self._optimize_full_sequence(q_init, targets_list)
        elif self.batch_config.strategy == "sliding_window":
            q_opt = self._optimize_sliding_windows(q_init, targets_list)
        else:
            raise ValueError(
                f"Unknown batch strategy: {self.batch_config.strategy!r} "
                "(use sliding_window or full)"
            )
        t_optimize = (time.perf_counter() - t0) * 1000.0 if profile else 0.0

        t0 = time.perf_counter() if profile else 0.0
        if self.batch_config.finalize_contact:
            q_out = []
            finalize_iter = zip(q_opt, prepared_list, human_frames)
            if self.batch_config.show_progress:
                finalize_iter = tqdm(
                    finalize_iter,
                    total=len(human_frames),
                    desc="[batch-to] finalize contact",
                    unit="frame",
                )
            for q, prepared, frame in finalize_iter:
                q_out.append(
                    self._finalize_qpos(q.copy(), prepared, frame, offset_to_ground)
                )
            q_out_arr = np.asarray(q_out)
        else:
            q_out_arr = np.asarray(q_opt)

        if (
            self._foot_penalties_active()
            and self.batch_config.w_root_xy_contact > 0.0
            and self.batch_config.lock_root_xy_on_contact
        ):
            q_out_arr = self._stabilize_root_xy_on_contact(q_out_arr, q_init)

        t_finalize = (time.perf_counter() - t0) * 1000.0 if profile else 0.0
        if profile:
            n = max(len(human_frames), 1)
            t_all = (time.perf_counter() - t_total) * 1000.0
            self.last_profile = {
                "prepare_ms": t_prepare,
                "bootstrap_ms": t_bootstrap,
                "optimize_ms": t_optimize,
                "finalize_ms": t_finalize,
                "total_ms": t_all,
                "ms_per_frame": t_all / n,
                "effective_fps": 1000.0 * n / max(t_all, 1e-9),
                "n_frames": float(len(human_frames)),
            }

        return q_out_arr

    def _stabilize_root_xy_on_contact(
        self,
        q_out: np.ndarray,
        q_ref: np.ndarray,
    ) -> np.ndarray:
        """Lock floating-base XY to IK on reference-contact frames (reduces stance slip)."""
        ref_foot = np.stack([self._foot_positions(q) for q in q_ref])
        contact = self._foot_contact_masks(ref_foot, ref_foot)
        out = q_out.copy()
        if self.model.nq < 2:
            return out
        for t in range(len(out)):
            if np.any(contact[t]):
                out[t, 0] = q_ref[t, 0]
                out[t, 1] = q_ref[t, 1]
        return out

    def _bootstrap_q_sequence(
        self,
        human_frames: Sequence[dict],
        prepared_list: Sequence[dict],
        offset_to_ground: bool,
    ) -> np.ndarray:
        n = len(human_frames)
        q_init = np.zeros((n, self.model.nq), dtype=float)
        if not self.batch_config.use_gmr_init:
            q0 = self.gmr.configuration.data.qpos.copy()
            for i in range(n):
                q_init[i] = q0
            return q_init

        frame_iter = enumerate(human_frames)
        if self.batch_config.show_progress:
            frame_iter = enumerate(
                tqdm(human_frames, desc="[batch-to] GMR bootstrap", unit="frame")
            )
        for i, frame in frame_iter:
            q_init[i] = self.gmr.retarget(frame, offset_to_ground=offset_to_ground)
        return q_init

    def _window_cost(
        self,
        q_window: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
        q_ref_window: np.ndarray | None = None,
    ) -> float:
        cost = 0.0
        pairs = zip(q_window, targets_list)
        if (
            self.batch_config.show_progress
            and getattr(self, "_in_lbfgs", False)
            and getattr(self, "_show_fk_progress", False)
        ):
            pairs = tqdm(
                pairs,
                total=len(targets_list),
                desc="  FK cost",
                leave=False,
                unit="frame",
            )
        for q, targets in pairs:
            cost += self._fk_tracking_cost(q, targets)

        if self.config.w_velocity > 0.0 and q_window.shape[0] >= 2:
            smooth_qidx = self._smooth_qidx
            if len(smooth_qidx) > 0:
                diffs = np.diff(q_window[:, smooth_qidx], axis=0)
                col = np.sum(diffs * diffs, axis=0) * self._eff_smooth_qidx_w
                cost += self.config.w_velocity * float(np.sum(col))

        if self.config.w_acceleration > 0.0 and q_window.shape[0] >= 3:
            smooth_qidx = self._smooth_qidx
            if len(smooth_qidx) > 0:
                acc = (
                    q_window[2:, smooth_qidx]
                    - 2.0 * q_window[1:-1, smooth_qidx]
                    + q_window[:-2, smooth_qidx]
                )
                col = np.sum(acc * acc, axis=0) * self._eff_smooth_qidx_w
                cost += self.config.w_acceleration * float(np.sum(col))

        anchor_w = getattr(self, "_window_anchor_w", self.config.w_anchor)
        if anchor_w > 0.0:
            delta = q_window[0] - anchor
            cost += anchor_w * float(np.dot(delta, delta))

        cost += self._window_foot_cost(q_window, q_ref_window)
        cost += self._window_torque_cost(q_window)
        return cost

    def _window_starts(self, n_frames: int) -> List[int]:
        H = self.batch_config.window_size
        S = self.batch_config.window_stride
        if n_frames <= H:
            return [0]
        starts = list(range(0, n_frames, S))
        last = n_frames - H
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _optimize_sliding_windows(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
    ) -> np.ndarray:
        n = len(q_init)
        H = self.batch_config.window_size
        S = self.batch_config.window_stride
        if n <= H:
            return self._optimize_window(
                q_init,
                targets_list,
                q_init[0].copy(),
                q_ref=q_init,
                desc="[batch-to] window",
                anchor_weight=self.batch_config.w_anchor,
            )

        starts = self._window_starts(n)
        q_out = q_init.copy()
        win_iter: Sequence[int] = starts
        if self.batch_config.show_progress:
            win_iter = tqdm(starts, desc="[batch-to] sliding windows", unit="win")

        for wi, start in enumerate(win_iter):
            end = min(start + H, n)
            win_len = end - start
            if win_len < 2:
                continue

            q_win = q_out[start:end].copy()
            tgt_win = targets_list[start:end]
            anchor = q_out[start].copy()
            anchor_w = self.batch_config.w_anchor
            if start > 0:
                anchor_w = max(anchor_w, self.batch_config.window_anchor_weight)

            self._window_frame_offset = start
            q_opt_win = self._optimize_window(
                q_win,
                tgt_win,
                anchor,
                q_ref=q_init[start:end],
                desc=f"win {wi + 1}/{len(starts)} [{start}:{end})",
                anchor_weight=anchor_w,
            )

            if wi == 0:
                commit_end = min(start + S, n)
            elif end >= n:
                commit_end = n
            else:
                commit_end = start + S

            q_out[start:commit_end] = q_opt_win[: commit_end - start]

        return q_out

    def _optimize_full_sequence(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
    ) -> np.ndarray:
        return self._optimize_window(
            q_init,
            targets_list,
            q_init[0].copy(),
            q_ref=q_init,
            desc="[batch-to] full window",
            anchor_weight=self.batch_config.w_anchor,
        )

    def _optimize_window(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
        *,
        q_ref: np.ndarray | None = None,
        desc: str = "[batch-to] window",
        anchor_weight: float | None = None,
        show_inner_progress: bool | None = None,
    ) -> np.ndarray:
        if self.batch_config.solver == "gn":
            return self._optimize_gn_window(
                q_init,
                targets_list,
                anchor,
                q_ref=q_ref,
                desc=desc,
                anchor_weight=anchor_weight,
            )
        return self._optimize_lbfgs_window(
            q_init,
            targets_list,
            anchor,
            q_ref=q_ref,
            desc=desc,
            anchor_weight=anchor_weight,
            show_inner_progress=show_inner_progress,
        )

    def _smooth_v_in_frame(self, vidx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q_to_v = {int(qadr): i for i, qadr in enumerate(self._smooth_qidx)}
        smooth_v: List[int] = []
        smooth_q: List[int] = []
        for vi, v in enumerate(vidx):
            j = self.model.dof_jntid[v]
            qadr = int(self.model.jnt_qposadr[j])
            if qadr in q_to_v:
                smooth_v.append(vi)
                smooth_q.append(qadr)
        return np.asarray(smooth_v, dtype=int), np.asarray(smooth_q, dtype=int)

    def _accumulate_frame_fk_gn(
        self,
        H: np.ndarray,
        g: np.ndarray,
        offset: int,
        q: np.ndarray,
        targets: dict,
        vidx: np.ndarray,
        jacp: np.ndarray,
        jacr: np.ndarray,
    ) -> None:
        m = len(vidx)
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

        for entry in self._track_entries:
            target = targets.get(entry.robot_frame)
            if target is None:
                continue
            pos_t, quat_t = target
            body_pos = self.data.xpos[entry.body_id]

            if entry.pos_weight > 0.0:
                pos_e = body_pos - pos_t
                mj.mj_jac(self.model, self.data, jacp, None, body_pos, entry.body_id)
                J = jacp[:, vidx]
                w = entry.pos_weight
                sl = slice(offset, offset + m)
                H[sl, sl] += w * (J.T @ J)
                g[sl] += w * (J.T @ pos_e)

            if entry.rot_weight > 0.0:
                rot_body = self.data.xmat[entry.body_id].reshape(3, 3)
                rot_t = R.from_quat(self.gmr._quat_wxyz_to_xyzw(quat_t))
                rot_err = (rot_t.inv() * R.from_matrix(rot_body)).as_rotvec()
                mj.mj_jac(self.model, self.data, None, jacr, body_pos, entry.body_id)
                J = jacr[:, vidx]
                w = entry.rot_weight
                sl = slice(offset, offset + m)
                H[sl, sl] += w * (J.T @ J)
                g[sl] += w * (J.T @ rot_err)

    def _accumulate_window_temporal_gn(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q_win: np.ndarray,
        smooth_v: np.ndarray,
        smooth_q: np.ndarray,
        m: int,
        w_velocity: float,
        w_acceleration: float,
    ) -> None:
        n_frames = len(q_win)
        if len(smooth_v) == 0:
            return

        tw = self._eff_smooth_w_by_dof

        if w_velocity > 0.0:
            for t in range(1, n_frames):
                off_t = t * m
                off_tm1 = (t - 1) * m
                e = q_win[t, smooth_q] - q_win[t - 1, smooth_q]
                for k, vi in enumerate(smooth_v):
                    i_t = off_t + vi
                    i_m = off_tm1 + vi
                    w = w_velocity * tw.get(vi, 1.0)
                    H[i_t, i_t] += w
                    H[i_m, i_m] += w
                    H[i_t, i_m] -= w
                    H[i_m, i_t] -= w
                    g[i_t] += w * e[k]
                    g[i_m] -= w * e[k]

        if w_acceleration > 0.0 and n_frames >= 3:
            for t in range(2, n_frames):
                acc = (
                    q_win[t, smooth_q]
                    - 2.0 * q_win[t - 1, smooth_q]
                    + q_win[t - 2, smooth_q]
                )
                offs = ((t - 2) * m, (t - 1) * m, t * m)
                for k, vi in enumerate(smooth_v):
                    i0 = offs[0] + vi
                    i1 = offs[1] + vi
                    i2 = offs[2] + vi
                    w = w_acceleration * tw.get(vi, 1.0)
                    e = acc[k]
                    H[i2, i2] += w
                    H[i1, i1] += 4.0 * w
                    H[i0, i0] += w
                    H[i2, i1] -= 2.0 * w
                    H[i1, i2] -= 2.0 * w
                    H[i2, i0] += w
                    H[i0, i2] += w
                    H[i1, i0] -= 2.0 * w
                    H[i0, i1] -= 2.0 * w
                    g[i2] += w * e
                    g[i1] -= 2.0 * w * e
                    g[i0] += w * e

    def _accumulate_window_anchor_gn(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q0: np.ndarray,
        anchor: np.ndarray,
        vidx: np.ndarray,
        anchor_weight: float,
    ) -> None:
        if anchor_weight <= 0.0:
            return
        m = len(vidx)
        for vi, v in enumerate(vidx):
            j = self.model.dof_jntid[v]
            qadr = int(self.model.jnt_qposadr[j])
            err = q0[qadr] - anchor[qadr]
            H[vi, vi] += anchor_weight
            g[vi] += anchor_weight * err

    def _optimize_gn_window(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
        *,
        q_ref: np.ndarray | None = None,
        desc: str = "[batch-to] GN",
        anchor_weight: float | None = None,
    ) -> np.ndarray:
        anchor = anchor.copy()
        anchor_w = self.batch_config.w_anchor if anchor_weight is None else anchor_weight
        q_ref_win = q_ref if q_ref is not None else q_init
        q_win = q_init.copy()
        n_frames = len(q_win)
        vidx = self._opt_vidx
        m = len(vidx)
        nvar = n_frames * m
        smooth_v, smooth_q = self._smooth_v_in_frame(vidx)
        damp = self.batch_config.gn_damping
        max_step = self.batch_config.gn_max_step
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        dq_v = np.zeros(self.model.nv)

        prev_anchor_w = getattr(self, "_window_anchor_w", self.config.w_anchor)
        self._window_anchor_w = anchor_w
        log_cost = self.batch_config.verbose
        try:
            cost_before = (
                self._window_cost(q_win, targets_list, anchor, q_ref_win)
                if log_cost
                else 0.0
            )
            alphas = self.batch_config.gn_line_search_alphas
            for step in range(self.batch_config.gn_steps):
                H = np.zeros((nvar, nvar), dtype=float)
                g = np.zeros(nvar, dtype=float)

                self._update_dynamic_smooth_weights(q_win)

                for t in range(n_frames):
                    self._accumulate_frame_fk_gn(
                        H,
                        g,
                        t * m,
                        q_win[t],
                        targets_list[t],
                        vidx,
                        jacp,
                        jacr,
                    )

                self._accumulate_window_anchor_gn(
                    H, g, q_win[0], anchor, vidx, anchor_w
                )
                self._accumulate_window_temporal_gn(
                    H, g, q_win, smooth_v, smooth_q, m, w_v, w_a
                )
                self._update_torque_limit_gate_from_window(q_win)
                self._accumulate_window_torque_limit_gn(H, g, q_win, m)
                self._accumulate_window_foot_gn(
                    H, g, q_win, vidx, m, jacp, q_ref_win
                )

                try:
                    dq_flat = np.linalg.solve(H + damp * np.eye(nvar), g)
                except np.linalg.LinAlgError:
                    if self.batch_config.verbose:
                        print(f"[batch-to] {desc} GN step {step + 1}: singular, stop")
                    break

                dq_flat = np.clip(dq_flat, -max_step, max_step)

                if len(alphas) <= 1:
                    alpha = alphas[0] if alphas else 1.0
                    for t in range(n_frames):
                        dq_v[:] = 0.0
                        dq_v[vidx] = -alpha * dq_flat[t * m : (t + 1) * m]
                        mj.mj_integratePos(self.model, q_win[t], dq_v, 1.0)
                        self._clip_hinge_qpos(q_win[t])
                    continue

                best_cost = self._window_cost(q_win, targets_list, anchor, q_ref_win)
                best_q = q_win.copy()
                for alpha in alphas:
                    q_trial = q_win.copy()
                    for t in range(n_frames):
                        dq_v[:] = 0.0
                        dq_v[vidx] = -alpha * dq_flat[t * m : (t + 1) * m]
                        mj.mj_integratePos(self.model, q_trial[t], dq_v, 1.0)
                        self._clip_hinge_qpos(q_trial[t])
                    trial_cost = self._window_cost(
                        q_trial, targets_list, anchor, q_ref_win
                    )
                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_q = q_trial
                q_win[:] = best_q

            cost_after = (
                self._window_cost(q_win, targets_list, anchor, q_ref_win)
                if log_cost
                else 0.0
            )
        finally:
            self._window_anchor_w = prev_anchor_w

        if log_cost:
            print(
                f"[batch-to] {desc} GN: steps={self.batch_config.gn_steps}, "
                f"cost {cost_before:.4f} -> {cost_after:.4f}"
            )
        return q_win

    def _optimize_lbfgs_window(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
        *,
        q_ref: np.ndarray | None = None,
        desc: str = "[batch-to] L-BFGS-B",
        anchor_weight: float | None = None,
        show_inner_progress: bool | None = None,
    ) -> np.ndarray:
        anchor = anchor.copy()
        q_ref_win = q_ref if q_ref is not None else q_init
        x0 = q_init.reshape(-1)
        bounds = self._flat_bounds(len(q_init))
        eval_count = [0]
        iter_count = [0]
        first_eval_s = [None]
        t0 = [None]
        pbar = None
        inner_progress = (
            self.batch_config.show_progress
            if show_inner_progress is None
            else show_inner_progress
        )
        if inner_progress:
            est_nfev = max(20, self.batch_config.max_opt_iter * 8)
            pbar = tqdm(
                total=est_nfev,
                desc=desc,
                unit="eval",
                dynamic_ncols=True,
                leave=show_inner_progress is not False,
            )

        prev_anchor_w = getattr(self, "_window_anchor_w", self.config.w_anchor)
        self._window_anchor_w = (
            self.config.w_anchor if anchor_weight is None else anchor_weight
        )
        self._in_lbfgs = True
        self._show_fk_progress = inner_progress

        def objective(x: np.ndarray) -> float:
            eval_count[0] += 1
            if t0[0] is None:
                t0[0] = time.perf_counter()
            q_window = x.reshape(q_init.shape)
            cost = self._window_cost(q_window, targets_list, anchor, q_ref_win)
            if first_eval_s[0] is None:
                first_eval_s[0] = time.perf_counter() - t0[0]
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    iter=iter_count[0],
                    cost=f"{cost:.2e}",
                    first_s=f"{first_eval_s[0]:.1f}",
                    refresh=False,
                )
            return cost

        def callback(_xk: np.ndarray) -> None:
            iter_count[0] += 1
            if pbar is not None:
                pbar.set_postfix(iter=iter_count[0], refresh=False)

        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback if pbar is not None else None,
                options={
                    "maxiter": self.batch_config.max_opt_iter,
                    "maxfun": max(500, self.batch_config.max_opt_iter * 30),
                    "ftol": self.batch_config.opt_tol,
                },
            )
        finally:
            self._in_lbfgs = False
            self._show_fk_progress = False
            self._window_anchor_w = prev_anchor_w
            if pbar is not None:
                pbar.close()

        q_opt = result.x.reshape(q_init.shape)
        if self.batch_config.verbose:
            status = "ok" if result.success else "partial"
            print(
                f"[batch-to] {desc} {status}: nit={result.nit}, "
                f"nfev={result.nfev}, cost={result.fun:.4f}"
            )
        return q_opt


# Deprecated aliases (clip → batch rename)
ClipTrajectoryConfig = BatchTrajectoryConfig
ClipTrajectoryRetargeter = BatchTrajectoryRetargeter
BatchTrajectoryRetargeter.retarget_clip = BatchTrajectoryRetargeter.retarget_batch
