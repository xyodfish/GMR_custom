"""MPC-like short-horizon Online QP retargeting.

Goal (product):
  - Keep absolute FK close to GMR (small loss OK)
  - Improve smoothness (dq / ddq / jerk) for trackability
  - Improve foot slip vs GMR (slip is treated as a GMR weakness)

Method:
  - Keep GMR human scaling + link targets
  - Soft warmstart: light IK (guide, not hard blend)
  - Linearize FK (+ foot + temporal) → convex QP in tangent space
  - Box + velocity constraints via DAQP (qpsolvers)
  - Receding horizon (default H=3 ≈ N=2); sequence mode uses preview

This is kinematic trajectory retargeting, not dynamics/control MPC: it has no
plant dynamics, control input, or feedback from the robot's executed state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Sequence

import mujoco as mj
import numpy as np
from qpsolvers import solve_qp

from .batch_trajectory_retarget import BatchTrajectoryConfig, BatchTrajectoryRetargeter
from .motion_retarget import GeneralMotionRetargeting


@dataclass
class OnlineQpConfig:
    """MPC-like short-horizon Online QP retargeting knobs."""

    preset: Literal["default", "smooth", "anti_slip"] = "default"
    horizon: int = 3
    sqp_iters: int = 3
    min_frames: int = 2
    w_velocity: float = 2.0
    w_acceleration: float = 8.0
    w_anchor: float = 1.0
    # Soft pull to light-IK / GMR seed (guide only; keep small).
    w_gmr: float = 0.35
    gn_damping: float = 1e-2
    gn_max_step: float = 0.08
    light_ik_iters: int = 4
    # Foot: strong slip to beat GMR; moderate height.
    enable_foot_penalties: bool = True
    w_foot_height: float = 40.0
    w_foot_slip: float = 900.0
    w_foot_ik_anchor: float = 30.0
    w_root_xy_contact: float = 20.0
    w_contact_joint_anchor: float = 40.0
    # Constraints
    dq_max: float = 4.0  # rad/s (hinge/slide)
    use_joint_limits: bool = True
    use_velocity_limits: bool = True
    # Keep committed hinge joints this many degrees away from hard limits (0 disables).
    joint_limit_margin_deg: float = 0.0
    # Control-feasibility: soft inverse-dynamics torque-limit barrier in the QP window
    # (penalise only torque beyond kappa*tau_max). See BatchTrajectoryConfig for semantics.
    torque_limit_constraint: bool = False
    torque_limit_margin: float = 0.1
    torque_limit_weight: float = 20.0
    torque_limit_scope: str = "upper"  # "upper" | "all"
    torque_limit_gate_mode: str = "soft"  # "off" | "soft" | "hard"
    torque_limit_gate_r_on: float = 0.85
    torque_limit_gate_r_full: float = 0.95
    torque_limit_gate_r_off: float = 0.85
    torque_limit_gate_min_on_frames: int = 5
    torque_limit_gate_min_off_frames: int = 10
    torque_limit_gate_floor: float = 0.0
    # Sequence lookahead (peek future human frames). Streaming falls back to causal.
    use_lookahead: bool = True
    qp_solver: str = "daqp"
    finalize_contact: bool = True
    bootstrap_gmr_frames: int = 2
    profile: bool = False
    verbose: bool = False

    @classmethod
    def from_preset(cls, preset: Literal["default", "smooth", "anti_slip"] = "default") -> "OnlineQpConfig":
        if preset == "smooth":
            return cls(
                preset="smooth",
                w_velocity=3.0,
                w_acceleration=12.0,
                w_gmr=0.25,
                w_foot_slip=700.0,
                sqp_iters=3,
            )
        if preset == "anti_slip":
            return cls(
                preset="anti_slip",
                w_velocity=1.5,
                w_acceleration=6.0,
                w_gmr=0.4,
                w_foot_slip=2000.0,
                w_foot_height=60.0,
                w_foot_ik_anchor=40.0,
                sqp_iters=3,
                finalize_contact=False,
            )
        return cls(preset="default")


class OnlineQpRetargeter(BatchTrajectoryRetargeter):
    """Causal / lookahead online retargeting via constrained QP."""

    def __init__(
        self,
        retargeter: GeneralMotionRetargeting,
        config: Optional[OnlineQpConfig] = None,
    ) -> None:
        self.qp_config = config or OnlineQpConfig.from_preset("default")
        oc = self.qp_config
        batch_cfg = BatchTrajectoryConfig(
            strategy="sliding_window",
            window_size=oc.horizon,
            window_stride=1,
            w_velocity=oc.w_velocity,
            w_acceleration=oc.w_acceleration,
            w_anchor=oc.w_anchor,
            solver="gn",
            gn_steps=oc.sqp_iters,
            gn_damping=oc.gn_damping,
            gn_max_step=oc.gn_max_step,
            gn_line_search_alphas=(1.0, 0.5, 0.25, 0.1),
            enable_foot_penalties=oc.enable_foot_penalties,
            w_foot_height=oc.w_foot_height,
            w_foot_slip=oc.w_foot_slip,
            w_foot_ik_anchor=oc.w_foot_ik_anchor,
            w_root_xy_contact=oc.w_root_xy_contact,
            w_contact_joint_anchor=oc.w_contact_joint_anchor,
            foot_contact_from_ref=True,
            smooth_root_xyz=False,
            use_gmr_init=False,
            finalize_contact=False,
            torque_limit_constraint=oc.torque_limit_constraint,
            torque_limit_margin=oc.torque_limit_margin,
            torque_limit_weight=oc.torque_limit_weight,
            torque_limit_scope=oc.torque_limit_scope,
            torque_limit_gate_mode=oc.torque_limit_gate_mode,
            torque_limit_gate_r_on=oc.torque_limit_gate_r_on,
            torque_limit_gate_r_full=oc.torque_limit_gate_r_full,
            torque_limit_gate_r_off=oc.torque_limit_gate_r_off,
            torque_limit_gate_min_on_frames=oc.torque_limit_gate_min_on_frames,
            torque_limit_gate_min_off_frames=oc.torque_limit_gate_min_off_frames,
            torque_limit_gate_floor=oc.torque_limit_gate_floor,
            verbose=oc.verbose,
            show_progress=False,
        )
        super().__init__(retargeter, batch_cfg)
        self.config.light_ik_warmstart_iters = oc.light_ik_iters
        self.config.w_velocity = oc.w_velocity
        self.config.w_acceleration = oc.w_acceleration

        H = oc.horizon
        self._prepared_buf: Deque[dict] = deque(maxlen=max(H, 8))
        self._targets_buf: Deque[dict] = deque(maxlen=max(H, 8))
        self._q_buf: Deque[np.ndarray] = deque(maxlen=max(H, 8))
        self._q_ref_buf: Deque[np.ndarray] = deque(maxlen=max(H, 8))
        self._frame_index = 0
        self.last_frame_ms: float = 0.0
        self.last_qp_status: str = ""

        # Precompute hinge/slide qadr ↔ local v index for constraints.
        self._hinge_pairs: List[tuple[int, int]] = []  # (local_v_index, qadr)
        for li, v in enumerate(self._opt_vidx):
            j = int(self.model.dof_jntid[v])
            jtype = self.model.jnt_type[j]
            if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                self._hinge_pairs.append((li, int(self.model.jnt_qposadr[j])))

        # Box bounds tightened by the safety margin (revolute joints only).
        self._qp_lower = self._qpos_lower.copy()
        self._qp_upper = self._qpos_upper.copy()
        margin = float(np.deg2rad(max(0.0, oc.joint_limit_margin_deg)))
        if margin > 0.0:
            for v in self._opt_vidx:
                j = int(self.model.dof_jntid[v])
                if self.model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                    continue
                if not self.model.jnt_limited[j]:
                    continue
                qadr = int(self.model.jnt_qposadr[j])
                lo, hi = float(self._qpos_lower[qadr]), float(self._qpos_upper[qadr])
                if hi - lo > 2.0 * margin:
                    self._qp_lower[qadr] = lo + margin
                    self._qp_upper[qadr] = hi - margin

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def reset(self) -> None:
        self._prepared_buf.clear()
        self._targets_buf.clear()
        self._q_buf.clear()
        self._q_ref_buf.clear()
        self._frame_index = 0
        self.last_frame_ms = 0.0
        self.last_qp_status = ""
        self._global_ref_contact = None
        self.reset_torque_limit_gate()

    def _soft_seed(
        self,
        human_data,
        prepared: dict,
        offset_to_ground: bool,
    ) -> np.ndarray:
        oc = self.qp_config
        if self._frame_index <= oc.bootstrap_gmr_frames:
            return self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)
        q0 = self._q_buf[-1].copy() if self._q_buf else self.gmr.configuration.data.qpos.copy()
        if oc.light_ik_iters > 0:
            return self._light_ik_warmstart(q0, prepared, human_data, offset_to_ground)
        return q0

    def _apply_margin_clip(self, q: np.ndarray) -> np.ndarray:
        """Clamp hinge joints into the margined band (covers non-QP bootstrap frames)."""
        if self.qp_config.joint_limit_margin_deg <= 0.0:
            return q
        for _, qadr in self._hinge_pairs:
            q[qadr] = min(max(float(q[qadr]), float(self._qp_lower[qadr])), float(self._qp_upper[qadr]))
        return q

    def _apply_penetration_fix(self, q: np.ndarray) -> np.ndarray:
        """Lift root Z so foot/trunk geoms clear the floor (idempotent).

        ``anti_slip`` disables full ``finalize_contact`` for speed, but still needs
        this lift — especially after offline ``ground_align`` parks feet near z=0.
        """
        if not self.gmr.contact_ground.fix_penetration:
            return q
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)
        self.gmr.contact_ground.fix_robot_penetration(self.model, self.data)
        return self.data.qpos.copy()

    def _build_qp_constraints(
        self,
        q_lin: np.ndarray,
        q_prev: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
        """Return G,h,lb,ub for dq variables (H * m)."""
        oc = self.qp_config
        Hn = q_lin.shape[0]
        m = len(self._opt_vidx)
        nvar = Hn * m
        lb = -np.full(nvar, oc.gn_max_step, dtype=float)
        ub = np.full(nvar, oc.gn_max_step, dtype=float)

        # Joint limits on hinge/slide (margined): q_lin + dq ∈ [qmin, qmax]
        if oc.use_joint_limits:
            for t in range(Hn):
                off = t * m
                for li, qadr in self._hinge_pairs:
                    lo = float(self._qp_lower[qadr])
                    hi = float(self._qp_upper[qadr])
                    q0 = float(q_lin[t, qadr])
                    lb[off + li] = max(lb[off + li], lo - q0)
                    ub[off + li] = min(ub[off + li], hi - q0)

        G_rows: List[np.ndarray] = []
        h_list: List[float] = []
        dt = max(float(self.dt), 1e-6)
        dq_lim = float(oc.dq_max) * dt

        if oc.use_velocity_limits and self._hinge_pairs:
            # |q0 - q_prev| and |q_t - q_{t-1}| ≤ dq_max*dt
            for t in range(Hn):
                off = t * m
                for li, qadr in self._hinge_pairs:
                    row_p = np.zeros(nvar)
                    row_m = np.zeros(nvar)
                    row_p[off + li] = 1.0
                    row_m[off + li] = -1.0
                    if t == 0:
                        if q_prev is None:
                            continue
                        base = float(q_lin[0, qadr] - q_prev[qadr])
                        # q_lin[0]+dq0 - q_prev ≤ dq_lim  →  dq0 ≤ dq_lim - (q_lin-q_prev)
                        h_list.append(dq_lim - base)
                        G_rows.append(row_p)
                        h_list.append(dq_lim + base)
                        G_rows.append(row_m)
                    else:
                        off_m = (t - 1) * m
                        row_p[off_m + li] -= 1.0
                        row_m[off_m + li] += 1.0
                        base = float(q_lin[t, qadr] - q_lin[t - 1, qadr])
                        h_list.append(dq_lim - base)
                        G_rows.append(row_p)
                        h_list.append(dq_lim + base)
                        G_rows.append(row_m)

        if G_rows:
            G = np.asarray(G_rows, dtype=float)
            h = np.asarray(h_list, dtype=float)
        else:
            G, h = None, None
        return G, h, lb, ub

    def _accumulate_gmr_prior_gn(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q_win: np.ndarray,
        q_ref: np.ndarray,
        vidx: np.ndarray,
        w: float,
    ) -> None:
        if w <= 0.0:
            return
        m = len(vidx)
        for t in range(len(q_win)):
            off = t * m
            for li, v in enumerate(vidx):
                j = int(self.model.dof_jntid[v])
                qadr = int(self.model.jnt_qposadr[j])
                # Skip free-joint quat components (non-Euclidean).
                jtype = self.model.jnt_type[j]
                if jtype == mj.mjtJoint.mjJNT_FREE and qadr >= 3:
                    continue
                err = float(q_win[t, qadr] - q_ref[t, qadr])
                idx = off + li
                H[idx, idx] += w
                g[idx] += w * err

    def _solve_qp_window(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        q_ref: np.ndarray,
        q_prev: np.ndarray | None,
        pin_frames: int = 0,
    ) -> np.ndarray:
        oc = self.qp_config
        q_win = q_init.copy()
        n_frames = len(q_win)
        vidx = self._opt_vidx
        m = len(vidx)
        nvar = n_frames * m
        smooth_v, smooth_q = self._smooth_v_in_frame(vidx)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        dq_v = np.zeros(self.model.nv)
        pin_frames = max(0, min(pin_frames, n_frames - 1))
        alphas = (1.0, 0.5, 0.25, 0.1)
        anchor = q_win[0].copy()
        w_v = self.config.w_velocity
        w_a = self.config.w_acceleration
        damp = oc.gn_damping

        self._window_anchor_w = oc.w_anchor
        # Do NOT clear _global_ref_contact here; sequence mode sets a full schedule.
        prev_offset = getattr(self, "_window_frame_offset", 0)

        for _ in range(oc.sqp_iters):
            H = np.zeros((nvar, nvar), dtype=float)
            g = np.zeros(nvar, dtype=float)
            for t in range(n_frames):
                self._accumulate_frame_fk_gn(
                    H, g, t * m, q_win[t], targets_list[t], vidx, jacp, jacr
                )
            self._accumulate_window_anchor_gn(H, g, q_win[0], anchor, vidx, oc.w_anchor)
            self._accumulate_window_temporal_gn(
                H, g, q_win, smooth_v, smooth_q, m, w_v, w_a
            )
            self._update_torque_limit_gate_from_window(q_win)
            self._accumulate_window_torque_limit_gn(H, g, q_win, m)
            self._accumulate_window_foot_gn(H, g, q_win, vidx, m, jacp, q_ref)
            self._accumulate_gmr_prior_gn(H, g, q_win, q_ref, vidx, oc.w_gmr)

            P = H + damp * np.eye(nvar)
            # Symmetrize for QP solvers.
            P = 0.5 * (P + P.T)
            G, h, lb, ub = self._build_qp_constraints(q_win, q_prev)
            if pin_frames > 0:
                # Force prefix dq ≈ 0 via tight bounds.
                lb[: pin_frames * m] = 0.0
                ub[: pin_frames * m] = 0.0

            try:
                dq = solve_qp(
                    P,
                    g,
                    G=G,
                    h=h,
                    lb=lb,
                    ub=ub,
                    solver=oc.qp_solver,
                    verbose=False,
                )
                self.last_qp_status = "ok" if dq is not None else "fail"
            except Exception as exc:  # noqa: BLE001
                self.last_qp_status = f"err:{type(exc).__name__}"
                dq = None

            if dq is None:
                # Fallback: unconstrained damped GN step.
                try:
                    dq = np.linalg.solve(P, g)
                    self.last_qp_status = "fallback_chol"
                except np.linalg.LinAlgError:
                    break

            dq = np.asarray(dq, dtype=float).reshape(-1)
            dq = np.clip(dq, -oc.gn_max_step, oc.gn_max_step)
            if pin_frames > 0:
                dq[: pin_frames * m] = 0.0

            best_cost = self._window_cost(q_win, targets_list, anchor, q_ref)
            best_q = q_win.copy()
            improved = False
            for alpha in alphas:
                q_trial = q_win.copy()
                for t in range(n_frames):
                    dq_v[:] = 0.0
                    dq_v[vidx] = -alpha * dq[t * m : (t + 1) * m]
                    mj.mj_integratePos(self.model, q_trial[t], dq_v, 1.0)
                    self._clip_hinge_qpos(q_trial[t])
                trial_cost = self._window_cost(q_trial, targets_list, anchor, q_ref)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_q = q_trial
                    improved = True
            q_win[:] = best_q
            if not improved:
                break

        return q_win

    def retarget(self, human_data, offset_to_ground: bool = False) -> np.ndarray:
        """Streaming API (causal window, no future frames)."""
        import time

        t0 = time.perf_counter()
        prepared = self.gmr._prepare_scaled_human_data(human_data, offset_to_ground)
        targets = self._targets_for_prepared(prepared)
        self._frame_index += 1
        q_seed = self._soft_seed(human_data, prepared, offset_to_ground)

        self._prepared_buf.append(prepared)
        self._targets_buf.append(targets)
        self._q_ref_buf.append(q_seed.copy())

        if self._frame_index <= self.qp_config.bootstrap_gmr_frames:
            q_out = q_seed.copy()
        elif len(self._q_buf) + 1 < self.qp_config.min_frames:
            q_out = q_seed.copy()
        else:
            q_list = list(self._q_buf) + [q_seed.copy()]
            Hn = min(self.qp_config.horizon, len(q_list))
            q_win = np.stack(q_list[-Hn:], axis=0)
            tgt_win = list(self._targets_buf)[-Hn:]
            ref_win = np.stack(list(self._q_ref_buf)[-Hn:], axis=0)
            q_prev = self._q_buf[-1] if self._q_buf else None
            # Pin all but last 1–2 frames for causal stability.
            trail = min(2, Hn)
            pin = Hn - trail
            q_opt = self._solve_qp_window(
                q_win, tgt_win, ref_win, q_prev, pin_frames=pin
            )
            q_out = q_opt[-1].copy()

        if self.qp_config.finalize_contact:
            q_out = self._finalize_qpos(q_out, prepared, human_data, offset_to_ground)
        else:
            # Still clear ground penetration when full finalize is off (anti_slip).
            q_out = self._apply_penetration_fix(q_out)

        q_out = self._apply_margin_clip(q_out)
        self._q_buf.append(q_out.copy())
        self.gmr.configuration.data.qpos[:] = q_out
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        self.gmr.scaled_human_data = self.gmr._build_scaled_human_data(prepared)
        self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
        return q_out

    def iter_retarget_sequence(
        self,
        human_frames: Sequence[dict],
        offset_to_ground: bool = False,
    ):
        """Yield one qpos per frame (online). Lookahead peeks future frames when enabled."""
        self.reset()
        if not self.qp_config.use_lookahead:
            for f in human_frames:
                yield self.retarget(f, offset_to_ground)
            return

        import time

        T = len(human_frames)
        Hn = self.qp_config.horizon
        prepared_all: list[dict | None] = [None] * T
        targets_all: list[dict | None] = [None] * T

        def ensure(i: int) -> None:
            if prepared_all[i] is None:
                prepared_all[i] = self.gmr._prepare_scaled_human_data(
                    human_frames[i], offset_to_ground
                )
                targets_all[i] = self._targets_for_prepared(prepared_all[i])

        q_prev: np.ndarray | None = None
        for k in range(T):
            t0 = time.perf_counter()
            self._frame_index = k + 1
            self._window_frame_offset = 0
            end = min(k + Hn, T)
            for i in range(k, end):
                ensure(i)
            if k > 0:
                ensure(k - 1)

            frames_slice = human_frames[k:end]
            prepared_slice = [prepared_all[i] for i in range(k, end)]
            targets_slice = [targets_all[i] for i in range(k, end)]

            seeds = []
            q_cursor = (
                q_prev.copy()
                if q_prev is not None
                else self.gmr.configuration.data.qpos.copy()
            )
            for i, (f, p) in enumerate(zip(frames_slice, prepared_slice)):
                if k < self.qp_config.bootstrap_gmr_frames and i == 0:
                    q_s = self.gmr.retarget(f, offset_to_ground=offset_to_ground)
                elif self.qp_config.light_ik_iters > 0:
                    q_s = self._light_ik_warmstart(q_cursor, p, f, offset_to_ground)
                else:
                    q_s = q_cursor.copy()
                seeds.append(q_s.copy())
                q_cursor = q_s

            q_win = np.stack(seeds, axis=0)
            ref_win = q_win.copy()
            if k < self.qp_config.bootstrap_gmr_frames:
                q_cmd = seeds[0].copy()
            else:
                if q_prev is not None and k > 0:
                    q_win = np.concatenate([q_prev[None, :], q_win], axis=0)
                    ref_win = np.concatenate([q_prev[None, :], ref_win], axis=0)
                    targets_slice = [targets_all[k - 1], *targets_slice]
                    pin = 1
                else:
                    pin = 0
                q_opt = self._solve_qp_window(
                    q_win, targets_slice, ref_win, q_prev, pin_frames=pin
                )
                q_cmd = q_opt[pin].copy() if pin else q_opt[0].copy()

            if self.qp_config.finalize_contact:
                q_cmd = self._finalize_qpos(
                    q_cmd, prepared_slice[0], frames_slice[0], offset_to_ground
                )
            else:
                q_cmd = self._apply_penetration_fix(q_cmd)

            q_cmd = self._apply_margin_clip(q_cmd)
            q_prev = q_cmd
            self._q_buf.append(q_cmd.copy())
            self.gmr.configuration.data.qpos[:] = q_cmd
            mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
            self.gmr.scaled_human_data = self.gmr._build_scaled_human_data(
                prepared_slice[0]
            )
            self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
            yield q_cmd

    def retarget_sequence(
        self,
        human_frames: Sequence[dict],
        offset_to_ground: bool = False,
    ) -> np.ndarray:
        """Full sequence; uses delayed short-horizon preview when enabled."""
        return np.stack(
            list(self.iter_retarget_sequence(human_frames, offset_to_ground)), axis=0
        )
