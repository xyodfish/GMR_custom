"""Online batch-lite retargeting: causal multi-frame GN inspired by Batch TO.

Combines Batch TO's FK tracking + temporal smoothness (+ optional foot penalties)
with a streaming ``retarget()`` API suitable for real-time use.

Speed strategy (vs offline Batch TO):
  - small causal window (default 4–5 frames)
  - pin committed prefix; GN updates only trailing 1–2 frames
  - 1 GN step, no line search by default
  - optional lightweight foot height penalty (slip off in balanced mode)

Warmstart:
  - ``seed_mode="gmr_ik"`` (default): light IK from previous q each frame
  - ``seed_mode="extrapolate"``: GMR IK only for the first ``gmr_bootstrap_frames``,
    then constant-velocity extrapolation from committed q (keeps GMR human
    scaling / link targets, drops per-frame GMR IK as seed)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional, Sequence

import mujoco as mj
import numpy as np

from .batch_trajectory_retarget import BatchTrajectoryConfig, BatchTrajectoryRetargeter
from .motion_retarget import GeneralMotionRetargeting


@dataclass
class OnlineBatchConfig:
    """Streaming batch-lite TO tuned for online use.

    FK vs smoothness knobs (most important first):
      ik_blend          0=full TO, 1=keep seed (strongest pull to warmstart)
      seed_mode         gmr_ik | extrapolate
      gmr_bootstrap_frames  full GMR IK for the first K frames (extrap mode)
      light_ik_iters    more → seed closer to GMR IK (gmr_ik mode only)
      w_velocity / w_acceleration  temporal smooth (higher → worse FK)
      enable_foot_penalties / w_foot_*  contact terms (can fight FK)
      gn_steps / window_size / opt_trailing_frames  optimizer capacity
      finalize_contact  post-lift root (can hurt FK slightly)
    """

    preset: Literal["fast", "balanced", "quality", "track", "extrap"] = "balanced"
    window_size: int = 5
    opt_trailing_frames: int = 2
    gn_steps: int = 1
    min_frames: int = 2
    w_velocity: float = 2.0
    w_acceleration: float = 10.0
    w_anchor: float = 2.0
    window_anchor_weight: float = 4.0
    gn_damping: float = 0.1
    gn_max_step: float = 0.05
    light_ik_iters: int = 3
    # Blend optimized qpos toward seed: raises fidelity to warmstart, lowers TO freedom.
    ik_blend: float = 0.0
    seed_mode: Literal["gmr_ik", "extrapolate"] = "gmr_ik"
    # Full GMR.retarget() for frames 1..K, then switch to seed_mode policy.
    gmr_bootstrap_frames: int = 1
    # During bootstrap, commit GMR q directly (skip window TO) to avoid early drift.
    bootstrap_commit_gmr: bool = True
    # After bootstrap: "velocity" = 2*q1-q0; "hold" = last committed q (safer).
    extrap_policy: Literal["velocity", "hold"] = "hold"
    # If post-TO FK cost exceeds this vs the seed FK, fall back to one GMR re-anchor.
    reanchor_fk_ratio: float = 8.0
    use_gmr_init_frame0: bool = True
    enable_foot_penalties: bool = True
    w_foot_height: float = 25.0
    w_foot_slip: float = 150.0
    w_foot_ik_anchor: float = 40.0
    w_root_xy_contact: float = 0.0
    w_contact_joint_anchor: float = 0.0
    finalize_contact: bool = True
    # Knee pre-bend (from robot_retargeter): enforce a minimum knee bend on the
    # human leg targets to avoid straight-leg IK singularity → smoother, more
    # trackable knee joints. 0 disables. Only near-straight legs are touched.
    knee_min_bend_deg: float = 0.0
    knee_prebend_legs: tuple[tuple[str, str, str], ...] = (
        ("left_hip", "left_knee", "left_foot"),
        ("right_hip", "right_knee", "right_foot"),
    )
    # Joint-limit safety margin (deg) kept away from hard hinge limits so the
    # committed trajectory stays inside the controllable range. 0 disables.
    joint_limit_margin_deg: float = 0.0
    profile: bool = False
    verbose: bool = False

    @classmethod
    def from_preset(
        cls,
        preset: Literal["fast", "balanced", "quality", "track", "extrap"] = "balanced",
    ) -> "OnlineBatchConfig":
        if preset == "fast":
            return cls(
                preset="fast",
                window_size=4,
                opt_trailing_frames=1,
                gn_steps=1,
                min_frames=2,
                enable_foot_penalties=False,
                w_anchor=1.0,
                light_ik_iters=2,
                seed_mode="gmr_ik",
                gmr_bootstrap_frames=1,
            )
        if preset == "quality":
            return cls(
                preset="quality",
                window_size=6,
                opt_trailing_frames=2,
                gn_steps=2,
                min_frames=2,
                enable_foot_penalties=True,
                w_foot_height=35.0,
                w_foot_slip=300.0,
                w_foot_ik_anchor=60.0,
                w_root_xy_contact=30.0,
                w_contact_joint_anchor=80.0,
                light_ik_iters=4,
                seed_mode="gmr_ik",
                gmr_bootstrap_frames=1,
            )
        if preset == "track":
            return cls(
                preset="track",
                window_size=5,
                opt_trailing_frames=2,
                gn_steps=2,
                min_frames=2,
                w_velocity=1.0,
                w_acceleration=3.0,
                w_anchor=4.0,
                window_anchor_weight=6.0,
                light_ik_iters=8,
                ik_blend=0.5,
                enable_foot_penalties=True,
                w_foot_height=15.0,
                w_foot_slip=80.0,
                w_foot_ik_anchor=80.0,
                finalize_contact=True,
                seed_mode="gmr_ik",
                gmr_bootstrap_frames=1,
            )
        if preset == "extrap":
            # No per-frame GMR IK after short bootstrap; more GN to compensate.
            return cls(
                preset="extrap",
                window_size=5,
                opt_trailing_frames=2,
                gn_steps=4,
                min_frames=2,
                w_velocity=1.0,
                w_acceleration=3.0,
                w_anchor=2.0,
                window_anchor_weight=4.0,
                gn_max_step=0.06,
                light_ik_iters=0,
                ik_blend=0.0,
                seed_mode="extrapolate",
                gmr_bootstrap_frames=5,
                bootstrap_commit_gmr=True,
                extrap_policy="hold",
                reanchor_fk_ratio=8.0,
                use_gmr_init_frame0=True,
                # Foot terms need a good q_ref; without IK seed keep them light.
                enable_foot_penalties=True,
                w_foot_height=15.0,
                w_foot_slip=50.0,
                w_foot_ik_anchor=20.0,
                finalize_contact=True,
            )
        return cls(preset="balanced")


class OnlineBatchRetargeter(BatchTrajectoryRetargeter):
    """Causal receding-horizon batch TO for online retargeting."""

    def __init__(
        self,
        retargeter: GeneralMotionRetargeting,
        config: Optional[OnlineBatchConfig] = None,
    ) -> None:
        self.online_config = config or OnlineBatchConfig.from_preset("balanced")
        oc = self.online_config
        line_search = (
            (1.0, 0.5, 0.25, 0.1)
            if oc.preset in ("quality", "track", "extrap")
            else (1.0,)
        )
        batch_cfg = BatchTrajectoryConfig(
            strategy="sliding_window",
            window_size=oc.window_size,
            window_stride=1,
            w_velocity=oc.w_velocity,
            w_acceleration=oc.w_acceleration,
            w_anchor=oc.w_anchor,
            window_anchor_weight=oc.window_anchor_weight,
            solver="gn",
            gn_steps=oc.gn_steps,
            gn_damping=oc.gn_damping,
            gn_max_step=oc.gn_max_step,
            gn_line_search_alphas=line_search,
            profile=oc.profile,
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
            verbose=oc.verbose,
            show_progress=False,
        )
        super().__init__(retargeter, batch_cfg)
        self.config.light_ik_warmstart_iters = oc.light_ik_iters

        H = oc.window_size
        self._prepared_buf: Deque[dict] = deque(maxlen=H)
        self._targets_buf: Deque[dict] = deque(maxlen=H)
        self._q_buf: Deque[np.ndarray] = deque(maxlen=H)
        self._q_ref_buf: Deque[np.ndarray] = deque(maxlen=H)
        self._frame_index = 0
        self.last_frame_ms: float = 0.0
        self.last_seed_source: str = ""

        # (qadr, lo, hi) margined bounds for revolute joints; empty when disabled.
        self._margin_bounds = self._build_margin_bounds(oc.joint_limit_margin_deg)

    def _build_margin_bounds(self, margin_deg: float) -> list[tuple[int, float, float]]:
        margin = float(np.deg2rad(max(0.0, float(margin_deg))))
        if margin <= 0.0:
            return []
        bounds: list[tuple[int, float, float]] = []
        for j in range(self.model.njnt):
            if not self.model.jnt_limited[j]:
                continue
            if self.model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                continue
            qadr = int(self.model.jnt_qposadr[j])
            lo, hi = float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])
            if hi - lo <= 2.0 * margin:
                continue  # range too tight for the margin; leave hard limits.
            bounds.append((qadr, lo + margin, hi - margin))
        return bounds

    def _clip_hinge_qpos(self, q: np.ndarray) -> None:
        super()._clip_hinge_qpos(q)
        for qadr, lo, hi in self._margin_bounds:
            q[qadr] = min(max(float(q[qadr]), lo), hi)

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
        self.last_seed_source = ""
        self._global_ref_contact = None

    @staticmethod
    def _normalize_root_quat(q: np.ndarray) -> np.ndarray:
        out = q.copy()
        nrm = float(np.linalg.norm(out[3:7]))
        if nrm > 1e-9:
            out[3:7] /= nrm
        return out

    @staticmethod
    def _two_bone_ik_knee(
        hip: np.ndarray,
        knee: np.ndarray,
        foot: np.ndarray,
        target_foot: np.ndarray,
    ) -> np.ndarray:
        """Knee position for a bent leg (thigh len a, calf len b) reaching target_foot.

        Preserves the original bend plane/direction to avoid knee reversal.
        Single-frame port of robot_retargeter's two-bone IK.
        """
        world_up = np.array([0.0, 0.0, 1.0])
        world_y = np.array([0.0, 1.0, 0.0])

        def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-8 else fallback

        a = float(np.linalg.norm(knee - hip))
        b = float(np.linalg.norm(foot - knee))
        orig = foot - hip
        u0 = _unit(orig, world_up)

        # Bend preference: knee offset off the original hip→foot line.
        knee_proj = hip + np.dot(knee - hip, u0) * u0
        bend_pref = knee - knee_proj
        bend_pref = _unit(bend_pref, np.cross(u0, world_up))
        bend_pref = _unit(bend_pref, np.cross(u0, world_y))

        plane_n = np.cross(knee - hip, orig)
        plane_n = _unit(plane_n, np.cross(u0, bend_pref))
        plane_n = _unit(plane_n, np.cross(u0, world_up))

        thf = target_foot - hip
        d = float(np.clip(np.linalg.norm(thf), 1e-8, a + b - 1e-8))
        tu = _unit(thf, u0)
        x = (a * a - b * b + d * d) / (2.0 * max(d, 1e-8))
        h = float(np.sqrt(max(a * a - x * x, 0.0)))

        bend_pref_proj = bend_pref - np.dot(bend_pref, tu) * tu
        bend_dir = np.cross(plane_n, tu)
        bend_dir = _unit(bend_dir, bend_pref_proj)
        bend_dir = _unit(bend_dir, np.cross(tu, world_y))
        if float(np.linalg.norm(bend_pref_proj)) < 1e-8:
            bend_pref_proj = bend_pref
        if float(np.dot(bend_dir, bend_pref_proj)) < 0.0:
            bend_dir = -bend_dir

        return hip + x * tu + h * bend_dir

    def _apply_knee_prebend(self, prepared: dict) -> dict:
        """Enforce a minimum knee bend on near-straight legs (in place on prepared)."""
        min_angle = float(np.deg2rad(self.online_config.knee_min_bend_deg))
        if min_angle <= 0.0:
            return prepared
        for hip_n, knee_n, foot_n in self.online_config.knee_prebend_legs:
            if hip_n not in prepared or knee_n not in prepared or foot_n not in prepared:
                continue
            hip = np.asarray(prepared[hip_n][0], dtype=float).reshape(3)
            knee = np.asarray(prepared[knee_n][0], dtype=float).reshape(3)
            foot = np.asarray(prepared[foot_n][0], dtype=float).reshape(3)
            upper = knee - hip
            lower = foot - knee
            a = float(np.linalg.norm(upper))
            b = float(np.linalg.norm(lower))
            if a < 1e-6 or b < 1e-6:
                continue
            # Turn angle between thigh and calf (0 = straight leg).
            cos_now = float(np.dot(upper, lower) / (a * b))
            angle_now = float(np.arccos(np.clip(cos_now, -1.0, 1.0)))
            if angle_now >= min_angle:
                continue  # already bent enough; leave stance/bent legs untouched.
            hf = foot - hip
            hf_len = float(np.linalg.norm(hf))
            if hf_len < 1e-6:
                continue
            d_new = float(np.sqrt(max(a * a + b * b + 2.0 * a * b * np.cos(min_angle), 0.0)))
            target_foot = hip + (hf / hf_len) * d_new
            new_knee = self._two_bone_ik_knee(hip, knee, foot, target_foot)
            prepared[foot_n] = [target_foot, prepared[foot_n][1]]
            prepared[knee_n] = [new_knee, prepared[knee_n][1]]
        return prepared

    def _seed_extrapolate(self) -> np.ndarray:
        """History warmstart: hold last q (default) or constant-velocity extrapolate."""
        if not self._q_buf:
            return self.gmr.configuration.data.qpos.copy()
        if (
            self.online_config.extrap_policy == "hold"
            or len(self._q_buf) == 1
        ):
            return self._q_buf[-1].copy()
        q0 = self._q_buf[-2]
        q1 = self._q_buf[-1]
        q = q1 + (q1 - q0)
        return self._normalize_root_quat(q)

    def _make_seed(
        self,
        human_data,
        prepared: dict,
        offset_to_ground: bool,
    ) -> np.ndarray:
        oc = self.online_config
        bootstrap_n = max(0, int(oc.gmr_bootstrap_frames))

        # First K frames: full GMR IK to land in a good basin.
        if bootstrap_n > 0 and self._frame_index <= bootstrap_n:
            self.last_seed_source = f"gmr_bootstrap[{self._frame_index}/{bootstrap_n}]"
            return self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)

        # Legacy single-frame0 flag when bootstrap_frames==0.
        if (
            bootstrap_n == 0
            and self._frame_index == 1
            and oc.use_gmr_init_frame0
            and oc.seed_mode == "gmr_ik"
        ):
            self.last_seed_source = "gmr_frame0"
            return self.gmr.retarget(human_data, offset_to_ground=offset_to_ground)

        if oc.seed_mode == "extrapolate":
            self.last_seed_source = "extrapolate"
            q_seed = self._seed_extrapolate()
            self._clip_hinge_qpos(q_seed)
            return q_seed

        # Default: light IK from previous committed q (or model qpos).
        self.last_seed_source = "gmr_light_ik"
        if self._q_buf:
            q_init = self._q_buf[-1].copy()
        else:
            q_init = self.gmr.configuration.data.qpos.copy()
        if oc.light_ik_iters > 0:
            return self._light_ik_warmstart(
                q_init, prepared, human_data, offset_to_ground
            )
        return q_init

    def _optimize_gn_pinned_prefix(
        self,
        q_init: np.ndarray,
        targets_list: Sequence[dict],
        anchor: np.ndarray,
        *,
        q_ref: np.ndarray | None = None,
        pin_frames: int,
    ) -> np.ndarray:
        """Multi-frame GN; prefix frames are fixed (zero update each step)."""
        pin_frames = max(0, min(pin_frames, len(q_init) - 1))
        anchor_w = self.online_config.w_anchor
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
        self._window_frame_offset = 0
        self._global_ref_contact = None
        alphas = tuple(self.batch_config.gn_line_search_alphas) or (1.0,)
        try:
            for _step in range(self.batch_config.gn_steps):
                H = np.zeros((nvar, nvar), dtype=float)
                g = np.zeros(nvar, dtype=float)

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
                self._accumulate_window_foot_gn(
                    H, g, q_win, vidx, m, jacp, q_ref_win
                )

                try:
                    dq_flat = np.linalg.solve(H + damp * np.eye(nvar), g)
                except np.linalg.LinAlgError:
                    break

                dq_flat = np.clip(dq_flat, -max_step, max_step)
                if pin_frames > 0:
                    dq_flat[: pin_frames * m] = 0.0

                if len(alphas) <= 1:
                    alpha = alphas[0]
                    for t in range(n_frames):
                        dq_v[:] = 0.0
                        dq_v[vidx] = -alpha * dq_flat[t * m : (t + 1) * m]
                        mj.mj_integratePos(self.model, q_win[t], dq_v, 1.0)
                        self._clip_hinge_qpos(q_win[t])
                    continue

                # Armijo-style: accept only cost-decreasing steps (critical w/o IK seed).
                best_cost = self._window_cost(q_win, targets_list, anchor, q_ref_win)
                best_q = q_win.copy()
                improved = False
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
                        improved = True
                if not improved:
                    break
                q_win[:] = best_q
        finally:
            self._window_anchor_w = prev_anchor_w

        return q_win

    def retarget(self, human_data, offset_to_ground: bool = False) -> np.ndarray:
        """Retarget one streaming human frame; returns optimized ``qpos``."""
        import time

        t0 = time.perf_counter()
        # Always keep GMR human scaling + link target mapping.
        prepared = self.gmr._prepare_scaled_human_data(human_data, offset_to_ground)
        if self.online_config.knee_min_bend_deg > 0.0:
            prepared = self._apply_knee_prebend(prepared)
        targets = self._targets_for_prepared(prepared)
        self._frame_index += 1

        q_seed = self._make_seed(human_data, prepared, offset_to_ground)
        bootstrap_n = max(0, int(self.online_config.gmr_bootstrap_frames))
        in_bootstrap = bootstrap_n > 0 and self._frame_index <= bootstrap_n

        self._prepared_buf.append(prepared)
        self._targets_buf.append(targets)
        self._q_ref_buf.append(q_seed.copy())

        if self._q_buf:
            q_list = list(self._q_buf) + [q_seed.copy()]
        else:
            q_list = [q_seed.copy()]

        # Bootstrap: commit pure GMR IK so the history basin stays clean.
        if in_bootstrap and self.online_config.bootstrap_commit_gmr:
            q_out = q_seed.copy()
        elif len(q_list) < self.online_config.min_frames:
            q_out = q_seed.copy()
        else:
            q_win = np.stack(q_list[-self.online_config.window_size :], axis=0)
            tgt_win = list(self._targets_buf)[-q_win.shape[0] :]
            ref_win = np.stack(list(self._q_ref_buf)[-q_win.shape[0] :], axis=0)
            anchor = q_win[0].copy()
            trail = min(
                self.online_config.opt_trailing_frames,
                max(1, q_win.shape[0] - 1),
            )
            pin_frames = q_win.shape[0] - trail

            q_opt = self._optimize_gn_pinned_prefix(
                q_win,
                tgt_win,
                anchor,
                q_ref=ref_win,
                pin_frames=pin_frames,
            )
            q_out = q_opt[-1].copy()
            blend = float(np.clip(self.online_config.ik_blend, 0.0, 1.0))
            if blend > 0.0:
                q_out = (1.0 - blend) * q_out + blend * q_seed
                q_out = self._normalize_root_quat(q_out)

            # Safety: if TO FK is much worse than previous committed frame, re-anchor.
            ratio = float(self.online_config.reanchor_fk_ratio)
            if (
                ratio > 0.0
                and self.online_config.seed_mode == "extrapolate"
                and len(self._q_buf) > 0
            ):
                prev_tg = (
                    list(self._targets_buf)[-2]
                    if len(self._targets_buf) >= 2
                    else targets
                )
                # Compare current output FK on *current* targets vs seed FK.
                fk_seed = self._fk_tracking_cost(q_seed, targets)
                fk_out = self._fk_tracking_cost(q_out, targets)
                # Also absolute guard: walking FK is typically O(1)–O(10).
                if (fk_seed > 1e-9 and fk_out > ratio * max(fk_seed, 1.0)) or fk_out > 50.0:
                    self.last_seed_source = "gmr_reanchor"
                    q_out = self.gmr.retarget(
                        human_data, offset_to_ground=offset_to_ground
                    )

        if self.online_config.finalize_contact:
            q_out = self._finalize_qpos(
                q_out, prepared, human_data, offset_to_ground
            )

        # Keep every committed frame inside the joint-limit safety margin.
        if self._margin_bounds:
            self._clip_hinge_qpos(q_out)

        self._q_buf.append(q_out.copy())
        self.gmr.configuration.data.qpos[:] = q_out
        mj.mj_forward(self.gmr.model, self.gmr.configuration.data)
        self.gmr.scaled_human_data = self.gmr._build_scaled_human_data(prepared)

        self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
        if self.online_config.profile:
            self.last_profile = {
                "frame_ms": self.last_frame_ms,
                "window_size": float(len(q_list)),
                "frame_index": float(self._frame_index),
                "seed_source": self.last_seed_source,
            }
        return q_out

    def retarget_sequence(
        self,
        human_frames: Sequence[dict],
        offset_to_ground: bool = False,
    ) -> np.ndarray:
        """Process a full sequence (resets state, returns all qpos)."""
        self.reset()
        out = []
        for frame in human_frames:
            out.append(self.retarget(frame, offset_to_ground=offset_to_ground))
        return np.stack(out, axis=0)
