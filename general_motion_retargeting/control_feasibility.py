"""Control-feasibility limiter for motion retargeting.

Native GMR solves a per-frame kinematic IK whose only temporal regulariser is the
solver damping. The resulting joint trajectory can be *kinematically fine* yet
*dynamically infeasible*: the inertial torque ``M(q) qddot`` needed to realise the
commanded acceleration exceeds the actuator torque limit, so no controller can
track it. This module upgrades the optimisation target from "kinematic smoothness"
to "control feasibility" by bounding, per joint, the commanded acceleration so that
the predicted joint torque

    tau_j = M_jj(q) * qddot_j + g_j(q)

stays inside ``kappa * tau_max_j`` (``kappa = 1 - margin``). It is a causal,
single-frame projection applied to the committed pose (mirroring the joint-limit
margin clip), using the previous committed frame velocity for the finite-difference
acceleration.

Two modes are provided so that the control-feasibility formulation can be compared
head-to-head against plain kinematic smoothing:

* ``"torque"``  – per-joint budget from ``M_jj`` and ``tau_max`` (control-feasible).
* ``"uniform"`` – a single acceleration cap shared by all joints (kinematic
  smoothing baseline used only for ablation).
"""

from __future__ import annotations

import mujoco as mj
import numpy as np


class TorqueFeasibilityLimiter:
    def __init__(
        self,
        model: mj.MjModel,
        *,
        margin: float = 0.2,
        mode: str = "torque",
        uniform_accel_cap: float = 30.0,
        fps: float = 30.0,
    ) -> None:
        self.model = model
        self.kappa = float(np.clip(1.0 - margin, 0.05, 1.0))
        self.mode = mode
        self.uniform_accel_cap = float(uniform_accel_cap)
        self.dt = 1.0 / float(fps)

        # Actuated, torque-limited hinge joints only (1-DoF, no quaternion book-keeping).
        self.qadr: list[int] = []
        self.vadr: list[int] = []
        self.taumax: list[float] = []
        for j in range(model.njnt):
            if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                continue
            lo, hi = model.jnt_actfrcrange[j]
            tmax = float(max(abs(lo), abs(hi)))
            if tmax <= 0.0:
                continue
            self.qadr.append(int(model.jnt_qposadr[j]))
            self.vadr.append(int(model.jnt_dofadr[j]))
            self.taumax.append(tmax)
        self.qadr = np.asarray(self.qadr, dtype=int)
        self.vadr = np.asarray(self.vadr, dtype=int)
        self.taumax = np.asarray(self.taumax, dtype=float)
        self._M = np.zeros((model.nv, model.nv))
        self.reset()

    def set_fps(self, fps: float) -> None:
        self.dt = 1.0 / float(fps)

    def reset(self) -> None:
        self._v_prev = None  # previous committed joint velocity (per limited dof)

    def project(self, data: mj.MjData, q_raw: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        """Clip ``q_raw`` (in place on a copy) so the implied joint torque is feasible.

        ``data`` is expected to already hold ``q_raw`` with forward kinematics run.
        ``q_prev`` is the previously committed full qpos.
        """
        if self.qadr.size == 0:
            return q_raw
        dt = self.dt
        q = q_raw.copy()

        # Effective per-joint inertia (diagonal of the joint-space mass matrix) and
        # gravity torque, evaluated at the raw pose.
        mj.mj_fullM(self.model, data, self._M)
        m_diag = np.maximum(np.diag(self._M)[self.vadr], 1e-4)
        grav = np.zeros(self.model.nv)
        qvel_save = data.qvel.copy()
        data.qvel[:] = 0.0
        mj.mj_forward(self.model, data)
        mj.mj_rne(self.model, data, 0, grav)
        data.qvel[:] = qvel_save
        g = grav[self.vadr]

        dq = q[self.qadr] - q_prev[self.qadr]
        v_new = dq / dt
        if self._v_prev is None:
            self._v_prev = np.zeros_like(v_new)
        v_prev = self._v_prev

        if self.mode == "uniform":
            a_lo = -self.uniform_accel_cap * np.ones_like(v_new)
            a_hi = self.uniform_accel_cap * np.ones_like(v_new)
        else:
            # |M*a + g| <= kappa*taumax  ->  a in [(-kt - g)/M, (kt - g)/M]
            kt = self.kappa * self.taumax
            a_lo = (-kt - g) / m_diag
            a_hi = (kt - g) / m_diag
            a_lo = np.minimum(a_lo, 0.0)  # never forbid decelerating toward feasibility
            a_hi = np.maximum(a_hi, 0.0)

        v_min = v_prev + dt * a_lo
        v_max = v_prev + dt * a_hi
        v_clip = np.clip(v_new, v_min, v_max)

        q[self.qadr] = q_prev[self.qadr] + dt * v_clip
        self._v_prev = v_clip
        return q
