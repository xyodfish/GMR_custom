#!/usr/bin/env python3
"""Deterministic quality gate for G1-bridged Unitree H2 trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


QUALITY_SCHEMA = "gmr_h2_quality_v4"


@dataclass(frozen=True)
class QualityThresholds:
    semantic_position_p95_warn_m: float = 0.25
    semantic_position_p95_reject_m: float = 0.45
    semantic_position_max_warn_m: float = 0.50
    semantic_position_max_reject_m: float = 0.90
    semantic_rotation_p95_warn_deg: float = 35.0
    semantic_rotation_p95_reject_deg: float = 70.0
    semantic_rotation_max_warn_deg: float = 75.0
    semantic_rotation_max_reject_deg: float = 140.0
    root_step_warn_m: float = 0.35
    root_step_reject_m: float = 1.0
    root_angle_step_warn_deg: float = 45.0
    root_angle_step_reject_deg: float = 120.0
    joint_step_warn_rad: float = 0.75
    joint_step_reject_rad: float = 1.57
    joint_speed_warn_rad_s: float = 15.0
    joint_speed_reject_rad_s: float = 35.0
    joint_accel_warn_rad_s2: float = 400.0
    clamp_peak_warn_deg: float = 10.0
    clamp_peak_reject_deg: float = 60.0
    clamp_frame_ratio_warn: float = 0.05
    clamp_frame_ratio_reject: float = 0.50
    persistent_clamp_peak_reject_deg: float = 30.0
    foot_slip_p95_warn_m_s: float = 0.50
    protected_body_penetration_warn_m: float = 0.01
    protected_body_penetration_reject_m: float = 0.03


def _joint_table(model: mujoco.MjModel) -> dict[str, tuple[int, int, float, float]]:
    result: dict[str, tuple[int, int, float, float]] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        low, high = model.jnt_range[joint_id]
        result[name] = (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            float(low),
            float(high),
        )

    return result


def _quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    dot = np.clip(np.abs(np.sum(first * second, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def material_quality_regressions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return material candidate regressions using the same limits as the pilot gate."""
    regressions: list[dict[str, Any]] = []
    baseline_rejects = {
        issue["code"] for issue in baseline["issues"] if issue["severity"] == "reject"
    }
    candidate_rejects = {
        issue["code"] for issue in candidate["issues"] if issue["severity"] == "reject"
    }
    new_rejects = sorted(candidate_rejects - baseline_rejects)
    if new_rejects:
        regressions.append({"metric": "new_reject_codes", "codes": new_rejects})

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    baseline_tracking = baseline_metrics["g1_semantic_tracking"]
    candidate_tracking = candidate_metrics["g1_semantic_tracking"]
    comparisons = {
        "position_p95_m": (
            float(baseline_tracking["position_p95_m"]),
            float(candidate_tracking["position_p95_m"]),
            1.10,
            0.03,
        ),
        "rotation_p95_deg": (
            float(baseline_tracking["rotation_p95_deg"]),
            float(candidate_tracking["rotation_p95_deg"]),
            1.10,
            5.0,
        ),
        "joint_acceleration_max_rad_s2": (
            float(baseline_metrics["joint_acceleration_max_rad_s2"]),
            float(candidate_metrics["joint_acceleration_max_rad_s2"]),
            1.25,
            100.0,
        ),
        "foot_slip_p95_m_s": (
            float(baseline_metrics["contact_foot_slip_p95_m_s"]),
            float(candidate_metrics["contact_foot_slip_p95_m_s"]),
            2.0,
            1.0,
        ),
    }
    for metric, (baseline_value, candidate_value, ratio, allowance) in comparisons.items():
        limit = max(baseline_value * ratio, baseline_value + allowance)
        if candidate_value > limit:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "limit": limit,
                }
            )

    return regressions


class H2MotionQualityGate:
    """Validate bridge fidelity, temporal continuity, limits and ground interaction."""

    def __init__(
        self,
        h2_xml: Path,
        g1_xml: Path,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        self.h2_model = mujoco.MjModel.from_xml_path(str(h2_xml))
        self.g1_model = mujoco.MjModel.from_xml_path(str(g1_xml))
        self.h2_data = mujoco.MjData(self.h2_model)
        self.h2_joints = _joint_table(self.h2_model)
        self.g1_joints = _joint_table(self.g1_model)
        self.thresholds = thresholds or QualityThresholds()
        if set(self.h2_joints) != set(self.g1_joints):
            raise ValueError("G1 and H2 quality models must expose the same named hinge joints")

        self.joint_names = list(self.h2_joints)
        self.h2_qpos_indices = np.asarray(
            [self.h2_joints[name][0] for name in self.joint_names], dtype=np.int32
        )
        self.h2_dof_indices = np.asarray(
            [self.h2_joints[name][1] for name in self.joint_names], dtype=np.int32
        )
        self.g1_qpos_indices = np.asarray(
            [self.g1_joints[name][0] for name in self.joint_names], dtype=np.int32
        )
        self.lower_limits = np.asarray(
            [self.h2_joints[name][2] for name in self.joint_names], dtype=np.float64
        )
        self.upper_limits = np.asarray(
            [self.h2_joints[name][3] for name in self.joint_names], dtype=np.float64
        )
        self.floor_geom = mujoco.mj_name2id(
            self.h2_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.protected_geoms = [
            mujoco.mj_name2id(self.h2_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("pelvis_collision", "torso_collision", "head_collision")
        ]
        self.foot_sites = [
            mujoco.mj_name2id(self.h2_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("left_foot", "right_foot")
        ]

    def evaluate(
        self,
        h2_qpos: np.ndarray,
        g1_qpos: np.ndarray,
        fps: float,
        g1_tracking_quality: dict[str, Any],
        require_bridge_identity: bool = True,
    ) -> dict[str, Any]:
        h2_qpos = np.asarray(h2_qpos, dtype=np.float64)
        g1_qpos = np.asarray(g1_qpos, dtype=np.float64)
        if h2_qpos.ndim != 2 or h2_qpos.shape[1] != self.h2_model.nq:
            raise ValueError(f"expected H2 qpos [T,{self.h2_model.nq}], got {h2_qpos.shape}")

        if g1_qpos.shape != (h2_qpos.shape[0], self.g1_model.nq):
            raise ValueError(
                f"expected matching G1 qpos [{h2_qpos.shape[0]},{self.g1_model.nq}], "
                f"got {g1_qpos.shape}"
            )

        if not np.isfinite(h2_qpos).all() or not np.isfinite(g1_qpos).all():
            raise ValueError("quality-gate input contains NaN or infinity")

        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {fps}")

        issues: list[dict[str, Any]] = []

        def add_issue(
            code: str,
            severity: str,
            message: str,
            value: float,
            threshold: float,
            frame: int | None = None,
        ) -> None:
            issue: dict[str, Any] = {
                "code": code,
                "severity": severity,
                "message": message,
                "value": float(value),
                "threshold": float(threshold),
            }
            if frame is not None:
                issue["frame"] = int(frame)

            issues.append(issue)

        required_tracking_metrics = (
            "position_mean_m",
            "position_p95_m",
            "position_max_m",
            "rotation_mean_deg",
            "rotation_p95_deg",
            "rotation_max_deg",
            "worst_position_frame",
            "worst_rotation_frame",
            "worst_position_body",
            "worst_rotation_body",
        )
        missing_tracking_metrics = [
            name for name in required_tracking_metrics if name not in g1_tracking_quality
        ]
        if missing_tracking_metrics:
            raise ValueError(
                "retarget output lacks G1 semantic tracking metrics: "
                + ", ".join(missing_tracking_metrics)
            )

        tracking_numeric = np.asarray(
            [
                g1_tracking_quality["position_mean_m"],
                g1_tracking_quality["position_p95_m"],
                g1_tracking_quality["position_max_m"],
                g1_tracking_quality["rotation_mean_deg"],
                g1_tracking_quality["rotation_p95_deg"],
                g1_tracking_quality["rotation_max_deg"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(tracking_numeric).all():
            raise ValueError("G1 semantic tracking metrics contain NaN or infinity")

        self._threshold_scalar(
            issues,
            "semantic_position_p95",
            "G1 body positions deviate from the scaled SMPL-X task targets",
            float(g1_tracking_quality["position_p95_m"]),
            self.thresholds.semantic_position_p95_warn_m,
            self.thresholds.semantic_position_p95_reject_m,
        )
        self._threshold_scalar(
            issues,
            "semantic_position_max",
            f"G1 body {g1_tracking_quality['worst_position_body']} has a large SMPL-X target error",
            float(g1_tracking_quality["position_max_m"]),
            self.thresholds.semantic_position_max_warn_m,
            self.thresholds.semantic_position_max_reject_m,
            int(g1_tracking_quality["worst_position_frame"]),
        )
        self._threshold_scalar(
            issues,
            "semantic_rotation_p95",
            "G1 body orientations deviate from the scaled SMPL-X task targets",
            float(g1_tracking_quality["rotation_p95_deg"]),
            self.thresholds.semantic_rotation_p95_warn_deg,
            self.thresholds.semantic_rotation_p95_reject_deg,
        )
        self._threshold_scalar(
            issues,
            "semantic_rotation_max",
            f"G1 body {g1_tracking_quality['worst_rotation_body']} has a large SMPL-X rotation error",
            float(g1_tracking_quality["rotation_max_deg"]),
            self.thresholds.semantic_rotation_max_warn_deg,
            self.thresholds.semantic_rotation_max_reject_deg,
            int(g1_tracking_quality["worst_rotation_frame"]),
        )

        frame_count = h2_qpos.shape[0]
        h2_joint = h2_qpos[:, self.h2_qpos_indices]
        g1_joint = g1_qpos[:, self.g1_qpos_indices]
        expected_joint = np.clip(g1_joint, self.lower_limits, self.upper_limits)
        mapping_error = np.abs(h2_joint - expected_joint)
        mapping_peak_index = np.unravel_index(int(np.argmax(mapping_error)), mapping_error.shape)
        mapping_peak = float(mapping_error[mapping_peak_index])
        if require_bridge_identity and mapping_peak > 1.0e-6:
            add_issue(
                "bridge_joint_mismatch",
                "reject",
                "H2 named joints do not match the clamped G1 bridge trajectory",
                mapping_peak,
                1.0e-6,
                mapping_peak_index[0],
            )

        root_xy_error = np.linalg.norm(h2_qpos[:, :2] - g1_qpos[:, :2], axis=1)
        root_xy_peak_frame = int(np.argmax(root_xy_error))
        root_xy_peak = float(root_xy_error[root_xy_peak_frame])
        if require_bridge_identity and root_xy_peak > 1.0e-6:
            add_issue(
                "bridge_root_xy_mismatch",
                "reject",
                "H2 root XY differs from the G1 bridge trajectory",
                root_xy_peak,
                1.0e-6,
                root_xy_peak_frame,
            )

        root_orientation_error = _quaternion_angle_deg(h2_qpos[:, 3:7], g1_qpos[:, 3:7])
        root_orientation_peak_frame = int(np.argmax(root_orientation_error))
        root_orientation_peak = float(root_orientation_error[root_orientation_peak_frame])
        if require_bridge_identity and root_orientation_peak > 1.0e-4:
            add_issue(
                "bridge_root_orientation_mismatch",
                "reject",
                "H2 root orientation differs from the G1 bridge trajectory",
                root_orientation_peak,
                1.0e-4,
                root_orientation_peak_frame,
            )

        lower_violation = self.lower_limits - h2_joint
        upper_violation = h2_joint - self.upper_limits
        limit_violation = np.maximum(np.maximum(lower_violation, upper_violation), 0.0)
        limit_peak_index = np.unravel_index(int(np.argmax(limit_violation)), limit_violation.shape)
        limit_peak = float(limit_violation[limit_peak_index])
        if limit_peak > 1.0e-6:
            add_issue(
                "h2_joint_limit_violation",
                "reject",
                "H2 output exceeds a model joint limit",
                limit_peak,
                1.0e-6,
                limit_peak_index[0],
            )

        clamp = np.abs(g1_joint - expected_joint)
        clamp_peak_index = np.unravel_index(int(np.argmax(clamp)), clamp.shape)
        clamp_peak_deg = float(np.rad2deg(clamp[clamp_peak_index]))
        clamped_frames = np.any(clamp > 1.0e-6, axis=1)
        clamp_frame_ratio = float(np.mean(clamped_frames))
        clamp_joint_frame_ratio = {
            name: float(np.mean(clamp[:, index] > 1.0e-6))
            for index, name in enumerate(self.joint_names)
            if np.any(clamp[:, index] > 1.0e-6)
        }
        if clamp_peak_deg > self.thresholds.clamp_peak_reject_deg:
            add_issue(
                "severe_h2_limit_clamp",
                "reject",
                f"G1 joint {self.joint_names[clamp_peak_index[1]]} needs a severe H2 limit clamp",
                clamp_peak_deg,
                self.thresholds.clamp_peak_reject_deg,
                clamp_peak_index[0],
            )
        elif clamp_peak_deg > self.thresholds.clamp_peak_warn_deg:
            add_issue(
                "h2_limit_clamp",
                "warning",
                f"G1 joint {self.joint_names[clamp_peak_index[1]]} is clipped to the H2 limit",
                clamp_peak_deg,
                self.thresholds.clamp_peak_warn_deg,
                clamp_peak_index[0],
            )

        if (
            clamp_frame_ratio > self.thresholds.clamp_frame_ratio_reject
            and clamp_peak_deg > self.thresholds.persistent_clamp_peak_reject_deg
        ):
            add_issue(
                "persistent_h2_limit_clamp",
                "reject",
                f"A clamp larger than {self.thresholds.persistent_clamp_peak_reject_deg:.0f} degrees "
                "persists through more than half of the trajectory",
                clamp_peak_deg,
                self.thresholds.persistent_clamp_peak_reject_deg,
            )
        elif clamp_frame_ratio > self.thresholds.clamp_frame_ratio_warn:
            add_issue(
                "frequent_h2_limit_clamp",
                "warning",
                "H2 limit clamping affects a noticeable part of the trajectory",
                clamp_frame_ratio,
                self.thresholds.clamp_frame_ratio_warn,
            )

        velocity = np.zeros((frame_count, self.h2_model.nv), dtype=np.float64)
        for frame in range(1, frame_count):
            mujoco.mj_differentiatePos(
                self.h2_model,
                velocity[frame],
                1.0 / fps,
                h2_qpos[frame - 1],
                h2_qpos[frame],
            )

        root_steps = np.linalg.norm(np.diff(h2_qpos[:, :3], axis=0), axis=1)
        root_angle_steps = _quaternion_angle_deg(h2_qpos[:-1, 3:7], h2_qpos[1:, 3:7])
        joint_steps = np.abs(np.diff(h2_joint, axis=0))
        joint_speed = np.abs(velocity[:, self.h2_dof_indices])
        joint_accel = np.abs(np.diff(velocity[:, self.h2_dof_indices], axis=0) * fps)

        self._threshold_peak(
            issues,
            "root_position_step",
            "Root position has a frame-to-frame discontinuity",
            root_steps,
            self.thresholds.root_step_warn_m,
            self.thresholds.root_step_reject_m,
        )
        self._threshold_peak(
            issues,
            "root_orientation_step",
            "Root orientation has a frame-to-frame discontinuity",
            root_angle_steps,
            self.thresholds.root_angle_step_warn_deg,
            self.thresholds.root_angle_step_reject_deg,
        )
        self._threshold_peak(
            issues,
            "joint_position_step",
            "A joint has a frame-to-frame discontinuity",
            joint_steps,
            self.thresholds.joint_step_warn_rad,
            self.thresholds.joint_step_reject_rad,
        )
        self._threshold_peak(
            issues,
            "joint_speed",
            "A joint speed is unusually high",
            joint_speed,
            self.thresholds.joint_speed_warn_rad_s,
            self.thresholds.joint_speed_reject_rad_s,
        )
        self._threshold_peak(
            issues,
            "joint_acceleration",
            "A joint acceleration is unusually high",
            joint_accel,
            self.thresholds.joint_accel_warn_rad_s2,
            None,
        )

        site_positions = np.empty((frame_count, len(self.foot_sites), 3), dtype=np.float64)
        protected_min_distance = np.full(frame_count, np.inf, dtype=np.float64)
        for frame, frame_qpos in enumerate(h2_qpos):
            self.h2_data.qpos[:] = frame_qpos
            mujoco.mj_forward(self.h2_model, self.h2_data)
            site_positions[frame] = self.h2_data.site_xpos[self.foot_sites]
            protected_min_distance[frame] = min(
                mujoco.mj_geomDistance(
                    self.h2_model, self.h2_data, self.floor_geom, geom_id, 10.0, None
                )
                for geom_id in self.protected_geoms
            )

        protected_penetration = np.maximum(-protected_min_distance, 0.0)
        penetration_frame = int(np.argmax(protected_penetration))
        penetration_peak = float(protected_penetration[penetration_frame])
        if penetration_peak > self.thresholds.protected_body_penetration_reject_m:
            add_issue(
                "protected_body_ground_penetration",
                "reject",
                "Pelvis, torso, or head penetrates the floor",
                penetration_peak,
                self.thresholds.protected_body_penetration_reject_m,
                penetration_frame,
            )
        elif penetration_peak > self.thresholds.protected_body_penetration_warn_m:
            add_issue(
                "protected_body_ground_penetration",
                "warning",
                "Pelvis, torso, or head is slightly below the floor",
                penetration_peak,
                self.thresholds.protected_body_penetration_warn_m,
                penetration_frame,
            )

        foot_velocity_xy = np.linalg.norm(np.diff(site_positions[:, :, :2], axis=0), axis=2) * fps
        foot_height = site_positions[:, :, 2]
        contact = foot_height[:-1] <= (np.min(foot_height, axis=0, keepdims=True) + 0.03)
        contact_slip = foot_velocity_xy[contact]
        contact_slip_p95 = _percentile(contact_slip, 95.0)
        if contact_slip_p95 > self.thresholds.foot_slip_p95_warn_m_s:
            add_issue(
                "possible_foot_slip",
                "warning",
                "Foot horizontal speed is high near its trajectory-low ground height",
                contact_slip_p95,
                self.thresholds.foot_slip_p95_warn_m_s,
            )

        status = "quarantine" if any(issue["severity"] == "reject" for issue in issues) else "accepted"
        warning_count = sum(issue["severity"] == "warning" for issue in issues)
        reject_count = sum(issue["severity"] == "reject" for issue in issues)
        anomaly_score = 100.0 * reject_count + 10.0 * warning_count
        metrics = {
            "frames": frame_count,
            "fps": float(fps),
            "duration_seconds": float(frame_count / fps),
            "bridge_joint_error_max_rad": mapping_peak,
            "bridge_root_xy_error_max_m": root_xy_peak,
            "bridge_root_orientation_error_max_deg": root_orientation_peak,
            "h2_joint_limit_violation_max_rad": limit_peak,
            "clamp_peak_deg": clamp_peak_deg,
            "clamp_frame_ratio": clamp_frame_ratio,
            "clamp_joint_frame_ratio": clamp_joint_frame_ratio,
            "root_step_max_m": float(np.max(root_steps)) if root_steps.size else 0.0,
            "root_angle_step_max_deg": float(np.max(root_angle_steps)) if root_angle_steps.size else 0.0,
            "joint_step_max_rad": float(np.max(joint_steps)) if joint_steps.size else 0.0,
            "joint_speed_max_rad_s": float(np.max(joint_speed)) if joint_speed.size else 0.0,
            "joint_acceleration_max_rad_s2": float(np.max(joint_accel)) if joint_accel.size else 0.0,
            "protected_body_ground_penetration_max_m": penetration_peak,
            "contact_foot_slip_p95_m_s": contact_slip_p95,
            "g1_semantic_tracking": g1_tracking_quality,
        }
        return {
            "quality_schema": QUALITY_SCHEMA,
            "status": status,
            "anomaly_score": anomaly_score,
            "issues": issues,
            "metrics": metrics,
            "thresholds": self.thresholds.__dict__,
        }

    @staticmethod
    def _threshold_scalar(
        issues: list[dict[str, Any]],
        code: str,
        message: str,
        value: float,
        warning_threshold: float,
        reject_threshold: float,
        frame: int | None = None,
    ) -> None:
        if value > reject_threshold:
            issue = {
                "code": code,
                "severity": "reject",
                "message": message,
                "value": value,
                "threshold": reject_threshold,
            }
        elif value > warning_threshold:
            issue = {
                "code": code,
                "severity": "warning",
                "message": message,
                "value": value,
                "threshold": warning_threshold,
            }
        else:
            return

        if frame is not None:
            issue["frame"] = frame

        issues.append(issue)

    @staticmethod
    def _threshold_peak(
        issues: list[dict[str, Any]],
        code: str,
        message: str,
        values: np.ndarray,
        warning_threshold: float,
        reject_threshold: float | None,
    ) -> None:
        if not values.size:
            return

        flat_index = int(np.argmax(values))
        peak = float(values.flat[flat_index])
        frame = int(np.unravel_index(flat_index, values.shape)[0]) + 1
        if reject_threshold is not None and peak > reject_threshold:
            issues.append(
                {
                    "code": code,
                    "severity": "reject",
                    "message": message,
                    "value": peak,
                    "threshold": reject_threshold,
                    "frame": frame,
                }
            )
        elif peak > warning_threshold:
            issues.append(
                {
                    "code": code,
                    "severity": "warning",
                    "message": message,
                    "value": peak,
                    "threshold": warning_threshold,
                    "frame": frame,
                }
            )
