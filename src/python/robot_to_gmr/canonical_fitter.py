"""Fit fixed-bone-length SMPL-X-compatible body poses from semantic site targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .semantic_site_map import CORE_BODIES, hemisphere_continue, quat_mul, quat_normalize, quat_rotate


def _as3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def _look_rotation(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build wxyz quaternion with +Z along forward, +Y toward up_hint projection."""
    f = forward / (np.linalg.norm(forward) + 1e-12)
    up = up_hint / (np.linalg.norm(up_hint) + 1e-12)
    x_axis = np.cross(up, f)
    if np.linalg.norm(x_axis) < 1e-8:
        up = np.array([0.0, 1.0, 0.0] if abs(f[2]) < 0.9 else [1.0, 0.0, 0.0])
        x_axis = np.cross(up, f)

    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(f, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    rot = np.column_stack((x_axis, y_axis, f))  # columns = body axes in world
    return _rotmat_to_quat_wxyz(rot)


def _rotmat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def _clamp_length(root: np.ndarray, target: np.ndarray, length: float) -> np.ndarray:
    delta = target - root
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return root + np.array([0.0, 0.0, -length if length > 0 else 0.0], dtype=np.float64)

    return root + delta / norm * length


def two_bone_ik(
    root: np.ndarray,
    target: np.ndarray,
    length1: float,
    length2: float,
    pole: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mid_joint_pos, reachable_target)."""
    delta = target - root
    dist = float(np.linalg.norm(delta))
    max_reach = length1 + length2 - 1e-4
    min_reach = abs(length1 - length2) + 1e-4
    if dist < 1e-8:
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dist = min_reach
    else:
        direction = delta / dist

    reachable_dist = float(np.clip(dist, min_reach, max_reach))
    reachable_target = root + direction * reachable_dist
    cos_a = (length1**2 + reachable_dist**2 - length2**2) / (2.0 * length1 * reachable_dist)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))

    pole_dir = pole - root
    pole_dir = pole_dir - direction * float(np.dot(pole_dir, direction))
    if np.linalg.norm(pole_dir) < 1e-8:
        helper = np.array([0.0, 1.0, 0.0] if abs(direction[1]) < 0.9 else [1.0, 0.0, 0.0])
        pole_dir = np.cross(direction, helper)

    pole_dir /= np.linalg.norm(pole_dir) + 1e-12
    mid = root + direction * (length1 * cos_a) + pole_dir * (length1 * sin_a)
    return mid, reachable_target


@dataclass
class FitQuality:
    semantic_position_rmse_m: float
    semantic_position_p95_m: float
    semantic_rotation_mean_deg: float
    contact_slip_mean_mps: float


class CanonicalTrajectoryFitter:
    """Generate fixed-bone-length canonical body world poses."""

    def __init__(self, mapping_yaml: Path):
        with Path(mapping_yaml).open("r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle)

        bones = self.cfg["canonical_bones_m"]
        self.height = float(self.cfg["canonical_height_m"])
        self.pelvis_to_spine3 = _as3(bones["pelvis_to_spine3"])
        self.pelvis_to_left_hip = _as3(bones["pelvis_to_left_hip"])
        self.pelvis_to_right_hip = _as3(bones["pelvis_to_right_hip"])
        self.thigh = float(bones["thigh"])
        self.shank = float(bones["shank"])
        self.pelvis_to_left_shoulder = _as3(bones["pelvis_to_left_shoulder"])
        self.pelvis_to_right_shoulder = _as3(bones["pelvis_to_right_shoulder"])
        self.upper_arm = float(bones["upper_arm"])
        self.forearm = float(bones["forearm"])
        contact = self.cfg.get("contact", {})
        self.foot_bodies = list(contact.get("foot_bodies", ["left_foot", "right_foot"]))
        self.height_threshold = float(contact.get("height_threshold_m", 0.04))
        self.speed_threshold = float(contact.get("speed_threshold_mps", 0.35))
        self.min_contact_frames = int(contact.get("min_contact_frames", 3))
        self.lock_contact_z = bool(contact.get("lock_contact_z_to_ground", False))
        self.smooth_window = int(self.cfg.get("smoothing", {}).get("window", 5))

    def fit(
        self,
        semantic_frames: list[dict[str, dict[str, list[float]]]],
        fps: float,
    ) -> tuple[list[dict[str, dict[str, list[float]]]], list[dict[str, bool]], FitQuality]:
        if not semantic_frames:
            raise ValueError("Empty semantic frames")

        contacts = self.infer_contacts(semantic_frames, fps)
        # Pass 1: per-frame IK. Pass 2: smooth targets, then re-fit so bone lengths stay fixed.
        smoothed_targets = self._smooth_frames(semantic_frames)
        smoothed_targets = self._apply_contact_targets(smoothed_targets, contacts)
        fitted = [self._fit_frame(frame) for frame in smoothed_targets]
        quality = self._measure_quality(semantic_frames, fitted, contacts, fps)
        return fitted, contacts, quality

    def infer_contacts(
        self, frames: list[dict[str, dict[str, list[float]]]], fps: float
    ) -> list[dict[str, bool]]:
        dt = 1.0 / float(fps)
        n = len(frames)
        positions = {
            name: np.array([frames[t][name]["position"] for t in range(n)], dtype=np.float64)
            for name in self.foot_bodies
        }
        speeds = {}
        for name, pos in positions.items():
            vel = np.zeros_like(pos)
            if n >= 2:
                vel[1:] = np.diff(pos, axis=0) / dt
                vel[0] = vel[1]

            speeds[name] = np.linalg.norm(vel[:, :2], axis=1)

        # Relative support: near the per-frame lower foot envelope (not absolute z).
        stacked_z = np.stack([positions[name][:, 2] for name in self.foot_bodies], axis=1)
        lower = np.min(stacked_z, axis=1)
        band = max(self.height_threshold, 0.025)

        raw = {name: np.zeros(n, dtype=bool) for name in self.foot_bodies}
        for name in self.foot_bodies:
            near_lower = positions[name][:, 2] <= (lower + band)
            slow = speeds[name] < self.speed_threshold
            raw[name] = near_lower & slow

        contacts: list[dict[str, bool]] = []
        cleaned = {name: raw[name].copy() for name in self.foot_bodies}
        for name in self.foot_bodies:
            arr = cleaned[name]
            t = 0
            while t < n:
                if not arr[t]:
                    t += 1
                    continue

                t0 = t
                while t < n and arr[t]:
                    t += 1

                if t - t0 < self.min_contact_frames:
                    arr[t0:t] = False

        for t in range(n):
            contacts.append({name: bool(cleaned[name][t]) for name in self.foot_bodies})

        return contacts

    def _fit_frame(self, target: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
        def pose(name: str) -> tuple[np.ndarray, np.ndarray]:
            return (
                _as3(target[name]["position"]),
                quat_normalize(np.asarray(target[name]["orientation"], dtype=np.float64)),
            )

        pelvis_pos, pelvis_quat = pose("pelvis")
        spine_pos_tgt, spine_quat = pose("spine3")
        spine_len = float(np.linalg.norm(self.pelvis_to_spine3))
        spine_pos = _clamp_length(pelvis_pos, spine_pos_tgt, spine_len)

        # Legs: fixed bone lengths via two-bone IK (foot contact / gait). Keep semantic orients.
        left_hip = _clamp_length(pelvis_pos, _as3(target["left_hip"]["position"]), float(np.linalg.norm(self.pelvis_to_left_hip)))
        right_hip = _clamp_length(
            pelvis_pos, _as3(target["right_hip"]["position"]), float(np.linalg.norm(self.pelvis_to_right_hip))
        )
        left_knee, left_foot = two_bone_ik(
            left_hip, _as3(target["left_foot"]["position"]), self.thigh, self.shank, _as3(target["left_knee"]["position"])
        )
        right_knee, right_foot = two_bone_ik(
            right_hip,
            _as3(target["right_foot"]["position"]),
            self.thigh,
            self.shank,
            _as3(target["right_knee"]["position"]),
        )

        # Arms: keep G1 semantic shoulder/elbow/wrist poses.
        # smplx_to_h2 relies on shoulder/elbow orientation; inventing look-at frames folds the elbows.
        left_shoulder, left_shoulder_quat = pose("left_shoulder")
        right_shoulder, right_shoulder_quat = pose("right_shoulder")
        left_elbow, left_elbow_quat = pose("left_elbow")
        right_elbow, right_elbow_quat = pose("right_elbow")
        left_wrist, left_wrist_quat = pose("left_wrist")
        right_wrist, right_wrist_quat = pose("right_wrist")
        left_hip_quat = quat_normalize(np.asarray(target["left_hip"]["orientation"], dtype=np.float64))
        right_hip_quat = quat_normalize(np.asarray(target["right_hip"]["orientation"], dtype=np.float64))
        left_knee_quat = quat_normalize(np.asarray(target["left_knee"]["orientation"], dtype=np.float64))
        right_knee_quat = quat_normalize(np.asarray(target["right_knee"]["orientation"], dtype=np.float64))
        left_foot_quat = quat_normalize(np.asarray(target["left_foot"]["orientation"], dtype=np.float64))
        right_foot_quat = quat_normalize(np.asarray(target["right_foot"]["orientation"], dtype=np.float64))

        out = {
            "pelvis": {"position": pelvis_pos.tolist(), "orientation": pelvis_quat.tolist()},
            "spine3": {"position": spine_pos.tolist(), "orientation": spine_quat.tolist()},
            "left_hip": {"position": left_hip.tolist(), "orientation": left_hip_quat.tolist()},
            "right_hip": {"position": right_hip.tolist(), "orientation": right_hip_quat.tolist()},
            "left_knee": {"position": left_knee.tolist(), "orientation": left_knee_quat.tolist()},
            "right_knee": {"position": right_knee.tolist(), "orientation": right_knee_quat.tolist()},
            "left_foot": {"position": left_foot.tolist(), "orientation": left_foot_quat.tolist()},
            "right_foot": {"position": right_foot.tolist(), "orientation": right_foot_quat.tolist()},
            "left_shoulder": {"position": left_shoulder.tolist(), "orientation": left_shoulder_quat.tolist()},
            "right_shoulder": {"position": right_shoulder.tolist(), "orientation": right_shoulder_quat.tolist()},
            "left_elbow": {"position": left_elbow.tolist(), "orientation": left_elbow_quat.tolist()},
            "right_elbow": {"position": right_elbow.tolist(), "orientation": right_elbow_quat.tolist()},
            "left_wrist": {"position": left_wrist.tolist(), "orientation": left_wrist_quat.tolist()},
            "right_wrist": {"position": right_wrist.tolist(), "orientation": right_wrist_quat.tolist()},
        }

        if "head" in target:
            out["head"] = {
                "position": list(target["head"]["position"]),
                "orientation": list(target["head"]["orientation"]),
            }

        for name in CORE_BODIES:
            if name not in out:
                raise RuntimeError(f"Canonical frame missing {name}")

        return out

    def _smooth_frames(
        self, frames: list[dict[str, dict[str, list[float]]]]
    ) -> list[dict[str, dict[str, list[float]]]]:
        if self.smooth_window <= 1 or len(frames) < 3:
            return frames

        half = self.smooth_window // 2
        n = len(frames)
        bodies = list(frames[0].keys())
        pos = {b: np.array([frames[t][b]["position"] for t in range(n)], dtype=np.float64) for b in bodies}
        quat = {b: np.array([frames[t][b]["orientation"] for t in range(n)], dtype=np.float64) for b in bodies}

        # Hemisphere-continue before averaging.
        for b in bodies:
            for t in range(1, n):
                quat[b][t] = hemisphere_continue(quat[b][t - 1], quat[b][t])

        out: list[dict[str, dict[str, list[float]]]] = []
        for t in range(n):
            t0 = max(0, t - half)
            t1 = min(n, t + half + 1)
            frame: dict[str, dict[str, list[float]]] = {}
            for b in bodies:
                p = pos[b][t0:t1].mean(axis=0)
                q = quat_normalize(quat[b][t0:t1].mean(axis=0))
                if t > 0:
                    q = hemisphere_continue(np.asarray(out[t - 1][b]["orientation"], dtype=np.float64), q)

                frame[b] = {"position": p.tolist(), "orientation": q.tolist()}

            out.append(frame)

        return out

    def _apply_contact_targets(
        self,
        frames: list[dict[str, dict[str, list[float]]]],
        contacts: list[dict[str, bool]],
    ) -> list[dict[str, dict[str, list[float]]]]:
        out = []
        for t, frame in enumerate(frames):
            updated = {name: dict(pose) for name, pose in frame.items()}
            for name in self.foot_bodies:
                if not contacts[t].get(name, False):
                    continue

                pos = _as3(updated[name]["position"])
                if t > 0 and contacts[t - 1].get(name, False):
                    prev = _as3(out[t - 1][name]["position"])
                    pos[0] = prev[0]
                    pos[1] = prev[1]
                    if self.lock_contact_z:
                        pos[2] = prev[2]

                if self.lock_contact_z and pos[2] < self.height_threshold:
                    pos[2] = 0.0

                updated[name] = {
                    "position": pos.tolist(),
                    "orientation": list(updated[name]["orientation"]),
                }

            out.append(updated)

        return out

    def _measure_quality(
        self,
        targets: list[dict[str, dict[str, list[float]]]],
        fitted: list[dict[str, dict[str, list[float]]]],
        contacts: list[dict[str, bool]],
        fps: float,
    ) -> FitQuality:
        key_bodies = ["pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist"]
        errors: list[float] = []
        rot_errors: list[float] = []
        for t, (tgt, fit) in enumerate(zip(targets, fitted)):
            for name in key_bodies:
                if name not in tgt or name not in fit:
                    continue

                dp = _as3(tgt[name]["position"]) - _as3(fit[name]["position"])
                errors.append(float(np.linalg.norm(dp)))
                q1 = quat_normalize(np.asarray(tgt[name]["orientation"], dtype=np.float64))
                q2 = quat_normalize(np.asarray(fit[name]["orientation"], dtype=np.float64))
                if float(np.dot(q1, q2)) < 0:
                    q2 = -q2

                qe = quat_mul(q1, np.array([q2[0], -q2[1], -q2[2], -q2[3]]))
                ang = 2.0 * np.arctan2(np.linalg.norm(qe[1:]), abs(qe[0]))
                rot_errors.append(float(np.degrees(ang)))

        slip = []
        dt = 1.0 / float(fps)
        for name in self.foot_bodies:
            for t in range(1, len(fitted)):
                if contacts[t].get(name, False) and contacts[t - 1].get(name, False):
                    d = _as3(fitted[t][name]["position"]) - _as3(fitted[t - 1][name]["position"])
                    slip.append(float(np.linalg.norm(d[:2]) / dt))

        err_arr = np.asarray(errors, dtype=np.float64) if errors else np.array([0.0])
        rot_arr = np.asarray(rot_errors, dtype=np.float64) if rot_errors else np.array([0.0])
        slip_arr = np.asarray(slip, dtype=np.float64) if slip else np.array([0.0])
        return FitQuality(
            semantic_position_rmse_m=float(np.sqrt(np.mean(err_arr**2))),
            semantic_position_p95_m=float(np.percentile(err_arr, 95)),
            semantic_rotation_mean_deg=float(np.mean(rot_arr)),
            contact_slip_mean_mps=float(np.mean(slip_arr)),
        )
