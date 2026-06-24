"""Streaming contact detection and ground handling for GMR (KCR-inspired).

Designed for causal / real-time use: no future frames, no full-sequence
interval means. Suitable for teleop and frame-by-frame offline playback.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import mujoco as mj
import numpy as np


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_foot_positions(human_data: dict, foot_bodies: list[str]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for name in foot_bodies:
        if name not in human_data:
            continue
        pos = np.asarray(human_data[name][0], dtype=np.float64).reshape(3)
        positions[name] = pos
    return positions


class StreamingContactDetector:
    """Causal foot contact detector using past-only velocity + height hysteresis."""

    def __init__(
        self,
        fps: float,
        vel_threshold: float = 0.5,
        height_threshold: float = 0.08,
        height_off_threshold: float = 0.12,
        vel_window: int = 6,
    ) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = float(fps)
        self.vel_threshold = float(vel_threshold)
        self.height_threshold = float(height_threshold)
        self.height_off_threshold = float(max(height_off_threshold, height_threshold))
        self.vel_window = max(2, int(vel_window))
        self._buf: deque[dict[str, np.ndarray]] = deque(maxlen=self.vel_window)
        self._last_contacts: dict[str, bool] = {}

    def update(self, foot_positions: dict[str, np.ndarray]) -> dict[str, bool]:
        self._buf.append({name: pos.copy() for name, pos in foot_positions.items()})
        contacts: dict[str, bool] = {}
        dt = max((len(self._buf) - 1) / self.fps, 1.0 / self.fps)

        for name, pos in foot_positions.items():
            was_contact = self._last_contacts.get(name, False)
            z_limit = self.height_off_threshold if was_contact else self.height_threshold
            z_ok = float(pos[2]) <= z_limit

            if len(self._buf) < 2:
                vel_ok = True
            else:
                displacement = pos - self._buf[0][name]
                speed = float(np.linalg.norm(displacement)) / dt
                vel_ok = speed <= self.vel_threshold

            contacts[name] = z_ok and vel_ok

        self._last_contacts = contacts
        return contacts


class StreamingGroundAligner:
    """Contact-gated vertical shift of human reference (freeze offset when airborne)."""

    def __init__(
        self,
        ground_z: float = 0.0,
        ground_margin: float = 0.02,
        lpf_alpha: float = 0.3,
        airborne_height_threshold: float = 0.15,
        airborne_offset_decay: float = 0.85,
    ) -> None:
        self.ground_z = float(ground_z)
        self.ground_margin = float(ground_margin)
        self.lpf_alpha = float(np.clip(lpf_alpha, 1e-6, 1.0))
        self.airborne_height_threshold = float(airborne_height_threshold)
        self.airborne_offset_decay = float(np.clip(airborne_offset_decay, 0.0, 1.0))
        self.last_offset = 0.0

    def update(
        self,
        human_data: dict,
        contacts: dict[str, bool],
        foot_positions: dict[str, np.ndarray] | None = None,
    ) -> dict:
        if foot_positions:
            max_foot_z = max(float(pos[2]) for pos in foot_positions.values())
            if max_foot_z > self.airborne_height_threshold:
                self.last_offset *= self.airborne_offset_decay

        active_z = [
            float(np.asarray(human_data[name][0], dtype=np.float64).reshape(3)[2])
            for name, in_contact in contacts.items()
            if in_contact and name in human_data
        ]

        if active_z:
            target_z = self.ground_z + self.ground_margin
            raw_offset = min(active_z) - target_z
            self.last_offset = (
                self.lpf_alpha * raw_offset + (1.0 - self.lpf_alpha) * self.last_offset
            )

        offset_vec = np.array([0.0, 0.0, self.last_offset], dtype=np.float64)
        aligned: dict = {}
        for body_name, (pos, quat) in human_data.items():
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
            aligned[body_name] = [pos_arr - offset_vec, quat]
        return aligned


class StreamingFootLocker:
    """EMA foot position lock while in contact to reduce foot sliding."""

    def __init__(self, ema_alpha: float = 0.05) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 1e-6, 1.0))
        self._locked: dict[str, np.ndarray] = {}

    def apply(self, human_data: dict, contacts: dict[str, bool]) -> dict:
        updated: dict = {}
        for body_name, (pos, quat) in human_data.items():
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
            in_contact = contacts.get(body_name, False)
            if in_contact:
                if body_name not in self._locked:
                    self._locked[body_name] = pos_arr.copy()
                else:
                    self._locked[body_name] = (
                        (1.0 - self.ema_alpha) * self._locked[body_name]
                        + self.ema_alpha * pos_arr
                    )
                updated[body_name] = [self._locked[body_name].copy(), quat]
            else:
                self._locked.pop(body_name, None)
                updated[body_name] = [pos_arr, quat]
        return updated


def resolve_foot_geom_ids(model: mj.MjModel, geom_names: list[str] | None = None) -> list[int]:
    if geom_names:
        ids: list[int] = []
        for name in geom_names:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.append(gid)
        return ids

    ids = []
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name is None:
            continue
        lower = name.lower()
        if "foot" in lower and "collision" in lower:
            ids.append(gid)
    return ids


def resolve_foot_body_ids(model: mj.MjModel, body_names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(bid)
    return ids


def collect_foot_body_subtree(model: mj.MjModel, root_body_names: list[str]) -> list[int]:
    """Collect root foot bodies and all descendants (e.g. toe links)."""
    body_ids: set[int] = set()
    for name in root_body_names:
        root_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if root_id < 0:
            continue
        stack = [root_id]
        while stack:
            body_id = stack.pop()
            if body_id in body_ids:
                continue
            body_ids.add(body_id)
            for child_id in range(1, model.nbody):
                if int(model.body_parentid[child_id]) == body_id:
                    stack.append(child_id)
    return sorted(body_ids)


def collect_geom_ids_for_bodies(model: mj.MjModel, body_ids: list[int]) -> list[int]:
    body_set = set(body_ids)
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in body_set:
            geom_ids.append(geom_id)
    return geom_ids


def measure_robot_foot_min_z(
    model: mj.MjModel,
    data: mj.MjData,
    foot_geom_ids: list[int],
    foot_body_ids: list[int],
) -> float:
    mj.mj_forward(model, data)
    min_z = np.inf
    for geom_id in foot_geom_ids:
        min_z = min(min_z, float(data.geom_xpos[geom_id, 2]))
    for body_id in foot_body_ids:
        min_z = min(min_z, float(data.xpos[body_id, 2]))
    return min_z


def fix_robot_foot_penetration(
    model: mj.MjModel,
    data: mj.MjData,
    foot_geom_ids: list[int],
    foot_body_ids: list[int] | None = None,
    floor_geom_name: str = "floor",
    penetration_margin: float = 0.01,
    max_iterations: int = 3,
) -> float:
    """Lift free-flyer root Z until foot geoms/bodies clear the ground plane."""
    if model.nq < 3:
        return 0.0
    if not foot_geom_ids and not foot_body_ids:
        return 0.0

    total_lift = 0.0
    foot_body_ids = foot_body_ids or []

    for _ in range(max(1, int(max_iterations))):
        min_z = measure_robot_foot_min_z(model, data, foot_geom_ids, foot_body_ids)
        if not np.isfinite(min_z) or min_z >= penetration_margin:
            break
        lift = penetration_margin - min_z
        data.qpos[2] += lift
        total_lift += lift

    if total_lift > 0.0:
        mj.mj_forward(model, data)
    return total_lift


class ContactGroundPipeline:
    """Orchestrates streaming contact, human ground align, foot lock, and robot penetration fix."""

    def __init__(self, cfg: dict[str, Any], model: mj.MjModel, fps: float = 30.0) -> None:
        self.enabled = bool(cfg.get("enabled", False))
        self.foot_bodies = list(cfg.get("foot_bodies", []))
        self.enable_foot_lock = bool(cfg.get("enable_foot_lock", True))
        self.fix_penetration = bool(cfg.get("fix_robot_penetration", True))
        self.penetration_margin = _as_float(cfg.get("penetration_margin", 0.01), 0.01)
        self.penetration_max_iterations = _as_int(cfg.get("penetration_max_iterations", 3), 3)
        self.floor_geom_name = str(cfg.get("floor_geom_name", "floor"))

        self.contact_detector = StreamingContactDetector(
            fps=fps,
            vel_threshold=_as_float(cfg.get("vel_threshold", 0.5), 0.5),
            height_threshold=_as_float(cfg.get("height_threshold", 0.08), 0.08),
            height_off_threshold=_as_float(cfg.get("height_off_threshold", 0.12), 0.12),
            vel_window=_as_int(cfg.get("vel_window", 6), 6),
        )
        self.ground_aligner = StreamingGroundAligner(
            ground_z=_as_float(cfg.get("ground_z", 0.0), 0.0),
            ground_margin=_as_float(cfg.get("ground_margin", 0.02), 0.02),
            lpf_alpha=_as_float(cfg.get("lpf_alpha", 0.3), 0.3),
            airborne_height_threshold=_as_float(cfg.get("airborne_height_threshold", 0.15), 0.15),
            airborne_offset_decay=_as_float(cfg.get("airborne_offset_decay", 0.85), 0.85),
        )
        self.foot_locker = StreamingFootLocker(
            ema_alpha=_as_float(cfg.get("foot_lock_ema_alpha", 0.05), 0.05),
        )
        geom_names = cfg.get("foot_collision_geoms")
        explicit_geom_ids = resolve_foot_geom_ids(
            model,
            list(geom_names) if geom_names else None,
        )
        robot_foot_bodies = list(
            cfg.get(
                "robot_foot_bodies",
                ["left_ankle_roll_link", "right_ankle_roll_link"],
            )
        )
        self.foot_body_ids = resolve_foot_body_ids(model, robot_foot_bodies)
        subtree_body_ids = collect_foot_body_subtree(model, robot_foot_bodies)
        subtree_geom_ids = collect_geom_ids_for_bodies(model, subtree_body_ids)
        self.foot_geom_ids = sorted(set(explicit_geom_ids + subtree_geom_ids))
        self.last_contacts: dict[str, bool] = {}
        self.last_root_lift = 0.0

    def set_fps(self, fps: float) -> None:
        if fps > 0.0:
            self.contact_detector.fps = float(fps)

    def process_human_frame(self, human_data: dict) -> dict:
        foot_positions = extract_foot_positions(human_data, self.foot_bodies)
        if not foot_positions:
            return human_data

        contacts = self.contact_detector.update(foot_positions)
        self.last_contacts = contacts

        aligned = self.ground_aligner.update(human_data, contacts, foot_positions)
        if self.enable_foot_lock:
            aligned = self.foot_locker.apply(aligned, contacts)
        return aligned

    def fix_robot_penetration(self, model: mj.MjModel, data: mj.MjData) -> float:
        if not self.fix_penetration:
            return 0.0
        self.last_root_lift = fix_robot_foot_penetration(
            model,
            data,
            self.foot_geom_ids,
            foot_body_ids=self.foot_body_ids,
            floor_geom_name=self.floor_geom_name,
            penetration_margin=self.penetration_margin,
            max_iterations=self.penetration_max_iterations,
        )
        return self.last_root_lift
