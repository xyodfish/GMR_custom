"""Extract calibrated semantic sites from robot-A MuJoCo FK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .source_trajectory import SourceTrajectory


CORE_BODIES = (
    "pelvis",
    "spine3",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass
class SiteSpec:
    name: str
    source_body: str
    position_offset_local: np.ndarray
    orientation_offset_wxyz: np.ndarray
    position_weight: float
    orientation_weight: float
    body_id: int = -1


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return q / n


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def hemisphere_continue(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    if float(np.dot(prev, curr)) < 0.0:
        return -curr

    return curr


class SemanticSiteMap:
    """Map robot-A FK bodies to calibrated canonical semantic sites."""

    def __init__(self, mapping_yaml: Path, puppet_root: Path):
        self.puppet_root = Path(puppet_root).resolve()
        with Path(mapping_yaml).open("r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle)

        robot_model = Path(self.cfg["robot_model"])
        if not robot_model.is_absolute():
            robot_model = self.puppet_root / robot_model

        self.model = mujoco.MjModel.from_xml_path(str(robot_model.resolve()))
        self.data = mujoco.MjData(self.model)
        self.canonical_height = float(self.cfg["canonical_height_m"])
        self.source_height = float(self.cfg["source_robot_reference_height_m"])
        if self.source_height <= 0:
            raise ValueError("source_robot_reference_height_m must be positive")

        self.global_scale = self.canonical_height / self.source_height
        self.sites: list[SiteSpec] = []
        for name, spec in self.cfg["sites"].items():
            body = str(spec["source_body"])
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0:
                raise ValueError(f"Unknown source body '{body}' for site '{name}'")

            self.sites.append(
                SiteSpec(
                    name=str(name),
                    source_body=body,
                    position_offset_local=np.asarray(spec["position_offset_local"], dtype=np.float64),
                    orientation_offset_wxyz=quat_normalize(
                        np.asarray(spec["orientation_offset_wxyz"], dtype=np.float64)
                    ),
                    position_weight=float(spec.get("position_weight", 1.0)),
                    orientation_weight=float(spec.get("orientation_weight", 1.0)),
                    body_id=int(body_id),
                )
            )

        missing = [name for name in CORE_BODIES if name not in {s.name for s in self.sites}]
        if missing:
            raise ValueError(f"Mapping missing core bodies: {missing}")

    def extract_frame(self, qpos: np.ndarray) -> dict[str, dict[str, list[float]]]:
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos length {qpos.shape[0]} != model.nq {self.model.nq}")

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        frame: dict[str, dict[str, list[float]]] = {}
        for site in self.sites:
            link_pos = np.asarray(self.data.xpos[site.body_id], dtype=np.float64)
            link_quat = quat_normalize(np.asarray(self.data.xquat[site.body_id], dtype=np.float64))
            semantic_pos = link_pos + quat_rotate(link_quat, site.position_offset_local)
            semantic_quat = quat_normalize(quat_mul(link_quat, site.orientation_offset_wxyz))
            frame[site.name] = {
                "position": semantic_pos.tolist(),
                "orientation": semantic_quat.tolist(),
            }

        return frame

    def extract_sequence(self, trajectory: SourceTrajectory) -> list[dict[str, dict[str, list[float]]]]:
        frames: list[dict[str, dict[str, list[float]]]] = []
        prev_quats: dict[str, np.ndarray] = {}
        root0 = None

        for t in range(trajectory.num_frames):
            raw = self.extract_frame(trajectory.qpos_frames[t])
            pelvis = np.asarray(raw["pelvis"]["position"], dtype=np.float64)
            if root0 is None:
                root0 = pelvis.copy()

            # One global_scale on root-relative motion (preserves speed in height/s).
            root0_scaled = root0 * np.array([1.0, 1.0, self.global_scale], dtype=np.float64)
            scaled: dict[str, dict[str, list[float]]] = {}
            for name, pose in raw.items():
                pos = np.asarray(pose["position"], dtype=np.float64)
                world = root0_scaled + (pos - root0) * self.global_scale
                quat = quat_normalize(np.asarray(pose["orientation"], dtype=np.float64))
                if name in prev_quats:
                    quat = hemisphere_continue(prev_quats[name], quat)

                prev_quats[name] = quat
                scaled[name] = {
                    "position": world.tolist(),
                    "orientation": quat.tolist(),
                }

            frames.append(scaled)

        return frames

    def site_weights(self) -> dict[str, tuple[float, float]]:
        return {s.name: (s.position_weight, s.orientation_weight) for s in self.sites}
