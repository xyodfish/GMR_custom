"""Read and validate robot-A joint trajectories (LAFAN1 G1 CSV / qpos JSON)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LAFAN1_G1_NQ = 36  # xyz + xyzw + 29 hinge joints
MUJOCO_G1_NQ = 36  # xyz + wxyz + 29 hinge joints


@dataclass
class SourceTrajectory:
    robot_model: str
    model_hash: str
    joint_order: list[str]
    qpos_frames: np.ndarray  # [T, nq] MuJoCo order (root quat wxyz)
    fps: float
    root_type: str
    coordinate_convention: dict[str, Any]
    source_id: str
    local_motion_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.qpos_frames.shape[0])


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()[:16]


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(out)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Root quaternion has near-zero norm")

    out /= norm
    return out


def _hemisphere_continue(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    if float(np.dot(prev, curr)) < 0.0:
        return -curr

    return curr


class SourceTrajectoryReader:
    """Convert external joint files into a validated MuJoCo qpos sequence."""

    def __init__(self, mapping_yaml: Path, puppet_root: Path):
        self.puppet_root = Path(puppet_root).resolve()
        self.mapping_path = Path(mapping_yaml).resolve()
        with self.mapping_path.open("r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle)

        robot_model = Path(self.cfg["robot_model"])
        if not robot_model.is_absolute():
            robot_model = self.puppet_root / robot_model

        self.robot_model = robot_model.resolve()
        if not self.robot_model.is_file():
            raise FileNotFoundError(f"Robot model not found: {self.robot_model}")

        self.model_hash = _sha256_file(self.robot_model)
        self.joint_order = list(self.cfg.get("joint_order_lafan1_csv", []))
        self.fps_default = float(self.cfg.get("fps_default", 30))
        self.root_type = str(self.cfg.get("root_type", "free"))
        self.coordinate_convention = dict(self.cfg.get("coordinate_convention", {}))

    def load(self, path: Path, *, fps: float | None = None, max_frames: int | None = None) -> SourceTrajectory:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            qpos = self._load_lafan1_csv(path)
        elif suffix == ".json":
            qpos = self._load_qpos_json(path)
        elif suffix == ".npy":
            qpos = np.load(path)
        else:
            raise ValueError(f"Unsupported trajectory format: {suffix}")

        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2:
            raise ValueError(f"Expected [T, nq] qpos, got shape {qpos.shape}")

        if max_frames is not None:
            qpos = qpos[: int(max_frames)]

        self._validate(qpos)
        resolved_fps = float(fps if fps is not None else self.fps_default)
        if resolved_fps <= 0:
            raise ValueError("fps must be positive")

        return SourceTrajectory(
            robot_model=str(self.robot_model),
            model_hash=self.model_hash,
            joint_order=self.joint_order,
            qpos_frames=qpos,
            fps=resolved_fps,
            root_type=self.root_type,
            coordinate_convention=self.coordinate_convention,
            source_id=path.stem,
            local_motion_only=self.root_type == "fixed",
            metadata={"input_path": str(path), "mapping": str(self.mapping_path)},
        )

    def _load_lafan1_csv(self, path: Path) -> np.ndarray:
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        if raw.shape[1] != LAFAN1_G1_NQ:
            raise ValueError(f"LAFAN1 G1 CSV expects {LAFAN1_G1_NQ} columns, got {raw.shape[1]}")

        qpos = np.zeros_like(raw)
        qpos[:, 0:3] = raw[:, 0:3]
        # CSV: qx qy qz qw  -> MuJoCo: qw qx qy qz
        qpos[:, 3] = raw[:, 6]
        qpos[:, 4] = raw[:, 3]
        qpos[:, 5] = raw[:, 4]
        qpos[:, 6] = raw[:, 5]
        qpos[:, 7:] = raw[:, 7:]

        prev_q = None
        for t in range(qpos.shape[0]):
            quat = _normalize_quat_wxyz(qpos[t, 3:7])
            if prev_q is not None:
                quat = _hemisphere_continue(prev_q, quat)

            qpos[t, 3:7] = quat
            prev_q = quat

        return qpos

    def _load_qpos_json(self, path: Path) -> np.ndarray:
        import json

        root = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(root, dict) and "qpos_frames" in root:
            frames = root["qpos_frames"]
        else:
            frames = root

        qpos = np.asarray(frames, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != MUJOCO_G1_NQ:
            raise ValueError(f"qpos JSON expects [T, {MUJOCO_G1_NQ}], got {qpos.shape}")

        return qpos

    def _validate(self, qpos: np.ndarray) -> None:
        if not np.isfinite(qpos).all():
            raise ValueError("Trajectory contains non-finite values")

        if qpos.shape[1] != MUJOCO_G1_NQ:
            raise ValueError(f"Expected nq={MUJOCO_G1_NQ}, got {qpos.shape[1]}")

        if self.root_type == "free":
            norms = np.linalg.norm(qpos[:, 3:7], axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-2):
                raise ValueError("Free-root quaternions are not unit length")

        # Reject single-frame root teleport (> 0.5 m / frame at default fps is extreme).
        if qpos.shape[0] >= 2:
            delta = np.linalg.norm(np.diff(qpos[:, 0:3], axis=0), axis=1)
            if float(np.max(delta)) > 0.5:
                raise ValueError(f"Detected root teleport (max step {float(np.max(delta)):.3f} m)")
