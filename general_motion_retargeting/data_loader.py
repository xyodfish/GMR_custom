import json
import pickle
from pathlib import Path

import numpy as np


def _qpos_arrays_from_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=float)
    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7]
    dof_pos = qpos[:, 7:]
    return qpos, root_pos, root_rot, dof_pos


def _load_robot_motion_pkl(motion_file: str | Path):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    motion_fps = motion_data["fps"]
    motion_root_pos = motion_data["root_pos"]
    motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    motion_dof_pos = motion_data["dof_pos"]
    motion_local_body_pos = motion_data.get("local_body_pos")
    motion_link_body_list = motion_data.get("link_body_list")
    motion_qpos = motion_data.get("qpos")
    if motion_qpos is not None:
        motion_qpos = np.asarray(motion_qpos, dtype=float)
    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
        motion_qpos,
    )


def _load_robot_motion_json(motion_file: str | Path):
    with open(motion_file, "r", encoding="utf-8") as f:
        motion_data = json.load(f)

    motion_fps = float(motion_data.get("fps", 30.0))
    motion_local_body_pos = motion_data.get("local_body_pos")
    motion_link_body_list = motion_data.get("link_body_list")

    if "qpos_frames" in motion_data:
        motion_qpos, motion_root_pos, motion_root_rot, motion_dof_pos = _qpos_arrays_from_qpos(
            motion_data["qpos_frames"]
        )
    elif "qpos" in motion_data:
        motion_qpos, motion_root_pos, motion_root_rot, motion_dof_pos = _qpos_arrays_from_qpos(
            motion_data["qpos"]
        )
    else:
        motion_root_pos = np.asarray(motion_data["root_pos"], dtype=float)
        root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=float)
        motion_root_rot = root_rot_xyzw[:, [3, 0, 1, 2]]
        motion_dof_pos = np.asarray(motion_data["dof_pos"], dtype=float)
        motion_qpos = np.hstack([motion_root_pos, motion_root_rot, motion_dof_pos])

    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
        motion_qpos,
    )


def load_robot_motion(motion_file):
    """Load robot motion from PKL (GMR) or JSON (batch TO / C++ export)."""
    path = Path(motion_file)
    if path.suffix.lower() == ".json":
        return _load_robot_motion_json(path)
    return _load_robot_motion_pkl(path)
