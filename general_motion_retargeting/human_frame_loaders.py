"""Load human motion frames from GVHMR .pt, SMPL-X, or BVH for retargeting."""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    get_smplx_data_offline_fast,
    load_gvhmr_pred_file,
    load_smplx_file,
)

InputType = Literal["gvhmr_pt", "smplx", "bvh_lafan1", "bvh_nokov", "auto"]


def detect_input_type(path: str | pathlib.Path) -> str:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".pt":
        return "gvhmr_pt"
    if suffix in (".npz", ".pkl"):
        return "smplx"
    if suffix == ".bvh":
        return "bvh_lafan1"
    raise ValueError(
        f"Cannot infer input type from extension '{suffix}'. "
        "Use --input_type gvhmr_pt|smplx|bvh_lafan1|bvh_nokov."
    )


def _normalize_frame(frame: dict) -> dict:
    out = {}
    for name, pose in frame.items():
        pos, quat = pose
        out[name] = (pos, quat)
    return out


def _load_bvh_frames(path: pathlib.Path, bvh_format: str) -> tuple[list[dict], float]:
    try:
        raw_frames, height = load_bvh_file(str(path), format=bvh_format)
        return [_normalize_frame(f) for f in raw_frames], float(height)
    except KeyError:
        import warnings

        warnings.warn(
            f"BVH '{path.name}' is not a full {bvh_format} skeleton; using generic bone names. "
            "Batch TO / IK overlay need standard LAFAN1 bones (LeftUpLeg, LeftFootMod, ...).",
            stacklevel=2,
        )
        import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
        from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

        data = read_bvh(str(path))
        global_data = lafan_utils.quat_fk(data.quats, data.pos, data.parents)
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0], dtype=np.float64)

        frames: list[dict] = []
        for frame_idx in range(data.pos.shape[0]):
            result: dict = {}
            for i, bone in enumerate(data.bones):
                orientation = lafan_utils.quat_mul(rotation_quat, global_data[0][frame_idx, i])
                position = global_data[1][frame_idx, i] @ rotation_matrix.T / 100.0
                result[bone] = (position, orientation)
            frames.append(result)
        return frames, 1.75


def load_human_motion_frames(
    input_file: str | pathlib.Path,
    *,
    input_type: InputType = "auto",
    body_model_dir: str | pathlib.Path | None = None,
    bvh_format: str = "lafan1",
    tgt_fps: int = 30,
    max_frames: int | None = None,
) -> tuple[list[dict], float, float, str]:
    """Return (human_frames, fps, actual_human_height, src_human)."""
    path = pathlib.Path(input_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    resolved_type = detect_input_type(path) if input_type == "auto" else input_type
    if resolved_type == "bvh_lafan1":
        bvh_format = "lafan1"
    elif resolved_type == "bvh_nokov":
        bvh_format = "nokov"

    if resolved_type == "gvhmr_pt":
        repo = pathlib.Path(__file__).resolve().parents[1]
        model_dir = pathlib.Path(body_model_dir or repo / "assets" / "body_models")
        smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(path, model_dir)
        frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        src_human = "smplx"
    elif resolved_type == "smplx":
        repo = pathlib.Path(__file__).resolve().parents[1]
        model_dir = pathlib.Path(body_model_dir or repo / "assets" / "body_models")
        smplx_data, body_model, smplx_output, height = load_smplx_file(path, model_dir)
        frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        src_human = "smplx"
    elif resolved_type in ("bvh_lafan1", "bvh_nokov"):
        frames, height = _load_bvh_frames(path, bvh_format)
        fps = float(tgt_fps)
        src_human = f"bvh_{bvh_format}"
    else:
        raise ValueError(f"Unsupported input_type: {resolved_type}")

    if max_frames is not None:
        frames = frames[:max_frames]
    return frames, float(fps), float(height), src_human


def frame_to_json_dict(frame: dict) -> dict:
    out = {}
    for name, (pos, quat_wxyz) in frame.items():
        out[name] = {
            "position": [float(x) for x in pos],
            "orientation": [float(x) for x in quat_wxyz],
        }
    return out
