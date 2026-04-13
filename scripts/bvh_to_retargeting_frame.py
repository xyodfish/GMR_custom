import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from general_motion_retargeting.utils.lafan1 import load_bvh_file
import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def _to_serializable_frames(
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> List[Dict[str, Dict[str, List[float]]]]:
    serialized = []
    for frame in frames:
        frame_out: Dict[str, Dict[str, List[float]]] = {}
        for body_name, (pos, quat_wxyz) in frame.items():
            frame_out[body_name] = {
                "position": np.asarray(pos, dtype=np.float64).tolist(),
                "orientation": np.asarray(quat_wxyz, dtype=np.float64).tolist(),
            }
        serialized.append(frame_out)
    return serialized


def _load_bvh_file_generic(
    bvh_file: str,
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float]:
    data = read_bvh(bvh_file)
    global_data = lafan_utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0], dtype=np.float64)  # wxyz

    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for frame in range(data.pos.shape[0]):
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for i, bone in enumerate(data.bones):
            orientation = lafan_utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100.0  # cm to m
            result[bone] = (position, orientation)
        frames.append(result)

    human_height = 1.75
    return frames, human_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", type=str, required=True, help="Path to input BVH file.")
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
        help="BVH skeleton format used by GMR mapping.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to output retargeting frame json.",
    )
    parser.add_argument(
        "--motion_fps",
        type=int,
        default=30,
        help="FPS stored in output json for downstream playback.",
    )
    args = parser.parse_args()

    try:
        frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    except KeyError:
        frames, actual_human_height = _load_bvh_file_generic(args.bvh_file)
    output = {
        "fps": int(args.motion_fps),
        "src_human": f"bvh_{args.format}",
        "actual_human_height": float(actual_human_height),
        "frames": _to_serializable_frames(frames),
    }

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Saved {len(frames)} frames to {args.save_path}")
