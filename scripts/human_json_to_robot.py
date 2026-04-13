import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer


def _is_body_state(value: Any) -> bool:
    if isinstance(value, dict):
        return "position" in value and "orientation" in value
    if isinstance(value, list) and len(value) == 2:
        return True
    return False


def _is_frame_dict(value: Any) -> bool:
    if not isinstance(value, dict) or len(value) == 0:
        return False
    return all(_is_body_state(v) for v in value.values())


def _to_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3D vector, got shape {arr.shape}")
    return arr


def _to_quat_wxyz(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"Expected a quaternion [w,x,y,z], got shape {arr.shape}")
    return arr


def _parse_frame(frame_obj: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    parsed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for body_name, body_value in frame_obj.items():
        if isinstance(body_value, dict):
            pos = _to_vec3(body_value["position"])
            quat = _to_quat_wxyz(body_value["orientation"])
        elif isinstance(body_value, list) and len(body_value) == 2:
            pos = _to_vec3(body_value[0])
            quat = _to_quat_wxyz(body_value[1])
        else:
            raise ValueError(
                f"Invalid body format for '{body_name}'. "
                "Use {'position':[x,y,z],'orientation':[w,x,y,z]} or [[x,y,z],[w,x,y,z]]."
            )
        parsed[body_name] = (pos, quat)
    return parsed


def _parse_frames_root(root: Any) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], int]:
    fps = 30

    if isinstance(root, dict) and "fps" in root:
        fps = int(root["fps"])

    if isinstance(root, dict) and "frames" in root:
        frames_node = root["frames"]
    else:
        frames_node = root

    if isinstance(frames_node, list):
        frames = [_parse_frame(frame) for frame in frames_node]
        return frames, fps

    if _is_frame_dict(frames_node):
        return [_parse_frame(frames_node)], fps

    if isinstance(frames_node, dict):
        # Support frame-index dict: {"0": {...}, "1": {...}}
        try:
            ordered_keys = sorted(frames_node.keys(), key=lambda x: int(x))
        except Exception:
            ordered_keys = sorted(frames_node.keys())
        frames = [_parse_frame(frames_node[k]) for k in ordered_keys]
        return frames, fps

    raise ValueError("Unsupported json structure for human frames.")


def load_human_frames(json_path: str) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], int]:
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    frames, fps = _parse_frames_root(root)
    if len(frames) == 0:
        raise ValueError("No frames found in the json file.")
    return frames, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_frame_json", type=str, required=True, help="Path to human frame json.")
    parser.add_argument(
        "--src_human",
        type=str,
        default="smplx",
        choices=["smplx", "bvh_lafan1", "bvh_nokov", "bvh_xsens", "fbx", "fbx_offline", "xrobot", "xsens_mvn"],
        help="Source human type for IK config selection.",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--actual_human_height", type=float, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--offset_to_ground", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max rendering steps; -1 means unlimited.")

    args = parser.parse_args()

    human_frames, motion_fps = load_human_frames(args.human_frame_json)

    retarget = GMR(
        src_human=args.src_human,
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
    )

    video_name = os.path.splitext(os.path.basename(args.human_frame_json))[0]
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{video_name}.mp4",
    )

    qpos_list = []
    i = 0
    step_count = 0
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        if args.max_steps > 0 and step_count >= args.max_steps:
            break

        if not args.loop and i >= len(human_frames):
            break

        frame_idx = i % len(human_frames)

        fps_counter += 1
        now = time.time()
        if now - fps_start_time >= 2.0:
            actual_fps = fps_counter / (now - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = now

        human_frame = human_frames[frame_idx]
        qpos = retarget.retarget(human_frame, offset_to_ground=args.offset_to_ground)

        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        if args.save_path is not None:
            qpos_list.append(qpos.copy())

        i += 1
        step_count += 1

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])

        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    viewer.close()
