from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, load_robot_motion, PLANAR_BASE_ROBOTS
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_RETARGET_DIR = Path(__file__).resolve().parents[1] / "retarget"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_RETARGET_DIR))

from human_json_to_robot import load_human_frames


def _required_human_bodies(retarget) -> set[str]:
    bodies: set[str] = set()
    for entry in retarget.task_frames1:
        bodies.add(entry["human_body"])
    if retarget.use_ik_match_table2:
        for entry in retarget.task_frames2:
            bodies.add(entry["human_body"])
    return bodies


def _missing_human_bodies(frame: dict, required: set[str]) -> list[str]:
    return sorted(body for body in required if body not in frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True,
                        help="Robot motion .pkl (GMR) or .json (batch TO / C++ export).")
    parser.add_argument(
        "--human_frame_json",
        type=str,
        default=None,
        help="Optional human frame json to visualize IK target anchors.",
    )
    parser.add_argument(
        "--src_human",
        type=str,
        default=None,
        help="Human format for IK config. Defaults to json metadata or bvh_lafan1.",
    )
    parser.add_argument(
        "--show_human_body_name",
        action="store_true",
        default=False,
        help="Show human body names on IK target anchors.",
    )
    parser.add_argument(
        "--show_raw_human_targets",
        action="store_true",
        default=False,
        help="Show unscaled human json targets instead of GMR IK targets.",
    )

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    human_frames = None
    retarget = None
    use_scaled_human_targets = False
    if args.human_frame_json is not None:
        if not os.path.exists(args.human_frame_json):
            raise FileNotFoundError(f"Human frame json {args.human_frame_json} not found")
        with open(args.human_frame_json, "r", encoding="utf-8") as f:
            human_json_root = json.load(f)
        human_frames, human_fps = load_human_frames(args.human_frame_json)
        src_human = args.src_human or human_json_root.get("src_human", "bvh_lafan1")
        actual_human_height = human_json_root.get("actual_human_height")
        if args.show_raw_human_targets:
            use_scaled_human_targets = False
        else:
            retarget = GMR(
                src_human=src_human,
                tgt_robot=robot_type,
                actual_human_height=actual_human_height,
                verbose=False,
            )
            missing = _missing_human_bodies(human_frames[0], _required_human_bodies(retarget))
            if missing:
                print(
                    f"[vis_robot_motion] Warning: human JSON missing {len(missing)} bodies for "
                    f"{src_human} IK overlay (e.g. {missing[:3]}). "
                    "Showing robot motion only. Use a full LAFAN1 BVH for IK anchors, "
                    "or pass --show_raw_human_targets for partial raw bones."
                )
                retarget = None
            else:
                use_scaled_human_targets = True
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list, motion_qpos = load_robot_motion(robot_motion_path)
    n_frames = len(motion_qpos) if motion_qpos is not None else len(motion_root_pos)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=robot_type not in PLANAR_BASE_ROBOTS,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        human_motion_data = None
        if human_frames is not None:
            human_frame = human_frames[frame_idx % len(human_frames)]
            if args.show_raw_human_targets:
                human_motion_data = human_frame
            elif use_scaled_human_targets and retarget is not None:
                retarget.update_targets(human_frame)
                human_motion_data = retarget.scaled_human_data

        if motion_qpos is not None:
            env.step(
                qpos=motion_qpos[frame_idx],
                human_motion_data=human_motion_data,
                show_human_body_name=args.show_human_body_name,
                rate_limit=True,
                follow_camera=env.camera_follow,
            )
        elif robot_type in PLANAR_BASE_ROBOTS:
            env.step(
                root_pos=motion_root_pos[frame_idx],
                dof_pos=motion_dof_pos[frame_idx],
                human_motion_data=human_motion_data,
                show_human_body_name=args.show_human_body_name,
                rate_limit=True,
                follow_camera=env.camera_follow,
            )
        else:
            env.step(
                motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx],
                human_motion_data=human_motion_data,
                show_human_body_name=args.show_human_body_name,
                rate_limit=True,
                follow_camera=env.camera_follow,
            )
        frame_idx += 1
        if frame_idx >= n_frames:
            frame_idx = 0
    env.close()