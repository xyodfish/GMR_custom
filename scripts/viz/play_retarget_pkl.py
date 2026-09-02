#!/usr/bin/env python3
"""Play retargeted robot motion (.pkl) in MuJoCo.

Examples:
  # single clip (robot auto-read from pkl when present)
  python scripts/viz/play_retarget_pkl.py \\
    --motion ~/Workspace/gmr_cg_batch_h2/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii_gmr.pkl

  # browse all pkls under a folder: [ ] switch clip, Space pause
  python scripts/viz/play_retarget_pkl.py --root ~/Workspace/gmr_cg_batch_h2/ACCAD

  # list clips
  python scripts/viz/play_retarget_pkl.py --root ~/Workspace/gmr_cg_batch_h2 --list

  # record mp4
  python scripts/viz/play_retarget_pkl.py --motion path/to/clip_gmr.pkl --record_video --video_path /tmp/clip.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import PLANAR_BASE_ROBOTS, RobotMotionViewer, load_robot_motion


def _infer_robot(motion_path: Path, robot_arg: str | None) -> str:
    if robot_arg:
        return robot_arg

    with motion_path.open("rb") as stream:
        motion_data = pickle.load(stream)

    robot = motion_data.get("robot") if isinstance(motion_data, dict) else None
    if robot:
        return str(robot)

    raise SystemExit(
        f"Could not infer robot from {motion_path}. Pass --robot unitree_h2 (or unitree_g1)."
    )


def _collect_pkls(root: Path) -> list[Path]:
    return sorted(root.rglob("*_gmr.pkl")) if root.is_dir() else [root]


def _print_clip_list(clips: list[Path], root: Path | None) -> None:
    for index, clip in enumerate(clips):
        label = str(clip.relative_to(root)) if root is not None else str(clip)
        print(f"[{index:4d}] {label}")


def _play_one(
    motion_path: Path,
    robot_type: str,
    *,
    record_video: bool,
    video_path: str,
    loop: bool,
    playback_speed: float,
    playback_control: Path | None,
    playback_status: Path | None,
) -> None:
    (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        _local_body_pos,
        _link_body_list,
        motion_qpos,
    ) = load_robot_motion(motion_path)

    n_frames = len(motion_qpos) if motion_qpos is not None else len(motion_root_pos)
    method = motion_data.get("method", "?") if isinstance(motion_data, dict) else "?"
    print(
        f"[play] {motion_path.name} | robot={robot_type} fps={motion_fps:.1f} "
        f"frames={n_frames} method={method}",
        flush=True,
    )

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps * playback_speed,
        camera_follow=robot_type not in PLANAR_BASE_ROBOTS,
        record_video=record_video,
        video_path=video_path,
    )

    def read_playback_command() -> dict:
        if playback_control is None:
            return {"command_id": -1, "paused": False, "seek_frame": None}

        try:
            return json.loads(playback_control.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"command_id": -1, "paused": False, "seek_frame": None}

    def write_playback_status(frame: int, paused: bool) -> None:
        if playback_status is None:
            return

        payload = {"frame": frame, "total_frames": n_frames, "paused": paused}
        temporary = playback_status.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(playback_status)
        except OSError:
            pass

    frame_idx = 0
    last_command_id = -1
    try:
        while env.viewer.is_running():
            command = read_playback_command()
            command_id = int(command.get("command_id", -1))
            if command_id != last_command_id:
                seek_frame = command.get("seek_frame")
                if seek_frame is not None:
                    frame_idx = min(max(0, int(seek_frame)), n_frames - 1)

                last_command_id = command_id

            paused = bool(command.get("paused", False))
            if paused:
                write_playback_status(frame_idx, True)
                env.viewer.sync()
                time.sleep(0.03)
                continue

            if motion_qpos is not None:
                env.step(
                    qpos=motion_qpos[frame_idx],
                    rate_limit=True,
                    follow_camera=env.camera_follow,
                )
            elif robot_type in PLANAR_BASE_ROBOTS:
                env.step(
                    root_pos=motion_root_pos[frame_idx],
                    dof_pos=motion_dof_pos[frame_idx],
                    rate_limit=True,
                    follow_camera=env.camera_follow,
                )
            else:
                env.step(
                    motion_root_pos[frame_idx],
                    motion_root_rot[frame_idx],
                    motion_dof_pos[frame_idx],
                    rate_limit=True,
                    follow_camera=env.camera_follow,
                )

            write_playback_status(frame_idx, False)
            frame_idx += 1
            if frame_idx >= n_frames:
                if not loop:
                    break

                frame_idx = 0

    finally:
        env.close()


def _play_folder(
    clips: list[Path],
    root: Path,
    robot_arg: str | None,
    *,
    start_index: int,
    record_video: bool,
    video_path: str,
    loop_clip: bool,
) -> None:
    if not clips:
        raise SystemExit(f"No *_gmr.pkl under {root}")

    motion_id = max(0, min(start_index, len(clips) - 1))
    current_id = -1
    frame_idx = 0
    paused = False

    env = None
    robot_type = robot_arg
    motion_fps = 30.0
    motion_root_pos = None
    motion_root_rot = None
    motion_dof_pos = None
    motion_qpos = None
    n_frames = 0

    def _load_clip(clip_id: int) -> None:
        nonlocal env, robot_type, motion_fps, motion_root_pos, motion_root_rot
        nonlocal motion_dof_pos, motion_qpos, n_frames, frame_idx

        clip = clips[clip_id]
        robot_type = _infer_robot(clip, robot_arg)
        (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            _local_body_pos,
            _link_body_list,
            motion_qpos,
        ) = load_robot_motion(clip)

        n_frames = len(motion_qpos) if motion_qpos is not None else len(motion_root_pos)
        method = motion_data.get("method", "?") if isinstance(motion_data, dict) else "?"
        label = clip.relative_to(root)
        print(
            f"[play] [{clip_id + 1}/{len(clips)}] {label} | robot={robot_type} "
            f"fps={motion_fps:.1f} frames={n_frames} method={method}",
            flush=True,
        )

        if env is not None:
            env.close()

        env = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            camera_follow=robot_type not in PLANAR_BASE_ROBOTS,
            record_video=record_video,
            video_path=video_path,
            keyboard_callback=_keyboard_callback,
        )
        frame_idx = 0

    def _keyboard_callback(keycode: int) -> None:
        nonlocal motion_id, current_id, paused, frame_idx

        key = chr(keycode)
        if key == " ":
            paused = not paused
            print(f"[play] {'paused' if paused else 'playing'}", flush=True)

        if key == "[":
            motion_id = (motion_id - 1) % len(clips)
            current_id = -1

        if key == "]":
            motion_id = (motion_id + 1) % len(clips)
            current_id = -1

    print(f"[play] {len(clips)} clips under {root}. Keys: [ ] switch, Space pause.", flush=True)
    _load_clip(motion_id)
    current_id = motion_id

    try:
        while True:
            if current_id != motion_id:
                _load_clip(motion_id)
                current_id = motion_id

            if not paused:
                if motion_qpos is not None:
                    env.step(
                        qpos=motion_qpos[frame_idx],
                        rate_limit=True,
                        follow_camera=env.camera_follow,
                    )
                elif robot_type in PLANAR_BASE_ROBOTS:
                    env.step(
                        root_pos=motion_root_pos[frame_idx],
                        dof_pos=motion_dof_pos[frame_idx],
                        rate_limit=True,
                        follow_camera=env.camera_follow,
                    )
                else:
                    env.step(
                        motion_root_pos[frame_idx],
                        motion_root_rot[frame_idx],
                        motion_dof_pos[frame_idx],
                        rate_limit=True,
                        follow_camera=env.camera_follow,
                    )

                frame_idx += 1
                if frame_idx >= n_frames:
                    if loop_clip:
                        frame_idx = 0
                    else:
                        motion_id = (motion_id + 1) % len(clips)
                        current_id = -1

    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, default=None, help="Single retargeted .pkl file.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Folder to browse (recursive *_gmr.pkl). Use [ ] to switch clips.",
    )
    parser.add_argument("--robot", type=str, default=None, help="Override robot (e.g. unitree_h2).")
    parser.add_argument("--index", type=int, default=0, help="Start clip index when using --root.")
    parser.add_argument("--list", action="store_true", help="List clips and exit.")
    parser.add_argument("--no_loop", action="store_true", help="Play once then exit (single clip).")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--playback_control", type=Path, default=None)
    parser.add_argument("--playback_status", type=Path, default=None)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/retarget_playback.mp4")
    args = parser.parse_args()

    if args.motion is None and args.root is None:
        parser.error("Pass --motion <file.pkl> or --root <folder>.")

    if args.motion is not None and args.root is not None:
        parser.error("Use either --motion or --root, not both.")

    if args.playback_speed <= 0.0:
        parser.error("--playback_speed must be positive.")

    if args.motion is not None:
        motion_path = args.motion.expanduser().resolve()
        if not motion_path.is_file():
            raise SystemExit(f"Motion file not found: {motion_path}")

        robot_type = _infer_robot(motion_path, args.robot)
        _play_one(
            motion_path,
            robot_type,
            record_video=args.record_video,
            video_path=args.video_path,
            loop=not args.no_loop,
            playback_speed=args.playback_speed,
            playback_control=args.playback_control,
            playback_status=args.playback_status,
        )
        return

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    clips = _collect_pkls(root)
    if args.list:
        _print_clip_list(clips, root if root.is_dir() else None)
        print(f"total={len(clips)}")
        return

    _play_folder(
        clips,
        root,
        args.robot,
        start_index=args.index,
        record_video=args.record_video,
        video_path=args.video_path,
        loop_clip=not args.no_loop,
    )


if __name__ == "__main__":
    main()
