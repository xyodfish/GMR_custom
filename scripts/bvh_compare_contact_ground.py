"""Generate side-by-side retargeting videos with/without contact-ground optimization.

Example:
    conda activate py310
    cd /data/open_src_code/GMR_custom
    python scripts/bvh_compare_contact_ground.py \\
        --bvh_file /data2/Documents/lafan1/multipleActions1_subject3.bvh
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import imageio
import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    GeneralMotionRetargeting as GMR,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file

from stitch_videos_side_by_side import stitch_videos_side_by_side

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def retarget_sequence(
    frames: list,
    *,
    robot: str,
    fmt: str,
    contact_ground: bool,
    motion_fps: int,
    actual_human_height: float,
) -> np.ndarray:
    retargeter = GMR(
        src_human=f"bvh_{fmt}",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        contact_ground=contact_ground,
        motion_fps=motion_fps,
        verbose=False,
    )
    retargeter.set_motion_fps(motion_fps)

    qpos_list: list[np.ndarray] = []
    label = "ON" if contact_ground else "OFF"
    for frame in tqdm(frames, desc=f"Retarget (contact_ground={label})"):
        qpos_list.append(retargeter.retarget(frame))
    return np.asarray(qpos_list, dtype=np.float64)


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    if not _HAS_CV2:
        return img
    out = img.copy()
    cv2.rectangle(out, (8, 8), (8 + 12 * len(text) + 16, 44), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_qpos_video(
    qpos_seq: np.ndarray,
    *,
    robot: str,
    video_path: str,
    motion_fps: int,
    video_width: int,
    video_height: int,
    overlay_label: str | None = None,
) -> None:
    xml_path = str(ROBOT_XML_DICT[robot])
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=video_height, width=video_width)

    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = float(VIEWER_CAM_DISTANCE_DICT[robot])
    camera.elevation = -10.0
    camera.azimuth = 90.0

    base_id = model.body(ROBOT_BASE_DICT[robot]).id
    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    writer = imageio.get_writer(video_path, fps=motion_fps)
    try:
        for qpos in tqdm(qpos_seq, desc=f"Render {pathlib.Path(video_path).name}"):
            data.qpos[:3] = qpos[:3]
            data.qpos[3:7] = qpos[3:7]
            data.qpos[7:] = qpos[7:]
            mj.mj_forward(model, data)

            camera.lookat[:] = data.xpos[base_id]
            renderer.update_scene(data, camera=camera)
            img = renderer.render()
            if overlay_label:
                img = _draw_label(img, overlay_label)
            writer.append_data(img)
    finally:
        writer.close()
        renderer.close()

    print(f"[green]Saved video:[/green] {video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record two retargeting videos: contact_ground OFF vs ON.",
    )
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--motion_fps", default=30, type=int)
    parser.add_argument(
        "--output_dir",
        default="videos/contact_compare",
        type=str,
        help="Directory for output mp4 files.",
    )
    parser.add_argument("--video_width", default=1280, type=int)
    parser.add_argument("--video_height", default=720, type=int)
    parser.add_argument(
        "--frame_start",
        default=0,
        type=int,
        help="First frame index (inclusive).",
    )
    parser.add_argument(
        "--frame_end",
        default=-1,
        type=int,
        help="Last frame index (exclusive). -1 means full sequence.",
    )
    parser.add_argument(
        "--skip_retarget",
        action="store_true",
        help="Skip retargeting and only render from saved .npy qpos files.",
    )
    parser.add_argument(
        "--side_by_side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export one left-right comparison mp4 (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bvh_path = pathlib.Path(args.bvh_file)
    stem = bvh_path.stem
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    off_video = str(output_dir / f"{stem}_contact_off.mp4")
    on_video = str(output_dir / f"{stem}_contact_on.mp4")
    side_by_side_video = str(output_dir / f"{stem}_side_by_side.mp4")
    off_qpos_path = output_dir / f"{stem}_contact_off_qpos.npy"
    on_qpos_path = output_dir / f"{stem}_contact_on_qpos.npy"

    frames, actual_human_height = load_bvh_file(str(bvh_path), format=args.format)
    frame_end = len(frames) if args.frame_end < 0 else min(args.frame_end, len(frames))
    frame_start = max(0, min(args.frame_start, frame_end))
    frames = frames[frame_start:frame_end]
    print(
        f"BVH: {bvh_path} | frames [{frame_start}:{frame_end}] "
        f"({len(frames)} frames @ {args.motion_fps} fps)"
    )

    if args.skip_retarget:
        qpos_off = np.load(off_qpos_path)
        qpos_on = np.load(on_qpos_path)
    else:
        qpos_off = retarget_sequence(
            frames,
            robot=args.robot,
            fmt=args.format,
            contact_ground=False,
            motion_fps=args.motion_fps,
            actual_human_height=actual_human_height,
        )
        qpos_on = retarget_sequence(
            frames,
            robot=args.robot,
            fmt=args.format,
            contact_ground=True,
            motion_fps=args.motion_fps,
            actual_human_height=actual_human_height,
        )
        np.save(off_qpos_path, qpos_off)
        np.save(on_qpos_path, qpos_on)
        print(f"Saved qpos: {off_qpos_path}")
        print(f"Saved qpos: {on_qpos_path}")

    render_qpos_video(
        qpos_off,
        robot=args.robot,
        video_path=off_video,
        motion_fps=args.motion_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        overlay_label="contact_ground: OFF",
    )
    render_qpos_video(
        qpos_on,
        robot=args.robot,
        video_path=on_video,
        motion_fps=args.motion_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        overlay_label="contact_ground: ON",
    )

    if args.side_by_side:
        stitch_videos_side_by_side(off_video, on_video, side_by_side_video)
        print(f"[green]Saved side-by-side:[/green] {side_by_side_video}")

    print("\n[bold green]Done.[/bold green]")
    print(f"  OFF: {off_video}")
    print(f"  ON : {on_video}")
    if args.side_by_side:
        print(f"  SxS: {side_by_side_video}")


if __name__ == "__main__":
    main()
