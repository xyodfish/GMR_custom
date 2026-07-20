#!/usr/bin/env python3
"""Three-panel compare: GVHMR incam | retarget (no float fix) | retarget (float+pen fix).

Both retarget panels use Online QP + torque_limit (same settings as dual-viz).

Example
-------
python scripts/viz/make_ground_align_compare_video.py \\
  --clip_dir data/gvhmr_test_videos/ma_girl_run \\
  --output videos/ground_align_compare/ma_girl_run_3panel.mp4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

import imageio
import mujoco as mj
import numpy as np
from tqdm import tqdm

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import (  # noqa: E402
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    GeneralMotionRetargeting as GMR,
)
from general_motion_retargeting.human_frame_loaders import load_human_motion_frames  # noqa: E402
from general_motion_retargeting.online_qp_retarget import (  # noqa: E402
    OnlineQpConfig,
    OnlineQpRetargeter,
)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    if not _HAS_CV2:
        return img
    out = img.copy()
    # Black bar + white text (ASCII-safe).
    bar_h = 48
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _resolve_clip(clip_dir: pathlib.Path | None, input_file: pathlib.Path | None, incam: pathlib.Path | None):
    if clip_dir is not None:
        clip_dir = clip_dir.expanduser().resolve()
        pt = clip_dir / "hmr4d_results.pt"
        inc = clip_dir / "1_incam.mp4"
        if not pt.is_file():
            raise FileNotFoundError(f"Missing {pt}")
        if not inc.is_file():
            raise FileNotFoundError(f"Missing {inc}")
        return pt, inc, clip_dir.name

    if input_file is None:
        raise ValueError("Provide --clip_dir or --input_file")
    pt = input_file.expanduser().resolve()
    if incam is not None:
        inc = incam.expanduser().resolve()
    else:
        inc = pt.parent / "1_incam.mp4"
    if not inc.is_file():
        raise FileNotFoundError(f"Missing incam video: {inc} (pass --incam)")
    stem = pt.parent.name if pt.name == "hmr4d_results.pt" else pt.stem
    return pt, inc, stem


def retarget_torque_limit(
    frames: list[dict],
    *,
    fps: float,
    height: float,
    src: str,
    robot: str,
    preset: str,
    weight: float,
    margin: float,
    scope: str,
    gate_mode: str,
    contact_ground: bool,
    label: str,
) -> np.ndarray:
    gmr = GMR(
        src_human=src,
        tgt_robot=robot,
        verbose=False,
        contact_ground=contact_ground,
        actual_human_height=height,
        motion_fps=fps,
    )
    cfg = OnlineQpConfig.from_preset(preset)
    cfg.torque_limit_constraint = True
    cfg.torque_limit_weight = weight
    cfg.torque_limit_margin = margin
    cfg.torque_limit_scope = scope
    cfg.torque_limit_gate_mode = gate_mode
    ret = OnlineQpRetargeter(gmr, cfg)
    ret.set_motion_fps(fps)
    print(f"[3panel] retargeting {label} ({len(frames)} frames) ...")
    return ret.retarget_sequence(frames)


def render_qpos_video(
    qpos_seq: np.ndarray,
    *,
    robot: str,
    video_path: pathlib.Path,
    motion_fps: float,
    video_width: int,
    video_height: int,
    overlay_label: str,
) -> None:
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=video_height, width=video_width)

    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = float(VIEWER_CAM_DISTANCE_DICT[robot])
    camera.elevation = -10.0
    camera.azimuth = 90.0
    base_id = model.body(ROBOT_BASE_DICT[robot]).id

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=float(motion_fps))
    try:
        for qpos in tqdm(qpos_seq, desc=f"Render {video_path.name}"):
            data.qpos[:3] = qpos[:3]
            data.qpos[3:7] = qpos[3:7]
            data.qpos[7:] = qpos[7:]
            mj.mj_forward(model, data)
            camera.lookat[:] = data.xpos[base_id]
            renderer.update_scene(data, camera=camera)
            img = _draw_label(renderer.render(), overlay_label)
            writer.append_data(img)
    finally:
        writer.close()
        renderer.close()
    print(f"[3panel] wrote {video_path}")


def label_incam_video(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    label: str,
    max_frames: int | None,
    fps: float,
) -> None:
    """Copy incam frames with a top label bar (optionally trim length)."""
    reader = imageio.get_reader(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(dst), fps=float(fps))
    n = 0
    try:
        for frame in tqdm(reader, desc=f"Label {dst.name}"):
            if max_frames is not None and n >= max_frames:
                break
            writer.append_data(_draw_label(np.asarray(frame), label))
            n += 1
    finally:
        writer.close()
        reader.close()
    print(f"[3panel] wrote {dst} ({n} frames)")


def stitch_three(
    left: pathlib.Path,
    mid: pathlib.Path,
    right: pathlib.Path,
    output: pathlib.Path,
    *,
    height: int = 720,
    crf: int = 18,
) -> pathlib.Path:
    """Scale to common height and hstack three videos."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # Scale preserving aspect; force even dims for yuv420p.
    fc = (
        f"[0:v]scale=-2:{height},setsar=1[v0];"
        f"[1:v]scale=-2:{height},setsar=1[v1];"
        f"[2:v]scale=-2:{height},setsar=1[v2];"
        f"[v0][v1][v2]hstack=inputs=3[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(mid),
        "-i",
        str(right),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n" + (result.stderr or result.stdout or "unknown error")
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip_dir", type=str, default=None, help="GVHMR output folder")
    parser.add_argument("--input_file", type=str, default=None, help="hmr4d_results.pt")
    parser.add_argument("--incam", type=str, default=None, help="1_incam.mp4 path")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--preset", choices=["default", "smooth", "anti_slip"], default="anti_slip")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--torque_limit_weight", type=float, default=10.0)
    parser.add_argument("--torque_limit_margin", type=float, default=0.1)
    parser.add_argument("--torque_limit_scope", choices=["upper", "all"], default="upper")
    parser.add_argument("--torque_limit_gate_mode", choices=["off", "soft", "hard"], default="soft")
    parser.add_argument("--contact_ground", action="store_true", default=True)
    parser.add_argument("--no-contact_ground", dest="contact_ground", action="store_false")
    parser.add_argument("--video_width", type=int, default=960)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--panel_height", type=int, default=720, help="Stitched panel height")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Final 3-panel mp4 (default: videos/ground_align_compare/<stem>_3panel.mp4)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Intermediate clips dir (default: next to --output)",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--skip_retarget",
        action="store_true",
        help="Reuse existing panel mp4s in work_dir and only stitch",
    )
    args = parser.parse_args()

    pt, incam, stem = _resolve_clip(
        pathlib.Path(args.clip_dir) if args.clip_dir else None,
        pathlib.Path(args.input_file) if args.input_file else None,
        pathlib.Path(args.incam) if args.incam else None,
    )

    out = pathlib.Path(
        args.output or f"videos/ground_align_compare/{stem}_3panel.mp4"
    ).expanduser().resolve()
    work = pathlib.Path(args.work_dir or (out.parent / f"{stem}_panels")).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    v_incam = work / "01_gvhmr_incam.mp4"
    v_raw = work / "02_retarget_no_ground_align.mp4"
    v_fix = work / "03_retarget_ground_align.mp4"

    if not args.skip_retarget:
        frames_raw, fps, height, src = load_human_motion_frames(
            pt, max_frames=args.max_frames, ground_align=False
        )
        frames_fix, _, _, _ = load_human_motion_frames(
            pt,
            max_frames=args.max_frames,
            ground_align="lower_envelope",
            ground_align_verbose=True,
        )
        n = min(len(frames_raw), len(frames_fix))
        frames_raw, frames_fix = frames_raw[:n], frames_fix[:n]

        common = dict(
            fps=fps,
            height=height,
            src=src,
            robot=args.robot,
            preset=args.preset,
            weight=args.torque_limit_weight,
            margin=args.torque_limit_margin,
            scope=args.torque_limit_scope,
            gate_mode=args.torque_limit_gate_mode,
            contact_ground=args.contact_ground,
        )
        q_raw = retarget_torque_limit(frames_raw, label="no ground_align", **common)
        q_fix = retarget_torque_limit(frames_fix, label="ground_align", **common)

        label_incam_video(
            incam,
            v_incam,
            label="1  GVHMR (incam)",
            max_frames=n,
            fps=fps,
        )
        render_qpos_video(
            q_raw,
            robot=args.robot,
            video_path=v_raw,
            motion_fps=fps,
            video_width=args.video_width,
            video_height=args.video_height,
            overlay_label="2  Retarget (no float fix)",
        )
        render_qpos_video(
            q_fix,
            robot=args.robot,
            video_path=v_fix,
            motion_fps=fps,
            video_width=args.video_width,
            video_height=args.video_height,
            overlay_label="3  Retarget (float + pen fix)",
        )
    else:
        for p in (v_incam, v_raw, v_fix):
            if not p.is_file():
                raise FileNotFoundError(f"--skip_retarget but missing {p}")

    stitch_three(v_incam, v_raw, v_fix, out, height=args.panel_height, crf=args.crf)
    print(f"[3panel] DONE -> {out}")
    print(f"[3panel] panels in {work}")


if __name__ == "__main__":
    # Avoid MuJoCo trying to open a window in headless environments.
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
