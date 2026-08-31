#!/usr/bin/env python3
"""Side-by-side G1 vs another GMR robot in one MuJoCo window.

Example
-------
python scripts/viz/vis_g1_robot_compare.py \\
  --g1_motion ~/Workspace/puppet/output/gmr_references/source/unitree_g1/lafan1/walk1_subject2.qpos.json \\
  --robot_b unitree_h2 \\
  --robot_b_motion ~/Workspace/puppet/output/gmr_references/robot_b/unitree_h2/lafan1/walk1_subject2.qpos.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "python"))

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.dual_robot_viewer import TwoRobotMotionViewer
from general_motion_retargeting.params import ROBOT_XML_DICT
from robot_to_gmr.source_trajectory import SourceTrajectoryReader


def _qpos_fps(path: str) -> tuple[np.ndarray, float]:
    _meta, fps, _rp, _rr, _dof, *_rest, qpos = load_robot_motion(path)
    if qpos is None:
        raise ValueError(f"No qpos in {path}")

    return np.asarray(qpos, dtype=float), float(fps)


def _source_qpos_fps(path: str) -> tuple[np.ndarray, float]:
    if pathlib.Path(path).suffix.lower() in {".json", ".pkl"}:
        return _qpos_fps(path)

    reader = SourceTrajectoryReader(
        REPO / "config/retarget/source/unitree_g1_to_smplx_proxy.yaml",
        REPO,
    )
    trajectory = reader.load(pathlib.Path(path))
    return np.asarray(trajectory.qpos_frames, dtype=float), float(trajectory.fps)


def _read_stream_frame(
    process: subprocess.Popen[str], expected_index: int
) -> tuple[np.ndarray, int, np.ndarray | None]:
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        returncode = process.poll()
        raise RuntimeError(
            f"Realtime retargeter stopped before frame {expected_index} "
            f"(returncode={returncode})"
        )

    message = json.loads(line)
    frame_index = int(message["frame_index"])
    if frame_index != expected_index:
        raise RuntimeError(
            f"Realtime frame order mismatch: expected {expected_index}, got {frame_index}"
        )

    latency = int(message.get("pipeline_latency_frames", 0))
    initial = message.get("initial_qpos")
    initial_qpos = None if initial is None else np.asarray(initial, dtype=float)
    return np.asarray(message["qpos"], dtype=float), latency, initial_qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1_motion", required=True, help="G1 motion (.qpos.json / .pkl)")
    parser.add_argument("--robot_b", required=True, help="Target robot key in ROBOT_XML_DICT")
    parser.add_argument("--robot_b_motion", help="Target robot motion or live output path")
    parser.add_argument("--live_retarget", action="store_true")
    parser.add_argument("--online_canonical", action="store_true")
    parser.add_argument(
        "--realtime_cli",
        default=str(REPO / "cpp/build/gmr_realtime_robot_to_robot_cli"),
    )
    parser.add_argument("--dump_source_json")
    parser.add_argument("--offset_y", type=float, default=1.2)
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/g1_robot_compare.mp4")
    parser.add_argument("--no-tint", action="store_true")
    args = parser.parse_args()

    if args.robot_b not in ROBOT_XML_DICT:
        raise SystemExit(f"Unknown robot_b={args.robot_b!r}. Keys: {sorted(ROBOT_XML_DICT)}")

    if not args.robot_b_motion:
        raise SystemExit("--robot_b_motion is required")

    q_g1, fps_g1 = _source_qpos_fps(args.g1_motion)
    fps = fps_g1
    q_b = None
    if not args.live_retarget:
        q_b, fps_b = _qpos_fps(args.robot_b_motion)
        if abs(fps_g1 - fps_b) > 1e-3:
            print(f"[g1-robot] warning: fps mismatch g1={fps_g1} b={fps_b}, using g1")

        n = min(len(q_g1), len(q_b))
        q_g1, q_b = q_g1[:n], q_b[:n]
    else:
        n = len(q_g1)

    if args.live_retarget:
        mode = "live Online Canonical-QP" if args.online_canonical else "live Direct-QP"
    else:
        mode = "file playback"

    print(f"[g1-robot] {mode} frames={n} @ {fps:.0f}Hz | G1 | {args.robot_b} +Y={args.offset_y}")
    print("[g1-robot] Close the window to exit.")

    tint = None if args.no_tint else (0.35, 0.55, 0.95, 1.0)
    viewer = TwoRobotMotionViewer(
        "unitree_g1",
        args.robot_b,
        motion_fps=fps,
        offset_b=(0.0, float(args.offset_y), 0.0),
        tint_b=tint,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    process = None
    live_frames: list[np.ndarray] = []
    playback_latency = 0
    playback_initial: np.ndarray | None = None
    try:
        if args.live_retarget:
            command = [
                args.realtime_cli,
                "--gmr_root",
                str(REPO),
                "--input",
                args.g1_motion,
                "--robot_b",
                args.robot_b,
                "--out_json",
                args.robot_b_motion,
                "--stream_jsonl",
            ]
            if args.dump_source_json:
                command.extend(["--dump_source_json", args.dump_source_json])

            if args.online_canonical:
                command.append("--online_canonical")

            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            first_target, playback_latency, initial_target = _read_stream_frame(process, 0)
            if initial_target is None:
                initial_target = first_target

            playback_initial = initial_target
            print(
                f"[g1-robot] actual timeline delay={playback_latency} frames "
                f"({1000.0 * playback_latency / fps:.0f} ms)"
            )
            for source_idx in range(min(playback_latency, n)):
                if not viewer.viewer.is_running():
                    break

                viewer.step(
                    q_g1[source_idx],
                    initial_target,
                    rate_limit=True,
                    follow_camera=True,
                )

            for idx in range(n):
                if not viewer.viewer.is_running():
                    break

                if idx == 0:
                    q_target = first_target
                else:
                    q_target, latency, _initial = _read_stream_frame(process, idx)
                    if latency != playback_latency:
                        raise RuntimeError(
                            f"Realtime latency changed: {playback_latency} -> {latency}"
                        )

                live_frames.append(q_target)
                source_idx = min(idx + playback_latency, n - 1)
                viewer.step(
                    q_g1[source_idx],
                    q_target,
                    rate_limit=True,
                    follow_camera=True,
                )

            if len(live_frames) == n:
                returncode = process.wait(timeout=10)
                if returncode != 0:
                    raise RuntimeError(f"Realtime retargeter failed with returncode={returncode}")

            q_b = np.asarray(live_frames)
            if not args.loop:
                return

        idx = 0
        while viewer.viewer.is_running():
            assert q_b is not None
            n = min(len(q_g1), len(q_b))
            timeline_frames = n + playback_latency
            if idx >= timeline_frames:
                if args.loop:
                    idx = 0
                else:
                    break

            if idx < playback_latency:
                source_idx = idx
                target_idx = 0
                target_qpos = playback_initial if playback_initial is not None else q_b[target_idx]
            else:
                source_idx = min(idx, n - 1)
                target_idx = idx - playback_latency
                target_qpos = q_b[target_idx]

            viewer.step(
                q_g1[source_idx],
                target_qpos,
                rate_limit=True,
                follow_camera=True,
            )
            idx += 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

        viewer.close()


if __name__ == "__main__":
    main()
