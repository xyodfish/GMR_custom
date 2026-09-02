#!/usr/bin/env python3
"""Web studio for selecting SMPL-X NPZ files and rendering G1 versus H2.

The browser supports both the existing Batch/bridge comparison and a pure per-frame
GMR comparison that retargets SMPL-X independently to G1 and H2.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
RUN_BATCH = REPO / "scripts" / "tools" / "run_cpp_batch_to.py"
GMR_BATCH_CLI = REPO / "cpp" / "build" / "gmr_retarget_batch_cli"
VIEWER = REPO / "scripts" / "viz" / "vis_g1_robot_compare.py"
PKL_VIEWER = REPO / "scripts" / "viz" / "play_retarget_pkl.py"
ROBOT_CONVERTER = REPO / "cpp" / "build" / "gmr_robot_to_robot_cli"
DEFAULT_DATA = Path.home() / "Workspace" / "data"
DEFAULT_OUTPUT = REPO / "output" / "smplx_h2_gui"
DEFAULT_QUALITY_ROOT = Path.home() / "Workspace" / "gmr_cg_batch_h2"
DEFAULT_PYTHON = Path(os.environ.get("GMR_PYTHON", Path.home() / "miniconda3/envs/gmr/bin/python"))
HOST = "127.0.0.1"
PORT = 8778

_DATA_ROOT = DEFAULT_DATA
_OUTPUT_ROOT = DEFAULT_OUTPUT
_QUALITY_ROOT = DEFAULT_QUALITY_ROOT
_PYTHON = DEFAULT_PYTHON
_MOTIONS: list[Path] = []
_MOTION_SET: set[Path] = set()
_DATASET_COUNTS: dict[str, int] = {}
_QUALITY_BY_RELATIVE_INPUT: dict[str, dict] = {}
_QUALITY_EVENTS_OFFSET = 0
_PLAY_PROC: subprocess.Popen[str] | None = None
_PLAY_PAUSED = False
_PLAY_COMMAND_ID = 0
_CONVERT_LOCK = threading.Lock()
_PLAY_CONTROL_LOCK = threading.Lock()


def playback_control_path() -> Path:
    return _OUTPUT_ROOT / ".viewer_playback_command.json"


def playback_status_path() -> Path:
    return _OUTPUT_ROOT / ".viewer_playback_status.json"


def write_playback_command(seek_frame: int | None = None) -> int:
    global _PLAY_COMMAND_ID
    path = playback_control_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _PLAY_CONTROL_LOCK:
        _PLAY_COMMAND_ID += 1
        command = {
            "command_id": _PLAY_COMMAND_ID,
            "paused": _PLAY_PAUSED,
            "seek_frame": seek_frame,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(command), encoding="utf-8")
        temporary.replace(path)

    return _PLAY_COMMAND_ID


def read_playback_status() -> dict:
    try:
        payload = json.loads(playback_status_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"frame": 0, "total_frames": 0}

    return {
        "frame": max(0, int(payload.get("frame", 0))),
        "total_frames": max(0, int(payload.get("total_frames", 0))),
    }


def index_motions() -> None:
    global _DATASET_COUNTS, _MOTIONS, _MOTION_SET
    _MOTIONS = sorted(path.resolve() for path in _DATA_ROOT.rglob("*_stageii.npz") if path.is_file())
    _MOTION_SET = set(_MOTIONS)
    _DATASET_COUNTS = {}
    for path in _MOTIONS:
        relative = path.relative_to(_DATA_ROOT)
        dataset = relative.parts[0] if len(relative.parts) > 1 else "(root)"
        _DATASET_COUNTS[dataset] = _DATASET_COUNTS.get(dataset, 0) + 1


def index_quality() -> None:
    global _QUALITY_BY_RELATIVE_INPUT, _QUALITY_EVENTS_OFFSET
    records_root = _QUALITY_ROOT / "quality" / "records"
    records: dict[str, dict] = {}
    if records_root.is_dir():
        for path in records_root.rglob("*.quality.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            relative_input = str(record.get("relative_input", ""))
            if relative_input:
                records[relative_input] = record

    _QUALITY_BY_RELATIVE_INPUT = records
    event_path = _QUALITY_ROOT / "quality" / "events.jsonl"
    _QUALITY_EVENTS_OFFSET = event_path.stat().st_size if event_path.is_file() else 0


def refresh_quality_events() -> None:
    global _QUALITY_EVENTS_OFFSET
    event_path = _QUALITY_ROOT / "quality" / "events.jsonl"
    if not event_path.is_file():
        return

    size = event_path.stat().st_size
    if size < _QUALITY_EVENTS_OFFSET:
        index_quality()
        return

    if size == _QUALITY_EVENTS_OFFSET:
        return

    with event_path.open("rb") as stream:
        stream.seek(_QUALITY_EVENTS_OFFSET)
        chunk = stream.read()

    consumed = 0
    for line in chunk.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break

        consumed += len(line)
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

        relative_input = str(record.get("relative_input", ""))
        if relative_input:
            _QUALITY_BY_RELATIVE_INPUT[relative_input] = record

    _QUALITY_EVENTS_OFFSET += consumed


def quality_summary(record: dict | None) -> dict | None:
    if record is None:
        return None

    issues = record.get("issues", [])
    return {
        "status": record["status"],
        "anomaly_score": float(record.get("anomaly_score", 0.0)),
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "reject_count": sum(issue.get("severity") == "reject" for issue in issues),
        "issue_codes": [str(issue.get("code", "unknown")) for issue in issues],
    }


def quality_record(path: Path) -> dict | None:
    refresh_quality_events()
    return _QUALITY_BY_RELATIVE_INPUT.get(str(path.relative_to(_DATA_ROOT)))


def motion_item(path: Path) -> dict:
    relative = path.relative_to(_DATA_ROOT)
    parts = relative.parts
    dataset = parts[0] if len(parts) > 1 else ""
    subject = "/".join(parts[1:-1])
    return {
        "path": str(path),
        "relative": str(relative),
        "dataset": dataset,
        "subject": subject,
        "clip": path.stem,
        "label": f"{relative.with_suffix('')}",
        "quality": quality_summary(_QUALITY_BY_RELATIVE_INPUT.get(str(relative))),
    }


def search_motions(
    query: str,
    dataset: str,
    quality_filter: str,
    offset: int,
    limit: int,
) -> tuple[list[dict], int]:
    refresh_quality_events()
    tokens = query.lower().split()
    matches: list[dict] = []
    matched_total = 0
    for path in _MOTIONS:
        relative_path = path.relative_to(_DATA_ROOT)
        path_dataset = relative_path.parts[0] if len(relative_path.parts) > 1 else "(root)"
        if dataset and path_dataset != dataset:
            continue

        relative = str(relative_path)
        quality = _QUALITY_BY_RELATIVE_INPUT.get(relative)
        if quality_filter == "pending" and quality is not None:
            continue

        if quality_filter in ("accepted", "quarantine") and (
            quality is None or quality.get("status") != quality_filter
        ):
            continue

        if quality_filter == "warning" and (
            quality is None
            or not any(issue.get("severity") == "warning" for issue in quality.get("issues", []))
        ):
            continue

        text = relative.lower()
        if tokens and not all(token in text for token in tokens):
            continue

        if offset <= matched_total < offset + limit:
            matches.append(motion_item(path))

        matched_total += 1

    return matches, matched_total


def selected_motion(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path not in _MOTION_SET:
        raise ValueError("请选择索引中的 Stage-II NPZ。")

    return path


def output_paths(source: Path) -> tuple[Path, Path, Path]:
    relative = source.relative_to(_DATA_ROOT).with_suffix("")
    output_dir = _OUTPUT_ROOT / relative.parent
    return (
        output_dir / f"{relative.name}_g1_bridge.qpos.json",
        output_dir / f"{relative.name}_h2_smplx_bridge.qpos.json",
        output_dir / f"{relative.name}_h2_from_g1_gui.qpos.json",
    )


def pure_gmr_output_paths(source: Path) -> tuple[Path, Path]:
    relative = source.relative_to(_DATA_ROOT).with_suffix("")
    output_dir = _OUTPUT_ROOT / relative.parent
    return (
        output_dir / f"{relative.name}_g1_pure_gmr.qpos.json",
        output_dir / f"{relative.name}_h2_pure_gmr.qpos.json",
    )


def stop_play() -> dict:
    global _PLAY_PAUSED, _PLAY_PROC
    proc = _PLAY_PROC
    _PLAY_PROC = None
    _PLAY_PAUSED = False
    write_playback_command()
    if proc is None:
        return {"ok": True, "message": "当前没有渲染窗口。"}

    if proc.poll() is not None:
        return {"ok": True, "message": "渲染窗口已经结束。"}

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.terminate()

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.kill()

    return {"ok": True, "message": "渲染窗口已关闭。"}


def start_viewer(
    robot_a: str,
    motion_a: Path,
    robot_b: str,
    motion_b: Path,
    payload: dict,
) -> dict:
    global _PLAY_PAUSED, _PLAY_PROC
    stop_play()
    _PLAY_PAUSED = False
    write_playback_command()
    playback_status_path().write_text('{"frame": 0, "total_frames": 0}\n', encoding="utf-8")
    command = [
        str(_PYTHON),
        str(VIEWER),
        "--g1_motion",
        str(motion_a),
        "--robot_a",
        robot_a,
        "--robot_b",
        robot_b,
        "--robot_b_motion",
        str(motion_b),
        "--offset_y",
        str(float(payload.get("offset_y", 1.35))),
        "--playback_speed",
        str(float(payload.get("playback_speed", 1.0))),
        "--playback_control",
        str(playback_control_path()),
        "--playback_status",
        str(playback_status_path()),
    ]
    if not bool(payload.get("loop", True)):
        command.append("--no-loop")

    if not bool(payload.get("tint", True)):
        command.append("--no-tint")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    _PLAY_PROC = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"pid": _PLAY_PROC.pid, "viewer_command": command}


def quality_motion_path(record: dict) -> Path:
    relative_output = Path(str(record.get("relative_output", "")))
    if not relative_output.parts or relative_output.is_absolute():
        raise ValueError("质量记录缺少合法的 relative_output。")

    path = (_QUALITY_ROOT / relative_output).resolve()
    if not path.is_relative_to(_QUALITY_ROOT.resolve()) or not path.is_file():
        raise ValueError(f"质量 PKL 不存在：{path}")

    return path


def play_quality_motion(payload: dict) -> dict:
    global _PLAY_PAUSED, _PLAY_PROC
    source = selected_motion(str(payload.get("source", "")))
    record = quality_record(source)
    if record is None:
        return {"ok": False, "error": "该动作尚未完成批处理质量判定。"}

    if not PKL_VIEWER.is_file():
        raise RuntimeError(f"缺少 H2 PKL viewer：{PKL_VIEWER}")

    playback_speed = float(payload.get("playback_speed", 1.0))
    if playback_speed <= 0.0:
        raise ValueError("播放倍率必须大于零。")

    motion = quality_motion_path(record)
    stop_play()
    _PLAY_PAUSED = False
    write_playback_command()
    playback_status_path().write_text('{"frame": 0, "total_frames": 0}\n', encoding="utf-8")
    command = [
        str(_PYTHON),
        str(PKL_VIEWER),
        "--motion",
        str(motion),
        "--robot",
        "unitree_h2",
        "--playback_speed",
        str(playback_speed),
        "--playback_control",
        str(playback_control_path()),
        "--playback_status",
        str(playback_status_path()),
    ]
    if not bool(payload.get("loop", True)):
        command.append("--no_loop")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    _PLAY_PROC = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "ok": True,
        "source": str(source),
        "motion": str(motion),
        "quality": record,
        "pid": _PLAY_PROC.pid,
        "viewer_command": command,
    }


def toggle_pause() -> dict:
    global _PLAY_PAUSED
    proc = _PLAY_PROC
    if proc is None or proc.poll() is not None:
        _PLAY_PAUSED = False
        return {"ok": False, "paused": False, "error": "当前没有正在播放的渲染窗口。"}

    _PLAY_PAUSED = not _PLAY_PAUSED
    write_playback_command()
    return {
        "ok": True,
        "paused": _PLAY_PAUSED,
        "pid": proc.pid,
        "message": "播放已暂停。" if _PLAY_PAUSED else "播放已继续。",
    }


def seek_playback(payload: dict) -> dict:
    proc = _PLAY_PROC
    if proc is None or proc.poll() is not None:
        return {"ok": False, "error": "当前没有正在播放的渲染窗口。"}

    frame = int(payload.get("frame", 0))
    playback = read_playback_status()
    total_frames = playback["total_frames"]
    if total_frames <= 0:
        return {"ok": False, "error": "播放器尚未准备好轨迹。"}

    frame = min(max(0, frame), total_frames - 1)
    write_playback_command(seek_frame=frame)
    return {
        "ok": True,
        "paused": _PLAY_PAUSED,
        "frame": frame,
        "total_frames": total_frames,
        "message": f"已跳转到第 {frame + 1} / {total_frames} 帧。",
    }


def convert_and_play(payload: dict) -> dict:
    source = selected_motion(str(payload.get("source", "")))
    g1_path, h2_smplx_path, h2_from_g1_path = output_paths(source)
    max_frames = int(payload.get("max_frames", 0) or 0)
    if max_frames < 0:
        raise ValueError("最大帧数不能为负数。")

    if not RUN_BATCH.is_file() or not VIEWER.is_file() or not ROBOT_CONVERTER.is_file():
        raise RuntimeError("缺少 batch wrapper、G1→H2 converter 或双机器人 viewer。")

    if not _CONVERT_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "已有转换任务正在运行，请稍候。"}

    g1_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(_PYTHON),
        str(RUN_BATCH),
        "--input_file",
        str(source),
        "--input_type",
        "smplx",
        "--robot",
        "unitree_h2",
        "--out_json",
        str(h2_smplx_path),
        "--dump_g1_bridge_json",
        str(g1_path),
        "--contact_ground",
    ]
    command.append("--fast" if bool(payload.get("fast", True)) else "--quality")
    if max_frames > 0:
        command += ["--max_frames", str(max_frames)]

    try:
        batch_result = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        batch_log = (batch_result.stdout + "\n" + batch_result.stderr).strip()
        if batch_result.returncode != 0:
            return {"ok": False, "error": "SMPL-X→H2 转换失败。", "log": batch_log[-8000:]}

        gui_command = [
            str(ROBOT_CONVERTER),
            "--gmr_root",
            str(REPO),
            "--input",
            str(g1_path),
            "--robot_b",
            "unitree_h2",
            "--out_json",
            str(h2_from_g1_path),
            "--fast",
        ]
        gui_result = subprocess.run(
            gui_command,
            cwd=REPO,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        gui_log = (gui_result.stdout + "\n" + gui_result.stderr).strip()
        if gui_result.returncode != 0:
            return {"ok": False, "error": "G1→H2 GUI 基线转换失败。", "log": gui_log[-8000:]}
    finally:
        _CONVERT_LOCK.release()

    viewer = start_viewer("unitree_g1", g1_path, "unitree_h2", h2_smplx_path, payload)
    return {
        "ok": True,
        "source": str(source),
        "g1": str(g1_path),
        "h2_smplx_bridge": str(h2_smplx_path),
        "h2_from_g1_gui": str(h2_from_g1_path),
        "method": "smplx_via_internal_g1_bridge",
        "max_frames": max_frames,
        "batch_log": batch_log[-5000:],
        "g1_to_h2_log": gui_log[-5000:],
        **viewer,
    }


def pure_gmr_command(source: Path, robot: str, output: Path, max_frames: int) -> list[str]:
    command = [
        str(_PYTHON),
        str(RUN_BATCH),
        "--input_file",
        str(source),
        "--input_type",
        "smplx",
        "--robot",
        robot,
        "--out_json",
        str(output),
        "--cpp_cli",
        str(GMR_BATCH_CLI),
        "--backend",
        "mujoco_se3",
        "--no_contact_ground",
    ]
    if max_frames > 0:
        command += ["--max_frames", str(max_frames)]

    return command


def convert_pure_gmr_and_play(payload: dict) -> dict:
    source = selected_motion(str(payload.get("source", "")))
    g1_path, h2_path = pure_gmr_output_paths(source)
    max_frames = int(payload.get("max_frames", 0) or 0)
    if max_frames < 0:
        raise ValueError("最大帧数不能为负数。")

    if not RUN_BATCH.is_file() or not GMR_BATCH_CLI.is_file() or not VIEWER.is_file():
        raise RuntimeError("缺少 GMR wrapper、逐帧 GMR CLI 或双机器人 viewer。")

    if not _CONVERT_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "已有转换任务正在运行，请稍候。"}

    g1_path.parent.mkdir(parents=True, exist_ok=True)
    logs: dict[str, str] = {}
    try:
        for robot, output in (("unitree_g1", g1_path), ("unitree_h2", h2_path)):
            result = subprocess.run(
                pure_gmr_command(source, robot, output, max_frames),
                cwd=REPO,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                check=False,
            )
            log = (result.stdout + "\n" + result.stderr).strip()
            logs[robot] = log[-5000:]
            if result.returncode != 0:
                return {
                    "ok": False,
                    "error": f"SMPL-X→{robot} 纯 GMR 转换失败。",
                    "log": log[-8000:],
                }
    finally:
        _CONVERT_LOCK.release()

    viewer = start_viewer("unitree_g1", g1_path, "unitree_h2", h2_path, payload)
    return {
        "ok": True,
        "source": str(source),
        "g1": str(g1_path),
        "h2": str(h2_path),
        "method": "pure_gmr_smplx_direct_g1_and_h2",
        "contact_ground": False,
        "max_frames": max_frames,
        "g1_log": logs["unitree_g1"],
        "h2_log": logs["unitree_h2"],
        **viewer,
    }


def play_existing_pure_gmr(payload: dict) -> dict:
    source = selected_motion(str(payload.get("source", "")))
    g1_path, h2_path = pure_gmr_output_paths(source)
    if not g1_path.is_file() or not h2_path.is_file():
        return {"ok": False, "error": "该 NPZ 尚无纯 GMR 缓存，请先生成纯 GMR G1 + H2。"}

    viewer = start_viewer("unitree_g1", g1_path, "unitree_h2", h2_path, payload)
    return {
        "ok": True,
        "source": str(source),
        "g1": str(g1_path),
        "h2": str(h2_path),
        "method": "pure_gmr_smplx_direct_g1_and_h2",
        **viewer,
    }


def play_existing(payload: dict) -> dict:
    source = selected_motion(str(payload.get("source", "")))
    g1_path, h2_smplx_path, h2_from_g1_path = output_paths(source)
    if not g1_path.is_file() or not h2_smplx_path.is_file() or not h2_from_g1_path.is_file():
        return {"ok": False, "error": "该 NPZ 尚无缓存结果，请先转换并播放。"}

    mode = str(payload.get("render_mode", "g1_h2"))
    if mode == "g1_h2":
        viewer = start_viewer("unitree_g1", g1_path, "unitree_h2", h2_smplx_path, payload)
    elif mode == "h2_h2":
        viewer = start_viewer("unitree_h2", h2_smplx_path, "unitree_h2", h2_from_g1_path, payload)
    else:
        raise ValueError("render_mode 必须是 g1_h2 或 h2_h2。")

    return {
        "ok": True,
        "source": str(source),
        "g1": str(g1_path),
        "h2_smplx_bridge": str(h2_smplx_path),
        "h2_from_g1_gui": str(h2_from_g1_path),
        "render_mode": mode,
        **viewer,
    }


def status() -> dict:
    proc = _PLAY_PROC
    if proc is None:
        return {"running": False, "converting": _CONVERT_LOCK.locked(), "log": ""}

    alive = proc.poll() is None
    log = ""
    if not alive and proc.stdout is not None:
        try:
            log = proc.stdout.read() or ""
        except OSError:
            log = ""

    playback = read_playback_status() if alive else {"frame": 0, "total_frames": 0}
    return {
        "running": alive,
        "paused": _PLAY_PAUSED if alive else False,
        "converting": _CONVERT_LOCK.locked(),
        "pid": proc.pid if alive else None,
        "returncode": None if alive else proc.returncode,
        "log": log[-5000:],
        **playback,
    }


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SMPL-X → H2 Studio</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; background: #101114; color: #eee; }
  header { padding: 18px 22px 9px; border-bottom: 1px solid #252832; }
  h1 { margin: 0; font-size: 21px; }
  .sub { color: #9da7b8; margin-top: 4px; }
  main { display: grid; grid-template-columns: minmax(360px, 430px) 1fr; gap: 16px; padding: 16px 22px 24px; }
  label { display: block; font-size: 12px; color: #a6afbf; margin: 13px 0 5px; }
  select, input, button { width: 100%; font: inherit; color: inherit; background: #191c23; border: 1px solid #343946; border-radius: 8px; padding: 9px 10px; }
  select { height: 360px; }
  select.compact { height: auto; }
  button { cursor: pointer; background: #265b98; border-color: #3978bd; margin-top: 9px; }
  button.secondary { background: #242833; border-color: #414756; }
  button:disabled { cursor: wait; opacity: .55; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
  .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 10px; }
  .checks label { display: flex; align-items: center; gap: 7px; margin: 0; color: #ddd; font-size: 13px; }
  .checks input { width: auto; }
  .progress-head { display: flex; justify-content: space-between; align-items: baseline; }
  .progress-head output { color: #91c9ff; font-size: 12px; }
  input[type="range"] { padding: 6px 0; accent-color: #4f9ee8; }
  .meta { min-height: 3em; margin-top: 7px; color: #91c9ff; font-size: 12px; word-break: break-all; }
  .hint { min-height: 2.8em; margin-top: 9px; color: #9bd49b; font-size: 12px; }
  .right { display: grid; grid-template-rows: minmax(260px, 1fr) minmax(180px, .65fr); gap: 10px; min-height: calc(100vh - 112px); }
  pre { margin: 0; overflow: auto; background: #090a0d; border: 1px solid #292d37; border-radius: 10px; padding: 13px; white-space: pre-wrap; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .badge { color: #a9d4ff; background: #18304a; border-radius: 999px; padding: 2px 9px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>SMPL-X → H2 Studio <span class="badge">Batch / Pure GMR</span></h1>
  <div class="sub">既可比较 Batch/bridge 路径，也可将同一 NPZ 通过纯逐帧 GMR 分别直接重定向到 G1 和 H2。</div>
</header>
<main>
  <aside>
    <label>数据集</label>
    <select class="compact" id="dataset"><option value="">全部数据集</option></select>
    <label>质量状态</label>
    <select class="compact" id="qualityFilter">
      <option value="">全部状态</option>
      <option value="quarantine">仅隔离</option>
      <option value="accepted">仅通过</option>
      <option value="warning">含警告</option>
      <option value="pending">尚未处理</option>
    </select>
    <label>搜索 13k+ NPZ（空格分隔多个关键词）</label>
    <input id="query" placeholder="例如 ACCAD Male2Running C3 run" autofocus />
    <label>匹配动作（每页 200 条）</label>
    <select id="motion" size="16"></select>
    <div class="row">
      <button class="secondary" id="previousPage">上一页</button>
      <button class="secondary" id="nextPage">下一页</button>
    </div>
    <div class="meta" id="meta">请选择动作…</div>
    <div class="row">
      <div><label>最大帧数（0=整段）</label><input id="maxFrames" type="number" min="0" step="1" value="180" /></div>
      <div><label>H2 横向间距</label><input id="offsetY" type="number" min="0" max="5" step="0.1" value="1.35" /></div>
    </div>
    <label>播放倍率（慢放 / 正常 / 快放）</label>
    <select id="playbackSpeed" style="height:auto">
      <option value="0.1">0.1×</option>
      <option value="0.25">0.25×</option>
      <option value="0.5">0.5×</option>
      <option value="0.75">0.75×</option>
      <option value="1" selected>1×</option>
      <option value="1.5">1.5×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
    <div class="checks">
      <label><input id="fast" type="checkbox" checked /> Fast Batch</label>
      <label><input id="loop" type="checkbox" checked /> 循环播放</label>
      <label><input id="tint" type="checkbox" checked /> H2 蓝色 Tint</label>
    </div>
    <div class="progress-head"><label for="progress">播放进度</label><output id="progressText">0 / 0</output></div>
    <input id="progress" type="range" min="0" max="0" step="1" value="0" disabled />
    <button id="convert">生成三条轨迹并播放 G1 ↔ H2</button>
    <button id="convertPureGmr">纯 GMR：生成 G1 + H2 并播放</button>
    <button class="secondary" id="playPureGmr">纯 GMR：播放缓存 G1 ↔ H2</button>
    <button class="secondary" id="playQuality">直接播放批处理 H2（通过/隔离均可）</button>
    <button class="secondary" id="playG1H2">渲染一：G1 ↔ SMPL-X→H2</button>
    <button class="secondary" id="playH2H2">渲染二：SMPL-X→H2 ↔ G1→H2</button>
    <button class="secondary" id="pause">暂停播放</button>
    <button class="secondary" id="stop">关闭窗口</button>
    <button class="secondary" id="refresh">重新扫描 NPZ</button>
    <div class="hint" id="hint"></div>
  </aside>
  <section class="right">
    <pre id="qualityOut">质量统计：请选择动作…</pre>
    <pre id="out">正在加载动作索引…</pre>
  </section>
</main>
<script>
let items = [];
let searchTimer = null;
let wasRunning = false;
let draggingProgress = false;
let resultOffset = 0;
let matchedTotal = 0;
const pageSize = 200;

function esc(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }

  return payload;
}

function selected() {
  return document.getElementById('motion').value;
}

function payload(renderMode = 'g1_h2') {
  return {
    source: selected(),
    max_frames: Number(document.getElementById('maxFrames').value || 0),
    offset_y: Number(document.getElementById('offsetY').value || 1.35),
    playback_speed: Number(document.getElementById('playbackSpeed').value || 1),
    fast: document.getElementById('fast').checked,
    loop: document.getElementById('loop').checked,
    tint: document.getElementById('tint').checked,
    render_mode: renderMode,
  };
}

function writeOut(value) {
  document.getElementById('out').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function qualityLabel(quality) {
  if (!quality) {
    return '[未处理]';
  }

  const status = quality.status === 'quarantine' ? '隔离' : '通过';
  return `[${status} ${quality.anomaly_score.toFixed(0)}]`;
}

async function updateMeta() {
  const item = items.find(candidate => candidate.path === selected());
  document.getElementById('meta').textContent = item
    ? `${qualityLabel(item.quality)} ${item.dataset} / ${item.subject}\n${item.clip}\n${item.path}`
    : '请选择动作…';
  if (!item) {
    document.getElementById('qualityOut').textContent = '质量统计：请选择动作…';
    return;
  }

  const data = await api(`/api/quality?source=${encodeURIComponent(item.path)}`);
  if (!data.record) {
    document.getElementById('qualityOut').textContent = '质量状态：尚未处理\n该动作还没有批处理 PKL 和质量报告。';
    return;
  }

  const record = data.record;
  const issues = record.issues.length
    ? record.issues.map(issue => ({
        severity: issue.severity,
        code: issue.code,
        frame: issue.frame ?? null,
        value: issue.value,
        threshold: issue.threshold,
        reason: issue.message,
      }))
    : ['无异常'];
  document.getElementById('qualityOut').textContent = JSON.stringify({
    status: record.status,
    anomaly_score: record.anomaly_score,
    output_file: record.output_file,
    issues,
    metrics: record.metrics,
  }, null, 2);
}

async function search(resetPage = false) {
  if (resetPage) {
    resultOffset = 0;
  }

  const query = encodeURIComponent(document.getElementById('query').value.trim());
  const dataset = encodeURIComponent(document.getElementById('dataset').value);
  const qualityFilter = encodeURIComponent(document.getElementById('qualityFilter').value);
  const data = await api(`/api/motions?q=${query}&dataset=${dataset}&quality=${qualityFilter}&offset=${resultOffset}&limit=${pageSize}`);
  items = data.items;
  matchedTotal = data.matched_total;
  const datasetSelect = document.getElementById('dataset');
  if (datasetSelect.options.length === 1) {
    datasetSelect.innerHTML = '<option value="">全部数据集</option>' + data.datasets
      .map(item => `<option value="${esc(item.name)}">${esc(item.name)} (${item.count})</option>`)
      .join('');
  }

  const select = document.getElementById('motion');
  const previous = select.value;
  select.innerHTML = items
    .map(item => `<option value="${esc(item.path)}">${esc(qualityLabel(item.quality))} ${esc(item.label)}</option>`)
    .join('');
  if (items.some(item => item.path === previous)) {
    select.value = previous;
  }

  await updateMeta();
  const first = items.length ? resultOffset + 1 : 0;
  const last = resultOffset + items.length;
  document.getElementById('previousPage').disabled = resultOffset === 0;
  document.getElementById('nextPage').disabled = last >= matchedTotal;
  document.getElementById('hint').textContent = `总索引 ${data.total} 条；匹配 ${matchedTotal} 条，显示 ${first}-${last}。`;
  writeOut({
    data_root: data.data_root,
    output_root: data.output_root,
    indexed: data.total,
    matched: matchedTotal,
    shown: items.length,
    offset: resultOffset,
  });
}

async function post(path, renderMode = 'g1_h2') {
  if (!selected()) {
    document.getElementById('hint').textContent = '请先选择一个 NPZ。';
    return;
  }

  const data = await api(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload(renderMode)),
  });
  writeOut(data);
  document.getElementById('hint').textContent = data.ok
    ? (renderMode === 'h2_h2'
      ? 'H2↔H2 窗口已启动：左侧 SMPL-X→H2，右侧独立 G1→H2。'
      : 'G1↔H2 窗口已启动：左侧 G1，右侧新 SMPL-X→H2。')
    : (data.error || '操作失败。');
}

async function convert() {
  const button = document.getElementById('convert');
  button.disabled = true;
  button.textContent = '转换中，请稍候…';
  document.getElementById('hint').textContent = '正在生成 G1、H2 和独立 G1→H2 三条轨迹…';
  try {
    await post('/api/convert');
  } catch (error) {
    document.getElementById('hint').textContent = `转换失败：${error.message}`;
    writeOut({ok: false, error: error.message});
  } finally {
    button.disabled = false;
    button.textContent = '生成三条轨迹并播放 G1 ↔ H2';
  }
}

async function convertPureGmr() {
  const button = document.getElementById('convertPureGmr');
  button.disabled = true;
  button.textContent = '纯 GMR 转换中，请稍候…';
  document.getElementById('hint').textContent = '正在分别执行 SMPL-X→G1 和 SMPL-X→H2 逐帧 GMR（无 Batch TO、无 G1 bridge）…';
  try {
    const data = await api('/api/convert-pure-gmr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload()),
    });
    writeOut(data);
    document.getElementById('hint').textContent = data.ok
      ? '纯 GMR G1↔H2 窗口已启动：两侧均由同一 SMPL-X 独立直接重定向。'
      : (data.error || '纯 GMR 转换失败。');
  } catch (error) {
    document.getElementById('hint').textContent = `纯 GMR 转换失败：${error.message}`;
    writeOut({ok: false, error: error.message});
  } finally {
    button.disabled = false;
    button.textContent = '纯 GMR：生成 G1 + H2 并播放';
  }
}

async function playPureGmr() {
  const data = await api('/api/play-pure-gmr', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload()),
  });
  writeOut(data);
  document.getElementById('hint').textContent = data.ok
    ? '正在播放纯 GMR：左侧 G1，右侧直接 H2。'
    : (data.error || '播放失败。');
}

async function playQualityMotion() {
  if (!selected()) {
    document.getElementById('hint').textContent = '请先选择一个已处理动作。';
    return;
  }

  const data = await api('/api/play-quality', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload()),
  });
  writeOut(data);
  document.getElementById('hint').textContent = data.ok
    ? `正在播放批处理 H2，质量状态：${data.quality.status}。隔离状态未改变。`
    : (data.error || '播放失败。');
}

async function stopViewer() {
  const data = await api('/api/stop', {method: 'POST'});
  writeOut(data);
  document.getElementById('hint').textContent = data.message || '渲染窗口已关闭。';
}

async function togglePause() {
  const data = await api('/api/toggle-pause', {method: 'POST'});
  writeOut(data);
  document.getElementById('pause').textContent = data.paused ? '继续播放' : '暂停播放';
  document.getElementById('hint').textContent = data.message || data.error;
}

function updateProgressText(frame, totalFrames) {
  document.getElementById('progressText').textContent = totalFrames > 0
    ? `${frame + 1} / ${totalFrames}`
    : '0 / 0';
}

async function seekProgress() {
  const progress = document.getElementById('progress');
  const data = await api('/api/seek', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame: Number(progress.value)}),
  });
  draggingProgress = false;
  writeOut(data);
  updateProgressText(data.frame, data.total_frames);
  document.getElementById('hint').textContent = data.message || data.error;
}

async function refreshIndex() {
  const data = await api('/api/refresh', {method: 'POST'});
  document.getElementById('dataset').innerHTML = '<option value="">全部数据集</option>';
  await search(true);
  document.getElementById('hint').textContent = `重新扫描完成，共 ${data.total} 条 Stage-II NPZ。`;
}

async function pollStatus() {
  const data = await api('/api/status');
  if (data.running) {
    wasRunning = true;
    const progress = document.getElementById('progress');
    if (data.total_frames > 0) {
      progress.disabled = false;
      progress.max = String(data.total_frames - 1);
      if (!draggingProgress) {
        progress.value = String(data.frame);
        updateProgressText(data.frame, data.total_frames);
      }

    } else {
      progress.disabled = true;
    }

    document.getElementById('pause').textContent = data.paused ? '继续播放' : '暂停播放';
    document.getElementById('hint').textContent = data.paused
      ? `MuJoCo 已暂停（PID ${data.pid}）。`
      : `MuJoCo 正在实时渲染（PID ${data.pid}）。`;
  } else if (wasRunning) {
    wasRunning = false;
    document.getElementById('pause').textContent = '暂停播放';
    document.getElementById('progress').disabled = true;
    updateProgressText(0, 0);
    document.getElementById('hint').textContent = '渲染窗口已结束，可以选择下一条动作。';
    if (data.log) {
      writeOut(data);
    }

  }

}

document.getElementById('query').oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => search(true).catch(error => writeOut({error: error.message})), 250);
};
document.getElementById('dataset').onchange = () => search(true).catch(error => writeOut({error: error.message}));
document.getElementById('qualityFilter').onchange = () => search(true).catch(error => writeOut({error: error.message}));
document.getElementById('motion').onchange = () => updateMeta().catch(error => writeOut({error: error.message}));
document.getElementById('previousPage').onclick = () => {
  resultOffset = Math.max(0, resultOffset - pageSize);
  search().catch(error => writeOut({error: error.message}));
};
document.getElementById('nextPage').onclick = () => {
  if (resultOffset + items.length < matchedTotal) {
    resultOffset += pageSize;
    search().catch(error => writeOut({error: error.message}));
  }
};
document.getElementById('convert').onclick = convert;
document.getElementById('convertPureGmr').onclick = convertPureGmr;
document.getElementById('playPureGmr').onclick = () => playPureGmr().catch(error => {
  document.getElementById('hint').textContent = `播放失败：${error.message}`;
  writeOut({error: error.message});
});
document.getElementById('playQuality').onclick = () => playQualityMotion().catch(error => {
  document.getElementById('hint').textContent = `播放失败：${error.message}`;
  writeOut({error: error.message});
});
document.getElementById('playG1H2').onclick = () => post('/api/play', 'g1_h2').catch(error => writeOut({error: error.message}));
document.getElementById('playH2H2').onclick = () => post('/api/play', 'h2_h2').catch(error => writeOut({error: error.message}));
document.getElementById('pause').onclick = () => togglePause().catch(error => writeOut({error: error.message}));
document.getElementById('progress').onpointerdown = () => { draggingProgress = true; };
document.getElementById('progress').oninput = event => {
  draggingProgress = true;
  updateProgressText(Number(event.target.value), Number(event.target.max) + 1);
};
document.getElementById('progress').onchange = () => seekProgress().catch(error => {
  draggingProgress = false;
  writeOut({error: error.message});
});
document.getElementById('stop').onclick = () => stopViewer().catch(error => writeOut({error: error.message}));
document.getElementById('refresh').onclick = () => refreshIndex().catch(error => writeOut({error: error.message}));

search(true).catch(error => writeOut({error: error.message}));
setInterval(() => pollStatus().catch(() => {}), 500);
</script>
</body>
</html>
"""


class StudioServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    def send_json(self, payload: dict, code: int = 200) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def send_html(self) -> None:
        blob = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_html()
            return

        if parsed.path == "/api/motions":
            query = parse_qs(parsed.query)
            text = query.get("q", [""])[0]
            dataset = query.get("dataset", [""])[0]
            quality_filter = query.get("quality", [""])[0]
            if dataset and dataset not in _DATASET_COUNTS:
                self.send_json({"ok": False, "error": f"未知数据集：{dataset}"}, 400)
                return

            if quality_filter not in ("", "accepted", "quarantine", "warning", "pending"):
                self.send_json({"ok": False, "error": f"未知质量状态：{quality_filter}"}, 400)
                return

            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(500, max(1, int(query.get("limit", ["200"])[0])))
            except ValueError:
                self.send_json({"ok": False, "error": "offset 和 limit 必须是整数。"}, 400)
                return

            items, matched_total = search_motions(text, dataset, quality_filter, offset, limit)
            self.send_json({
                "items": items,
                "total": len(_MOTIONS),
                "matched_total": matched_total,
                "offset": offset,
                "datasets": [
                    {"name": name, "count": count}
                    for name, count in sorted(_DATASET_COUNTS.items())
                ],
                "data_root": str(_DATA_ROOT),
                "output_root": str(_OUTPUT_ROOT),
            })
            return

        if parsed.path == "/api/quality":
            query = parse_qs(parsed.query)
            try:
                source = selected_motion(query.get("source", [""])[0])
            except ValueError as error:
                self.send_json({"ok": False, "error": str(error)}, 400)
                return

            self.send_json({"ok": True, "record": quality_record(source)})
            return

        if parsed.path == "/api/status":
            self.send_json(status())
            return

        self.send_json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/convert":
                self.send_json(convert_and_play(self.read_json()))
                return

            if parsed.path == "/api/convert-pure-gmr":
                self.send_json(convert_pure_gmr_and_play(self.read_json()))
                return

            if parsed.path == "/api/play":
                self.send_json(play_existing(self.read_json()))
                return

            if parsed.path == "/api/play-pure-gmr":
                self.send_json(play_existing_pure_gmr(self.read_json()))
                return

            if parsed.path == "/api/play-quality":
                self.send_json(play_quality_motion(self.read_json()))
                return

            if parsed.path == "/api/stop":
                self.send_json(stop_play())
                return

            if parsed.path == "/api/toggle-pause":
                self.send_json(toggle_pause())
                return

            if parsed.path == "/api/seek":
                self.send_json(seek_playback(self.read_json()))
                return

            if parsed.path == "/api/refresh":
                index_motions()
                index_quality()
                self.send_json({"ok": True, "total": len(_MOTIONS)})
                return

            self.send_json({"ok": False, "error": "not found"}, 404)
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            self.send_json({"ok": False, "error": str(error)}, 400)


def main() -> None:
    global _DATA_ROOT, _OUTPUT_ROOT, _PYTHON, _QUALITY_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    _DATA_ROOT = args.data_root.expanduser().resolve()
    _OUTPUT_ROOT = args.output_root.expanduser().resolve()
    _QUALITY_ROOT = args.quality_root.expanduser().resolve()
    _PYTHON = args.python.expanduser().resolve()
    if not _DATA_ROOT.is_dir():
        raise SystemExit(f"Data root does not exist: {_DATA_ROOT}")

    if not _PYTHON.is_file():
        raise SystemExit(f"Python executable does not exist: {_PYTHON}")

    index_motions()
    index_quality()
    server = StudioServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[smplx-h2-studio] indexed {len(_MOTIONS)} motions under {_DATA_ROOT}")
    print(f"[smplx-h2-studio] outputs: {_OUTPUT_ROOT}")
    print(f"[smplx-h2-studio] quality: {_QUALITY_ROOT} ({len(_QUALITY_BY_RELATIVE_INPUT)} reports)")
    print(f"[smplx-h2-studio] open {url}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[smplx-h2-studio] bye")
    finally:
        stop_play()
        server.server_close()


if __name__ == "__main__":
    main()
