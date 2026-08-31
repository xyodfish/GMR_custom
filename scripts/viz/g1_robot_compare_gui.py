#!/usr/bin/env python3
"""G1 ↔ Robot-B compare studio (GMT Studio–style local web UI).

  cd ~/Workspace/open_source_code/GMR_custom
  python scripts/viz/g1_robot_compare_gui.py

  # or from puppet
  python3 app/python/g1_robot_compare_gui.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "python"))

from general_motion_retargeting.data_loader import load_robot_motion
from robot_to_gmr import (
    flatten_stance_feet_ik,
    infer_foot_contacts_from_soles,
    snap_robot_qpos_to_ground,
)

VIZ_SCRIPT = Path(__file__).resolve().parent / "vis_g1_robot_compare.py"
CPP_CONVERTER = REPO / "cpp" / "build" / "gmr_robot_to_robot_cli"
REALTIME_CPP_CONVERTER = REPO / "cpp" / "build" / "gmr_realtime_robot_to_robot_cli"
DEFAULT_REFS = Path.home() / "Workspace" / "puppet" / "output" / "gmr_references"
GUI_REFS = REPO / "output" / "robot_to_robot_gui"
GMR_CG_BATCH = Path.home() / "Workspace" / "gmr_cg_batch"
G1_MODEL = REPO / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HOST = "127.0.0.1"
PORT = 8777

_QPOS_SUFFIXES = (
    ".post_minimal.qpos.json",
    ".post_none.qpos.json",
    ".post_full.qpos.json",
    ".raw.qpos.json",
    ".qpos.json",
)

_PLAY_PROC: subprocess.Popen | None = None
_CONVERT_LOCK = threading.Lock()
_REFS = DEFAULT_REFS
_ROBOT_REFS = [DEFAULT_REFS]


def _split_clip_name(path: Path) -> tuple[str, str]:
    name = path.name
    for suffix in _QPOS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix

    return path.stem, ""


def _kind_label(suffix: str) -> str:
    return {
        ".qpos.json": "minimal",
        ".raw.qpos.json": "raw/none",
        ".post_minimal.qpos.json": "post_minimal",
        ".post_none.qpos.json": "post_none",
        ".post_full.qpos.json": "post_full",
    }.get(suffix, suffix.strip(".") or "qpos")


def _motion_meta(path: Path, refs: Path, robot: str) -> dict:
    clip, suffix = _split_clip_name(path)
    try:
        rel = path.relative_to(refs / "robot_b" / robot)
        dataset = rel.parts[0] if len(rel.parts) > 1 else path.parent.name
    except ValueError:
        dataset = path.parent.name

    kind = _kind_label(suffix)
    return {
        "id": f"{refs.name}/{dataset}/{clip}:{kind}",
        "path": str(path),
        "clip": clip,
        "dataset": dataset,
        "collection": refs.name,
        "kind": kind,
        "label": f"{refs.name} · {dataset} · {clip} · {kind}",
    }


def discover_robots(refs_roots: list[Path]) -> list[str]:
    names: set[str] = set()
    for refs in refs_roots:
        root = refs / "robot_b"
        if not root.is_dir():
            continue

        for path in root.iterdir():
            if path.is_dir() and any(path.rglob("*.qpos.json")):
                names.add(path.name)

    return sorted(names)


def supported_target_robots() -> list[str]:
    header = REPO / "cpp" / "include" / "gmr" / "retarget" / "repo_paths.h"
    text = header.read_text(encoding="utf-8")
    xml_names = set(re.findall(r'\{"([a-z0-9_]+)", "assets/', text))
    smplx_names = set(re.findall(r'\{"([a-z0-9_]+)", "general_motion_retargeting/ik_configs/smplx_', text))
    return sorted(xml_names & smplx_names)


def discover_source_inputs(refs_roots: list[Path]) -> list[dict]:
    candidates: list[Path] = []
    for refs in refs_roots:
        source = refs / "source" / "unitree_g1"
        if source.is_dir():
            candidates.extend(source.rglob("*.qpos.json"))

    csv_root = REPO / "output" / "csv"
    if csv_root.is_dir():
        candidates.extend(csv_root.glob("*.csv"))

    if GMR_CG_BATCH.is_dir():
        candidates.extend(GMR_CG_BATCH.rglob("*.pkl"))

    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    out = []
    for path in unique:
        kind = {".csv": "CSV", ".pkl": "GMR PKL"}.get(path.suffix.lower(), "qpos JSON")
        label = _split_clip_name(path)[0]
        if path.suffix.lower() == ".pkl":
            label = str(path.relative_to(GMR_CG_BATCH).with_suffix(""))

        out.append({"path": str(path), "clip": _split_clip_name(path)[0], "kind": kind, "label": f"{label} · {kind}"})

    return out


def prepare_cpp_input(source: Path, output: Path) -> Path:
    if source.suffix.lower() != ".pkl":
        return source

    _meta, fps, root_pos, root_rot, dof_pos, *_rest, qpos = load_robot_motion(source)
    if qpos is None:
        qpos = np.hstack([root_pos, root_rot, dof_pos])

    qpos = np.asarray(qpos, dtype=float)
    contacts = infer_foot_contacts_from_soles(qpos, str(G1_MODEL), fps=float(fps))
    qpos = flatten_stance_feet_ik(qpos, str(G1_MODEL), contacts)
    qpos = snap_robot_qpos_to_ground(qpos, str(G1_MODEL), contacts=contacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "robot": "unitree_g1",
                "fps": float(fps),
                "nq": int(qpos.shape[1]),
                "num_frames": int(qpos.shape[0]),
                "qpos_frames": qpos.tolist(),
                "source_postprocess": "contact_aware_foot_comfort",
            }
        ),
        encoding="utf-8",
    )
    return output


def discover_motions(refs_roots: list[Path], robot: str, *, main_only: bool = True) -> list[dict]:
    """List robot-B motions.

    ``main_only`` keeps ``<clip>.qpos.json`` (pipeline default / minimal) and drops
    ``.raw`` / ``.post_*`` / experimental ``*_wrist_*`` sidecars.
    """
    rank = {
        ".qpos.json": 0,
        ".post_minimal.qpos.json": 1,
        ".raw.qpos.json": 2,
        ".post_none.qpos.json": 3,
        ".post_full.qpos.json": 4,
    }
    out: list[dict] = []
    for refs in refs_roots:
        root = refs / "robot_b" / robot
        if not root.is_dir():
            continue

        def sort_key(path: Path) -> tuple:
            clip, suffix = _split_clip_name(path)
            return (str(path.parent), clip, rank.get(suffix, 9), path.name)

        for path in sorted(root.rglob("*.qpos.json"), key=sort_key):
            clip, suffix = _split_clip_name(path)
            if main_only:
                if suffix != ".qpos.json":
                    continue

                if "_wrist_" in clip:
                    continue

            out.append(_motion_meta(path, refs, robot))

    return out


def find_g1_motion(robot_motion: Path) -> Path | None:
    clip, _suffix = _split_clip_name(robot_motion)
    ordered_roots = []
    for refs in _ROBOT_REFS:
        try:
            robot_motion.relative_to(refs / "robot_b")
            ordered_roots.append(refs)
        except ValueError:
            continue

    ordered_roots.extend(refs for refs in _ROBOT_REFS if refs not in ordered_roots)
    for refs in ordered_roots:
        source = refs / "source" / "unitree_g1"
        if not source.is_dir():
            continue

        try:
            rel = robot_motion.relative_to(refs / "robot_b")
            dataset = rel.parts[1] if len(rel.parts) > 2 else None
        except ValueError:
            dataset = robot_motion.parent.name

        candidates: list[Path] = []
        if dataset:
            candidates.append(source / dataset / f"{clip}.qpos.json")

        candidates.extend(sorted(source.rglob(f"{clip}.qpos.json")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None


def catalog(*, main_only: bool = True) -> dict:
    refs = _REFS
    robots = sorted(set(discover_robots(_ROBOT_REFS)) | set(supported_target_robots()))
    by_robot = {
        name: discover_motions(_ROBOT_REFS, name, main_only=main_only) for name in robots
    }
    source_inputs = discover_source_inputs(_ROBOT_REFS)
    return {
        "refs": str(refs),
        "robot_refs": [str(path) for path in _ROBOT_REFS],
        "robots": robots,
        "motions": by_robot,
        "source_inputs": source_inputs,
        "source_clips": sorted({item["clip"] for item in source_inputs}),
        "main_only": main_only,
        "defaults": {
            "robot": "unitree_h2" if "unitree_h2" in robots else (robots[0] if robots else ""),
            "offset_y": 1.2,
            "loop": True,
            "tint": True,
            "main_only": True,
        },
    }


def stop_play() -> dict:
    global _PLAY_PROC
    proc = _PLAY_PROC
    _PLAY_PROC = None
    if proc is None:
        return {"ok": False, "error": "没有在跑的窗口。"}

    if proc.poll() is not None:
        return {"ok": True, "message": "窗口已结束。"}

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

    return {"ok": True, "message": "窗口已关。"}


def play(payload: dict) -> dict:
    global _PLAY_PROC
    robot = str(payload.get("robot", "")).strip()
    motion = Path(str(payload.get("motion", ""))).expanduser()
    if not robot or not motion.is_file():
        return {"ok": False, "error": "请选择有效的机器人与轨迹。"}

    g1 = find_g1_motion(motion)
    if g1 is None or not g1.is_file():
        return {
            "ok": False,
            "error": f"找不到匹配的 G1 source：{_REFS / 'source' / 'unitree_g1'} / {motion.name}",
        }

    if not VIZ_SCRIPT.is_file():
        return {"ok": False, "error": f"找不到 viewer：{VIZ_SCRIPT}"}

    stop_play()
    cmd = [
        sys.executable,
        str(VIZ_SCRIPT),
        "--g1_motion",
        str(g1),
        "--robot_b",
        robot,
        "--robot_b_motion",
        str(motion),
        "--offset_y",
        str(float(payload.get("offset_y", 1.2))),
    ]
    if not bool(payload.get("loop", True)):
        cmd.append("--no-loop")

    if not bool(payload.get("tint", True)):
        cmd.append("--no-tint")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    _PLAY_PROC = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "ok": True,
        "pid": _PLAY_PROC.pid,
        "robot": robot,
        "g1": str(g1),
        "motion": str(motion),
        "cmd": " ".join(cmd),
    }


def convert(payload: dict) -> dict:
    global _PLAY_PROC
    source = Path(str(payload.get("source", ""))).expanduser().resolve()
    robot = str(payload.get("robot", "")).strip()
    realtime = bool(payload.get("realtime", False))
    online_canonical = realtime and bool(payload.get("online_canonical", False))
    allowed_sources = {Path(item["path"]) for item in discover_source_inputs(_ROBOT_REFS)}
    if source not in allowed_sources:
        return {"ok": False, "error": "请选择列表中的有效 G1 输入轨迹。"}

    if robot not in supported_target_robots():
        return {"ok": False, "error": f"目标机器人不支持 SMPL-X 重定向：{robot}"}

    converter = REALTIME_CPP_CONVERTER if realtime else CPP_CONVERTER
    if not converter.is_file():
        target = converter.name
        return {
            "ok": False,
            "error": f"纯 C++ 转换器尚未编译。请先在 IDE 中构建 {target}。",
        }

    clip_source = _split_clip_name(source)[0]
    if source.suffix.lower() == ".pkl":
        clip_source = str(source.relative_to(GMR_CG_BATCH).with_suffix(""))

    clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_source).strip("._")
    if not clip:
        return {"ok": False, "error": "输入轨迹文件名无法生成有效 clip 名。"}

    if online_canonical:
        output_clip = f"{clip}_realtime_canonical"
    elif realtime:
        output_clip = f"{clip}_realtime"
    else:
        output_clip = clip

    source_output = GUI_REFS / "source" / "unitree_g1" / "gui" / f"{output_clip}.qpos.json"
    robot_output = GUI_REFS / "robot_b" / robot / "gui" / f"{output_clip}.qpos.json"
    cpp_input = prepare_cpp_input(source, source_output)
    command = [
        str(converter),
        "--gmr_root",
        str(REPO),
        "--input",
        str(cpp_input),
        "--robot_b",
        robot,
        "--out_json",
        str(robot_output),
        "--dump_source_json",
        str(source_output),
    ]
    if not realtime:
        command.append("--fast")

    if realtime:
        stop_play()
        viewer_command = [
            sys.executable,
            str(VIZ_SCRIPT),
            "--g1_motion",
            str(cpp_input),
            "--robot_b",
            robot,
            "--robot_b_motion",
            str(robot_output),
            "--offset_y",
            str(float(payload.get("offset_y", 1.2))),
            "--live_retarget",
            "--realtime_cli",
            str(REALTIME_CPP_CONVERTER),
            "--dump_source_json",
            str(source_output),
        ]
        if not bool(payload.get("loop", True)):
            viewer_command.append("--no-loop")

        if not bool(payload.get("tint", True)):
            viewer_command.append("--no-tint")

        if online_canonical:
            viewer_command.append("--online_canonical")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        _PLAY_PROC = subprocess.Popen(
            viewer_command,
            cwd=str(REPO),
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return {
            "ok": True,
            "live": True,
            "pid": _PLAY_PROC.pid,
            "robot": robot,
            "source": str(source_output),
            "motion": str(robot_output),
            "method": "realtime_online_canonical_qp_stream" if online_canonical else "realtime_direct_qp_stream",
            "cmd": " ".join(viewer_command),
        }

    if not _CONVERT_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "已有转换任务正在运行，请稍候。"}

    try:
        env = os.environ.copy()
        library_dirs = [Path("/opt/robot/devel/x86_64_gcc114/lib"), Path("/opt/robot/devel/lib")]
        available = [str(path) for path in library_dirs if path.is_dir()]
        if available:
            available.append(env.get("LD_LIBRARY_PATH", ""))
            env["LD_LIBRARY_PATH"] = ":".join(available)

        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        _CONVERT_LOCK.release()

    log = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        return {"ok": False, "error": "纯 C++ 转换失败。", "log": log[-6000:]}

    return {
        "ok": True,
        "robot": robot,
        "source": str(source_output),
        "motion": str(robot_output),
        "method": "realtime_direct_qp" if realtime else "canonical_batch",
        "log": log[-6000:],
    }


def status() -> dict:
    proc = _PLAY_PROC
    if proc is None:
        return {"running": False, "log": ""}

    alive = proc.poll() is None
    log = ""
    if proc.stdout is not None and not alive:
        try:
            log = proc.stdout.read() or ""
        except OSError:
            log = ""

    return {
        "running": alive,
        "pid": proc.pid if alive else None,
        "returncode": None if alive else proc.returncode,
        "log": log[-4000:],
    }


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>GMR Compare Studio</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }
  header { padding: 16px 20px 8px; }
  h1 { margin: 0; font-size: 20px; font-weight: 650; }
  .sub { color: #9aa; margin-top: 4px; }
  main { display: grid; grid-template-columns: 340px 1fr; gap: 16px; padding: 12px 20px 24px; }
  label { display: block; font-size: 12px; color: #9aa; margin: 12px 0 4px; }
  select, input, button { width: 100%; font: inherit; color: inherit; background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 8px 10px; }
  button { cursor: pointer; background: #2a4a7a; border-color: #3a6aaa; margin-top: 8px; }
  button.secondary { background: #2a2a2a; border-color: #444; }
  button:disabled { cursor: wait; opacity: 0.55; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .hint { font-size: 12px; color: #8a8; margin-top: 8px; min-height: 2.8em; }
  .meta { font-size: 12px; color: #9cf; margin-top: 6px; min-height: 1.4em; word-break: break-all; }
  .checks { display: grid; gap: 6px; margin-top: 8px; }
  .checks label { display: flex; gap: 8px; align-items: center; margin: 0; color: #ddd; font-size: 14px; }
  .checks input { width: auto; }
  pre { margin: 0; height: calc(100vh - 120px); overflow: auto; background: #0b0b0b; border: 1px solid #2a2a2a; border-radius: 10px; padding: 12px; white-space: pre-wrap; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #1a2a3a; color: #9cf; font-size: 11px; margin-left: 8px; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>GMR Compare Studio <span class="badge">G1 ↔ Robot-B</span></h1>
  <div class="sub">左侧选机器人和轨迹，点「打开窗口」并排看 G1 与目标机。灰蓝=目标机 tint。</div>
</header>
<main>
  <aside>
    <label>结果目录（G1 source 使用第一项）</label>
    <input id="refs" readonly />
    <label>G1 输入轨迹</label>
    <select id="source" size="7" style="height:150px"></select>
    <label>Robot B</label>
    <select id="robot"></select>
    <button id="convertPlay">纯 C++ 转换并播放</button>
    <button id="realtimeConvertPlay">实时 Direct-QP 转换并播放</button>
    <button id="onlineCanonicalPlay">实时 Canonical-QP 转换并播放</button>
    <label>搜索轨迹</label>
    <input id="q" placeholder="clip 名，例如 walk1" />
    <label>参考轨迹</label>
    <select id="motion" size="14" style="height:280px"></select>
    <div class="meta" id="motionMeta">选择一条轨迹…</div>
    <label>Offset Y</label>
    <input id="offsetY" type="number" min="0" max="5" step="0.1" value="1.2" />
    <div class="checks">
      <label><input id="loop" type="checkbox" checked> 循环播放</label>
      <label><input id="tint" type="checkbox" checked> Tint Robot B</label>
      <label><input id="mainOnly" type="checkbox" checked> 只显示主文件（隐藏 raw/post）</label>
    </div>
    <div class="row">
      <button id="play">打开窗口</button>
      <button class="secondary" id="stop">关掉窗口</button>
    </div>
    <button class="secondary" id="refresh">刷新目录</button>
    <div class="hint" id="hint"></div>
  </aside>
  <pre id="out">加载中…</pre>
</main>
<script>
let catalog = {robots: [], motions: {}, defaults: {}};

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return await res.json();
}

function robotMotions() {
  const robot = document.getElementById('robot').value;
  return catalog.motions[robot] || [];
}

function filteredMotions() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const items = robotMotions();
  if (!q) return items;
  return items.filter(m => (m.label + ' ' + m.path).toLowerCase().includes(q));
}

function fillRobots() {
  const el = document.getElementById('robot');
  el.innerHTML = catalog.robots.map(r => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
  const preferred = localStorage.getItem('gmrCompareRobot') || catalog.defaults.robot || '';
  if (preferred && catalog.robots.includes(preferred)) el.value = preferred;
}

function fillSources() {
  const el = document.getElementById('source');
  const items = catalog.source_inputs || [];
  el.innerHTML = items.map(item => `<option value="${esc(item.path)}">${esc(item.label)}</option>`).join('');
  const preferred = localStorage.getItem('gmrCompareSource') || '';
  if (preferred && items.some(item => item.path === preferred)) {
    el.value = preferred;
  } else if (items.length) {
    el.value = items[0].path;
  }
}

function fillMotions(keepPath) {
  const el = document.getElementById('motion');
  const items = filteredMotions();
  el.innerHTML = items.map(m => `<option value="${esc(m.path)}">${esc(m.label)}</option>`).join('');
  if (keepPath && items.some(m => m.path === keepPath)) {
    el.value = keepPath;
  } else if (items.length) {
    const preferred = items.find(m => m.kind === 'minimal') || items[0];
    el.value = preferred.path;
  }
  updateMeta();
}

function updateMeta() {
  const robot = document.getElementById('robot').value;
  const path = document.getElementById('motion').value;
  const item = (catalog.motions[robot] || []).find(m => m.path === path);
  const box = document.getElementById('motionMeta');
  if (!item) {
    box.textContent = '选择一条轨迹…';
    return;
  }
  box.textContent = `${item.dataset} / ${item.clip} · ${item.kind}\n${item.path}`;
}

function writeOut(obj) {
  document.getElementById('out').textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
}

async function loadCatalog() {
  const mainOnly = document.getElementById('mainOnly').checked;
  catalog = await api('/api/catalog?main_only=' + (mainOnly ? '1' : '0'));
  document.getElementById('refs').value = (catalog.robot_refs || [catalog.refs]).join(' | ');
  document.getElementById('offsetY').value = catalog.defaults.offset_y ?? 1.2;
  document.getElementById('loop').checked = !!catalog.defaults.loop;
  document.getElementById('tint').checked = !!catalog.defaults.tint;
  fillRobots();
  fillSources();
  fillMotions(localStorage.getItem('gmrCompareMotion') || '');
  const n = robotMotions().length;
  const src = (catalog.source_clips || []).length;
  writeOut({
    refs: catalog.refs,
    robot_refs: catalog.robot_refs,
    robots: catalog.robots.length,
    selected_robot: document.getElementById('robot').value,
    motion_count: n,
    g1_source_clips: src,
    tip: n <= 1
      ? '该机器人目前几乎只有 walk1_subject2。批量导出更多 clip 时，点「刷新目录」更新列表。'
      : '选好后点「打开窗口」。关闭 MuJoCo 窗或点「关掉窗口」。',
  });
  document.getElementById('hint').textContent =
    `机器人 ${catalog.robots.length} · 当前可选轨迹 ${n}` +
    (src ? ` · G1 source ${src} 条` : '');
}

async function convertAndPlay(realtime, onlineCanonical = false) {
  const buttonId = onlineCanonical
    ? 'onlineCanonicalPlay'
    : (realtime ? 'realtimeConvertPlay' : 'convertPlay');
  const button = document.getElementById(buttonId);
  const source = document.getElementById('source').value;
  const robot = document.getElementById('robot').value;
  if (!source || !robot) {
    document.getElementById('hint').textContent = '请选择 G1 输入轨迹和 Robot B。';
    return;
  }

  localStorage.setItem('gmrCompareSource', source);
  localStorage.setItem('gmrCompareRobot', robot);
  button.disabled = true;
  button.textContent = '转换中…';
  document.getElementById('hint').textContent = onlineCanonical
    ? '正在按帧运行 Online Canonical-QP…'
    : (realtime ? '正在按帧运行实时 Direct-QP…' : '正在运行 canonical Batch pipeline…');
  try {
    const data = await api('/api/convert', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        source,
        robot,
        realtime,
        online_canonical: onlineCanonical,
        offset_y: Number(document.getElementById('offsetY').value || 1.2),
        loop: document.getElementById('loop').checked,
        tint: document.getElementById('tint').checked,
      }),
    });
    writeOut(data);
    if (!data.ok) {
      document.getElementById('hint').textContent = data.error || '转换失败';
      return;
    }

    if (data.live) {
      document.getElementById('hint').textContent =
        onlineCanonical
          ? '实时窗口已启动：Canonical-QP 每产出一帧立即显示；整段结束后自动循环已缓存结果。'
          : '实时窗口已启动：Direct-QP 每产出一帧立即显示；整段结束后自动循环已缓存结果。';
      return;
    }

    await loadCatalog();
    document.getElementById('robot').value = data.robot;
    fillMotions(data.motion);
    document.getElementById('motion').value = data.motion;
    updateMeta();
    await play();
  } catch (error) {
    document.getElementById('hint').textContent = `转换失败：${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = onlineCanonical
      ? '实时 Canonical-QP 转换并播放'
      : (realtime ? '实时 Direct-QP 转换并播放' : '纯 C++ 转换并播放');
  }
}

async function play() {
  const hint = document.getElementById('hint');
  const body = {
    robot: document.getElementById('robot').value,
    motion: document.getElementById('motion').value,
    offset_y: Number(document.getElementById('offsetY').value || 1.2),
    loop: document.getElementById('loop').checked,
    tint: document.getElementById('tint').checked,
  };
  localStorage.setItem('gmrCompareRobot', body.robot);
  localStorage.setItem('gmrCompareMotion', body.motion);
  const data = await api('/api/play', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  writeOut(data);
  hint.textContent = data.ok
    ? `已打开窗口 · ${data.robot} · pid ${data.pid}`
    : (data.error || '启动失败');
}

async function stopPlay() {
  const data = await api('/api/stop', {method: 'POST'});
  document.getElementById('hint').textContent = data.message || data.error || '';
  writeOut(data);
}

async function pollStatus() {
  try {
    const data = await api('/api/status');
    if (data.running) {
      document.getElementById('hint').textContent = `窗口运行中 · pid ${data.pid}`;
    } else if (data.log) {
      writeOut({finished: true, returncode: data.returncode, log: data.log});
    }
  } catch (_) {}
}

document.getElementById('play').onclick = play;
document.getElementById('convertPlay').onclick = () => convertAndPlay(false);
document.getElementById('realtimeConvertPlay').onclick = () => convertAndPlay(true);
document.getElementById('onlineCanonicalPlay').onclick = () => convertAndPlay(true, true);
document.getElementById('stop').onclick = stopPlay;
document.getElementById('refresh').onclick = loadCatalog;
document.getElementById('mainOnly').onchange = loadCatalog;
document.getElementById('robot').onchange = () => {
  localStorage.setItem('gmrCompareRobot', document.getElementById('robot').value);
  fillMotions('');
  document.getElementById('hint').textContent =
    `当前 ${document.getElementById('robot').value} · 可选轨迹 ${robotMotions().length}`;
};
document.getElementById('motion').onchange = () => {
  localStorage.setItem('gmrCompareMotion', document.getElementById('motion').value);
  updateMeta();
};
document.getElementById('source').onchange = () => {
  localStorage.setItem('gmrCompareSource', document.getElementById('source').value);
};
document.getElementById('q').oninput = () => fillMotions(document.getElementById('motion').value);

loadCatalog();
setInterval(pollStatus, 2500);
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

    def _json(self, payload: dict, code: int = 200) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _html(self, text: str) -> None:
        blob = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._html(PAGE)
            return

        if parsed.path == "/api/catalog":
            qs = parse_qs(parsed.query)
            main_only = qs.get("main_only", ["1"])[0] != "0"
            self._json(catalog(main_only=main_only))
            return

        if parsed.path == "/api/status":
            self._json(status())
            return

        self._json({"ok": False, "error": "not found"}, code=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/play":
            self._json(play(self._read_json()))
            return

        if parsed.path == "/api/convert":
            self._json(convert(self._read_json()))
            return

        if parsed.path == "/api/stop":
            self._json(stop_play())
            return

        self._json({"ok": False, "error": "not found"}, code=404)


def main() -> None:
    global _REFS, _ROBOT_REFS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    parser.add_argument(
        "--robot-refs",
        type=Path,
        action="append",
        default=[],
        help="Additional result root containing robot_b/ (repeatable)",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    _REFS = args.refs.expanduser().resolve()
    robot_refs = [_REFS]
    automatic_contact_anchor = _REFS.with_name(f"{_REFS.name}_contact_anchor")
    if automatic_contact_anchor.is_dir():
        robot_refs.append(automatic_contact_anchor)

    robot_refs.append(GUI_REFS)
    robot_refs.extend(path.expanduser().resolve() for path in args.robot_refs)
    _ROBOT_REFS = list(dict.fromkeys(robot_refs))

    server = StudioServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[gmr-compare-studio] source refs={_REFS}")
    print(f"[gmr-compare-studio] robot refs={', '.join(map(str, _ROBOT_REFS))}")
    print(f"[gmr-compare-studio] open {url}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gmr-compare-studio] bye")
    finally:
        stop_play()
        server.server_close()


if __name__ == "__main__":
    main()
