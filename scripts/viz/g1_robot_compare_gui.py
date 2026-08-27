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
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
VIZ_SCRIPT = Path(__file__).resolve().parent / "vis_g1_robot_compare.py"
DEFAULT_REFS = Path.home() / "Workspace" / "puppet" / "output" / "gmr_references"
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
_REFS = DEFAULT_REFS


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
        "id": f"{dataset}/{clip}:{kind}",
        "path": str(path),
        "clip": clip,
        "dataset": dataset,
        "kind": kind,
        "label": f"{dataset} · {clip} · {kind}",
    }


def discover_robots(refs: Path) -> list[str]:
    root = refs / "robot_b"
    if not root.is_dir():
        return []

    names: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(p.rglob("*.qpos.json")):
            names.append(p.name)

    return names


def discover_motions(refs: Path, robot: str, *, main_only: bool = True) -> list[dict]:
    """List robot-B motions.

    ``main_only`` keeps ``<clip>.qpos.json`` (pipeline default / minimal) and drops
    ``.raw`` / ``.post_*`` / experimental ``*_wrist_*`` sidecars.
    """
    root = refs / "robot_b" / robot
    if not root.is_dir():
        return []

    rank = {
        ".qpos.json": 0,
        ".post_minimal.qpos.json": 1,
        ".raw.qpos.json": 2,
        ".post_none.qpos.json": 3,
        ".post_full.qpos.json": 4,
    }
    paths = sorted(root.rglob("*.qpos.json"))

    def sort_key(p: Path) -> tuple:
        clip, suffix = _split_clip_name(p)
        return (str(p.parent), clip, rank.get(suffix, 9), p.name)

    out: list[dict] = []
    for path in sorted(paths, key=sort_key):
        clip, suffix = _split_clip_name(path)
        if main_only:
            if suffix != ".qpos.json":
                continue

            if "_wrist_" in clip:
                continue

        out.append(_motion_meta(path, refs, robot))

    return out


def find_g1_motion(refs: Path, robot_motion: Path) -> Path | None:
    clip, _suffix = _split_clip_name(robot_motion)
    source = refs / "source" / "unitree_g1"
    if not source.is_dir():
        return None

    try:
        rel = robot_motion.relative_to(refs / "robot_b")
        dataset = rel.parts[1] if len(rel.parts) > 2 else None
    except ValueError:
        dataset = robot_motion.parent.name

    candidates: list[Path] = []
    if dataset:
        candidates.append(source / dataset / f"{clip}.qpos.json")

    candidates.extend(sorted(source.rglob(f"{clip}.qpos.json")))
    for c in candidates:
        if c.is_file():
            return c

    return None


def catalog(*, main_only: bool = True) -> dict:
    refs = _REFS
    robots = discover_robots(refs)
    by_robot = {name: discover_motions(refs, name, main_only=main_only) for name in robots}
    source_dir = refs / "source" / "unitree_g1"
    source_clips = (
        sorted(
            {
                p.name[: -len(".qpos.json")]
                for p in source_dir.rglob("*.qpos.json")
                if p.name.endswith(".qpos.json")
            }
        )
        if source_dir.is_dir()
        else []
    )
    return {
        "refs": str(refs),
        "robots": robots,
        "motions": by_robot,
        "source_clips": source_clips,
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

    g1 = find_g1_motion(_REFS, motion)
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
    <label>Refs 根目录</label>
    <input id="refs" readonly />
    <label>Robot B</label>
    <select id="robot"></select>
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
  document.getElementById('refs').value = catalog.refs;
  document.getElementById('offsetY').value = catalog.defaults.offset_y ?? 1.2;
  document.getElementById('loop').checked = !!catalog.defaults.loop;
  document.getElementById('tint').checked = !!catalog.defaults.tint;
  fillRobots();
  fillMotions(localStorage.getItem('gmrCompareMotion') || '');
  const n = robotMotions().length;
  const src = (catalog.source_clips || []).length;
  writeOut({
    refs: catalog.refs,
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

        if parsed.path == "/api/stop":
            self._json(stop_play())
            return

        self._json({"ok": False, "error": "not found"}, code=404)


def main() -> None:
    global _REFS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    _REFS = args.refs.expanduser().resolve()

    server = StudioServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[gmr-compare-studio] refs={_REFS}")
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
