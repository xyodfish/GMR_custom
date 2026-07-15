#!/usr/bin/env python3
"""Run GVHMR demo on multiple videos and collect hmr4d_results.pt paths."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid importing general_motion_retargeting.__init__ (needs mink) when using gvhmr env.
import importlib.util

_gvhmr_env_path = REPO_ROOT / "general_motion_retargeting" / "utils" / "gvhmr_env.py"
_spec = importlib.util.spec_from_file_location("gmr_gvhmr_env", _gvhmr_env_path)
_gvhmr_env = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gvhmr_env)
DEFAULT_GVHMR_ROOT = _gvhmr_env.DEFAULT_GVHMR_ROOT
resolve_gvhmr_python = _gvhmr_env.resolve_gvhmr_python


def gvhmr_pred_path(gvhmr_root: Path, video_path: Path) -> Path:
    return gvhmr_root / "outputs" / "demo" / video_path.stem / "hmr4d_results.pt"


def run_one(
    video: Path,
    gvhmr_root: Path,
    gvhmr_python: str,
    static_cam: bool,
    copy_to: Path | None,
    force: bool,
) -> Path | None:
    if not video.is_file():
        print(f"[skip] missing video: {video}")
        return None

    pred = gvhmr_pred_path(gvhmr_root, video)
    if pred.is_file() and not force:
        print(f"[reuse] {pred}")
    else:
        cmd = [gvhmr_python, str(gvhmr_root / "tools" / "demo" / "demo.py"), f"--video={video}"]
        if static_cam:
            cmd.append("-s")
        print(f"[GVHMR] {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(gvhmr_root))
        if proc.returncode != 0:
            if pred.is_file():
                print(
                    f"[warn] GVHMR exit {proc.returncode} but {pred.name} exists "
                    "(inference ok, render likely failed)"
                )
            else:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        if not pred.is_file():
            raise FileNotFoundError(f"GVHMR finished but no output: {pred}")

    if copy_to is not None:
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred, copy_to)
        print(f"[copy] -> {copy_to}")
        return copy_to
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch GVHMR inference on videos.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(Path.home() / "Videos"),
        help="Directory to scan for videos (or pass --videos explicitly).",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Explicit video paths. If omitted, scans video_dir for *.mp4/*.mkv.",
    )
    parser.add_argument("--gvhmr_root", type=str, default=str(DEFAULT_GVHMR_ROOT))
    parser.add_argument("--gvhmr_python", type=str, default="")
    parser.add_argument("--static_cam", action="store_true", default=True)
    parser.add_argument("--no-static_cam", dest="static_cam", action="store_false")
    parser.add_argument(
        "--copy_into_video_dir",
        action="store_true",
        default=True,
        help="Copy hmr4d_results.pt next to each video under <stem>/hmr4d_results.pt",
    )
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(REPO_ROOT / "output" / "gvhmr_manifest.txt"),
    )
    args = parser.parse_args()

    gvhmr_root = Path(args.gvhmr_root).expanduser().resolve()
    gvhmr_python = resolve_gvhmr_python(gvhmr_root, args.gvhmr_python)

    if args.videos:
        videos = [Path(v).expanduser().resolve() for v in args.videos]
    else:
        video_dir = Path(args.video_dir).expanduser()
        videos = sorted(
            list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mkv")),
            key=lambda p: p.stat().st_size,
        )

    manifest_lines: list[str] = []
    for video in videos:
        copy_to = None
        if args.copy_into_video_dir:
            copy_to = video.parent / video.stem / "hmr4d_results.pt"
        try:
            pt = run_one(video, gvhmr_root, gvhmr_python, args.static_cam, copy_to, args.force)
            if pt is not None:
                manifest_lines.append(f"{video}\t{pt}")
        except subprocess.CalledProcessError as exc:
            print(f"[fail] {video}: exit {exc.returncode}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    print(f"[done] manifest: {manifest} ({len(manifest_lines)} entries)")


if __name__ == "__main__":
    main()
