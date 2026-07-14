"""GVHMR environment discovery (Python executable with required deps)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_GVHMR_ROOT = Path.home() / "Workspace/xeeform_motion_generation/GVHMR"
GVHMR_ENV_NAMES = ("gvhmr", "GVHMR", "hmr4d")
GVHMR_IMPORT_PROBE = "import pytorch_lightning"


def python_has_gvhmr_deps(python_path: str | Path) -> bool:
    path = str(python_path)
    if not Path(path).is_file():
        return False
    try:
        subprocess.run(
            [path, "-c", GVHMR_IMPORT_PROBE],
            capture_output=True,
            check=True,
            timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False


def _conda_base_from_python(python_path: Path) -> Path | None:
    resolved = python_path.resolve()
    if resolved.parent.name == "bin" and resolved.parent.parent.name == "envs":
        return resolved.parent.parent.parent
    return None


def gvhmr_python_candidates(gvhmr_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    env_python = os.environ.get("GVHMR_PYTHON", "").strip()
    if env_python:
        add(Path(env_python))

    root = (gvhmr_root or DEFAULT_GVHMR_ROOT).expanduser()
    for rel in (".venv/bin/python", "venv/bin/python"):
        add(root / rel)

    for base_python in (Path(sys.executable),):
        conda_base = _conda_base_from_python(base_python)
        if conda_base is not None:
            for env_name in GVHMR_ENV_NAMES:
                add(conda_base / "envs" / env_name / "bin" / "python")

    for base_dir in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path.home() / "mambaforge"):
        if base_dir.is_dir():
            for env_name in GVHMR_ENV_NAMES:
                add(base_dir / "envs" / env_name / "bin" / "python")

    return candidates


def resolve_gvhmr_python(gvhmr_root: Path | str | None = None, explicit: str | None = None) -> str:
    root = Path(gvhmr_root).expanduser() if gvhmr_root else DEFAULT_GVHMR_ROOT

    if explicit and explicit.strip():
        chosen = explicit.strip()
        if not python_has_gvhmr_deps(chosen):
            raise RuntimeError(
                f"指定的 GVHMR Python 缺少依赖（需要 pytorch_lightning）: {chosen}"
            )
        return chosen

    tried: list[str] = []
    for candidate in gvhmr_python_candidates(root):
        tried.append(str(candidate))
        if python_has_gvhmr_deps(candidate):
            return str(candidate)

    raise RuntimeError(
        "未找到可用的 GVHMR Python 环境。请安装 GVHMR 依赖，或在 GUI / 命令行指定 "
        f"--gvhmr_python（例如 {Path.home()}/miniconda3/envs/gvhmr/bin/python）。\n"
        f"已尝试: {', '.join(tried) if tried else '(无)'}"
    )


def default_gvhmr_python(gvhmr_root: Path | str | None = None) -> str:
    try:
        return resolve_gvhmr_python(gvhmr_root=gvhmr_root, explicit=None)
    except RuntimeError:
        for candidate in gvhmr_python_candidates(
            Path(gvhmr_root).expanduser() if gvhmr_root else DEFAULT_GVHMR_ROOT
        ):
            return str(candidate)
        return str(Path.home() / "miniconda3/envs/gvhmr/bin/python")
