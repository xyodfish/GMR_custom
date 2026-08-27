"""Resolve GMR_custom target robots (XML + smplx IK) without importing mink."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class GmrTargetRobot:
    name: str
    model_xml: Path
    ik_config: Path
    base_body: str
    planar_base: bool


def _load_gmr_params(gmr_root: Path) -> ModuleType:
    params_path = gmr_root / "general_motion_retargeting" / "params.py"
    if not params_path.is_file():
        raise FileNotFoundError(f"GMR params not found: {params_path}")

    spec = importlib.util.spec_from_file_location("_gmr_params_standalone", params_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {params_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_smplx_target_robots(gmr_root: Path) -> list[str]:
    """Robots that have both a MuJoCo XML and an ``smplx_to_*`` IK config."""
    params = _load_gmr_params(gmr_root)
    smplx = dict(params.IK_CONFIG_DICT.get("smplx", {}))
    xmls = dict(params.ROBOT_XML_DICT)
    names: list[str] = []
    for name in sorted(smplx.keys()):
        xml = Path(xmls[name]) if name in xmls else None
        ik = Path(smplx[name])
        if xml is not None and xml.is_file() and ik.is_file():
            names.append(name)

    return names


def resolve_target_robot(gmr_root: Path, robot: str) -> GmrTargetRobot:
    params = _load_gmr_params(gmr_root)
    name = robot.strip()
    smplx = dict(params.IK_CONFIG_DICT.get("smplx", {}))
    if name not in smplx:
        available = ", ".join(list_smplx_target_robots(gmr_root))
        raise ValueError(f"Robot '{name}' has no smplx IK config. Available: {available}")

    if name not in params.ROBOT_XML_DICT:
        raise ValueError(f"Robot '{name}' missing from ROBOT_XML_DICT")

    model_xml = Path(params.ROBOT_XML_DICT[name]).resolve()
    ik_config = Path(smplx[name]).resolve()
    if not model_xml.is_file():
        raise FileNotFoundError(f"Robot model XML missing: {model_xml}")

    if not ik_config.is_file():
        raise FileNotFoundError(f"IK config missing: {ik_config}")

    base_body = str(params.ROBOT_BASE_DICT.get(name, "pelvis"))
    planar = name in set(getattr(params, "PLANAR_BASE_ROBOTS", ()))
    return GmrTargetRobot(
        name=name,
        model_xml=model_xml,
        ik_config=ik_config,
        base_body=base_body,
        planar_base=planar,
    )


def parse_robot_b_list(gmr_root: Path, robot_b: str) -> list[GmrTargetRobot]:
    """Parse ``--robot-b``: comma-separated names, or ``all`` for every smplx target."""
    text = robot_b.strip()
    if not text:
        raise ValueError("--robot-b is empty")

    if text.lower() in {"all", "*"}:
        names = list_smplx_target_robots(gmr_root)
    else:
        names = [part.strip() for part in text.split(",") if part.strip()]

    if not names:
        raise ValueError(f"No robots resolved from --robot-b={robot_b!r}")

    return [resolve_target_robot(gmr_root, name) for name in names]


def with_overrides(
    target: GmrTargetRobot,
    *,
    model_xml: Path | None = None,
    ik_config: Path | None = None,
) -> GmrTargetRobot:
    kwargs = {}
    if model_xml is not None:
        kwargs["model_xml"] = model_xml

    if ik_config is not None:
        kwargs["ik_config"] = ik_config

    return replace(target, **kwargs) if kwargs else target


def model_has_wrist_pitch_yaw(model_xml: Path) -> bool:
    """True if MJCF defines both wrist_pitch and wrist_yaw joints (either side)."""
    text = Path(model_xml).read_text(encoding="utf-8", errors="ignore")
    return ("wrist_pitch_joint" in text) and ("wrist_yaw_joint" in text)
