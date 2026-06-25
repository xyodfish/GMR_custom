"""Merge per-robot contact_ground presets with IK JSON overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco as mj

_PRESETS_PATH = Path(__file__).resolve().parent / "ik_configs" / "contact_ground_presets.json"
_PRESETS_CACHE: dict[str, Any] | None = None


def _load_presets_file() -> dict[str, Any]:
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        with open(_PRESETS_PATH, encoding="utf-8") as f:
            _PRESETS_CACHE = json.load(f)
    return _PRESETS_CACHE


def _resolve_preset_entry(entry: dict[str, Any], presets: dict[str, Any]) -> dict[str, Any]:
    if "preset" not in entry:
        return dict(entry)
    base_name = str(entry["preset"])
    if base_name not in presets:
        raise KeyError(f"contact_ground preset '{base_name}' not found")
    base = _resolve_preset_entry(
        {k: v for k, v in presets[base_name].items() if not str(k).startswith("_")},
        presets,
    )
    override = {k: v for k, v in entry.items() if k != "preset"}
    return _deep_merge(base, override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def robot_preset(tgt_robot: str) -> dict[str, Any]:
    presets = _load_presets_file()
    default = {
        k: v
        for k, v in presets.get("_default", {}).items()
        if not str(k).startswith("_")
    }
    robot_entry = presets.get(tgt_robot)
    if robot_entry is None:
        return deepcopy(default)
    robot_cfg = _resolve_preset_entry(robot_entry, presets)
    return _deep_merge(default, robot_cfg)


def build_contact_ground_config(
    ik_config: dict[str, Any],
    tgt_robot: str,
    cli_override: bool | None = None,
) -> dict[str, Any]:
    cfg = robot_preset(tgt_robot)
    cfg = _deep_merge(cfg, dict(ik_config.get("contact_ground", {})))
    if "human_root_name" not in cfg:
        cfg["human_root_name"] = ik_config.get("human_root_name", "Hips")
    if cli_override is not None:
        cfg["enabled"] = bool(cli_override)
    return cfg


def validate_contact_ground_config(
    cfg: dict[str, Any],
    model: mj.MjModel,
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for key in ("robot_foot_bodies", "robot_trunk_bodies", "robot_leg_bodies", "robot_arm_bodies"):
        names = list(cfg.get(key, []))
        bad = [
            name
            for name in names
            if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) < 0
        ]
        if bad:
            missing[key] = bad
    floor_name = str(cfg.get("floor_geom_name", "floor"))
    if mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, floor_name) < 0:
        missing["floor_geom_name"] = [floor_name]
    return missing
