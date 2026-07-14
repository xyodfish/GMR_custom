"""Shared command-building logic for GMR GUI launchers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

INPUT_TYPES = {
    "bvh_lafan1": {
        "label": "BVH (LAFAN1)",
        "extensions": (".bvh",),
        "script": "bvh_to_robot.py",
        "batch_script": "bvh_to_robot_dataset.py",
        "src_human": "bvh_lafan1",
        "bvh_format": "lafan1",
        "supports_contact": True,
        "supports_batch": True,
    },
    "bvh_nokov": {
        "label": "BVH (Nokov)",
        "extensions": (".bvh",),
        "script": "bvh_to_robot.py",
        "batch_script": "bvh_to_robot_dataset.py",
        "src_human": "bvh_nokov",
        "bvh_format": "nokov",
        "supports_contact": True,
        "supports_batch": False,
    },
    "smplx": {
        "label": "SMPL-X (.npz)",
        "extensions": (".npz", ".pkl"),
        "script": "smplx_to_robot.py",
        "batch_script": "smplx_to_robot_dataset.py",
        "src_human": "smplx",
        "supports_contact": True,
        "supports_batch": True,
    },
    "human_json": {
        "label": "Human JSON",
        "extensions": (".json",),
        "script": "human_json_to_robot.py",
        "src_human": None,
        "supports_contact": True,
        "supports_batch": False,
    },
    "playback_pkl": {
        "label": "Playback PKL",
        "extensions": (".pkl",),
        "script": "vis_robot_motion.py",
        "supports_contact": False,
        "supports_batch": False,
    },
}

INPUT_TYPE_LABELS = {key: cfg["label"] for key, cfg in INPUT_TYPES.items()}
LABEL_TO_INPUT_TYPE = {cfg["label"]: key for key, cfg in INPUT_TYPES.items()}

TRI_STATE_LABELS = ("IK 默认", "开启", "关闭")
ALL_ROBOTS = sorted(ROBOT_XML_DICT.keys())


@dataclass
class GMRRunConfig:
    input_type: str
    run_mode: str
    input_path: str
    robot: str
    motion_fps: str = "30"
    human_json_path: str = ""
    save_path: str = ""
    video_path: str = "videos/gmr_gui_output.mp4"
    rate_limit: bool = True
    loop: bool = True
    save_output: bool = False
    record_video: bool = False
    show_ik_anchors: bool = False
    show_body_names: bool = False
    contact_ground: str = "IK 默认"
    fix_robot_penetration: str = "IK 默认"
    foot_ground_limit: str = "IK 默认"


def robots_for_input(input_type: str) -> list[str]:
    if input_type == "playback_pkl":
        return ALL_ROBOTS
    cfg = INPUT_TYPES[input_type]
    src_human = cfg.get("src_human")
    if src_human is None:
        return ALL_ROBOTS
    supported = sorted(IK_CONFIG_DICT.get(src_human, {}).keys())
    return supported or ALL_ROBOTS


def tri_state_to_args(name: str, value: str) -> list[str]:
    if value == "开启":
        return [f"--{name}"]
    if value == "关闭":
        return [f"--no-{name}"]
    return []


def default_save_path(input_path: str, robot: str) -> str:
    stem = Path(input_path).stem
    out_dir = REPO_ROOT / "retargeting_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{stem}_{robot}.pkl")


def validate_config(cfg: GMRRunConfig) -> str | None:
    path = cfg.input_path.strip()
    if not path:
        return "请填写输入路径（本地绝对路径）"
    p = Path(path)
    if cfg.run_mode == "batch":
        if not p.is_dir():
            return "批量模式需要有效目录路径"
    elif not p.is_file():
        return "单文件模式需要有效文件路径"
    if cfg.save_output and not cfg.save_path.strip():
        return "已开启保存 PKL，请填写保存路径"
    if cfg.input_type == "playback_pkl" and cfg.show_ik_anchors and not cfg.human_json_path.strip():
        return "显示 IK 锚点时需要人体 JSON 路径"
    return None


def build_command(cfg: GMRRunConfig) -> list[str]:
    input_type = cfg.input_type
    meta = INPUT_TYPES[input_type]
    path = cfg.input_path.strip()
    robot = cfg.robot
    cmd = [sys.executable]

    if cfg.run_mode == "batch":
        script = SCRIPTS_DIR / meta["batch_script"]
        cmd += [
            str(script),
            "--src_folder",
            path,
            "--tgt_folder",
            path.rstrip("/") + "_robot",
            "--robot",
            robot,
        ]
        return cmd

    script = SCRIPTS_DIR / meta["script"]
    cmd.append(str(script))

    if input_type in ("bvh_lafan1", "bvh_nokov"):
        cmd += ["--bvh_file", path, "--robot", robot, "--format", meta["bvh_format"]]
        cmd += ["--motion_fps", (cfg.motion_fps or "30").strip()]
    elif input_type == "smplx":
        cmd += ["--smplx_file", path, "--robot", robot]
    elif input_type == "human_json":
        cmd += ["--human_frame_json", path, "--robot", robot]
    elif input_type == "playback_pkl":
        cmd += ["--robot_motion_path", path, "--robot", robot]
        if cfg.show_ik_anchors and cfg.human_json_path.strip():
            cmd += ["--human_frame_json", cfg.human_json_path.strip()]
        if cfg.show_body_names:
            cmd += ["--show_human_body_name"]

    if meta.get("supports_contact"):
        for name, value in (
            ("contact_ground", cfg.contact_ground),
            ("fix_robot_penetration", cfg.fix_robot_penetration),
            ("foot_ground_limit", cfg.foot_ground_limit),
        ):
            cmd += tri_state_to_args(name, value)

    if cfg.rate_limit and input_type != "playback_pkl":
        cmd.append("--rate_limit")
    if cfg.loop and input_type != "playback_pkl":
        cmd.append("--loop")
    if cfg.save_output and input_type != "playback_pkl":
        cmd += ["--save_path", cfg.save_path.strip()]
    if cfg.record_video:
        cmd.append("--record_video")
        cmd += ["--video_path", cfg.video_path.strip()]

    return cmd
