"""Shared command-building logic for GMR GUI launchers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.gvhmr_env import DEFAULT_GVHMR_ROOT

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

INPUT_TYPES = {
    "bvh_lafan1": {
        "label": "BVH (LAFAN1)",
        "extensions": (".bvh",),
        "script": "retarget/bvh_to_robot.py",
        "batch_script": "retarget/bvh_to_robot_dataset.py",
        "src_human": "bvh_lafan1",
        "bvh_format": "lafan1",
        "supports_contact": True,
        "supports_batch": True,
    },
    "bvh_nokov": {
        "label": "BVH (Nokov)",
        "extensions": (".bvh",),
        "script": "retarget/bvh_to_robot.py",
        "batch_script": "retarget/bvh_to_robot_dataset.py",
        "src_human": "bvh_nokov",
        "bvh_format": "nokov",
        "supports_contact": True,
        "supports_batch": False,
    },
    "smplx": {
        "label": "SMPL-X (.npz)",
        "extensions": (".npz", ".pkl"),
        "script": "retarget/smplx_to_robot.py",
        "batch_script": "retarget/smplx_to_robot_dataset.py",
        "src_human": "smplx",
        "supports_contact": True,
        "supports_batch": True,
    },
    "gvhmr_pt": {
        "label": "GVHMR (.pt)",
        "extensions": (".pt",),
        "script": "gvhmr/to_robot.py",
        "src_human": "smplx",
        "supports_contact": True,
        "supports_batch": False,
    },
    "video_gvhmr": {
        "label": "Video → GVHMR → GMR",
        "extensions": VIDEO_EXTENSIONS,
        "script": "gvhmr/video_to_robot.py",
        "src_human": "smplx",
        "supports_contact": True,
        "supports_batch": False,
        "needs_gvhmr": True,
    },
    "human_json": {
        "label": "Human JSON",
        "extensions": (".json",),
        "script": "retarget/human_json_to_robot.py",
        "src_human": None,
        "supports_contact": True,
        "supports_batch": False,
    },
    "playback_pkl": {
        "label": "Playback PKL/JSON",
        "extensions": (".pkl", ".json"),
        "script": "viz/vis_robot_motion.py",
        "supports_contact": False,
        "supports_batch": False,
    },
}

INPUT_TYPE_LABELS = {key: cfg["label"] for key, cfg in INPUT_TYPES.items()}
LABEL_TO_INPUT_TYPE = {cfg["label"]: key for key, cfg in INPUT_TYPES.items()}
INVERSE_RATE_LIMIT_TYPES = frozenset({"gvhmr_pt", "video_gvhmr"})

RETARGET_ALGOS = {
    "ik": {"label": "Per-frame IK (GMR)"},
    "online_batch": {"label": "Online Batch-Lite (在线 · 推荐)"},
    "batch_to": {"label": "Batch TO (Python)"},
    "cpp_batch_to": {"label": "Batch TO (C++ · 一键回放)"},
}
RETARGET_ALGO_LABELS = {key: cfg["label"] for key, cfg in RETARGET_ALGOS.items()}
LABEL_TO_RETARGET_ALGO = {cfg["label"]: key for key, cfg in RETARGET_ALGOS.items()}

ONLINE_BATCH_SCRIPT_BY_INPUT = {
    "gvhmr_pt": "gvhmr/to_robot_online_batch.py",
}
ONLINE_BATCH_SUPPORTED_INPUT_TYPES = frozenset(ONLINE_BATCH_SCRIPT_BY_INPUT)

BATCH_TO_SCRIPT_BY_INPUT = {
    "gvhmr_pt": "retarget/to_robot_batch.py",
    "smplx": "retarget/to_robot_batch.py",
    "bvh_lafan1": "retarget/to_robot_batch.py",
    "bvh_nokov": "retarget/to_robot_batch.py",
}
BATCH_TO_SUPPORTED_INPUT_TYPES = frozenset(BATCH_TO_SCRIPT_BY_INPUT)
CPP_TO_SUPPORTED_INPUT_TYPES = BATCH_TO_SUPPORTED_INPUT_TYPES

CPP_TO_VIEWER_SCRIPT = "tools/run_cpp_to_viewer.py"
DEFAULT_CPP_VIEWER = REPO_ROOT / "cpp" / "build" / "gmr_retarget_viewer"

TRI_STATE_LABELS = ("IK 默认", "开启", "关闭")
ALL_ROBOTS = sorted(ROBOT_XML_DICT.keys())
GUI_APP_TITLE = "GMR Retargeting"


@dataclass
class GMRRunConfig:
    input_type: str
    run_mode: str
    input_path: str
    robot: str
    retarget_algo: str = "ik"
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
    gvhmr_root: str = ""
    gvhmr_python: str = ""
    gvhmr_static_cam: bool = True
    batch_to_fast: bool = False
    batch_to_window_size: int = 16
    batch_to_window_stride: int = 8
    batch_to_gn_steps: int = 3
    online_batch_preset: str = "balanced"


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


def supports_batch_to(input_type: str) -> bool:
    return input_type in BATCH_TO_SUPPORTED_INPUT_TYPES


def supports_cpp_to(input_type: str) -> bool:
    return input_type in CPP_TO_SUPPORTED_INPUT_TYPES


def is_cpp_retarget_algo(retarget_algo: str) -> bool:
    return retarget_algo == "cpp_batch_to"


def supports_online_batch(input_type: str) -> bool:
    return input_type in ONLINE_BATCH_SUPPORTED_INPUT_TYPES


def resolve_retarget_script(input_type: str, retarget_algo: str) -> str:
    if retarget_algo == "online_batch":
        if input_type not in ONLINE_BATCH_SCRIPT_BY_INPUT:
            raise ValueError(f"Online Batch not supported for input type: {input_type}")
        return ONLINE_BATCH_SCRIPT_BY_INPUT[input_type]
    if retarget_algo == "batch_to":
        if input_type not in BATCH_TO_SCRIPT_BY_INPUT:
            raise ValueError(f"Batch TO not supported for input type: {input_type}")
        return BATCH_TO_SCRIPT_BY_INPUT[input_type]
    return INPUT_TYPES[input_type]["script"]


def append_online_batch_args(cmd: list[str], cfg: GMRRunConfig) -> None:
    if cfg.retarget_algo != "online_batch":
        return
    cmd += ["--preset", str(cfg.online_batch_preset)]
    if cfg.loop:
        cmd.append("--loop")
    if cfg.rate_limit:
        cmd.append("--rate_limit")
    else:
        cmd.append("--no-rate-limit")


def append_batch_to_args(cmd: list[str], cfg: GMRRunConfig) -> None:
    if cfg.retarget_algo != "batch_to":
        return
    cmd += ["--window_size", str(int(cfg.batch_to_window_size))]
    cmd += ["--window_stride", str(int(cfg.batch_to_window_stride))]
    cmd += ["--gn_steps", str(int(cfg.batch_to_gn_steps))]
    if cfg.batch_to_fast:
        cmd.append("--fast")
    if cfg.loop or cfg.rate_limit:
        cmd.append("--view")
    if cfg.loop:
        cmd.append("--loop")
    if cfg.rate_limit:
        cmd.append("--rate_limit")
    else:
        cmd.append("--no-rate-limit")


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
    if INPUT_TYPES[cfg.input_type].get("needs_gvhmr"):
        gvhmr_root = cfg.gvhmr_root.strip() or str(DEFAULT_GVHMR_ROOT)
        if not Path(gvhmr_root).is_dir():
            return f"GVHMR 根目录不存在: {gvhmr_root}"
        demo_script = Path(gvhmr_root) / "tools" / "demo" / "demo.py"
        if not demo_script.is_file():
            return f"未找到 GVHMR demo 脚本: {demo_script}"
    if cfg.input_type == "video_gvhmr" and p.suffix.lower() not in VIDEO_EXTENSIONS:
        return f"视频模式需要 {', '.join(VIDEO_EXTENSIONS)} 格式文件"
    if cfg.retarget_algo == "online_batch":
        if cfg.run_mode == "batch":
            return "Online Batch-Lite 暂不支持批量模式"
        if not supports_online_batch(cfg.input_type):
            return "当前数据类型不支持 Online Batch（支持 GVHMR .pt）"
    if cfg.retarget_algo == "batch_to":
        if cfg.run_mode == "batch":
            return "Batch TO 暂不支持批量模式"
        if not supports_batch_to(cfg.input_type):
            return "当前数据类型不支持 Batch TO（支持 GVHMR .pt / SMPL-X / BVH）"
    if is_cpp_retarget_algo(cfg.retarget_algo):
        if cfg.run_mode == "batch":
            return "C++ TO 暂不支持批量模式"
        if not supports_cpp_to(cfg.input_type):
            return "当前数据类型不支持 C++ TO（支持 GVHMR .pt / SMPL-X / BVH）"
        if not DEFAULT_CPP_VIEWER.is_file():
            return f"未找到 C++ viewer: {DEFAULT_CPP_VIEWER}（请先 cmake --build cpp/build）"
        if cfg.record_video:
            return "C++ TO 暂不支持 GUI 内录制视频，请用 Python 流程或后续扩展"
    return None


def append_cpp_to_viewer_args(cmd: list[str], cfg: GMRRunConfig) -> None:
    if not is_cpp_retarget_algo(cfg.retarget_algo):
        return
    method = "batch_to"
    cmd += ["--method", method]
    input_type = cfg.input_type
    path = cfg.input_path.strip()
    cmd += ["--input_file", path]
    if input_type in ("bvh_lafan1", "bvh_nokov"):
        meta = INPUT_TYPES[input_type]
        cmd += ["--input_type", input_type, "--format", meta["bvh_format"]]
        cmd += ["--motion_fps", (cfg.motion_fps or "30").strip()]
    elif input_type == "smplx":
        cmd += ["--input_type", "smplx"]
        cmd += ["--motion_fps", (cfg.motion_fps or "30").strip()]
    elif input_type == "gvhmr_pt":
        cmd += ["--input_type", "gvhmr_pt"]
    if cfg.retarget_algo == "cpp_batch_to":
        cmd += [
            "--window_size",
            str(int(cfg.batch_to_window_size)),
            "--window_stride",
            str(int(cfg.batch_to_window_stride)),
            "--gn_steps",
            str(int(cfg.batch_to_gn_steps)),
        ]
        if cfg.batch_to_fast:
            cmd.append("--fast")
        else:
            cmd.append("--quality")
    if cfg.loop:
        cmd.append("--loop")
    else:
        cmd.append("--no_loop")
    for name, value in (
        ("contact_ground", cfg.contact_ground),
        ("fix_robot_penetration", cfg.fix_robot_penetration),
        ("foot_ground_limit", cfg.foot_ground_limit),
    ):
        if value != "IK 默认":
            cmd += [f"--{name}", value]
    if cfg.save_output and cfg.retarget_algo == "cpp_batch_to":
        out = cfg.save_path.strip()
        if out.endswith(".pkl"):
            out = out[:-4] + ".json"
        elif not out.endswith(".json"):
            out = out + ".json"
        cmd += ["--out_json", out]


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

    script = SCRIPTS_DIR / resolve_retarget_script(input_type, cfg.retarget_algo)
    if is_cpp_retarget_algo(cfg.retarget_algo):
        cmd.append(str(SCRIPTS_DIR / CPP_TO_VIEWER_SCRIPT))
        append_cpp_to_viewer_args(cmd, cfg)
        cmd += ["--robot", robot, "--gmr_root", str(REPO_ROOT)]
        return cmd

    cmd.append(str(script))

    if cfg.retarget_algo == "batch_to" and input_type in BATCH_TO_SCRIPT_BY_INPUT:
        cmd += ["--input_file", path, "--robot", robot]
        if input_type in ("bvh_lafan1", "bvh_nokov"):
            cmd += ["--input_type", input_type, "--format", meta["bvh_format"]]
            cmd += ["--motion_fps", (cfg.motion_fps or "30").strip()]
        elif input_type == "smplx":
            cmd += ["--input_type", "smplx"]
        elif input_type == "gvhmr_pt":
            cmd += ["--input_type", "gvhmr_pt"]
    elif input_type in ("bvh_lafan1", "bvh_nokov"):
        cmd += ["--bvh_file", path, "--robot", robot, "--format", meta["bvh_format"]]
        cmd += ["--motion_fps", (cfg.motion_fps or "30").strip()]
    elif input_type == "smplx":
        cmd += ["--smplx_file", path, "--robot", robot]
    elif input_type == "gvhmr_pt":
        cmd += ["--gvhmr_pred_file", path, "--robot", robot]
    elif input_type == "video_gvhmr":
        cmd += ["--video", path, "--robot", robot]
        gvhmr_root = cfg.gvhmr_root.strip() or str(DEFAULT_GVHMR_ROOT)
        cmd += ["--gvhmr_root", gvhmr_root]
        if cfg.gvhmr_python.strip():
            cmd += ["--gvhmr_python", cfg.gvhmr_python.strip()]
        if cfg.gvhmr_static_cam:
            cmd.append("--static_cam")
        else:
            cmd.append("--no-static_cam")
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

    if input_type != "playback_pkl":
        if input_type in INVERSE_RATE_LIMIT_TYPES:
            if cfg.rate_limit:
                cmd.append("--rate_limit")
            else:
                cmd.append("--no-rate-limit")
        elif cfg.rate_limit:
            cmd.append("--rate_limit")
    if cfg.loop and input_type != "playback_pkl":
        cmd.append("--loop")
    if cfg.save_output and input_type != "playback_pkl":
        cmd += ["--save_path", cfg.save_path.strip()]
    if cfg.record_video:
        cmd.append("--record_video")
        cmd += ["--video_path", cfg.video_path.strip()]

    if cfg.retarget_algo == "batch_to" and input_type != "playback_pkl":
        append_batch_to_args(cmd, cfg)
    if cfg.retarget_algo == "online_batch" and input_type != "playback_pkl":
        append_online_batch_args(cmd, cfg)

    return cmd
