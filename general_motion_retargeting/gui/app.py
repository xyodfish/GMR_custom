#!/usr/bin/env python3
"""GMR retargeting GUI — Gradio web interface."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

from general_motion_retargeting.gui.core import (
    DEFAULT_GVHMR_ROOT,
    GMRRunConfig,
    GUI_APP_TITLE,
    INPUT_TYPE_LABELS,
    INPUT_TYPES,
    LABEL_TO_INPUT_TYPE,
    LABEL_TO_RETARGET_ALGO,
    REPO_ROOT,
    RETARGET_ALGO_LABELS,
    TRI_STATE_LABELS,
    build_command,
    is_cpp_retarget_algo,
    robots_for_input,
    supports_batch_to,
    supports_cpp_to,
    supports_online_batch,
    validate_config,
)
from general_motion_retargeting.utils.gvhmr_env import default_gvhmr_python

_ACTIVE_PROC: subprocess.Popen | None = None

DEFAULT_BROWSE_ROOT = Path("/home/xiayu/Workspace/data/lafan1")
if not DEFAULT_BROWSE_ROOT.exists():
    DEFAULT_BROWSE_ROOT = Path.home()

CUSTOM_CSS = """
.gradio-container { max-width: 1440px !important; margin: auto; }
#gmr-header { text-align: left; margin-bottom: 0; }
#gmr-header h1 { font-size: 1.35rem; font-weight: 700; margin: 0; }
#gmr-header p { display: none; }
#run-btn { min-height: 2.5rem; font-size: 1rem !important; }
#log-box textarea {
    font-family: "JetBrains Mono", "Consolas", "Menlo", monospace !important;
    font-size: 0.82rem !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
}
#config-panel .block { padding: 6px 8px !important; }
#config-panel .form { gap: 0.4rem !important; }
#more-options .tabs { margin-top: 0 !important; }
"""


def _resolve_input_type(label_or_key: str) -> str:
    if label_or_key in INPUT_TYPES:
        return label_or_key
    return LABEL_TO_INPUT_TYPE.get(label_or_key, "bvh_lafan1")


def _glob_for_input_type(input_type: str) -> str:
    exts = INPUT_TYPES[input_type]["extensions"]
    if len(exts) == 1:
        return f"**/*{exts[0]}"
    return "**/*"


def _explorer_root() -> Path:
    return DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.exists() else Path.home()


def _normalize_explorer_path(
    selection: str | list[str] | None,
    run_mode: str,
    root_dir: str | Path | None = None,
) -> str:
    """Resolve FileExplorer value (absolute path, relative path, or list) to a usable path."""
    if selection is None or selection == "":
        return ""
    path = selection[0] if isinstance(selection, list) else selection
    path = str(path).strip()
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        root = Path(root_dir) if root_dir else _explorer_root()
        p = (root / path).resolve()
    else:
        p = p.resolve()
    if run_mode == "batch" and p.is_file():
        return str(p.parent)
    return str(p)


def on_file_explorer_select(evt: gr.SelectData, run_mode: str) -> str:
    """FileExplorer fires `.select` when clicking a row; `.change` only fires on checkbox value updates."""
    if not evt.selected or not evt.value:
        return ""
    return _normalize_explorer_path(str(evt.value), run_mode)


def _native_file_dialog(input_label: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    input_type = _resolve_input_type(input_label)
    exts = INPUT_TYPES[input_type]["extensions"]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [(f"*{ext}", f"*{ext}") for ext in exts] + [("All files", "*.*")]
    initial = str(DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.exists() else REPO_ROOT)
    path = filedialog.askopenfilename(title="选择输入文件", initialdir=initial, filetypes=filetypes)
    root.destroy()
    return path or ""


def _native_dir_dialog() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = str(DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.exists() else REPO_ROOT)
    path = filedialog.askdirectory(title="选择输入目录", initialdir=initial)
    root.destroy()
    return path or ""


def _native_json_dialog() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择人体 JSON",
        initialdir=str(REPO_ROOT),
        filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return path or ""


def _native_save_pkl_dialog() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    out_dir = REPO_ROOT / "retargeting_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = filedialog.asksaveasfilename(
        title="保存机器人动作",
        initialdir=str(out_dir),
        defaultextension=".pkl",
        filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
    )
    root.destroy()
    return path or ""


def _resolve_retarget_algo(label_or_key: str) -> str:
    if label_or_key in RETARGET_ALGO_LABELS:
        return label_or_key
    return LABEL_TO_RETARGET_ALGO.get(label_or_key, "ik")


def _algo_choices_for_input(input_type: str) -> list[str]:
    choices = [RETARGET_ALGO_LABELS["ik"]]
    if supports_online_batch(input_type):
        choices.append(RETARGET_ALGO_LABELS["online_batch"])
    if supports_batch_to(input_type):
        choices.append(RETARGET_ALGO_LABELS["batch_to"])
    if supports_cpp_to(input_type):
        choices.append(RETARGET_ALGO_LABELS["cpp_batch_to"])
    return choices


def on_input_type_change(input_label: str, current_robot: str, current_algo: str):
    input_type = _resolve_input_type(input_label)
    robots = robots_for_input(input_type)
    value = current_robot if current_robot in robots else (robots[0] if robots else "unitree_g1")
    batch_choices = ["single", "batch"] if INPUT_TYPES[input_type].get("supports_batch") else ["single"]
    supports_contact = INPUT_TYPES[input_type].get("supports_contact", False)
    is_playback = input_type == "playback_pkl"
    needs_gvhmr = INPUT_TYPES[input_type].get("needs_gvhmr", False)
    contact_default = "开启" if input_type in ("gvhmr_pt", "video_gvhmr") else "IK 默认"
    batch_to_supported = supports_batch_to(input_type)
    algo_key = _resolve_retarget_algo(current_algo)
    if not batch_to_supported and algo_key == "batch_to":
        algo_key = "ik"
    if not supports_online_batch(input_type) and algo_key == "online_batch":
        algo_key = "ik"
    if not supports_cpp_to(input_type) and is_cpp_retarget_algo(algo_key):
        algo_key = "ik"
    algo_label = RETARGET_ALGO_LABELS[algo_key]
    algo_choices = _algo_choices_for_input(input_type)
    show_batch_panel = batch_to_supported and algo_key in ("batch_to", "cpp_batch_to")
    return (
        gr.Dropdown(choices=robots, value=value),
        gr.Radio(choices=batch_choices, value="single"),
        gr.Checkbox(value=not is_playback, interactive=not is_playback),
        gr.Checkbox(interactive=is_playback),
        gr.Checkbox(interactive=is_playback),
        gr.Dropdown(value=contact_default, interactive=supports_contact),
        gr.Dropdown(interactive=supports_contact),
        gr.Dropdown(interactive=supports_contact),
        gr.FileExplorer(glob=_glob_for_input_type(input_type)),
        gr.update(visible=needs_gvhmr),
        gr.update(value=default_gvhmr_python()),
        gr.update(open=needs_gvhmr),
        gr.Dropdown(choices=algo_choices, value=algo_label),
        gr.update(visible=show_batch_panel),
    )


def on_retarget_algo_change(algo_label: str, input_label: str):
    input_type = _resolve_input_type(input_label)
    algo_key = _resolve_retarget_algo(algo_label)
    show_batch_to = supports_batch_to(input_type) and algo_key in ("batch_to", "cpp_batch_to")
    return gr.update(visible=show_batch_to)


def make_config(
    input_label: str,
    run_mode: str,
    input_path: str,
    robot: str,
    retarget_algo_label: str,
    motion_fps,
    human_json_path: str,
    save_path: str,
    video_path: str,
    rate_limit: bool,
    loop: bool,
    save_output: bool,
    record_video: bool,
    show_ik_anchors: bool,
    show_body_names: bool,
    contact_ground: str,
    fix_robot_penetration: str,
    foot_ground_limit: str,
    gvhmr_root: str,
    gvhmr_python: str,
    gvhmr_static_cam: bool,
    batch_to_fast: bool,
    batch_to_window_size,
    batch_to_window_stride,
    batch_to_gn_steps,
) -> GMRRunConfig:
    fps = "30"
    if motion_fps is not None and motion_fps != "":
        fps = str(int(float(motion_fps)))
    return GMRRunConfig(
        input_type=_resolve_input_type(input_label),
        run_mode=run_mode,
        input_path=input_path or "",
        robot=robot,
        retarget_algo=_resolve_retarget_algo(retarget_algo_label),
        motion_fps=fps,
        human_json_path=human_json_path or "",
        save_path=save_path or "",
        video_path=video_path or "videos/gmr_gui_output.mp4",
        rate_limit=bool(rate_limit),
        loop=bool(loop),
        save_output=bool(save_output),
        record_video=bool(record_video),
        show_ik_anchors=bool(show_ik_anchors),
        show_body_names=bool(show_body_names),
        contact_ground=contact_ground,
        fix_robot_penetration=fix_robot_penetration,
        foot_ground_limit=foot_ground_limit,
        gvhmr_root=gvhmr_root or "",
        gvhmr_python=gvhmr_python or "",
        gvhmr_static_cam=bool(gvhmr_static_cam),
        batch_to_fast=bool(batch_to_fast),
        batch_to_window_size=int(float(batch_to_window_size or 16)),
        batch_to_window_stride=int(float(batch_to_window_stride or 8)),
        batch_to_gn_steps=int(float(batch_to_gn_steps or 3)),
    )


def preview_command(*args) -> str:
    cfg = make_config(*args)
    err = validate_config(cfg)
    if err:
        return f"# 参数错误: {err}"
    return shlex.join(build_command(cfg))


def stop_process(current_log: str) -> str:
    global _ACTIVE_PROC
    if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
        _ACTIVE_PROC.terminate()
        return (current_log or "") + "\n[terminated by user]\n"
    return current_log or ""


def run_stream(*args):
    global _ACTIVE_PROC
    if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
        yield "已有任务在运行，请先点击「停止」。"
        return

    cfg = make_config(*args)
    err = validate_config(cfg)
    if err:
        yield f"ERROR: {err}"
        return

    cmd = build_command(cfg)
    log = "$ " + shlex.join(cmd) + "\n\n"
    yield log

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    devel_lib = "/opt/robot/devel/lib"
    if Path(devel_lib).is_dir():
        env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    _ACTIVE_PROC = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert _ACTIVE_PROC.stdout is not None
    for line in _ACTIVE_PROC.stdout:
        log += line
        yield log
    code = _ACTIVE_PROC.wait()
    log += f"\n[exit code {code}]\n"
    yield log


def build_app() -> gr.Blocks:
    input_labels = [INPUT_TYPE_LABELS[k] for k in INPUT_TYPES]
    default_robots = robots_for_input("bvh_lafan1")
    explorer_root = str(_explorer_root())

    with gr.Blocks(title=GUI_APP_TITLE) as demo:
        gr.HTML(
            f"""
            <div id="gmr-header">
              <h1>{GUI_APP_TITLE}</h1>
              <p>运动重定向 · IK / Online Batch / Batch TO</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="config-panel"):
                with gr.Row():
                    input_type = gr.Dropdown(
                        choices=input_labels,
                        value=INPUT_TYPE_LABELS["bvh_lafan1"],
                        label="数据类型",
                        scale=2,
                    )
                    run_mode = gr.Radio(
                        ["single", "batch"],
                        value="single",
                        label="模式",
                        scale=1,
                    )
                    robot = gr.Dropdown(
                        choices=default_robots,
                        value="unitree_h2" if "unitree_h2" in default_robots else default_robots[0],
                        label="机器人",
                        scale=2,
                    )
                    retarget_algo = gr.Dropdown(
                        choices=list(RETARGET_ALGO_LABELS.values()),
                        value=RETARGET_ALGO_LABELS["ik"],
                        label="重定向算法",
                        scale=2,
                    )
                    motion_fps = gr.Number(
                        value=30, precision=0, label="FPS", minimum=1, maximum=240, scale=1,
                    )

                with gr.Row():
                    input_path = gr.Textbox(
                        label="输入路径",
                        placeholder="绝对路径，或下方浏览 / 系统文件对话框",
                        scale=5,
                    )
                    pick_file_btn = gr.Button("📁 文件", scale=1, min_width=80)
                    pick_dir_btn = gr.Button("📂 目录", scale=1, min_width=80)

                with gr.Row():
                    run_btn = gr.Button("▶ 运行", variant="primary", scale=3, elem_id="run-btn")
                    stop_btn = gr.Button("停止", variant="stop", scale=1)
                    preview_btn = gr.Button("预览命令", scale=1)

                with gr.Accordion("更多选项", open=False, elem_id="more-options") as more_options:
                    with gr.Tabs():
                        with gr.Tab("输出"):
                            with gr.Row():
                                rate_limit = gr.Checkbox(value=True, label="实时限速")
                                loop = gr.Checkbox(value=True, label="循环播放")
                                save_output = gr.Checkbox(value=False, label="保存 PKL")
                                record_video = gr.Checkbox(value=False, label="录制视频")
                            with gr.Row():
                                save_path = gr.Textbox(
                                    label="PKL 路径", placeholder="retargeting_data/xxx.pkl", scale=4,
                                )
                                pick_save_btn = gr.Button("浏览", scale=1, min_width=60)
                            video_path = gr.Textbox(label="录制视频路径", value="videos/gmr_gui_output.mp4")

                        with gr.Tab("接触 / 地面"):
                            with gr.Row():
                                contact_ground = gr.Dropdown(TRI_STATE_LABELS, value="开启", label="接触对齐")
                                fix_robot_penetration = gr.Dropdown(TRI_STATE_LABELS, value="开启", label="修复穿地")
                                foot_ground_limit = gr.Dropdown(TRI_STATE_LABELS, value="IK 默认", label="脚地 QP")

                        with gr.Tab("Batch TO", visible=False) as batch_to_panel:
                            gr.Markdown(
                                "Python **Batch TO** 与 **C++ Batch TO** 共用窗口/GN 参数；"
                                "C++ 从 `.pt` / `.npz` / `.bvh` 一键加载：先在终端完成优化，再打开 MuJoCo 回放。"
                                "长 BVH 可先截短或降低帧数。"
                            )
                            batch_to_fast = gr.Checkbox(
                                value=False,
                                label="Fast 档 (--fast)",
                                info="gn_steps=2, 单 alpha, window 16/8",
                            )
                            with gr.Row():
                                batch_to_window_size = gr.Number(
                                    value=16, precision=0, minimum=4, maximum=64, label="window_size",
                                )
                                batch_to_window_stride = gr.Number(
                                    value=8, precision=0, minimum=1, maximum=32, label="window_stride",
                                )
                                batch_to_gn_steps = gr.Number(
                                    value=3, precision=0, minimum=1, maximum=10, label="gn_steps",
                                )

                        with gr.Tab("GVHMR"):
                            with gr.Column(visible=False) as gvhmr_panel:
                                gvhmr_root = gr.Textbox(
                                    label="GVHMR 根目录",
                                    value=str(DEFAULT_GVHMR_ROOT),
                                )
                                gvhmr_python = gr.Textbox(
                                    label="GVHMR Python",
                                    value=default_gvhmr_python(),
                                )
                                gvhmr_static_cam = gr.Checkbox(
                                    value=True,
                                    label="静态相机 (-s)",
                                )

                        with gr.Tab("回放 (PKL/JSON)"):
                            with gr.Row():
                                human_json_path = gr.Textbox(
                                    label="人体 JSON", placeholder="IK 锚点用", scale=4,
                                )
                                pick_json_btn = gr.Button("浏览", scale=1, min_width=60)
                            with gr.Row():
                                show_ik_anchors = gr.Checkbox(value=False, label="显示 IK 锚点")
                                show_body_names = gr.Checkbox(value=False, label="显示部位名称")

                        with gr.Tab("文件浏览"):
                            file_explorer = gr.FileExplorer(
                                glob="**/*.bvh",
                                file_count="single",
                                root_dir=explorer_root,
                                label="本地文件树（点击文件名填入路径）",
                                height=160,
                            )

            with gr.Column(scale=4):
                log_box = gr.Textbox(
                    label="终端输出",
                    lines=32,
                    max_lines=48,
                    interactive=False,
                    elem_id="log-box",
                )
                with gr.Accordion("命令预览", open=False):
                    cmd_preview = gr.Code(label=None, language="shell", interactive=False, lines=4)

        widget_args = [
            input_type,
            run_mode,
            input_path,
            robot,
            retarget_algo,
            motion_fps,
            human_json_path,
            save_path,
            video_path,
            rate_limit,
            loop,
            save_output,
            record_video,
            show_ik_anchors,
            show_body_names,
            contact_ground,
            fix_robot_penetration,
            foot_ground_limit,
            gvhmr_root,
            gvhmr_python,
            gvhmr_static_cam,
            batch_to_fast,
            batch_to_window_size,
            batch_to_window_stride,
            batch_to_gn_steps,
        ]

        input_type.change(
            fn=on_input_type_change,
            inputs=[input_type, robot, retarget_algo],
            outputs=[
                robot,
                run_mode,
                loop,
                show_ik_anchors,
                show_body_names,
                contact_ground,
                fix_robot_penetration,
                foot_ground_limit,
                file_explorer,
                gvhmr_panel,
                gvhmr_python,
                more_options,
                retarget_algo,
                batch_to_panel,
            ],
        )

        retarget_algo.change(
            fn=on_retarget_algo_change,
            inputs=[retarget_algo, input_type],
            outputs=[batch_to_panel],
        )

        pick_file_btn.click(fn=_native_file_dialog, inputs=[input_type], outputs=input_path)
        pick_dir_btn.click(fn=_native_dir_dialog, outputs=input_path).then(
            lambda p: ("batch" if p else "single"),
            inputs=[input_path],
            outputs=run_mode,
        )
        _explorer_event_kwargs = dict(show_progress="hidden", queue=False)
        file_explorer.select(
            fn=on_file_explorer_select,
            inputs=[run_mode],
            outputs=input_path,
            **_explorer_event_kwargs,
        )
        file_explorer.input(
            fn=_normalize_explorer_path,
            inputs=[file_explorer, run_mode, gr.State(explorer_root)],
            outputs=input_path,
            **_explorer_event_kwargs,
        )
        file_explorer.change(
            fn=_normalize_explorer_path,
            inputs=[file_explorer, run_mode, gr.State(explorer_root)],
            outputs=input_path,
            **_explorer_event_kwargs,
        )
        pick_save_btn.click(fn=_native_save_pkl_dialog, outputs=save_path)
        pick_json_btn.click(fn=_native_json_dialog, outputs=human_json_path)

        preview_btn.click(fn=preview_command, inputs=widget_args, outputs=cmd_preview)
        run_btn.click(fn=run_stream, inputs=widget_args, outputs=log_box)
        stop_btn.click(fn=stop_process, inputs=[log_box], outputs=log_box)

    return demo


def main() -> None:
    os.chdir(REPO_ROOT)
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#f1f5f9",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(15,23,42,0.06)",
        button_large_padding="12px 20px",
    )
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
