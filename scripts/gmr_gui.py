#!/usr/bin/env python3
"""GMR retargeting GUI — Gradio web interface."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

from scripts.gmr_gui_core import (
    GMRRunConfig,
    INPUT_TYPE_LABELS,
    INPUT_TYPES,
    LABEL_TO_INPUT_TYPE,
    REPO_ROOT,
    TRI_STATE_LABELS,
    build_command,
    robots_for_input,
    validate_config,
)

_ACTIVE_PROC: subprocess.Popen | None = None

DEFAULT_BROWSE_ROOT = Path("/home/xiayu/Workspace/data/lafan1")
if not DEFAULT_BROWSE_ROOT.exists():
    DEFAULT_BROWSE_ROOT = Path.home()

CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#gmr-header { text-align: left; margin-bottom: 0.25rem; }
#gmr-header h1 { font-size: 1.75rem; font-weight: 700; margin: 0; }
#gmr-header p { color: #64748b; margin: 0.25rem 0 0 0; }
#run-btn { min-height: 3rem; font-size: 1.05rem !important; }
#log-box textarea {
    font-family: "JetBrains Mono", "Consolas", "Menlo", monospace !important;
    font-size: 0.85rem !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
}
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


def on_input_type_change(input_label: str, current_robot: str):
    input_type = _resolve_input_type(input_label)
    robots = robots_for_input(input_type)
    value = current_robot if current_robot in robots else (robots[0] if robots else "unitree_g1")
    batch_choices = ["single", "batch"] if INPUT_TYPES[input_type].get("supports_batch") else ["single"]
    supports_contact = INPUT_TYPES[input_type].get("supports_contact", False)
    is_playback = input_type == "playback_pkl"
    return (
        gr.Dropdown(choices=robots, value=value),
        gr.Radio(choices=batch_choices, value="single"),
        gr.Checkbox(value=not is_playback, interactive=not is_playback),
        gr.Checkbox(interactive=is_playback),
        gr.Checkbox(interactive=is_playback),
        gr.Dropdown(interactive=supports_contact),
        gr.Dropdown(interactive=supports_contact),
        gr.Dropdown(interactive=supports_contact),
        gr.FileExplorer(glob=_glob_for_input_type(input_type)),
    )


def make_config(
    input_label: str,
    run_mode: str,
    input_path: str,
    robot: str,
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
) -> GMRRunConfig:
    fps = "30"
    if motion_fps is not None and motion_fps != "":
        fps = str(int(float(motion_fps)))
    return GMRRunConfig(
        input_type=_resolve_input_type(input_label),
        run_mode=run_mode,
        input_path=input_path or "",
        robot=robot,
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

    with gr.Blocks(title="GMR Retargeting") as demo:
        gr.HTML(
            """
            <div id="gmr-header">
              <h1>GMR Retargeting</h1>
              <p>运动重定向调试台 · 子进程调用 scripts/ 现有流程 · 点击浏览或左侧文件树选择路径</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### 输入")
                    input_type = gr.Dropdown(
                        choices=input_labels,
                        value=INPUT_TYPE_LABELS["bvh_lafan1"],
                        label="数据类型",
                    )
                    run_mode = gr.Radio(["single", "batch"], value="single", label="运行模式")
                    with gr.Row():
                        pick_file_btn = gr.Button("📁 选择文件", scale=1)
                        pick_dir_btn = gr.Button("📂 选择目录", scale=1)
                    input_path = gr.Textbox(
                        label="已选路径",
                        placeholder="点击上方按钮或下方文件树选择",
                        interactive=True,
                    )
                    explorer_root = str(_explorer_root())
                    file_explorer = gr.FileExplorer(
                        glob="**/*.bvh",
                        file_count="single",
                        root_dir=explorer_root,
                        label="浏览本地文件（点击文件名填入路径）",
                        height=220,
                    )

                with gr.Group():
                    gr.Markdown("### 机器人")
                    with gr.Row():
                        robot = gr.Dropdown(
                            choices=default_robots,
                            value="unitree_h2" if "unitree_h2" in default_robots else default_robots[0],
                            label="型号",
                        )
                        motion_fps = gr.Number(value=30, precision=0, label="FPS", minimum=1, maximum=240)

                with gr.Accordion("运行选项", open=True):
                    with gr.Row():
                        rate_limit = gr.Checkbox(value=True, label="实时限速")
                        loop = gr.Checkbox(value=True, label="循环播放")
                        save_output = gr.Checkbox(value=False, label="保存 PKL")
                        record_video = gr.Checkbox(value=False, label="录制视频")
                    save_path = gr.Textbox(label="保存路径 (.pkl)", placeholder="retargeting_data/xxx.pkl")
                    pick_save_btn = gr.Button("浏览保存位置")
                    video_path = gr.Textbox(label="视频路径", value="videos/gmr_gui_output.mp4")

                with gr.Accordion("接触 / 地面", open=True):
                    with gr.Row():
                        contact_ground = gr.Dropdown(TRI_STATE_LABELS, value="开启", label="接触对齐")
                        fix_robot_penetration = gr.Dropdown(TRI_STATE_LABELS, value="开启", label="修复穿地")
                        foot_ground_limit = gr.Dropdown(TRI_STATE_LABELS, value="IK 默认", label="脚地 QP")

                with gr.Accordion("回放叠加（仅 PKL）", open=False):
                    with gr.Row():
                        human_json_path = gr.Textbox(label="人体 JSON 路径", placeholder="可选，用于 IK 锚点", scale=4)
                        pick_json_btn = gr.Button("浏览 JSON", scale=1)
                    with gr.Row():
                        show_ik_anchors = gr.Checkbox(value=False, label="显示 IK 锚点")
                        show_body_names = gr.Checkbox(value=False, label="显示部位名称")

                with gr.Row():
                    run_btn = gr.Button("运行 Retargeting", variant="primary", scale=2, elem_id="run-btn")
                    stop_btn = gr.Button("停止", variant="stop", scale=1)
                    preview_btn = gr.Button("预览命令", scale=1)

            with gr.Column(scale=4):
                log_box = gr.Textbox(
                    label="终端输出",
                    lines=28,
                    max_lines=40,
                    interactive=False,
                    elem_id="log-box",
                )
                cmd_preview = gr.Code(label="命令预览", language="shell", interactive=False)

        widget_args = [
            input_type,
            run_mode,
            input_path,
            robot,
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
        ]

        input_type.change(
            fn=on_input_type_change,
            inputs=[input_type, robot],
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
            ],
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

        gr.Markdown(
            "---\n**说明**：MuJoCo 窗口关闭后任务结束；批量模式无可视化。"
        )

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
