# Scripts 目录说明

CLI 脚本按功能分子目录组织。

```
scripts/
├── retarget/          # 人体动作 → 机器人重定向
├── gvhmr/             # GVHMR .pt / 单目视频流水线
├── viz/               # 机器人动作可视化
├── analysis/          # 接触/地面模式分析
├── tools/             # 格式转换、预处理、导出
├── gmr_gui.py         # GUI 入口（转发到 general_motion_retargeting.gui）
├── _paths.py          # 仓库路径常量
└── README.md
```

## 常用命令

```bash
# GUI
python scripts/gmr_gui.py
# 或: gmr-gui
# 或: python -m general_motion_retargeting.gui.app

python scripts/retarget/bvh_to_robot.py --bvh_file ... --robot unitree_h2
python scripts/retarget/smplx_to_robot.py --smplx_file ... --robot unitree_g1
python scripts/gvhmr/to_robot.py --gvhmr_pred_file ... --robot unitree_h2
python scripts/gvhmr/video_to_robot.py --video ... --robot unitree_h2
python scripts/viz/vis_robot_motion.py --robot ... --robot_motion_path ...
python scripts/analysis/analyze_contact_modes.py ...
```

## 库代码

| 功能 | 路径 |
|------|------|
| GUI 逻辑 | `general_motion_retargeting/gui/core.py` |
| GUI 界面 | `general_motion_retargeting/gui/app.py` |
| GVHMR 环境检测 | `general_motion_retargeting/utils/gvhmr_env.py` |
