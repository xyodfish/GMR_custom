# Scripts 目录说明

```
scripts/
├── retarget/          # 人体动作 → 机器人重定向
├── gvhmr/             # GVHMR .pt / 单目视频流水线
├── viz/               # 机器人动作可视化
├── analysis/          # 指标对比、benchmark
├── tools/             # 格式转换、导出
├── gmr_gui.py         # GUI 入口
└── _paths.py
```

## 常用命令

```bash
python scripts/gmr_gui.py
```

`gmr_gui` 可在界面选择 **Per-frame IK**、**Online Batch**、**Batch TO (Python/C++)** 等（BVH / SMPL-X / GVHMR / 视频）。应用标题常量：`general_motion_retargeting/gui/core.py` → `GUI_APP_TITLE`。

```bash
python scripts/retarget/bvh_to_robot.py --bvh_file ... --robot unitree_g1
python scripts/retarget/smplx_to_robot.py --smplx_file ... --robot unitree_g1
python scripts/retarget/to_robot_batch.py --input_file ... --robot unitree_g1
python scripts/gvhmr/to_robot.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/gvhmr/to_robot_online_batch.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/gvhmr/to_robot_online_qp.py --input_file ... --robot unitree_g1
python scripts/viz/vis_robot_motion.py --robot ... --robot_motion_path ...
python scripts/analysis/compare_joint_trajectories.py --baseline ... --candidate ... --robot unitree_g1
python scripts/analysis/benchmark_gvhmr_retarget_methods.py --pt_glob 'data/gvhmr_test_videos/*/hmr4d_results.pt'
```

接地参数：`--contact_ground --foot_ground_limit`。详见 [`docs/contact_ground.md`](../docs/contact_ground.md)。
