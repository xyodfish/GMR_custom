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

python scripts/retarget/bvh_to_robot.py --bvh_file ... --robot unitree_g1
python scripts/retarget/bvh_to_robot_trajectory_opt.py --bvh_file ... --robot unitree_g1
python scripts/retarget/smplx_to_robot.py --smplx_file ... --robot unitree_g1
python scripts/retarget/smplx_to_robot_sliding_window.py --smplx_file ... --robot unitree_g1
python scripts/gvhmr/to_robot.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/gvhmr/to_robot_sliding_window.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/viz/vis_robot_motion.py --robot ... --robot_motion_path ...
python scripts/analysis/batch_lafan1_retarget_compare.py --bvh_dir ... --robot unitree_g1
python scripts/analysis/compare_joint_trajectories.py --smplx_file ... --robot unitree_g1
python scripts/analysis/benchmark_retarget_timing.py --gvhmr_pred_file ... --robot unitree_g1
```

接地参数：`--contact_ground --foot_ground_limit`。详见 [`docs/contact_ground.md`](../docs/contact_ground.md)。
