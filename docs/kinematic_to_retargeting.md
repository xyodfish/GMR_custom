# 运动学 TO 重定向

独立的 q-space / task-space 短片段实验脚本已移除。

当前离线轨迹优化请使用：

- 核心类：`general_motion_retargeting/trajectory_optimization_retarget.py`
- CLI：`scripts/retarget/smplx_to_robot_trajectory_opt.py`、`scripts/gvhmr/to_robot_trajectory_opt.py`
- 文档：[`trajectory_optimization_retargeting.md`](trajectory_optimization_retargeting.md)

因果 sliding-window 见 [`sliding_window_retargeting.md`](sliding_window_retargeting.md)（仅 SMPL-X / GVHMR 入口保留）。
