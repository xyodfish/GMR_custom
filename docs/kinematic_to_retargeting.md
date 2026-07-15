# 运动学 TO 重定向

独立的 q-space / task-space 短片段实验脚本已移除。

当前运动学 TO 路线：

| 场景 | 核心类 | 文档 |
|------|--------|------|
| **离线 batch 多帧 GN（推荐离线质量）** | `batch_trajectory_retarget.py` | [`batch_trajectory_retargeting.md`](batch_trajectory_retargeting.md) |
| 因果在线 TO | `trajectory_optimization_retarget.py` | [`trajectory_optimization_retargeting.md`](trajectory_optimization_retargeting.md) |
| 因果 sliding-window（GMR FrameTask） | `sliding_window_retarget.py` | [`sliding_window_retargeting.md`](sliding_window_retargeting.md) |

CLI 入口：

- Batch TO：`scripts/gvhmr/to_robot_batch.py`；C++ `cpp/build/gmr_batch_to_cli`
- 因果 TO：`scripts/gvhmr/to_robot_trajectory_opt.py`、`scripts/retarget/smplx_to_robot_trajectory_opt.py`
