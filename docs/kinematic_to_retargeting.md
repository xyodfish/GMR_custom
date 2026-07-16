# 运动学 TO 重定向

当前运动学 TO 路线：

| 场景 | 核心类 | 文档 |
|------|--------|------|
| **离线 batch 多帧 GN（推荐离线质量）** | `batch_trajectory_retarget.py` | [`batch_trajectory_retargeting.md`](batch_trajectory_retargeting.md) |
| **在线 batch-lite（推荐在线）** | `online_batch_retarget.py` | [`online_batch_retargeting.md`](online_batch_retargeting.md) |
| **在线 QP-MPC** | `online_qp_retarget.py` | [`online_batch_retargeting.md`](online_batch_retargeting.md)（Online QP 节） |
| 共享 FK / 边界工具（非独立算法） | `trajectory_optimization_retarget.py` | Batch TO 内部基类 |

CLI 入口：

- Batch TO：`scripts/retarget/to_robot_batch.py`；C++ `cpp/build/gmr_batch_to_cli`
- Online Batch-Lite：`scripts/gvhmr/to_robot_online_batch.py`；demo `scripts/tools/demo_online_batch.sh`
- Online QP：`scripts/gvhmr/to_robot_online_qp.py`；C++ `gmr_online_qp_cli` / viewer `--method online_qp`
