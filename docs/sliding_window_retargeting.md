# 滑动窗口运动学 TO 重定向

本文记录 `SlidingWindowRetargeter` 的设计、用法和评估方法。

## 定位

在 GMR 逐帧 IK 之上，增加**因果滑动窗口**运动学优化，降低关节 jitter，同时保持实时性。

```text
人体帧流
  -> GMR 预处理 (scale / contact_ground)
  -> 轻量 IK warm start (上一帧 qpos)
  -> 单帧 L-BFGS-B 平滑优化 (fast 模式)
  -> 输出 qpos[t]
```

**不是**完整动力学 TO/MPC；无真机反馈时，用上一帧优化后的 `qpos` 作为 kinematic state feedback。

## 两种模式

| 模式 | 说明 | 典型耗时 (G1) |
|------|------|---------------|
| `fast` (默认) | 只优化当前帧，历史帧提供速度/加速度先验 | ~20 ms/帧 |
| `full` | 联合优化整个窗口 (L-BFGS-B) | ~秒级/帧 |

## 快速开始

### SMPL-X / AMASS

```bash
python scripts/retarget/smplx_to_robot_sliding_window.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/motion_sw.pkl
```

### GVHMR

```bash
python scripts/gvhmr/to_robot_sliding_window.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/walking_sw.pkl
```

### Python API

```python
from general_motion_retargeting import (
    GeneralMotionRetargeting,
    SlidingWindowRetargeter,
    SlidingWindowConfig,
)

gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1")
sw = SlidingWindowRetargeter(gmr, SlidingWindowConfig(window_size=8))

for human_frame in stream:
    qpos = sw.retarget(human_frame)
```

新序列开始前调用 `sw.reset()`。

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `window_size` | 8 | 历史窗口长度 (帧) |
| `mode` | `fast` | `fast` 或 `full` |
| `w_velocity` | 2.0 | 速度平滑权重 |
| `w_acceleration` | 10.0 | 加速度平滑权重 |
| `ik_warmstart_iters` | 3 | fast 模式 IK 预热步数 |
| `fast_opt_iter` | 5 | fast 模式 L-BFGS-B 迭代上限 |

## 与旧 GMR 对比

### 1. 耗时 benchmark（不含可视化）

```bash
python scripts/analysis/benchmark_retarget_timing.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground
```

### 2. 关节轨迹曲线

```bash
python scripts/analysis/compare_joint_trajectories.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --output output/joint_compare.png
```

生成 position / velocity / acceleration 三组对比图；重点看 **acceleration** 曲线上的 jitter。

### 3. 已保存 motion 的平滑度 + 脚滑指标

```bash
python scripts/analysis/analyze_saved_motion_metrics.py \
  --robot unitree_g1 \
  --motion output/walking_ik.pkl \
  --compare output/walking_sw.pkl \
  --labels "per-frame IK" "sliding-window"
```

### 4. 并排视频

录视频时使用固定相机（`RobotMotionViewer` 录制不再读取交互式 viewer.cam）：

```bash
python scripts/gvhmr/to_robot.py \
  --gvhmr_pred_file ... --contact_ground \
  --record_video --video_path videos/walking_ik.mp4

python scripts/gvhmr/to_robot_sliding_window.py \
  --gvhmr_pred_file ... --contact_ground \
  --record_video --video_path videos/walking_sw.mp4

python scripts/analysis/stitch_videos_side_by_side.py \
  --left videos/walking_ik.mp4 \
  --right videos/walking_sw.mp4 \
  --output videos/walking_compare.mp4
```

## 适用场景

- **收益有限**：ACCAD 等高质量 mocap + `contact_ground` 已开 — 全局指标往往接近
- **更有价值**：GVHMR 单目视频、实时遥操作、上身 jitter 明显、未开 contact_ground 时

per-joint acceleration 曲线比全局 vel/acc 均值更能看出差异。

## 代码位置

| 组件 | 路径 |
|------|------|
| 核心类 | `general_motion_retargeting/sliding_window_retarget.py` |
| SMPL-X CLI | `scripts/retarget/smplx_to_robot_sliding_window.py` |
| GVHMR CLI | `scripts/gvhmr/to_robot_sliding_window.py` |
| 耗时 benchmark | `scripts/analysis/benchmark_retarget_timing.py` |
| 关节曲线对比 | `scripts/analysis/compare_joint_trajectories.py` |
| motion 指标分析 | `scripts/analysis/analyze_saved_motion_metrics.py` |
