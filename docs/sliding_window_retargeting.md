# 滑动窗口运动学 TO 重定向

本文档描述 GMR 中 **SlidingWindowRetargeter** 的算法原理、使用方法，以及与旧版逐帧 IK 的性能/效果对比。

> **注意**：若需要**脱离 GMR 逐帧 IK 的独立因果 TO**（直接 FK 跟踪 + 平滑，优化器内不调用 mink），请参见 [`trajectory_optimization_retargeting.md`](trajectory_optimization_retargeting.md) 中的 `TrajectoryOptimizationRetargeter`。  
> 若需要 **离线整段 motion 多帧联合 GN**（paper-style batch retargeting），请参见 [`batch_trajectory_retargeting.md`](batch_trajectory_retargeting.md) 中的 `BatchTrajectoryRetargeter`。

---

## 1. 背景与定位

### 1.1 旧 GMR 在做什么

旧版 GMR 对每一帧人体数据独立求解逆运动学（IK）：

```text
人体帧 t  →  scale / contact_ground  →  mink 两阶段 IK  →  qpos[t]
```

本质是**瞬时优化**（velocity-level QP），帧与帧之间没有显式时序约束，仅通过上一帧 `qpos` 做 warm start。

### 1.2 滑动窗口 TO 在做什么

在 IK 之上增加**因果滑动窗口**运动学轨迹优化，对当前帧输出做时序平滑：

```text
人体帧流
  → GMR 预处理 (scale / contact_ground)
  → 上一帧 qpos 作为 kinematic feedback（无真机时用模型内部状态）
  → 轻量 IK warm start
  → 单帧平滑优化 (fast 模式) 或 窗口联合优化 (full 模式)
  → 输出 qpos[t]
```

**不是**完整动力学 TO/MPC（无力矩约束、无 MuJoCo rollout）。与飞书 Know-How 中「重定向做成 TO」的关系：

- 当前实现：**运动学 TO**（kinematic TO），在 q 空间做短时域平滑
- 进阶方向：动力学 TO / 实时 MPC，需接 Pinocchio + OCS2/WBC 等运控栈

### 1.3 无真机时的 feedback

没有硬件编码器时，用**上一帧优化后的 `qpos`**（存在 MuJoCo `configuration` 中）作为状态反馈，即 open-loop receding horizon + model-based warm start。

---

## 2. 算法描述

### 2.1 优化变量

| 模式 | 优化变量 | 维度 (G1 例) |
|------|----------|--------------|
| `fast`（默认） | 仅当前帧 `q[t]` | 36 |
| `full` | 窗口内 `q[t-H+1], …, q[t]` | H × 36 |

### 2.2 目标函数（fast 模式）

在 IK warm start 初值附近，最小化：

```text
J(q_t) = || IK_error(q_t, human_t) ||²
       + w_v  || q_t - q_{t-1} ||²
       + w_a  || q_t - 2 q_{t-1} + q_{t-2} ||²
```

- **IK_error**：与 GMR 相同的 `FrameTask` 跟踪误差（沿用 `ik_match_table`）
- **w_v, w_a**：速度、加速度平滑项，抑制 jitter

约束：MuJoCo 关节限位（hinge/slide）。

求解器：L-BFGS-B，`fast_opt_iter=5`（默认）。

### 2.3 目标函数（full 模式）

对窗口内 H 帧联合优化：

```text
min  Σ_t || IK_error(q_t) ||²
   + w_v Σ || q_{t+1} - q_t ||²
   + w_a Σ || q_{t+2} - 2 q_{t+1} + q_t ||²
   + w_anchor || q_0 - q_0^prev ||²
```

变量维度 H×nq，L-BFGS-B 数值梯度，**非常慢**（~秒级/帧），仅用于离线实验。

### 2.4 处理流程（fast，每帧）

```mermaid
flowchart TD
    A[人体帧 t] --> B[GMR 预处理]
    B --> C[q_init = q_{t-1}]
    C --> D[设置 IK 目标]
    D --> E[轻量 IK: 3 步 x 两阶段]
    E --> F[L-BFGS-B 优化 q_t]
    F --> G[planar base / 穿地修正]
    G --> H[输出 qpos[t]]
    H --> I[写入 buffer 供下一帧]
```

---

## 3. 使用方法

### 3.1 依赖

```bash
conda activate gmr
pip install matplotlib   # 关节曲线对比脚本需要
```

### 3.2 SMPL-X / ACCAD

```bash
python scripts/retarget/smplx_to_robot_sliding_window.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/motion_sw.pkl
```

与旧 IK 对比（同脚本内双跑 + 打印全局 vel/acc）：

```bash
python scripts/retarget/smplx_to_robot_sliding_window.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --contact_ground \
  --compare_ik \
  --save_path output/motion_sw.pkl
```

### 3.3 GVHMR

```bash
python scripts/gvhmr/to_robot_sliding_window.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/walking_sw.pkl
```

GVHMR 建议始终开启 `--contact_ground`，修正单目 root 漂移和脚浮空。

### 3.4 Python API

```python
from general_motion_retargeting import (
    GeneralMotionRetargeting,
    SlidingWindowRetargeter,
    SlidingWindowConfig,
)

gmr = GeneralMotionRetargeting(
    src_human="smplx",
    tgt_robot="unitree_g1",
    contact_ground=True,
)
sw = SlidingWindowRetargeter(
    gmr,
    SlidingWindowConfig(
        window_size=8,
        mode="fast",
        w_velocity=2.0,
        w_acceleration=10.0,
    ),
)

for human_frame in stream:
    qpos = sw.retarget(human_frame)

# 新序列开始前
sw.reset()
```

### 3.5 主要参数

| 参数 | CLI 名 | 默认 | 说明 |
|------|--------|------|------|
| `window_size` | `--window_size` | 8 | 历史缓冲长度（帧） |
| `mode` | `--mode` | `fast` | `fast` 或 `full` |
| `w_velocity` | `--w_velocity` | 2.0 | 速度平滑权重 |
| `w_acceleration` | `--w_acceleration` | 10.0 | 加速度平滑权重 |
| `ik_warmstart_iters` | `--ik_warmstart_iters` | 3 | fast 模式 IK 预热迭代数 |
| `fast_opt_iter` | `--fast_opt_iter` | 5 | fast 模式 L-BFGS-B 上限 |
| `max_opt_iter` | `--max_opt_iter` | 25 | full 模式 L-BFGS-B 上限 |

加速建议：`--fast_opt_iter 3 --ik_warmstart_iters 3` → ~60 ms/帧。  
更强平滑：`--w_velocity 10 --w_acceleration 50`。

---

## 4. 评估方法

### 4.1 耗时 benchmark（纯 retarget，无可视化）

```bash
python scripts/analysis/benchmark_retarget_timing.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground
```

输出：mean / median / p95 / min / max（ms），等效 FPS，是否满足 30 fps 实时。

### 4.2 关节轨迹曲线（推荐）

```bash
python scripts/analysis/compare_joint_trajectories.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --output output/joint_compare.png
```

生成三张图：

- `*_position.png` — 关节位置
- `*_velocity.png` — 关节速度
- `*_acceleration.png` — 关节加速度（**看 jitter 最有用**）

实线 = 旧 IK，虚线 = sliding-window。默认绘制差异最大的 top-12 关节。

### 4.3 已保存 motion 指标

```bash
python scripts/analysis/analyze_saved_motion_metrics.py \
  --robot unitree_g1 \
  --motion output/walking_ik.pkl \
  --compare output/walking_sw.pkl \
  --labels "per-frame IK" "sliding-window"
```

输出：dq / ddq / jerk 范数统计，脚部 contact 期间 slip 距离。

### 4.4 可视化 retargeting 效果

```bash
# 旧 IK（人体锚点 + 机器人）
python scripts/gvhmr/to_robot.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 --contact_ground --no-rate-limit

# sliding-window
python scripts/gvhmr/to_robot_sliding_window.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 --contact_ground --no-rate-limit
```

MuJoCo 窗口中：小坐标轴 = 人体 IK 目标，机器人 = 重定向结果。

### 4.5 并排视频

录制时使用固定相机（不读交互 viewer.cam），两脚本统一 `follow_camera=True`：

```bash
python scripts/gvhmr/to_robot.py \
  --gvhmr_pred_file ... --contact_ground \
  --record_video --video_path videos/walking_ik.mp4 --no-rate-limit

python scripts/gvhmr/to_robot_sliding_window.py \
  --gvhmr_pred_file ... --contact_ground \
  --record_video --video_path videos/walking_sw.mp4 --no-rate-limit

python scripts/analysis/stitch_videos_side_by_side.py \
  --left videos/walking_ik.mp4 \
  --right videos/walking_sw.mp4 \
  --output videos/walking_compare.mp4
```

---

## 5. 性能对比（实测）

测试环境：GMR conda 环境，unitree_g1，开启 `contact_ground`。  
benchmark 脚本不含 MuJoCo 窗口渲染。

### 5.1 耗时（GVHMR walking，367 帧，fast 模式）

| 指标 | 旧 IK | sliding-window (fast) | 倍数 |
|------|-------|------------------------|------|
| mean | **1.93 ms** | **22.07 ms** | 11.4× 慢 |
| median | 1.86 ms | 22.01 ms | |
| p95 | 2.58 ms | 25.18 ms | |
| max | 4.14 ms | 30.05 ms | |
| 等效 FPS | ~518 | ~45 | |
| 362 帧总耗时 | 0.70 s | 7.99 s | |

**实时性（30 fps 需 ≤ 33.3 ms/帧）：两者均满足。**

| 模式 | mean 耗时 | 能否 30fps 实时 |
|------|-----------|-----------------|
| 旧 IK | ~2 ms | 是 |
| SW fast | ~22 ms | 是 |
| SW full | ~10 s/帧 | 否 |

> 注：早期 full 模式 + 可视化 + `--compare_ik` 叠加可达 ~2600 ms/帧，不代表 fast 模式性能。

### 5.2 全局平滑度指标

全局 mean joint velocity / acceleration 范数（`--compare_ik` 输出）：

**ACCAD EricCamper04**（高质量 mocap）：

| 方法 | vel | acc |
|------|-----|-----|
| sliding-window | 0.18124 | 0.09128 |
| 旧 IK | 0.18205 | 0.09833 |
| 变化 | -0.4% | -7% |

**GVHMR walking**（单目视频）：

| 方法 | vel | acc |
|------|-----|-----|
| sliding-window | 0.29570 | 0.17122 |
| 旧 IK | 0.29962 | 0.19425 |
| 变化 | -1.3% | **-12%** |

结论：**全局均值差异很小**，GVHMR 上 acc 改善略大，但肉眼整体动作仍接近。

### 5.3 逐关节对比（GVHMR walking，acceleration std）

| 关节 | acc_std IK | acc_std SW | 改善 |
|------|------------|------------|------|
| right_shoulder_roll_joint | 39.2 | 28.0 | **~28%** |
| right_elbow_joint | 39.5 | 27.5 | **~30%** |
| right_shoulder_yaw_joint | 37.9 | 27.2 | **~28%** |
| left_shoulder_yaw_joint | 26.8 | 25.5 | ~5% |
| left_shoulder_roll_joint | 23.1 | 25.1 | SW 略差 |

结论：**差异主要体现在手臂关节的 jitter（acceleration）**；腿和 root 差别不大。  
**per-joint 曲线比全局均值更能反映实际收益。**

### 5.4 效果 vs 耗时权衡

| | 旧 IK | SW fast |
|--|-------|---------|
| 速度 | 极快 (~2 ms) | 快 (~22 ms) |
| 全局平滑度 | baseline | 略好 |
| 手臂 jitter | baseline | 明显更好 |
| 整体动作形态 | baseline | 几乎相同 |
| 实现复杂度 | 低 | 中 |

---

## 6. 结论与建议

### 6.1 何时用 sliding-window

**建议使用：**

- GVHMR / 实时遥操作等输入噪声较大
- 上身/手臂 jitter 在 acceleration 曲线上明显
- 未开 `contact_ground`，需要额外时序平滑

**可不使用：**

- ACCAD 等高质量 mocap + 已开 `contact_ground`（收益 < 5%）
- 只关心腿部 locomotion，不关心手臂
- 极致低延迟（旧 IK 快 11 倍）

### 6.2 与 contact_ground 的关系

`contact_ground` 已做流式接触/地面对齐，与 sliding-window 的时序平滑**部分重叠**。  
两者同时开启时，sliding-window 的边际收益会变小——这是 GVHMR walking 上全局指标接近的主要原因。

### 6.3 推荐工作流

```text
1. 默认：旧 GMR + contact_ground
2. 若 joint acceleration 曲线毛刺多 → 试 SW fast
3. compare_joint_trajectories.py 看 per-joint 差异
4. benchmark_retarget_timing.py 确认实时性
5. 需要更强平滑 → 增大 w_velocity / w_acceleration
```

---

## 7. 代码索引

| 组件 | 路径 |
|------|------|
| 核心类 | `general_motion_retargeting/sliding_window_retarget.py` |
| SMPL-X CLI | `scripts/retarget/smplx_to_robot_sliding_window.py` |
| GVHMR CLI | `scripts/gvhmr/to_robot_sliding_window.py` |
| 耗时 benchmark | `scripts/analysis/benchmark_retarget_timing.py` |
| 关节曲线对比 | `scripts/analysis/compare_joint_trajectories.py` |
| motion 指标 | `scripts/analysis/analyze_saved_motion_metrics.py` |
| 固定相机录制 | `general_motion_retargeting/robot_motion_viewer.py` |

---

## 8. 术语

| 术语 | 含义 |
|------|------|
| **jitter** | 关节轨迹中不该有的高频小幅跳动；acceleration 曲线上表现为毛刺 |
| **kinematic TO** | 只在关节/任务空间做平滑，不考虑动力学可行性 |
| **warm start** | 用上一帧 qpos 作为当前帧优化初值 |
| **因果 (causal)** | 只使用当前及过去帧，不用未来帧（零输出延迟） |
