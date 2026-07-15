# 独立因果轨迹优化重定向 (TrajectoryOptimizationRetargeter)

本文档描述 **脱离 GMR 逐帧 IK 求解** 的因果滑动窗口 TO 实现。

与 [`sliding_window_retargeting.md`](sliding_window_retargeting.md) 中的 `SlidingWindowRetargeter` 不同：

| | SlidingWindowRetargeter | TrajectoryOptimizationRetargeter |
|--|-------------------------|--------------------------------|
| 优化时是否调用 `GMR.retarget()` / mink | **是** | **否** |
| 跟踪代价 | GMR `FrameTask` | 直接 MuJoCo FK vs 人体目标 |
| 窗口优化 | fast 仅当前帧；full 联合窗口 | **默认联合优化整个因果窗口** |
| GMR 角色 | 求解核心 | 仅预处理 + 可选初值 |

> **离线 batch**：整段 motion 重叠滑窗 + 多帧 GN，见 [`batch_trajectory_retargeting.md`](batch_trajectory_retargeting.md)（`BatchTrajectoryRetargeter`，非因果）。

## 1. 算法

### 1.1 因果窗口

维护过去 `H` 帧人体数据：

```text
buffer = [human_{t-H+1}, …, human_t]
```

每来一帧，联合优化：

```text
Q = [q_{t-H+1}, …, q_t]
```

只输出 `q_t`，窗口前移。

### 1.2 目标函数（fast 模式，默认）

因果链（与 sliding-window 同结构，但 full 模式仍用 FK 整窗）：

```text
frame 0:  q₀ = GMR.retarget()           # 仅 bootstrap
frame t:  q_seed = q_{t-1} + light IK(5 iter)
          q_t  = argmin  ||mink_task(q)||²
                        + w_v || q - q_{t-1} ||²
                        + w_a || q - 2q_{t-1} + q_{t-2} ||²
```

- **不再**每帧调用完整 `GMR.retarget()`（否则无法做时序平滑）
- **不再**在 GMR 解上用 FK GN 二次优化（会把 IK 解拉偏，smoothness 反而变差）
- 默认 `solver=lbfgs`，单帧 5 iter；可选 `solver=gn`（需正确 `dt`）
- 实测 **~7 ms/帧**（unitree_g1, GVHMR walking），可 30fps 实时

### 1.3 目标函数（full 模式，离线）

联合优化因果窗口 `[t-H+1 … t]`，scipy L-BFGS-B，~13 s/帧，仅离线质量优先时使用。

- **FK 跟踪**：MuJoCo `xpos / xmat` 与 IK config 解析出的人体目标比较
  - 位置：`||p_body - p_target||²`，权重来自 `ik_match_table` 的 `pos_weight`
  - 姿态：`|| Log(R_target^T R_body) ||²`，权重来自 `rot_weight`
- **smoothness**：窗口内速度 / 加速度正则
- **anchor**：窗口滑动连续性

### 1.3 约束

- 关节限位：`q_min ≤ q_i ≤ q_max`（hinge / slide）
- 优化器：L-BFGS-B（scipy），**不调用 mink**

### 1.4 GMR 仅用于

1. `_prepare_scaled_human_data()` — scale / contact_ground
2. `_resolve_ik_target()` — 从 IK config 得到每帧 task target
3. 可选 `--use_gmr_init`：仅 **第 0 帧** 调用 `GMR.retarget()` 作 bootstrap；之后从 `q_{t-1}` 因果递推

fast 模式优化循环内 **不调用** 完整 `GMR.retarget()`；light IK warmstart 仍用 mink（5 iter）。

---

## 2. 使用方法

### GVHMR

```bash
python scripts/gvhmr/to_robot_trajectory_opt.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --compare_ik \
  --save_path output/walking_to.pkl
```

### SMPL-X

```bash
python scripts/retarget/smplx_to_robot_trajectory_opt.py \
  --smplx_file /path/to/motion.npz \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/motion_to.pkl
```

### BVH (LAFAN1 / Nokov)

```bash
python scripts/retarget/bvh_to_robot_trajectory_opt.py \
  --bvh_file /path/to/walk.bvh \
  --format lafan1 \
  --robot unitree_g1 \
  --contact_ground --foot_ground_limit \
  --rate_limit --loop
```

```bash
python scripts/analysis/compare_joint_trajectories.py \
  --bvh_file /path/to/walk.bvh \
  --robot unitree_g1 \
  --candidate_method to \
  --contact_ground --foot_ground_limit \
  --labels "per-frame IK" "trajectory opt" \
  --output output/walk_ik_vs_to.png
```

### Python API

```python
from general_motion_retargeting import (
    GeneralMotionRetargeting,
    TrajectoryOptimizationRetargeter,
    TrajectoryOptimizationConfig,
)

gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1", contact_ground=True)
to = TrajectoryOptimizationRetargeter(
    gmr,
    TrajectoryOptimizationConfig(window_size=8, use_gmr_init=True),
)

qpos = to.retarget(human_frame)
```

### 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `mode` | `"fast"` | fast=单帧 GN（实时）；full=整窗 L-BFGS（离线） |
| `window_size` | 8 | 人体缓冲长度 |
| `gn_steps` | 2 | position Gauss-Newton 步数 |
| `gn_rot_steps` | 1 | rotation GN 步数（高权重 body 子集） |
| `w_velocity` | 2.0 | 速度平滑 |
| `w_acceleration` | 10.0 | 加速度平滑 |
| `use_gmr_init` | **True** | 每帧完整 GMR IK 初值 + GN 平滑（推荐） |
| `light_ik_warmstart_iters` | 5 | `--no-use_gmr_init` 时用 5 步 mink 轻量初值（非完整 retarget） |
| `gn_steps_no_init` | 25 | 仅当 `light_ik_warmstart_iters=0` 时的纯 FK GN 步数（实验性，易发散） |

### 初值模式说明

| 模式 | 行为 | 质量 |
|------|------|------|
| 默认 `use_gmr_init=True` | 完整 GMR IK → GN 时序平滑 | 最好，~1.0 qpos 差 vs IK |
| `--no-use_gmr_init` | 5 步 mink 轻量 seed → GN 平滑 | 可用，~1.1 qpos 差 |
| 纯 FK GN（`light_ik_warmstart_iters=0`） | 无 IK，仅 Jacobian GN | **不推荐**，会乱动 |

人形高维 FK 跟踪是强非凸问题，**没有 IK 类初值很难收敛**。优化器内仍不调用 `GMR.retarget()`；初值在 GN 之前单独完成。

---

## 3. 三种方法对比 benchmark

```bash
python scripts/analysis/benchmark_retarget_timing.py \
  --gvhmr_pred_file ~/Videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --methods ik sw to
```

---

## 4. 代码位置

| 组件 | 路径 |
|------|------|
| 核心类 | `general_motion_retargeting/trajectory_optimization_retarget.py` |
| GVHMR CLI | `scripts/gvhmr/to_robot_trajectory_opt.py` |
| SMPL-X CLI | `scripts/retarget/smplx_to_robot_trajectory_opt.py` |
| BVH CLI | `scripts/retarget/bvh_to_robot_trajectory_opt.py` |

---

## 5. 性能参考 (unitree_g1, GVHMR walking)

| 模式 | mean / frame | 说明 |
|------|-------------|------|
| per-frame IK | ~2 ms | 基线 |
| **TO fast (GN)** | **~1.5 ms** | 默认，可 30fps 实时 |
| sliding-window fast | ~22 ms | GMR + 单帧 L-BFGS |
| TO full window | ~13 s | 离线整窗 L-BFGS |

旧版 `fix_window_prefix` + scipy 误对整窗每步评估 FK，导致 ~3 s/帧；已废弃，请用默认 `mode=fast`。

---

## 6. 后续扩展

当前为 **运动学 FK-TO**。未包含：

- 动力学约束 / 力矩限
- 硬接触约束（当前仅有 GMR 预处理侧 contact_ground）
- CasADi / acados MPC

可作为 Phase 2 接入。
