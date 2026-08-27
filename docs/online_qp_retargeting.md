# Online QP：MPC-like 短时域重定向（OnlineQpRetargeter）

本文描述仓库中的 **Online QP + GMR** 基线算法：在保留 GMR 人体缩放与 link 跟踪目标的前提下，用滚动时域约束 QP 改善平滑度与足滑，供在线 / 近实时部署。

这里的 “MPC-like” 只表示有限 preview、滑动时间窗、多帧联合优化和逐帧提交。它是运动学轨迹重定向，不包含机器人动力学状态转移、控制量优化或真实执行状态反馈，因此不宣称是控制意义上的 MPC。

| | 逐帧 GMR IK | Online Batch-Lite | **Online QP（MPC-like，本文）** | Batch TO |
|--|-------------|-------------------|---------------------------|----------|
| 类 | `GeneralMotionRetargeting` | `OnlineBatchRetargeter` | **`OnlineQpRetargeter`** | `BatchTrajectoryRetargeter` |
| 时域 | 单帧 | 因果短窗 GN | **因果 / lookahead 短窗 QP** | 离线长窗 GN |
| 求解 | mink | 多帧 GN | **线性化 FK → DAQP** | 多帧 GN |
| 典型耗时 (G1) | ~2 ms | ~7.5 ms | **Py ~40 ms / C++ ~8 ms** | Py ~30 ms / C++ ~3.5 ms |
| 文档 | — | [`online_batch_retargeting.md`](online_batch_retargeting.md) | **本文** | [`batch_trajectory_retargeting.md`](batch_trajectory_retargeting.md) |

力矩可行扩展（soft RNE torque-limit）见配置项说明与相关评估笔记；**默认关闭**，不改变本文所述基线目标。

---

## 1. 设计动机

逐帧 GMR IK 的局限：

- 时间正则弱（主要靠求解器阻尼）→ 关节 jerk 大、难跟踪  
- 足端在接触相易滑（IK 不显式惩罚脚 XY）  
- 绝对 FK 尚可，但「可部署性」不足  

Online QP 的产品目标（见 `online_qp_retarget.py` 模块头注释）：

1. **FK 贴近 GMR**（允许极小损失）  
2. **更平滑**（速度 / 加速度 / jerk 更利于跟踪）  
3. **明显减滑脚**（相对 GMR 的弱点）  

手段：保留 GMR 的人体预处理与目标构造；在短窗口上对 `q` 做约束 QP，而不是每帧独立 mink。

---

## 2. 与 GMR 的关系

```text
人体帧 human_data
    │
    ▼
GMR._prepare_scaled_human_data   ← 身高缩放、contact_ground 人体对齐等
    │
    ▼
从 ik_match_table 解析 link 目标 targets
    │
    ├─ soft seed：bootstrap 用 GMR.retarget；其后 light IK（少步 mink）
    │
    ▼
Online QP 窗口优化（主循环内不再依赖 mink 收敛）
    │
    ▼
可选 finalize / 穿地抬升 / 关节余量裁剪
    │
    ▼
提交 q_t
```

要点：

- **GMR 负责**：缩放、目标、初值引导、接触流水线挂接  
- **QP 负责**：多帧 FK + 时间平滑 + 足部惩罚 + 盒约束  
- `w_gmr` 只是对 light-IK / GMR seed 的**弱先验**，不是硬混合（无强 `ik_blend`）

---

## 3. 算法流程

### 3.0 什么叫「在线 / 实时」（命名约定）

| 说法 | 正确含义 | 错误理解 |
|------|----------|----------|
| Online QP **算法** | 因果短窗，可 `retargetFrame` 一帧一帧解 | — |
| Viewer `online_qp` | **人体帧按到达喂入求解器**（文件只作数据源） | ~~整段预载再播~~（已删除） |
| `causal` | 来一帧解一帧 | — |
| `lookahead` | 到达缓冲；满 `horizon` 后提交最老帧，严格延迟 \(H-1\) 帧 | 预读整段 motion 做前瞻 |

文件仍可能一次性读进内存（JSON 回放源），但 **求解器状态机只看见已 push 的到达帧**。真 live（相机/mocap）调 `pushArrivedFrame` / `retargetFrame`。CLI 的 `retargetSequence` 也只是按帧 push 的便捷封装，**不再 peek 未到达帧**。

若求解比源 FPS 快：可以用短到达缓冲做 lookahead（等未来几帧到齐再提交当前命令），**不必**也不应该把整段原始数据预载进求解器。

### 3.1 单帧流式（因果）

默认窗口长度 \(H=3\)（`horizon`）。

```text
Frame t ≤ bootstrap_gmr_frames (默认 2):
    q_seed = GMR.retarget(human)          # 全量 IK 启动

Frame t > bootstrap:
    q_seed = light_IK(q_{t-1}, targets)   # 少量迭代，仅 warmstart

维护 deque：prepared / targets / q_ref / q
取最近 H 帧构成窗口 q_win
钉住前缀 pin_frames（通常 H-2），只松弛末 1–2 帧
对窗口做 sqp_iters 次：线性化 → 约束 QP → line search
输出末帧 q_t，写入 buffer
```

因果含义：流式 API `retarget(frame)` **不看未来人体帧**；滑脚耦合靠「钉住历史帧 + 窗口内惩罚」完成。

### 3.2 延迟前瞻（到达缓冲，非整段预载）

`use_lookahead=True` 且走 **arrival API** 时：

- 源（文件/传感器）每到一帧 → `pushArrivedFrame`  
- **缓冲未满**（`< horizon`）：`canStepArrived()` 返回 false，不输出新的关节命令
- **缓冲满**约 \(H\) 帧后 → `stepArrived` 用缓冲内未来帧做 MPC-like 短窗联合优化，提交**最老一帧**的 `q`，再 pop
- 代价：从启动开始始终承受约 \(H-1\) 帧延迟，但不会先输出未来帧、再回头提交旧帧
- 前提：求解足够快，缓冲跟得上源速率  

**已删除**：`beginSequence` / `stepSequence` 以及按整段 peek 未来帧的旧 lookahead。唯一实时入口是 `retargetFrame` 与 `pushArrivedFrame` / `stepArrived`。

Viewer / CLI 默认 **arrival + lookahead**（短缓冲延迟 QP）；要零延迟用 `--online_qp_mode causal`。

### 3.3 单次 SQP 迭代在做什么

对当前线性化点 \(q_{\text{lin}}\)，构造关于切空间增量 \(\Delta q\) 的凸 QP：

\[
\min_{\Delta q}\; \tfrac12 \Delta q^\top P\,\Delta q + g^\top\Delta q
\quad\text{s.t.}\quad
lb \le \Delta q \le ub,\;\; G\Delta q \le h
\]

其中 \(P \approx H + \lambda I\)，\(H,g\) 由下列 Gauss-Newton 项累加（与 Batch TO 同源实现，见 `BatchTrajectoryRetargeter`）：

C++ 实现会先把 seed 投影到关节/速度可行区间；固定历史帧不重复施加速度约束。QP 成功后使用受约束 primal solution，并转换成 `applyGnStepToWindow()` 的下降方向符号。求解失败时不再退回无约束 LDLT：已有安全命令则保持上一帧，并累计 `qp_fallback_count`；没有历史命令时错误继续上抛。

| 项 | 作用 |
|----|------|
| FK tracking | link 位姿跟人体目标（权重来自 IK JSON） |
| temporal | \(w_v\) 速度 + \(w_a\) 加速度平滑 |
| anchor | 窗口首帧贴近上一提交 / 锚点 |
| foot penalties | 接触相脚高、脚滑、脚 IK 锚、可选 root/关节锚 |
| GMR prior | 弱拉向 light-IK seed（`w_gmr`） |
| （可选）torque-limit | 默认关；见第 7 节 |

时间项使用 `s = (1/30) / dt = motion_fps / 30` 做帧率归一化：速度残差乘 `s`，加速度残差乘 `s²`，foot-slip 帧间位移也乘 `s`。这保留了 30 FPS 下已有 preset 的权重语义，同时避免同一动作仅因采样帧率变化而改变时间正则强度。

求解器：`qpsolvers` + **DAQP**（`qp_solver="daqp"`）。  
多步 SQP + 简单 line search（`gn_max_step` 限制单步）。

### 3.4 硬约束（相对 Batch GN 的关键差别）

Online QP 显式加盒 / 不等式，而不是仅靠 GN 后投影：

| 约束 | 含义 |
|------|------|
| 步长盒 | \(\|\Delta q\|_\infty \le\) `gn_max_step` |
| 关节限位 | hinge/slide：\(q+\Delta q \in [q_{\min},q_{\max}]\)（可带 `joint_limit_margin_deg`） |
| 速度限位 | \(\|q_t - q_{t-1}\| \le\) `dq_max` \(\cdot\Delta t\) |

`anti_slip` 预设里 `finalize_contact=False` 以省时间；穿地仍可走轻量 root Z 抬升（`_apply_penetration_fix`）。

---

## 4. 足部惩罚（减滑脚核心）

接触掩码默认来自 **参考轨迹脚高**（`foot_contact_from_ref`，与 Batch TO 一致）：脚靠近地面视为接触。

接触为真时惩罚（权重随 preset 变）：

| 项 | 典型权重 (anti_slip) | 含义 |
|----|---------------------|------|
| foot height | 60 | 支撑脚 z → 地面 |
| foot slip | **2000** | 支撑脚 XY 帧间位移 → 0 |
| foot IK anchor | 40 | 脚 XY 贴近 IK 参考 |
| root XY / joint anchor | 视配置 | 接触相稳住骨盆与关节 |

**设计取舍**：极强 `w_foot_slip` 是相对 GMR 减滑的主要来源；过强可能在接触切换处引入 jerk（步态上需配合接触调度，见 `retarget_ideas_and_directions.md`）。

---

## 5. Preset

| Preset | 倾向 | 要点 |
|--------|------|------|
| `default` | 均衡 | 中等平滑与脚滑 |
| `smooth` | 更顺 | \(w_v,w_a\) 更大，`w_foot_slip` 略降 |
| **`anti_slip`（常用）** | 减滑 | `w_foot_slip=2000`，脚高/锚加强；`finalize_contact=False` |

选用建议：在线展示 / 对比滑脚 → `anti_slip`；更在意关节顺滑 → `smooth`。

---

## 6. 实现与入口

| 层级 | 路径 |
|------|------|
| Python 核心 | `general_motion_retargeting/online_qp_retarget.py` |
| 共享 GN/脚惩罚 | `general_motion_retargeting/batch_trajectory_retarget.py` |
| C++ | `cpp/src/retarget/online_qp_retarget.cpp`，CLI `gmr_online_qp_cli` |
| Python 脚本 | `scripts/gvhmr/to_robot_online_qp.py` |
| C++ 封装 | `scripts/tools/run_cpp_online_qp.py` |
| 方法对比数据 | [`retarget_methods_comparison.md`](retarget_methods_comparison.md) |

### 快速命令

```bash
# Python 可视化 / 调参
python scripts/gvhmr/to_robot_online_qp.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --preset anti_slip --contact_ground

# C++ 实时（约 5× Python）
python scripts/tools/run_cpp_online_qp.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --preset anti_slip --contact_ground --benchmark
```

质量与速度对照（摘自方法对比文档，unitree_g1）：

| 实现 | ms/帧量级 | 相对 GMR 滑脚 |
|------|----------:|--------------|
| py_online_qp | ~39 | 明显改善 |
| **cpp_online_qp** | **~8** | **改善最大之一，可 30FPS** |

---

## 7. 可选扩展：Torque-limit（非基线默认）

基线 Online QP **不**把力矩可行性放进目标。可选打开 soft inverse-dynamics barrier：

- 配置：`torque_limit_constraint` / `weight` / `margin` / `scope` / `gate_mode`  
- 语义：窗口内 `mj_rne` 估 \(\tau\)，仅惩罚超过 \(\kappa\tau_{\max}\) 的部分；默认 `scope=upper`（无足端 GRF）  
- 门控：接近饱和才加大权重，避免平时乱压姿态  

实现挂在同一套窗口 \(H,g\) 累加路径上（Python / C++ 对齐）。详细算法与改进动机建议单独成文（与 Batch TO 共用同一套 torque-limit 模块）。

---

## 8. 局限与后续方向

1. **Python 偏慢**：实时请用 C++；Python 适合对照与 viz。  
2. **短窗 \(H=3\)**：无法像离线 Batch TO 那样全局修轨迹。  
3. **二值接触**：接触开关可引起 jerk；连续接触置信度是已知改进点。  
4. **无默认力矩约束**：大力矩动作需显式开 torque-limit 或后处理。  
5. **依赖 GMR/light-IK 初值**：完全去掉每帧 IK 引导会漂移（与 Online Batch `extrap` 实验结论一致）。

---

## 9. 一句话摘要

**Online QP-GMR = GMR 提供人体目标与软初值 + 短时域约束 QP（FK + 平滑 + 强脚滑惩罚）→ 在可接受 FK 代价下显著改善平滑与滑脚，C++ 路径可实时。**
