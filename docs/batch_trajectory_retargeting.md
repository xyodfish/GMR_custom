# 离线 Batch 轨迹优化重定向 (BatchTrajectoryRetargeter)

本文档描述 **paper-style 离线多帧联合 q 优化**：在整段 motion sequence 上滑动窗口，每窗联合优化 `q_{t…t+H-1}`，再按 overlap 融合输出。

与仓库内其它 TO 路线的关系：

| | 逐帧 IK | 因果 TO | Sliding Window | **Batch TO（本文）** |
|--|---------|---------|----------------|---------------------|
| 类 / 脚本 | `GeneralMotionRetargeting` | `TrajectoryOptimizationRetargeter` | `SlidingWindowRetargeter` | **`BatchTrajectoryRetargeter`** |
| 时域 | 无 | 因果 H 帧 | 因果 H 帧 | **离线整段 motion，重叠滑窗** |
| 每步优化变量 | 单帧 `q_t` | 窗口 `q_{t-H+1…t}` | fast: 单帧；full: 窗口 | **窗口 `q_{start…start+H-1}`** |
| 求解器 | mink QP | GN / L-BFGS | GN / L-BFGS | **多帧 GN（默认）** |
| 典型用途 | 实时 teleop | 在线 30fps 平滑 | 在线平滑 | **离线数据生成 / 质量优先** |
| 文档 | — | [`trajectory_optimization_retargeting.md`](trajectory_optimization_retargeting.md) | [`sliding_window_retargeting.md`](sliding_window_retargeting.md) | **本文** |

---

## 1. 算法

### 1.1 流程

```text
人体序列 [frame_0 … frame_{N-1}]
  → GMR 预处理 (scale / contact_ground)
  → Bootstrap：逐帧 IK 得到 q_ref（可选 use_gmr_init）
  → 从 IK 轨迹提取全局 foot contact mask（foot_contact_from_ref）
  → 滑动窗口 GN 优化
  → 窗口 overlap 加权融合
  → finalize_contact（可选）
  → 输出 q_batch[0…N-1]
```

**GMR 的角色**：预处理、IK bootstrap、FK 目标解析、foot contact 参考；**优化主循环内不调用 mink**。

### 1.2 滑动窗口

默认 `strategy=sliding_window`：

```text
windows: [0, H), [S, S+H), [2S, 2S+H), …
         stride S，末尾不足一窗则截断
```

每窗独立做 `gn_steps` 步 Gauss-Newton，窗口间用 `window_anchor_weight` 锚定上一窗结果，最终对重叠帧做加权平均。

`strategy=full`：整段 motion 一次 L-BFGS（极慢，仅作离线 baseline）。

### 1.3 目标函数（单窗）

```text
J(Q) = Σ_t  FK_tracking(q_t, human_target_t)     # pos + rot，权重来自 ik_match_table
     + w_v   Σ_t || q_{t+1} - q_t ||²              # 速度平滑（默认不含 root XYZ）
     + w_a   Σ_t || q_{t+2} - 2q_{t+1} + q_t ||²  # 加速度平滑
     + w_anchor || q_0 - q_0^prev ||²              # 窗间连续
     + foot_penalties(Q)                           # 见下节
```

- **FK_tracking**：MuJoCo `xpos / xmat` vs IK config 解析的人体目标
- **smooth_root_xyz**：默认 `False`，walking 时不平滑 floating-base 平移，避免拖慢 root
- 约束：每步 GN 后将 q **投影**到 MuJoCo 关节限位（非算法命名中的 clip）

### 1.4 足部惩罚（默认开启）

contact mask 来自 **IK bootstrap 轨迹** 的脚高（`foot_contact_from_ref=True`），与 `foot_slip_metrics` 一致。

| 项 | 权重默认 | 作用 |
|----|----------|------|
| foot height | `w_foot_height=50` | contact 期间脚 z → ground |
| foot slip | `w_foot_slip=2000` | contact 期间脚 XY 不滑 |
| foot IK anchor | `w_foot_ik_anchor=200` | contact 脚 XY 贴近 IK 参考 |
| root XY contact | `w_root_xy_contact=100` | contact 期间 root XY 贴近 IK |
| contact joint anchor | `w_contact_joint_anchor=400` | contact 期间关节贴近 IK bootstrap |

关闭：`--no_foot_penalties`（Python）或 `enableFootPenalties=false`（C++ config）。

### 1.5 GN 求解

每窗：

1. 在当前 `q_win` 上堆叠 Jacobian，构建 normal equation `(JᵀJ + λI) Δq = -Jᵀr`
2. 解 `Δq`（Python: `scipy.linalg.lstsq`；C++: Eigen LDLT）
3. `gn_max_step` 截断 + line search（默认 alphas: 1.0, 0.5, 0.25, 0.125）
4. 重复 `gn_steps` 次

---

## 2. Python 使用

### 2.1 核心代码

- 类：`general_motion_retargeting/batch_trajectory_retarget.py` → `BatchTrajectoryRetargeter`
- GVHMR CLI：`scripts/gvhmr/to_robot_batch.py`（兼容包装）
- 统一 CLI：`scripts/retarget/to_robot_batch.py`（GVHMR `.pt` / SMPL-X `.npz` / LAFAN1 `.bvh`）

### 2.2 示例

**GVHMR `.pt`**

```bash
conda activate gmr

python scripts/retarget/to_robot_batch.py \
  --input_file output/gvhmr_pt/cxk-ball_hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --max_frames 120 \
  --save_path output/walking_batch.pkl
```

**SMPL-X（AMASS `.npz`）**

```bash
python scripts/retarget/to_robot_batch.py \
  --input_file assets/motions/walk.npz \
  --input_type smplx \
  --robot unitree_g1 \
  --save_path output/walk_smplx_batch.pkl
```

**LAFAN1 BVH**

```bash
python scripts/retarget/to_robot_batch.py \
  --input_file assets/motions/walk1_subject1.bvh \
  --input_type bvh_lafan1 \
  --robot unitree_g1 \
  --contact_ground \
  --save_path output/walk_bvh_batch.pkl
```

（`--input_type` 可省略，按扩展名自动推断：`.pt` → GVHMR，`.npz`/`.pkl` → SMPL-X，`.bvh` → LAFAN1）

### 2.3 预设档位

`scripts/analysis/benchmark_batch_gn_perf.py` 定义三档：

| 预设 | window | gn_steps | line search | foot | 说明 |
|------|--------|----------|-------------|------|------|
| **quality** | 16 / 8 | 3 | 4 alphas | 开 | 推荐默认（`--fast` 未指定时 Python CLI 仍用 32/16，benchmark 用 16/8） |
| **fast** | 16 / 8 | 2 | 1 alpha | 开 | `--fast`，更快，FK 跟踪略差 |
| **ceiling** | 16 / 16 | 1 | 1 alpha | 关 | 性能上限测试，非生产配置 |

Python CLI `--fast` 等效：`window 16/8`, `gn_steps=2`, 单 alpha line search。

### 2.4 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--window_size` | 32 | 窗长 H |
| `--window_stride` | 16 | 步长 S |
| `--gn_steps` | 3 | 每窗 GN 迭代次数 |
| `--solver` | gn | gn 或 lbfgs（full strategy） |
| `--foot_contact_from_ref` | on | IK 轨迹定 contact |
| `--smooth_root_xyz` | off | walking 建议保持 off |
| `--no_foot_penalties` | off | 关闭全部 foot 项 |

---

## 3. C++ 实现

### 3.1 文件

| 路径 | 说明 |
|------|------|
| `cpp/include/gmr/retarget/batch_trajectory_config.h` | 配置与 profiling 结构体 |
| `cpp/include/gmr/retarget/batch_trajectory_retarget.h` | 类声明 |
| `cpp/src/retarget/batch_trajectory_retarget.cpp` | GN + foot penalty 实现 |
| `cpp/src/main_batch_to_cli.cpp` | CLI → `gmr_batch_to_cli` |

### 3.2 编译

依赖 prefix 默认 `/opt/robot/devel`。详见 [`cpp/README.md`](../cpp/README.md)。

```bash
./cpp/scripts/install_devel_cmake_packages.sh /opt/robot/devel   # 若 find_package(mujoco) 失败
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
export LD_LIBRARY_PATH=/opt/robot/devel/lib:$LD_LIBRARY_PATH
```

### 3.3 运行

C++ CLI 读 JSON 人体帧。可直接用统一包装脚本从原始文件运行：

```bash
# 一条命令：.pt / .npz / .bvh → C++ batch TO
python scripts/tools/run_cpp_batch_to.py \
  --input_file output/gvhmr_pt/cxk-ball_hmr4d_results.pt \
  --robot unitree_g1 \
  --contact_ground \
  --out_json output/cxk_ball_batch_cpp.json
```

或先导出 JSON 再调用 `gmr_batch_to_cli`（JSON 含 `src_human` / `actual_human_height` 时 CLI 可自动读取）：

```bash
# 1) 任意格式 → human_frame_json
python scripts/tools/export_human_frames_json.py \
  --input_file walk.npz \
  --out_json output/walk_human_frames.json

# 2) C++ batch TO
cpp/build/gmr_batch_to_cli \
  --gmr_root . --robot unitree_g1 \
  --human_frame_json output/walk_human_frames.json \
  --out_json output/walk_batch_cpp.json \
  --contact_ground
```

**LAFAN1 BVH 示例**

```bash
python scripts/tools/run_cpp_batch_to.py \
  --input_file assets/motions/walk1_subject1.bvh \
  --input_type bvh_lafan1 \
  --robot unitree_g1 \
  --contact_ground \
  --out_json output/walk_bvh_batch_cpp.json
```

### 3.4 C++ 默认配置 = Quality，不是 Fast

`BatchTrajectoryConfig` 默认值：

```cpp
windowSize=16, windowStride=8, gnSteps=3
gnLineSearchAlphas = {1.0, 0.5, 0.25, 0.125}
enableFootPenalties = true   // 全部 foot 权重与 Python 对齐
```

仅当 CLI 传入 **`--fast`** 时变为 `gnSteps=2` + 单 alpha line search。

**结论：C++ 默认跑的是 quality 档完整 GN，加速来自 native MuJoCo / Eigen，不是砍迭代换速度。**

### 3.5 C++ vs Python 差异（parity 未完成项）

| 项目 | Python | C++ | 影响 |
|------|--------|-----|------|
| 多 alpha line search cost | 完整 `_window_cost` | 完整 `windowCost`（已对齐） | — |
| finalize_contact | `contact_ground.fix_penetration` | `Retargeter::finalizeContact()` | 需 `--contact_ground` 开启 |
| IK bootstrap | 完整 GMR | C++ `Retargeter` | qpos RMSE ~0.03 |
| contact_ground CLI | 默认随 GMR | 需 `--contact_ground` / `--no_contact_ground` | 对比时注意对齐 |

---

## 4. 性能参考

测试条件：`unitree_g1`，120 帧 @ 30fps，quality 档（window 16/8, gn_steps=3, foot penalties on）。

### 4.1 Python（GVHMR）

| 预设 | ms/帧 | FPS | 备注 |
|------|-------|-----|------|
| per-frame IK | ~2 | ~500 | 基线，无轨迹平滑 |
| batch **quality** | ~29–32 | ~31–34 | optimize 占 ~89% |
| batch **fast** | ~7 | ~138 | FK 跟踪误差 +14–44% vs IK |
| batch **ceiling** | ~3.5 | ~288 | 无 foot penalty |

### 4.2 C++ vs Python（cxk-ball, quality 参数）

| 阶段 | Python | C++ | 加速比 |
|------|--------|-----|--------|
| optimize | ~3118 ms | ~330 ms | **~9.5×** |
| total | ~3367 ms (28 ms/f) | ~392 ms (3.3 ms/f, **~306 FPS**) | **~8.6×** |
| qpos RMSE vs Python | — | **0.034** | 同参数下的实现差，非 fast 牺牲 |

### 4.3 质量参考（Python quality, 120f）

相对逐帧 IK（`contact_ground` on）：

| motion | jerk ↓ | foot slip ↓ |
|------|--------|-------------|
| walking | ~12% | ~19% |
| tennis | ~2% | ~14% |
| cxk-ball | ~5% | ~31% |

（foot slip 修复前 walking 会 +61%；加入 foot penalty 后转为改善。）

---

## 5. 分析与对比脚本

| 脚本 | 用途 |
|------|------|
| `scripts/analysis/benchmark_batch_gn_perf.py` | quality / fast / ceiling 吞吐与 phase profile |
| `scripts/analysis/compare_ik_vs_batch_preset.py` | IK vs batch 预设质量指标 |
| `scripts/analysis/compare_ik_vs_batch_gn_metrics.py` | jerk / foot slip 等 |
| `scripts/analysis/benchmark_gvhmr_batch_to.py` | GVHMR 批量 IK vs batch |
| `scripts/analysis/compare_py_vs_cpp_batch.py` | 同输入一键 Py vs C++ batch TO |
| `scripts/tools/export_gvhmr_frames_json.py` | `.pt` → C++ 输入 JSON |

可视化（PKL 或 C++ batch JSON）：

```bash
# Python batch TO 输出
python scripts/viz/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path output/walking_batch.pkl \
  --record_video --video_path videos/walking_batch.mp4

# C++ gmr_batch_to_cli 输出
python scripts/viz/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path output/cxk_ball_batch_cpp.json

# GVHMR 离线 batch TO + 直接回放
python scripts/gvhmr/to_robot_batch.py \
  --gvhmr_pred_file output/gvhmr_pt/cxk-ball_hmr4d_results.pt \
  --robot unitree_g1 --contact_ground --max_frames 120 --view --loop

# C++ batch TO + MuJoCo 窗口（不写 JSON）
export LD_LIBRARY_PATH=/opt/robot/devel/lib:$LD_LIBRARY_PATH
cpp/build/gmr_retarget_viewer \
  --backend mujoco_se3 --method batch_to \
  --gmr_root . --robot unitree_g1 \
  --human_frame_json output/cxk_ball_human_frames.json \
  --actual_human_height 1.7 --contact_ground --max_frames 120 --loop
```

GUI：`scripts/gmr_gui.py` → 数据类型 **GVHMR (.pt)** → 算法 **Batch TO (offline)**。

---

## 6. 选型建议

```text
需要实时 / 在线 30fps？
  → 因果 TO (TrajectoryOptimizationRetargeter) 或 Sliding Window fast

需要离线最高质量、可接受秒级延迟？
  → Batch TO quality（Python 或 C++）

需要批量生成训练数据、追求吞吐？
  → C++ gmr_batch_to_cli（默认 quality，~300 FPS 量级）

只要速度、可接受跟踪变差？
  → --fast（Python / C++ 均支持）
```

---

## 7. 命名说明

算法原名 **clip TO**（motion clip），易与 `np.clip` / 关节限位投影混淆，已统一改为 **batch TO**：

- **Batch** = 离线对多帧 `q` 联合优化（paper-style batch retargeting）
- 与 **因果 TO**（逐帧在线）、**逐帧 IK** 形成对比

旧名仍可用（deprecated）：

| 旧名 | 新名 |
|------|------|
| `ClipTrajectoryRetargeter` | `BatchTrajectoryRetargeter` |
| `retarget_clip()` | `retarget_batch()` |
| `to_robot_clip.py` | `to_robot_batch.py` |
| `gmr_clip_to_cli` | `gmr_batch_to_cli` |

---

## 8. 后续工作

- [x] C++ line search 使用完整 window cost
- [x] C++ finalize_contact 对齐 Python `fix_penetration`
- [x] 统一 benchmark 脚本（`compare_py_vs_cpp_batch.py`）
- [x] GUI / viewer 集成 batch TO 回放
- [ ] pybind 绑定（可选，替代 subprocess CLI）
- [ ] 稀疏 / banded 求解器进一步提速
