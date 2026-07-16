# Retarget 算法一览与相对基线对比

> 基线：**逐帧 GMR IK**（`GeneralMotionRetargeting`）  
> 评测：`unitree_g1`，GVHMR 三段 clip（walking / gait_track_12 / tennis），各 **120 帧**，`contact_ground=True`  
> 指标相对 IK：`FK%` / `jerk%` / `slip%`（负更好）；`ms/f`；`rt30` = 是否 ≤ 33.3 ms/帧  
> 原始 JSON：`output/retarget_methods_comparison.json`  
> 日期：2026-07-16

---

## 1. 仓库里有哪些算法

| # | 算法 | 类 / 实现 | 语言 | 在线/离线 | 说明 | 入口 |
|---|------|-----------|------|-----------|------|------|
| 0 | **GMR 逐帧 IK（基线）** | `GeneralMotionRetargeting` / C++ `Retargeter` | Py + C++ | 在线 | 每帧任务空间 IK + 可选 contact/ground | `scripts/gvhmr/to_robot.py`，`gmr_retarget_cli` |
| 1 | **Batch TO** | `BatchTrajectoryRetargeter` | Py + C++ | **离线** | 整段滑动窗多帧 GN，质量最好之一 | `to_robot_batch.py`，`gmr_batch_to_cli` |
| 2 | **Online Batch-Lite** | `OnlineBatchRetargeter` | Python | 在线 | 因果递推窗 GN（Batch TO 的在线版） | `to_robot_online_batch.py` |
| 3 | **Online QP-MPC** | `OnlineQpRetargeter` | Py + C++ | 在线 | 线性化 FK → DAQP 约束 QP，强脚滑惩罚 | `to_robot_online_qp.py`，`gmr_online_qp_cli`，viewer `--method online_qp` |

已弃用 / 非算法：

- `ClipTrajectoryRetargeter` → 已改名为 Batch TO  
- `ContactGroundPipeline` → 各算法共用的接触/贴地模块  
- Sliding Window / Causal TO → 已从公开算法中移除（`trajectory_optimization_retarget.py` 仅保留 Batch 共享 FK 基类）

---

## 2. 相对基线：三段均值

预设：Online Batch 测了 `balanced` / `quality`；Online QP 用 `anti_slip` + lookahead；Batch TO 为 quality（dense GN + best line search）。

| 方法 | ms/帧 | FK Δ | jerk Δ | slip Δ | 30FPS? | 定位 |
|------|------:|-----:|-------:|-------:|:------:|------|
| **ik（Python 基线）** | **1.7** | 0 | 0 | 0 | ✅ | 最快 Python；脚滑偏大 |
| **cpp_ik（mujoco_se3）** | **0.57** | ≈0† | — | — | ✅ | **~3× Python IK**；与 Py RMSE≈0.003 |
| cpp_ik（pin_ik） | 0.22 | — | — | — | ✅ | 更快，后端不同（仅帧率参考） |
| **cpp_batch_to** | **3.5** | **+0.1%** | -4.9% | **-33%** | ✅ | **离线首选**：质量≈Py Batch，速度快 |
| py_batch_to | 29.1 | +0.1% | -4.9% | -33% | ✅* | 与 C++ Batch 质量一致，更慢 |
| online_batch_balanced | 7.6 | +38% | -5% | +30% | ✅ | 快但 FK/slip 不稳 |
| online_batch_quality | 40.7 | **-0.1%** | -7% | +13% | ❌ | FK 贴基线，略超实时 |
| py_online_qp | 38.9 | -0.4% | -4% | -28% | ❌ | 质量好，Python 偏慢 |
| **cpp_online_qp** | **7.7** | **+3.2%** | -4% | **-55%** | ✅ | **在线首选**：滑脚改善最大，可实时 |

\* py_batch_to 均值 29 ms ≈ 卡在 30FPS 边缘；更长窗会更慢。  
† C++ IK 帧率：`output/cpp_ik_fps_benchmark.json`（同 120f×3 clip，`contact_ground`，`GLOG_minloglevel=2`）；质量未重跑全套指标，仅 qpos RMSE vs Python。

---

## 3. 分 clip 明细（相对 IK）

### walking

| 方法 | ms | FK% | jerk% | slip% |
|------|---:|----:|------:|------:|
| ik | 2.1 | 0 | 0 | 0 |
| online_batch_balanced | 8.0 | +22 | -14 | +50 |
| online_batch_quality | 53.8 | +0.3 | -13 | +28 |
| py_batch_to / cpp_batch_to | 30 / 3.9 | +0.8 | -12 | **-19** |
| py_online_qp | 43.3 | +0.3 | -19 | **-79** |
| **cpp_online_qp** | **8.8** | +3.6 | **-23** | **-76** |

### gait_track_12

| 方法 | ms | FK% | jerk% | slip% |
|------|---:|----:|------:|------:|
| ik | 1.4 | 0 | 0 | 0 |
| online_batch_balanced | 7.2 | +63 | -4 | -37 |
| online_batch_quality | 27.9 | 0 | -3 | 0 |
| py_batch_to / cpp_batch_to | 28 / 3.3 | +0.7 | 0 | **-67** |
| py_online_qp | 44.9 | -1.6 | +9 | -9 |
| **cpp_online_qp** | **6.9** | +4.4 | +12 | **-49** |

### tennis

| 方法 | ms | FK% | jerk% | slip% |
|------|---:|----:|------:|------:|
| ik | 1.6 | 0 | 0 | 0 |
| online_batch_balanced | 7.5 | +28 | +3 | +76 |
| online_batch_quality | 40.4 | -0.8 | -7 | +11 |
| py_batch_to / cpp_batch_to | 29 / 3.3 | -1.2 | -2 | **-14** |
| py_online_qp | 28.5 | 0 | -3 | +3 |
| **cpp_online_qp** | **7.5** | +1.6 | -1 | **-39** |

---

## 4. 怎么选

```text
只要快、能接受脚滑          → cpp_ik（或 Python ik）
离线、要质量 + 减滑脚        → cpp_batch_to（或 py_batch_to）
在线、要减滑脚 + 可 30FPS    → cpp_online_qp (anti_slip)
在线、要 FK 尽量贴 IK        → online_batch quality（可能不够实时）
实验 / 调试                  → Python 对应实现
```

---

## 5. 复现

```bash
# 重新跑对比并写出 JSON
bash scripts/tools/run_gvhmr_retarget_benchmark.sh

# C++ Online QP 可视化（流式）
python scripts/gvhmr/to_robot_online_qp.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --preset anti_slip --contact_ground --loop

# C++ Batch TO
python scripts/tools/run_cpp_batch_to.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --out_json /tmp/batch.json --contact_ground --quality --benchmark
```

相关文档：

- `docs/online_batch_retargeting.md` — Online Batch / Online QP  
- `docs/batch_trajectory_retargeting.md` — Batch TO  
- `docs/kinematic_to_retargeting.md` — 运动学 TO 总览  

---

## 6. 简图

```text
                    ┌─────────────┐
                    │  GMR IK     │  基线 · 最快
                    │  (online)   │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
   Online Batch      Online QP-MPC    Batch TO
   (online, Py)      (online, Py/C++) (offline, Py/C++)
           │               │               │
           └───────────────┴───────────────┘
                     共用 contact/ground、FK 代价
```
