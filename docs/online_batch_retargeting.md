# Online Batch-Lite 在线重定向

## 背景与目标

在 **Batch TO**（离线多帧 GN，效果好、~30ms/帧 Python）与 **逐帧 IK**（~2ms/帧、实时但滑脚/抖动）之间，新增两类在线方法：

1. **Online Batch-Lite**（GN）：见下文历史结果  
2. **Online QP-MPC**（推荐实验）：线性化 FK → 带约束 QP（DAQP），目标是 **小幅 FK 损失 + 更平滑 + 明显减滑脚**

## Online QP-MPC（新）

实现：
- Python：`general_motion_retargeting/online_qp_retarget.py`
- **C++（推荐实时）**：`cpp/` → `gmr_online_qp_cli`，封装 `scripts/tools/run_cpp_online_qp.py`

- 保留 GMR 人点缩放 / link 目标；light IK 仅作 **软初值**（弱 `w_gmr`，无 `ik_blend`）
- 滚动窗 H=3，lookahead MPC；钉住上一帧以耦合滑脚
- 目标：FK + 时序平滑 + **强 foot slip** + 关节/速度盒约束
- 求解：DAQP 约束 QP，多步 SQP + line search

### 用法

```bash
conda activate gmr
# Python（预览 / 对照）
python scripts/gvhmr/to_robot_online_qp.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --preset anti_slip --mode lookahead --contact_ground --rate_limit

# C++（提速，~5× Python；walking 约 8 ms/帧，可 30FPS）
cmake --build cpp/build -j --target gmr_online_qp_cli
python scripts/tools/run_cpp_online_qp.py \
  --input_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --out_json /tmp/online_qp.json \
  --preset anti_slip --contact_ground --benchmark --max_frames 120
```

### 120 帧相对逐帧 GMR（unitree_g1，walking）

| 实现 | ms/帧 | vs Python | 备注 |
|------|------:|----------:|------|
| Python `anti_slip` | ~43 | 1× | FK≈持平，jerk/slip 改善 |
| **C++ `anti_slip`** | **~8** | **~5×** | 实时@30；qpos RMSE vs Py ≈0.02 |

Python 质量参考（相对 GMR IK）：

| clip | preset | ms/帧 | FK Δ | jerk Δ | slip Δ |
|------|--------|------:|-----:|-------:|-------:|
| walking | **anti_slip** | ~42 | **+0.4%** | **-19%** | **-70%** |
| gait_track_12 | **anti_slip** | ~44 | **-1.5%** | +9% | **-58%** |
| walking | smooth | ~41 | +0.3% | -25% | +21% |

解读：`anti_slip` 符合「FK 几乎不掉、滑脚明显好于 GMR、关节更顺」。实时部署用 **C++**；Python 用于调参与可视化。

---

## Online Batch-Lite（GN，既有）

在 **Batch TO**（离线多帧 GN，效果好、~30ms/帧 Python）与 **逐帧 IK**（~2ms/帧、实时但滑脚/抖动）之间，新增 **Online Batch-Lite**：

- **在线**：逐帧 `retarget(frame)` 流式 API，无需整段序列
- **实时**：G1 + GVHMR 测试集 **~7.5 ms/帧**，6/6 clip 满足 30 FPS
- **效果高于 GMR**：jerk 平均 **-13%**；步态 clip 滑脚可改善（如 gait_track_12 **-42%**）
- **牺牲相对 Batch TO**：RMSE vs IK ~0.15（Batch ~0.006）；部分 clip 滑脚仍差于 IK

## 算法概要

```text
Frame 0:  GMR.retarget() 全量 IK bootstrap
Frame t:  q_{t-1} + light IK (3 iter) → seed
          维护长度 H 的 causal buffer (q, targets, q_ref)
          对 buffer 做 multi-frame FK GN（Batch TO 同款目标）
          **固定 prefix，只更新 trailing 1–2 帧**（receding horizon）
          输出 q_t；可选 penetration finalize
```

与 Batch TO 的差异：

| 项 | Batch TO (offline) | Online Batch-Lite |
|----|-------------------|-------------------|
| 窗口 | H=16, stride=8 | H=5, 每帧 commit 末帧 |
| GN 步数 | 3 + line search | 1，无 line search |
| 优化变量 | 整窗 | 仅 trailing 2 帧 |
| Foot penalty | 全套 (slip 2000) | 轻量 (height 25, slip 150) |
| 延迟 | 需整段 | ~7.5 ms/帧 |

核心实现：`general_motion_retargeting/online_batch_retarget.py`  
`OnlineBatchRetargeter._optimize_gn_pinned_prefix()` — prefix 帧 GN 更新置零。

## Preset

| Preset | window | trailing | gn | foot | ik_blend | 用途 |
|--------|--------|----------|-----|------|----------|------|
| `fast` | 4 | 1 | 1 | off | 0 | 最低延迟 teleop |
| `balanced` | 5 | 2 | 1 | light | 0 | 速度/平滑折中（默认） |
| `quality` | 6 | 2 | 2 | medium | 0 | 更重脚部约束，仍流式 |
| `track` | 5 | 2 | 2 | soft | **0.5** | **FK 优先**（向 light-IK seed 回拉） |
| `extrap` | 5 | 2 | 4 | soft | 0 | **实验**：前 K 帧 GMR bootstrap，之后历史 hold 作初值（不用每帧 GMR IK） |

### 实验结论：`extrap`（不用 GMR IK 作初值）

设定：保留 GMR 人点缩放/link 目标；前 5 帧 `GMR.retarget` bootstrap 并直接 commit；之后 `hold` 上一帧作 seed + 窗口 GN（Armijo 线搜索）；跟丢则 GMR re-anchor。

walking 120 帧（相对逐帧 GMR）：

| 配置 | ms/帧 | 绝对 FK Δ | 现象 |
|------|------:|----------:|------|
| balanced（每帧 light IK） | ~7.5 | ~+20% 量级 | 稳定 |
| extrap + reanchor | ~56 | **~+180%** | 约每数帧 reanchor 一次 |
| extrap 无 reanchor | ~50 | **~+1700%** | 持续漂移 |

**结论：当前窗口 GN 离开 GMR/light-IK 初值后无法在线稳住人体 FK**；仅靠历史外推/hold 会漂移。要做「无 GMR 初值」需要更强的单帧 FK 求解（mink/QP 多步 SQP）或频繁 reanchor（又变回依赖 GMR）。

FK 不可接受时优先用 `track`，或手动加大 `--ik_blend`（0.5→0.7）。

```bash
# 实验：无每帧 GMR IK 初值
python scripts/gvhmr/to_robot_online_batch.py \
  --gvhmr_pred_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --preset extrap --rate_limit --compare_ik
```

关键可调参数见 `OnlineBatchConfig`：`seed_mode`、`gmr_bootstrap_frames`、`extrap_policy`、`bootstrap_commit_gmr`、`ik_blend`、`light_ik_iters`、`w_velocity` / `w_acceleration`、`w_foot_*`、`gn_steps`、`finalize_contact`。

## 用法

### GVHMR Demo（可视化）

```bash
conda activate gmr
python scripts/gvhmr/to_robot_online_batch.py \
  --gvhmr_pred_file data/gvhmr_test_videos/walking/hmr4d_results.pt \
  --robot unitree_g1 \
  --preset balanced \
  --compare_ik \
  --rate_limit
```

### Python API

```python
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import OnlineBatchRetargeter, OnlineBatchConfig

gmr = GMR(src_human="smplx", tgt_robot="unitree_g1", contact_ground=True, motion_fps=30.0)
online = OnlineBatchRetargeter(gmr, OnlineBatchConfig.from_preset("balanced"))

for frame in human_frames:
    q = online.retarget(frame)
    print(f"frame {online.frame_index}: {online.last_frame_ms:.1f} ms")
```

### GUI

算法下拉选择 **「Online Batch-Lite (在线 · 推荐)」**，输入类型选 GVHMR `.pt`。

## Benchmark 结果（2026-07-15）

数据：`data/gvhmr_test_videos/*/hmr4d_results.pt`，6 clips × 200 frames，G1，`contact_ground=True`。

复现：

```bash
bash scripts/tools/run_gvhmr_retarget_benchmark.sh
# 或
python scripts/analysis/benchmark_gvhmr_retarget_methods.py \
  --pt_glob 'data/gvhmr_test_videos/*/hmr4d_results.pt' \
  --methods ik,online_batch,py_batch_to,cpp_batch_to \
  --output_json output/gvhmr_retarget_benchmark.json
```

### 评判指标

相对逐帧 IK 的变化率（Δ%）及绝对值：

| 指标 | 含义 |
|------|------|
| **dq / ddq / jerk** | 关节速度、加速度、jerk 范数均值（平滑度） |
| **fk** | 加权 FK 跟踪代价（与 Batch TO 相同：pos/rot task error） |
| **foot_slip** | 接触期脚 XY 滑动总量 |
| **rmse_vs_ik** | qpos 相对 IK 的 RMSE（偏离基线程度） |
| **ms/帧 · 30FPS** | 实时性 |

### 汇总（6 clips 平均，2026-07-15；dq/ddq/fk 需重跑 benchmark）

| 方法 | 类型 | ms/帧 | 30FPS | jerk Δ | foot slip Δ | RMSE vs IK |
|------|------|-------|-------|--------|-------------|------------|
| **ik** | online | 1.9 | 6/6 | 0% | 0% | 0 |
| **online_batch** | online | **7.5** | **6/6** | **-13%** | +59%* | 0.155 |
| py_batch_to | offline | 30.0 | 5/6 | -6.6% | **-20%** | **0.006** |
| cpp_batch_to | offline | 4.2 | 6/6 | -6.6% | **-20%** | **0.006** |

\* foot slip 在 walking / gait_track_12 上优于或接近 IK；sports2d_walk 等 clip 仍偏高，后续可调 `w_foot_slip` 或 contact mask。

重跑后 JSON 的 `summary` 会额外包含 `mean_dq_change_pct`、`mean_ddq_change_pct`、`mean_fk_change_pct` 及对应绝对值。

### 结论

1. **Online Batch-Lite 达成实时目标**：~4× IK 延迟，远低于 33 ms/帧预算。
2. **相对 GMR 有明确收益**：jerk  consistently 更好；步态类 clip 滑脚可优于 IK。
3. **相对 offline Batch TO 有预期牺牲**：tracking RMSE 与滑脚不及 C++ Batch TO，符合设计 trade-off。
4. **在线减滑脚首选**：见 Online QP（`cpp_online_qp` / `anti_slip`）。

## 后续优化方向

- [ ] C++ 端口（复用 `optimizeGnWindow` + pinned prefix）→ 目标 ~3–4 ms/帧
- [ ] 自适应 preset：teleop 用 `fast`，录 demo 用 `quality`
- [ ] 轻量 soft contact variable（见 `GMR_custom_改进评估与下一阶段建议.md` §3.1）
- [ ] SMPL-X / BVH 在线 CLI（当前 GUI 仅 GVHMR .pt）

## 文件索引

| 路径 | 说明 |
|------|------|
| `general_motion_retargeting/online_batch_retarget.py` | 核心类 |
| `scripts/gvhmr/to_robot_online_batch.py` | GVHMR 可视化 CLI |
| `scripts/analysis/benchmark_gvhmr_retarget_methods.py` | 多方法 benchmark |
| `output/gvhmr_retarget_benchmark.json` | 最新 benchmark 报告 |
| `general_motion_retargeting/gui/core.py` | GUI 集成 |
