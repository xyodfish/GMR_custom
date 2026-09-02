# Scripts 目录说明

```
scripts/
├── retarget/          # 人体动作 → 机器人重定向
├── gvhmr/             # GVHMR .pt / 单目视频流水线
├── viz/               # 机器人动作可视化
├── analysis/          # 指标对比、benchmark
├── tools/             # 格式转换、导出
├── gmr_gui.py         # GUI 入口
└── _paths.py
```

## 常用命令

```bash
python scripts/gmr_gui.py
```

`gmr_gui` 可在界面选择 **Per-frame IK**、**Online Batch**、**Batch TO (Python/C++)** 等（BVH / SMPL-X / GVHMR / 视频）。应用标题常量：`general_motion_retargeting/gui/core.py` → `GUI_APP_TITLE`。

```bash
python scripts/retarget/bvh_to_robot.py --bvh_file ... --robot unitree_g1
python scripts/retarget/smplx_to_robot.py --smplx_file ... --robot unitree_g1
python scripts/retarget/to_robot_batch.py --input_file ... --robot unitree_g1
python scripts/gvhmr/to_robot.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/gvhmr/to_robot_online_batch.py --gvhmr_pred_file ... --robot unitree_g1
python scripts/gvhmr/to_robot_online_qp.py --input_file ... --robot unitree_g1
python scripts/viz/vis_robot_motion.py --robot ... --robot_motion_path ...
python scripts/viz/smplx_h2_compare_gui.py  # 数据集/质量筛选、统计详情，以及 G1/H2/隔离 PKL 播放
python scripts/retarget/smplx_to_h2_dataset.py  # ~/Workspace/data -> 可断点续跑的 H2 训练 PKL
python scripts/analysis/compare_joint_trajectories.py --baseline ... --candidate ... --robot unitree_g1
python scripts/analysis/benchmark_gvhmr_retarget_methods.py --pt_glob 'data/gvhmr_test_videos/*/hmr4d_results.pt'
```

接地参数：`--contact_ground --foot_ground_limit`。详见 [`docs/contact_ground.md`](../docs/contact_ground.md)。
Online QP（MPC-like 短时域重定向）算法说明：[`docs/online_qp_retargeting.md`](../docs/online_qp_retargeting.md)。
方法对比：[`docs/retarget_methods_comparison.md`](../docs/retarget_methods_comparison.md)。

## SMPL-X 全量生成 H2 训练数据

```bash
python scripts/retarget/smplx_to_h2_dataset.py \
  --input-root ~/Workspace/data \
  --output-root ~/Workspace/gmr_cg_batch_h2 \
  --quality
```

使用本仓库的 `SMPL-X -> G1 Batch TO -> H2 同名关节映射` 与接地约束；`--quality`
启用离线高质量预设，不加时使用 fast 预设。
映射保留 G1 的 floating-base 姿态和 29 个同轴关节角，按 H2 物理限位裁剪，
并仅调整 root Z 以对齐两台机器人的实际脚底碰撞净空。
质量门禁会同时检查 SMPL-X 任务目标、本次生成的 G1 中间轨迹和 H2 结果：
G1 对缩放后人体关键身体位置/旋转的跟踪误差、同名关节/根姿态映射契约、H2 限位
及裁剪强度、位置与姿态跳变、关节速度/加速度、关键躯干穿地和疑似滑脚。
关节加速度异常只记录 warning，不会单独隔离整条轨迹；如果同一轨迹还有穿地、
映射错误、严重限位裁剪等 reject，仍会进入 `quarantine/`。
硬异常进入 `quarantine/`，其余训练 PKL 进入 `accepted/`；两者均保留原 NPZ 目录树。
警告不会自动隔离，但会写入 PKL 和质量报告，便于按异常分数抽检。
PKL 包含 `root_pos` / `root_rot` / `dof_pos` / `qpos`，以及 MuJoCo FK 计算的
`local_body_pos` / `link_body_list`。训练时只读取 `accepted/`。

质量与运行记录：

- `retarget_manifest.json`：实时进度、当前文件、通过/隔离/失败计数。
- `retarget_failures.jsonl`：无法完成重定向或质量计算的错误。
- `quality/records/**/*.quality.json`：每条轨迹的完整指标、异常原因和触发帧。
- `quality/ranking.json`：按异常分数从高到低排列的总表。
- `quality/accepted.txt`、`quality/quarantine.txt`：可直接供数据加载器读取的 PKL 清单。
- `quality/events.jsonl`：生成过程中的逐条、只追加审计日志。

`scripts/viz/smplx_h2_compare_gui.py` 会实时读取上述质量事件。网页可按“通过 / 隔离 /
含警告 / 尚未处理”筛选；选中动作后显示异常原因、阈值、触发帧和完整指标。
“直接播放批处理 H2”可以播放 `accepted/` 或 `quarantine/` 中的原始 PKL，播放不会
改变质量状态，并继续支持倍速、暂停和进度拖动。
