# 接触与地面穿透修复（`contact_ground`）

为 GMR 提供流式接触检测、人体地面对齐、足部锁定，以及**机器人地面穿透修正**。设计思路参考 `robot_retargeter` 中的 KCR（Keypoint-Contact Retargeting），并适配因果 / 实时、逐帧 retargeting 场景。

## 解决的问题

| 阶段 | 问题 | 处理方式 |
|------|------|----------|
| 人体参考 | 脚相对地面漂移 / 悬空 | 接触门控：对所有人体关键点做竖直平移 |
| 人体参考 | 站立时脚滑移 | 接触期间对脚位置做 EMA 锁定 |
| 机器人结果 | IK 后脚穿入地面 | 抬高 root `qpos[2]`，直到脚部 geom 离开地面 |
| 机器人结果 | 躺下时骨盆 / 背部穿地 | 额外监控躯干 geom；髋部较低时再监控腿部 geom，并使用更大安全边距 |

对机器人**仅施加竖直 root 平移**，不修改 IK 任务权重。

## 流水线（逐帧）

```
human frame
    │
    ▼
StreamingContactDetector     ← 脚速 + 高度（因果，仅用过去帧）
    │
    ▼
StreamingGroundAligner       ← 接触时将人体关键点对齐到地面
    │
    ▼
StreamingFootLocker          ← 可选：接触脚的 EMA 锁定
    │
    ▼
GMR IK（两阶段 mink）
    │
    ▼
fix_robot_ground_penetration ← 用 mj_geomDistance 抬高机器人 root Z
    │
    ▼
qpos 输出
```

## 代码结构

| 文件 | 作用 |
|------|------|
| `general_motion_retargeting/contact_ground.py` | 核心流水线：接触、对齐、锁定、穿透修复 |
| `general_motion_retargeting/contact_ground_config.py` | 合并机器人 preset 与 IK JSON 覆盖项 |
| `general_motion_retargeting/ik_configs/contact_ground_presets.json` | **各机器人 body 名称与默认阈值** |
| `general_motion_retargeting/motion_retarget.py` | 在 `update_targets()` 与 `retarget()` 中挂接流水线 |
| `scripts/bvh_to_robot.py` | CLI：`--contact_ground` / `--no-contact_ground` |
| `scripts/bvh_compare_contact_ground.py` | 录制 OFF / ON 对比视频 |
| `scripts/stitch_videos_side_by_side.py` | 将两个 mp4 左右拼接 |
| `scripts/inspect_contact_ground.py` | 校验某机器人的合并配置 |

### C++ 实现（`cpp/`）

| 文件 | 作用 |
|------|------|
| `cpp/include/gmr/retarget/contact_ground.h` | `ContactGroundPipeline` 声明 |
| `cpp/src/retarget/contact_ground.cpp` | 与 Python 对齐的流水线实现 |
| `cpp/include/gmr/retarget/contact_ground_config.h` | 读取 preset 并合并 IK JSON |
| `cpp/src/retarget/contact_ground_config.cpp` | 配置合并逻辑 |
| `cpp/src/retarget/retargeter_mujoco*.cpp` | MuJoCo 后端：人体修复 + IK 后穿透修正 |
| `cpp/src/retarget/retargeter_pinocchio*.cpp` | Pinocchio 后端：仅人体侧修复（无 MuJoCo 穿透修正） |

C++ 与 Python **共用** `general_motion_retargeting/ik_configs/contact_ground_presets.json` 与各 IK JSON 中的 `contact_ground` 块。

**CLI 参数**（`gmr_retarget_cli` / `gmr_retarget_batch_cli` / `gmr_retarget_viewer`）：

```bash
--contact_ground        # 强制启用
--no_contact_ground     # 强制关闭
```

Viewer YAML 示例：`contact_ground: true`

**注意**：机器人穿透修正（`fixRobotPenetration`）需要 **MuJoCo 后端**（`mujoco_se3` / `mujoco_jacobian_legacy`）。Pinocchio 后端只做人体参考修复。

### `motion_retarget.py` 中的集成点

1. **初始化** — `build_contact_ground_config(ik_config, tgt_robot, cli_override)` 构造 `ContactGroundPipeline`。
2. **`update_targets()`** — 若启用，在设置 IK 目标前调用 `process_human_frame(human_data)`。
3. **`retarget()`** — IK 迭代结束后调用 `fix_robot_penetration(model, data)` 调整 `qpos[2]`。

## 配置层级

配置按以下顺序合并（后者覆盖前者）：

1. `contact_ground_presets.json` 中的 `_default`
2. `contact_ground_presets.json` 中的机器人条目（如 `unitree_g1`、`fourier_n1`）
3. IK JSON 中的 `contact_ground` 块（如 `bvh_lafan1_to_g1.json`）
4. CLI 参数 `--contact_ground` / `--no-contact_ground`（仅覆盖 `enabled`）

### IK JSON 最简写法（推荐）

G1 使用共享 preset 启用：

```json
"contact_ground": {
    "enabled": true
}
```

其他机器人启用（body 名称来自 preset）：

```json
"contact_ground": {
    "enabled": true
}
```

仅覆盖需要的字段：

```json
"contact_ground": {
    "enabled": true,
    "lying_penetration_margin": 0.03,
    "robot_foot_bodies": ["left_foot_roll_link", "right_foot_roll_link"]
}
```

### 完整参数说明

| 键 | 默认值 | 说明 |
|----|--------|------|
| `enabled` | `false`（`unitree_g1` preset 为 `true`） | 总开关 |
| `foot_bodies` | `["LeftFootMod", "RightFootMod"]` | 用于接触检测的人体脚关键点 |
| `human_root_name` | 来自 IK 的 `human_root_name` | 用于检测躺姿（髋部高度） |
| `vel_threshold` | `0.5` | 判定接触的最大脚速（m/s） |
| `height_threshold` | `0.08` | 进入接触的脚高度（m） |
| `height_off_threshold` | `0.12` | 离开接触的脚相对已估计地面高度（m，迟滞） |
| `ground_z` | `0.0` | 人体对齐的目标地面高度 |
| `ground_margin` | `0.02` | 人体对齐时相对地面的余量 |
| `airborne_height_threshold` | `0.15` | 无接触时用于识别腾空的脚高度（m） |
| `airborne_offset_decay` | `1.0` | 腾空时保留地面对齐 offset；小于 1 会逐帧衰减 |
| `enable_foot_lock` | `true` | 接触期间启用 EMA 脚锁定 |
| `fix_robot_penetration` | `true` | IK 后抬高 root |
| `penetration_margin` | `0.01` | 站立 / 正常姿态的安全距离（m） |
| `lying_hip_height_threshold` | `0.45` | 人体髋 Z 低于此值 → 低姿态模式 |
| `low_pose_foot_height_threshold` | `0.20` | 脚较低且髋部不高时也可进入低姿态模式 |
| `low_pose_max_hip_height` | `0.65` | 脚触地时允许进入低姿态的最大髋高 |
| `lying_penetration_margin` | `0.02` | 低姿态模式安全距离（m） |
| `penetration_max_iterations` | `5` | 每帧 root 抬升最大迭代次数 |
| `floor_geom_name` | `"floor"` | MuJoCo 地面 geom 名称 |
| `robot_foot_bodies` | 按机器人 | 脚部 link 根节点；会包含其子树 geom |
| `robot_trunk_bodies` | 按机器人 | 骨盆 / 躯干 link，用于背部穿透 |
| `robot_leg_bodies` | 按机器人 | 髋 / 膝 link，仅在低姿态模式使用 |
| `robot_arm_bodies` | 按机器人（G1 已配置） | 肩 / 肘 / 腕 / 手，仅在低姿态模式使用 |
| `foot_collision_geoms` | `[]` | 可选：显式 geom 名（为空则用 body 子树） |

## 各机器人 preset

定义于 `ik_configs/contact_ground_presets.json`：

| 机器人 | `robot_foot_bodies` | `robot_trunk_bodies` |
|--------|---------------------|----------------------|
| `unitree_g1` | `left/right_ankle_roll_link` | `pelvis`, `waist_yaw/roll`, `torso_link` |
| `booster_t1_29dof` | `left/right_foot_link` | `Waist`, `Trunk` |
| `engineai_pm01` | `LINK_ANKLE_ROLL_L/R` | `LINK_BASE`, `LINK_TORSO_YAW` |
| `stanford_toddy` | `ank_roll_link`, `ank_roll_link_2` | `waist_link`, `torso`, `waist_gears` |
| `fourier_n1` | `left/right_foot_roll_link` | `base_link`, `waist_yaw_link`, `torso_link` |
| `pal_talos` | `leg_left/right_6_link` | `base_link`, `torso_1/2_link` |

`unitree_g1_with_hands` 复用 `unitree_g1` 的 preset。

### 接入新机器人

1. 打开机器人 MJCF，找出带**脚 mesh**、**躯干 mesh**、**髋 / 膝 mesh** 的 body。
2. 在 `contact_ground_presets.json` 中新增条目，填写 `robot_foot_bodies`、`robot_trunk_bodies`、`robot_leg_bodies`。
3. 在 IK JSON 中加入 `"contact_ground": { "enabled": false }`（验证通过后可改为 `true`）。
4. 校验：

```bash
python scripts/inspect_contact_ground.py --robot <robot_name> --src_human bvh_lafan1
```

**注意**

- 优先选**实际挂载脚 mesh geom** 的 link，而非空的 toe 参考点（G1 的 `toe_link` 无 mesh，应使用 `ankle_roll_link`）。
- 穿透检测对 listed body 上的全部 geom（及脚子树）调用 `mj_geomDistance(geom, floor)`。
- 若视觉 mesh 仍有轻微穿模，可调大 `lying_penetration_margin`（凸包碰撞体与渲染 mesh 可能不一致）。

## 使用方法

### BVH retargeting

```bash
python scripts/bvh_to_robot.py \
  --bvh_file /path/to/motion.bvh \
  --robot unitree_g1 \
  --contact_ground
```

### C++ batch retargeting

```bash
./cpp/build/gmr_retarget_batch_cli \
  --gmr_root /data/open_src_code/GMR_custom \
  --robot unitree_g1 \
  --backend mujoco_se3 \
  --src_human bvh_lafan1 \
  --human_frame_json /path/to/human_frames.json \
  --contact_ground \
  --out_json /tmp/qpos.json
```

显式关闭：

```bash
python scripts/bvh_to_robot.py ... --no-contact_ground
```

其他机器人（需先在 IK JSON 或 preset 中启用）：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file motion.bvh \
  --robot fourier_n1 \
  --contact_ground
```

### 对比视频

```bash
python scripts/bvh_compare_contact_ground.py \
  --bvh_file /path/to/motion.bvh \
  --robot unitree_g1

python scripts/stitch_videos_side_by_side.py \
  --stem motion_stem \
  --dir videos/contact_compare
```

## 穿透算法细节

`contact_ground.py` 中的 `fix_robot_ground_penetration()`：

1. 选择 geom 集合：
   - **正常模式**：`robot_foot_bodies` + `robot_trunk_bodies` 的 geom
   - **躺姿模式**（人体髋 Z ≤ `lying_hip_height_threshold`）：额外加入 `robot_leg_bodies` 的 geom，并使用 `lying_penetration_margin`
2. 对每个被监控 geom 计算 `mj_geomDistance(geom, floor_geom)`。
3. 若距离 < margin，则抬升量 = `margin - distance`。
4. 执行 `data.qpos[2] += lift`（最多重复 `penetration_max_iterations` 次）。

## 已知限制

- **形态差异**：人可以平躺在地，人形机器人往往需要较大 root 抬升。
- **仅竖直修正**：不做水平平移，也不根据姿态调整 IK 权重。
- **Mesh 与碰撞体**：距离基于 MuJoCo 碰撞 mesh / 凸包，与视觉 mesh 可能略有偏差。
- **手臂**：未纳入监控，躺姿时手臂仍可能穿地。
- **按机器人调参**：preset 仅为起点，需在你的动作数据上验证。

## 变更记录（GMR_custom）

1. 初始版本：流式接触检测 + 人体地面对齐 + 脚锁定。
2. 脚部穿透修复：通过 ankle 子树 geom 检测（G1 脚 mesh 在 `ankle_roll_link` 上）。
3. 躯干 geom 用于背 / 骨盆；躺姿模式加入腿部 geom 与更大边距。
4. **机器人 preset**（`contact_ground_presets.json`）+ IK 覆盖合并，支持多机器人。
5. **C++ 端口**：`ContactGroundPipeline` 接入四个 retarget 后端；MuJoCo 后端支持完整穿透修正。
6. **全局高度漂移修复**：接触高度相对已估计地面 offset 计算，脚速仍使用原始轨迹；腾空时默认冻结 offset，避免短暂失联后持续浮空。
