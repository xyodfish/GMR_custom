# Contact Ground 模式定量分析

本文记录 `contact_ground`、`foot_ground_limit`、`fix_robot_penetration` 三个开关对 retargeting 的影响。

完整统计数据不在本文展开，见：

- `analysis/contact_modes_lafan1_representative_6x500/summary.csv`
- `analysis/contact_modes_lafan1_representative_6x500/per_bvh_summary.csv`
- `analysis/contact_modes_lafan1_representative_6x500/summary.json`

## 统计范围

本次使用 6 条 LaFAN1 代表性 BVH，每条取前 500 帧，共 3000 帧：

- `dance1_subject2.bvh`
- `fallAndGetUp1_subject4.bvh`
- `jumps1_subject1.bvh`
- `obstacles1_subject5.bvh`
- `pushAndStumble1_subject3.bvh`
- `walk1_subject5.bvh`

运行命令：

```bash
conda run -n py310 python scripts/analyze_contact_modes.py \
  --bvh_files \
  /data2/Documents/lafan1/dance1_subject2.bvh \
  /data2/Documents/lafan1/fallAndGetUp1_subject4.bvh \
  /data2/Documents/lafan1/jumps1_subject1.bvh \
  /data2/Documents/lafan1/obstacles1_subject5.bvh \
  /data2/Documents/lafan1/pushAndStumble1_subject3.bvh \
  /data2/Documents/lafan1/walk1_subject5.bvh \
  --max_frames 500 \
  --robot unitree_g1 \
  --format lafan1 \
  --motion_fps 30 \
  --out_dir analysis/contact_modes_lafan1_representative_6x500
```

## 缩写

- `cg`: `contact_ground`
- `fgl`: `foot_ground_limit`
- `fix`: `fix_robot_penetration`

例如：

```text
cg1_fgl0_fix1 = contact_ground 开，foot_ground_limit 关，fix_robot_penetration 开
```

## 关键指标

- `all_max_pen`: 全身最大穿透深度。
- `pen_ratio`: 出现任意穿透的帧比例。
- `root_lift`: `fix_robot_penetration` 产生的 root 抬升累计量。
- `foot_slip`: contact 期间 robot foot 的 XY 滑动累计量。
- `mean_e1`: IK tracking error1 均值。

穿透统计使用 MuJoCo `mj_geomDistance(geom, floor)`。

## 关键结果摘录

### baseline

```text
cg0_fgl0_fix0:
  all_max_pen = 0.24567
  pen_ratio   = 0.218
  foot_slip   = 13.379
  mean_e1     = 0.70521
```

### 单开 contact_ground

```text
cg1_fgl0_fix0:
  all_max_pen = 0.28805
  pen_ratio   = 0.874
  foot_slip   = 12.879
  mean_e1     = 0.70564
```

`contact_ground` 单独开后，foot slip 略降，但全身穿透明显变多。它当前更像是 contact 修正 / foot lock 策略，不是全身防穿透策略。

### 单开 foot_ground_limit

```text
cg0_fgl1_fix0:
  all_max_pen = 0.21918
  pen_ratio   = 0.133
  foot_slip   = 13.409
  mean_e1     = 0.70772
```

`foot_ground_limit` 单独开有轻微改善，但不能解决腿、躯干、手臂等非足底 geom 穿透。

### 打开 fix_robot_penetration

```text
cg0_fgl0_fix1:
  all_max_pen = 0
  pen_ratio   = 0
  root_lift   = 22.862
```

所有 `fix=1` 的组合里，`all_max_pen` 和 `pen_ratio` 都接近 0。当前真正稳定清全身穿透的是 `fix_robot_penetration`，代价是 root 会被后处理抬高。

### 旧策略 + root lift

```text
cg1_fgl0_fix1:
  all_max_pen = 0
  pen_ratio   = 0
  root_lift   = 119.557
  foot_slip   = 12.927
  mean_e1     = 0.67123
```

这组 foot slip 和 tracking 在当前统计中最好，但 root lift 累计量最大。

### 全开

```text
cg1_fgl1_fix1:
  all_max_pen = 0
  pen_ratio   = 0
  root_lift   = 26.716
  foot_slip   = 14.046
  mean_e1     = 0.71057
```

全开也能消穿透，但 foot slip 和 tracking 不是最优，因此不建议直接作为默认策略。

## 结论

1. `fix_robot_penetration` 是当前唯一稳定消除全身穿透的策略。
2. `contact_ground` 单独开不能防全身穿透，只能改善部分 contact / foot lock 行为。
3. `foot_ground_limit` 单独开效果有限，当前更适合作为实验开关。
4. 不建议默认全开，因为全开没有带来更好的 foot slip 或 tracking。

## 当前建议

默认建议：

```json
{
  "contact_ground": {
    "enabled": true,
    "fix_robot_penetration": true
  },
  "foot_ground_limit": {
    "enabled": false
  }
}
```

如果更关注 root lift 尽量小，可以对比：

```json
{
  "contact_ground": {
    "enabled": false,
    "fix_robot_penetration": true
  },
  "foot_ground_limit": {
    "enabled": true
  }
}
```

## 限制

- 本文是 6 条代表性 motion 的前 500 帧抽样，不是 77 条 LaFAN1 全量统计。
- `foot_slip` 依赖 `contact_ground.last_contacts`，当前结果中右脚 contact 没有命中，需要后续检查 contact 命名或检测逻辑。
- `root_lift` 是累计量，只适合不同 mode 之间相对比较。
- `mj_geomDistance` 不是纯 z 向距离，复杂地形下需要更严格的距离和法向处理。
