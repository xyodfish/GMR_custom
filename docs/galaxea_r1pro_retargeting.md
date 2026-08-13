# Galaxea R1 Pro 轮式双臂重定向

`galaxea_r1pro` 是继 Galbot One Golf 之后，第二个使用通用 `mobile_upper_body` 形态协议的轮式双臂机器人。它验证了这套方法不依赖 Galbot 的 link 名称、五轴腿腰链或独立头部。

## 输出布局

运行时模型为 `assets/galaxea_r1pro/r1_pro.xml`，MuJoCo `qpos` 共 27 维：

```text
[base_x, base_y, base_yaw,
 steer_motor_joint1, wheel_motor_joint1,
 steer_motor_joint2, wheel_motor_joint2,
 steer_motor_joint3, wheel_motor_joint3,
 torso_joint1..4,
 left_arm_joint1..7,
 right_arm_joint1..7]
```

人体重定向输出中的六个舵向/车轮关节保持中性值。人体 root 的平面轨迹由 `base_x/base_y/base_yaw` 表示；连接真实底盘时，应由底盘控制器把这条平面轨迹转换为轮速和舵向命令，而不是直接把人体关节动作发送给车轮。

## 语义映射

- 人体 pelvis 的平面位置和 heading 映射为虚拟底盘 `x/y/yaw`。
- `torso_link4` 跟随人体 `spine3` 的高度、有限 pitch 和 yaw；R1 Pro 没有 torso roll 能力，因此该分量显式限制为零。
- 左右臂根据人体 shoulder→elbow 和 elbow→wrist 方向，使用 R1 Pro 自身上臂、前臂长度重建目标。
- R1 Pro 没有独立头部关节，配置中省略 head task；这是通用协议支持的合法形态，不使用虚构 frame 兜底。
- wrist orientation 被显式关闭，优先保留肩肘和前臂动作轮廓。冗余腕关节由低权重 posture task 稳定。
- 所有有限位旋转关节保留 `2°` 安全余量。

旧配置曾把人体左脚映射到底盘，并把左髋、左膝映射到躯干链。新配置已经移除这些双足拓扑残留，人体迈腿不会再驱动 R1 Pro 躯干或车轮关节。

## 运行

摆臂动作：

```bash
conda run --no-capture-output -n gmr \
python scripts/retarget/smplx_to_robot.py \
  --smplx_file /home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz \
  --robot galaxea_r1pro \
  --save_path output/galaxea_r1pro_eval/swing_arms.pkl \
  --loop \
  --rate_limit
```

回放已经保存的动作：

```bash
conda run --no-capture-output -n gmr \
python scripts/viz/vis_robot_motion.py \
  --robot galaxea_r1pro \
  --robot_motion_path output/galaxea_r1pro_eval/swing_arms.pkl
```

左直拳：

```bash
conda run --no-capture-output -n gmr \
python scripts/retarget/smplx_to_robot.py \
  --smplx_file /home/xiayu/Workspace/data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz \
  --robot galaxea_r1pro \
  --loop \
  --rate_limit
```

配置位于 `general_motion_retargeting/ik_configs/smplx_to_r1pro.json`。

## 验证结果

统一使用站立、摆臂、行走和左直拳 4 组动作，共 533 帧：

- 所有 27 维输出均为有限值；
- 舵向和车轮关节全程保持 `0`；
- 关节硬限位最小余量为 `2°`；
- 站立和摆臂中，上臂平均方向误差约为 `3.7°–9.0°`，前臂约为 `2.1°–3.4°`；
- 行走中，上臂平均方向误差约为 `5.9°–8.6°`，前臂约为 `4.3°–5.4°`；
- 连续帧平均 IK 用时约 `8 ms`。

左直拳会触及 R1 Pro 单侧肩、肘和腕的硬限位，方向误差高于普通动作。这是模型可达域造成的有界降级，而不是关节方向或左右映射错误；输出仍严格保留安全余量。

验证产物：

- [摆臂视频](../videos/galaxea_r1pro_General_A3_-_Swing_Arms_While_Stand_stageii.mp4)
- [左直拳视频](../videos/galaxea_r1pro_E1_-__Jab_left_stageii.mp4)
- [摆臂接触表](../output/galaxea_r1pro_eval/swing_contact_sheet.png)
- [左直拳接触表](../output/galaxea_r1pro_eval/jab_contact_sheet.png)

## 回归测试

```bash
conda run --no-capture-output -n gmr \
python -m unittest -v tests/test_galaxea_r1pro_retarget.py
```

测试覆盖 27 维输出、有限值、六个轮组关节保持中性、无头部形态、双臂方向和 `2°` 关节限位余量。

通用设计与第三台机器人接入流程见 [GMR 新机器人重定向接入方法与验收指南](new_robot_retargeting_guide_zh.md)。
