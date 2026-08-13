# Galbot One Golf 轮式双臂重定向

本案例中关于模型审计、语义映射、frame 标定、调参顺序与验收指标的通用经验，已整理为
[GMR 新机器人重定向接入方法与验收指南](new_robot_retargeting_guide_zh.md)。后续接入不同构型机器人时应优先遵循该指南，再调整本机器人专用配置。

`galbot_one_golf` 使用 SMPL-X 动作输入，输出 24 维 MuJoCo `qpos`：

```text
[base_x, base_y, base_yaw,
 leg_joint1..5,
 left_arm_joint1..7,
 right_arm_joint1..7,
 head_joint1..2]
```

## 设计原则

轮式双臂机器人不能直接照搬双足机器人的髋、膝、脚任务。当前映射采用以下语义：

- 底盘 `x/y/yaw` 跟随与 Unitree G1 相同的人体 pelvis 平移缩放和朝向。
- 五轴腿腰机构只跟随人体 `spine3` 的高度与有限幅度的上身姿态，不再接收单侧髋、膝旋转。
- 双臂先根据人体的肩到肘、肘到腕方向构造与 Galbot 本体臂长一致的目标，再用低权重腕姿态补足冗余自由度。
- 头部只跟随头相对躯干的旋转，并限制到机器人可表达的范围。
- 所有旋转关节输出与 URDF 硬限位至少保留 2 度安全余量。

该设计避免人体腿部摆动驱动 Galbot 升降机构，也避免因人体和机器人臂长不同而把手臂拉到错误方向。

## 运行

```bash
conda activate gmr
python scripts/retarget/smplx_to_robot.py \
  --smplx_file /home/xiayu/Workspace/data/ACCAD/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii.npz \
  --robot galbot_one_golf \
  --save_path output/galbot_one_golf_swing.pkl \
  --rate_limit
```

回放保存的动作：

```bash
python scripts/viz/vis_robot_motion.py \
  --robot galbot_one_golf \
  --robot_motion_path output/galbot_one_golf_swing.pkl
```

## 模型来源与配置

仓库中的 `assets/galbot_one_golf/galbot_one_golf.xml` 是用于 GMR 的平面虚拟底盘版本。其关节树、轴向和限位已与
`~/Workspace/data/galbot_one_golf_description/galbot_one_golf.urdf` 核对；视觉 mesh 来自同一 description 包。

机器人专用参数位于
`general_motion_retargeting/ik_configs/smplx_to_galbot_one_golf.json` 的
`mobile_upper_body` 段。调参时优先调整高度范围、姿态限幅和任务权重，不要重新加入髋、膝到五轴腿腰机构的直接映射。

## 回归测试

```bash
conda run -n gmr python -m unittest -v \
  tests/test_galbot_one_golf_retarget.py
```

测试覆盖双臂方向、输出有限值、2 度关节限位余量，以及退化人体骨段的明确报错。
