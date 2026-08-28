# GMR_custom：带约束的通用运动重定向与 C++ 实现

[English](README.md)

#### GMR 核心特性：
- 实时高质量运动重定向，支持全身遥操作等场景，例如 [TWIST](https://github.com/YanjieZe/TWIST)。
- 针对 RL 跟踪策略进行了细致调参。
- 支持多种人形机器人和多种人体动作数据格式（见下表）。

#### 本仓库 fork 的扩展特性：
- 基于上游 GMR，在 [`cpp/`](cpp/) 下提供实验性 C++ 重定向流水线。
- C++ 重定向后端与渲染目标解耦（MuJoCo / ROS / 自定义 viewer）。
- 复用 `general_motion_retargeting/ik_configs/*.json` 中的 IK 配置，保持 Python/C++ 配置一致。
- 提供面向不同构型机器人的[新机器人重定向接入方法与验收指南](docs/new_robot_retargeting_guide_zh.md)，覆盖模型审计、语义映射、frame 标定、分层 IK、调参顺序和量化验收。
- 增加自碰撞约束，以及基于地面接触的脚滑优化。
- 关节限位安全余量（`joint_limit_margin_deg`）：让提交的旋转关节始终离机械硬限位保持可配置的度数，改善可跟踪性。已接入轨迹优化类重定向器（Python `OnlineBatchRetargeter` / `OnlineQpRetargeter`；C++ `OnlineQpRetargeter` 与离线 `BatchTrajectoryRetargeter`）。详见 [`docs/retarget_methods_comparison.md`](docs/retarget_methods_comparison.md)。
- Python 侧将 `contact_ground`、实验性 `foot_ground_limit`、`fix_robot_penetration` 解耦，便于独立评估各模式。
- 提供 BVH 重定向的离线接触模式分析工具，输出穿透、root 抬升、脚滑、IK 跟踪误差等 CSV/JSON 汇总。
- 提供纯 C++ 的[机器人关节轨迹到机器人关节轨迹](docs/robot_joint_trajectory_retargeting.md)入口：Unitree G1 qpos 经 canonical SMPL-X 代理、Batch TO 和接触感知后处理转换到 H2 等目标机器人；运行时不依赖 Python 或 Puppet。

> [!NOTE]
> 如需支持新机器人或新人体动作格式，请将机器人文件（`.xml`、`.urdf` 及 mesh）/ 人体动作数据发送至 <a href="mailto:lastyanjieze@gmail.com">Yanjie Ze</a> 或提交 issue，我们会尽快支持。请确保所提供机器人文件可在本仓库开源。

本仓库采用 [MIT License](LICENSE) 许可。


# 新闻与更新

- **2026-08-13：** 将 `mobile_upper_body` 扩展为可选 head/wrist 姿态的通用轮式双臂协议，并完成第二台机器人 **Galaxea R1 Pro** 的语义重定向；移除旧配置中左脚/髋/膝到轮式机构的错误映射。详见 [`docs/galaxea_r1pro_retargeting.md`](docs/galaxea_r1pro_retargeting.md)。
- **2026-08-13：** 总结 Galbot 轮式双臂适配中的可复用经验，新增[新机器人重定向接入方法与验收指南](docs/new_robot_retargeting_guide_zh.md)：核心原则是按动作语义和机器人能力设计映射，并以模型、单帧、动作序列、定量指标和视觉对比共同验收。
- **2026-08-12：** 完成 **Galbot One Golf**（`galbot_one_golf`）轮式双臂 SMPL-X 重定向：底盘复用 G1 根运动语义，五轴腿腰机构跟随受限上身姿态，双臂按人体骨段方向重建目标，并保留 2 度关节限位余量。案例见 [`docs/galbot_one_golf_retargeting.md`](docs/galbot_one_golf_retargeting.md)，可复用方法见[新机器人重定向接入方法与验收指南](docs/new_robot_retargeting_guide_zh.md)。
- **2026-07-20：** 为轨迹优化类重定向器新增**关节限位安全余量**（`joint_limit_margin_deg`）：让提交的旋转关节离硬限位保持设定度数（0 关闭），可消除限位饱和且跟踪代价极小。Python（`OnlineBatchRetargeter`、`OnlineQpRetargeter`）与 C++（`gmr_online_qp_cli`、`gmr_batch_to_cli`、`gmr_retarget_viewer`）均支持。见 [`docs/retarget_methods_comparison.md`](docs/retarget_methods_comparison.md)。
- **2026-07-16：** 移除不可用的 Sliding Window / Causal TO 公开算法；在线推荐 Online QP / Online Batch，离线推荐 Batch TO。见 [`docs/retarget_methods_comparison.md`](docs/retarget_methods_comparison.md)。
- **2026-07-15：** 新增 **Online Batch-Lite** 在线多帧 GN 重定向（`OnlineBatchRetargeter`）：~7.7 ms/帧、30 FPS 实时，jerk 优于 IK。见 [`docs/online_batch_retargeting.md`](docs/online_batch_retargeting.md)。
- **2026-07-13：** 新增 **Unitree H2**（`unitree_h2`），包含 SMPL-X 与 LAFAN1 BVH 的 IK 配置、`contact_ground` preset，以及调优后的 `bvh_lafan1_to_h2.json`。可通过 `bvh_to_robot.py` / `human_json_to_robot.py` 实时重定向；通过 `vis_robot_motion.py --human_frame_json` 回放并叠加 IK 目标锚点。
- **2026-06-26：** 新增解耦的接触/地面模式控制与 BVH 离线分析工具。详见 [`docs/contact_ground.md`](docs/contact_ground.md)、[`docs/contact_modes_analysis.md`](docs/contact_modes_analysis.md)。
- **2026-04-15：** 基于 GMR 新增实验性 C++ 重定向功能，见「C++ 功能（实验性）」章节及 [`cpp/README.md`](cpp/README.md)。
- **2026-01-21：** 支持 [Xsens](https://www.xsens.com/) BVH 离线数据。
- **2026-01-12：** 支持 [Fourier GR3](https://www.fftai.com/)，为本仓库第 17 款人形机器人。
- **2025-12-02：** 支持 [TWIST2](https://yanjieze.com/TWIST2)，使用 [XRoboToolkit SDK](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)。
- **2025-11-17：** 加入社区讨论可添加微信 [二维码](https://yanjieze.com/TWIST2/images/my_wechat.jpg)，备注格式：「[GMR] [姓名] [单位]」。
- **2025-11-08：** Jason Peng 的 [MimicKit](https://github.com/xbpeng/MimicKit/tree/main/tools/gmr_to_mimickit) 已支持 GMR 格式。
- **2025-10-15：** 支持 [PAL Robotics Talos](https://pal-robotics.com/robot/talos/)，第 15 款人形机器人。
- **2025-10-14：** 支持 [Nokov](https://www.nokov.com/) BVH 数据。
- **2025-10-14：** 新增 IK 配置文档，见 [DOC.md](DOC.md)。
- **2025-10-09：** [TWIST](https://github.com/YanjieZe/TWIST) 开源代码可用于 RL 动作跟踪。
- **2025-10-02：** GMR 技术报告已发布于 [arXiv](https://arxiv.org/abs/2510.02252)。
- **2025-10-01：** 支持将 GMR pickle 转为 CSV（用于 beyondmimic），见 `scripts/tools/batch_gmr_pkl_to_csv.py`。
- **2025-09-25：** GMR 介绍视频见 [Bilibili](https://www.bilibili.com/video/BV1p1nazeEzC/?share_source=copy_web&vd_source=c76e3ab14ac3f7219a9006b96b4b0f76)。
- **2025-09-16：** 支持通过 [GVHMR](https://github.com/zju3dv/GVHMR) 从**单目视频**提取人体姿态并重定向到机器人。
- **2025-09-12：** 支持 [Tienkung](https://github.com/Open-X-Humanoid/TienKung-Lab)，第 14 款人形机器人。
- **2025-08-30：** 支持 [Unitree H1 2](https://www.unitree.com/cn/h1) 与 [PND Adam Lite](https://pndbotics.com/)，第 12、13 款人形机器人。
- **2025-08-28：** 支持 [Booster T1](https://www.boosterobotics.com/) 23/29 DOF 版本。
- **2025-08-28：** 支持从 [OptiTrack](https://www.optitrack.com/) 导出的离线 FBX 动作数据。
- **2025-08-27：** 支持 [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets)，第 11 款人形机器人。
- **2025-08-24：** 支持 [Unitree H1](https://www.unitree.com/h1/)，第 10 款人形机器人。
- **2025-08-24：** `GeneralMotionRetargeting` 默认启用电机速度限制（`use_velocity_limit=True`，默认上限 `3*pi`）；默认打印机器人 DoF/Body/Motor 名称与 ID，可通过 `robot_dof_names`、`robot_body_names`、`robot_motor_names` 访问。
- **2025-08-10：** 支持 [Booster K1](https://www.boosterobotics.com/)，第 9 款机器人。
- **2025-08-09：** 支持 *Unitree G1 + Dex31 灵巧手*。
- **2025-08-07：** 支持 [Galaxea R1 Pro](https://galaxea-dynamics.com/)（轮式人形）与 [KUAVO](https://www.kuavo.ai/)，第 7、8 款人形机器人。
- **2025-08-06：** 支持 [HighTorque Hi](https://www.hightorquerobotics.com/hi/)，第 6 款人形机器人。
- **2025-08-04：** GMR 首次发布，见 [Twitter 帖子](https://x.com/ZeYanjie/status/1952446745696469334)。

## 支持的机器人与数据格式



| 编号 | 机器人/数据格式 | 机器人自由度 | SMPLX（[AMASS](https://amass.is.tue.mpg.de/)、[OMOMO](https://github.com/lijiaman/omomo_release)） | BVH [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | FBX（[OptiTrack](https://www.optitrack.com/)） | BVH [Nokov](https://www.nokov.com/) | PICO（[XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)） | 更多格式敬请期待 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Unitree G1 `unitree_g1` | 腿(2\*6)+腰(3)+臂(2\*7)=29 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1 | Unitree G1 灵巧手版 `unitree_g1_with_hands` | 腿(2\*6)+腰(3)+臂(2\*7)+手(2\*7)=43 | ✅ | ✅ | ✅ | TBD | TBD |
| 2 | Unitree H1 `unitree_h1` | 腿(2\*5)+腰(1)+臂(2\*4)=19 | ✅ | TBD | TBD | TBD | TBD |
| 3 | Unitree H1 2 `unitree_h1_2` | 腿(2\*6)+腰(1)+臂(2\*7)=27 | ✅ | TBD | TBD | TBD | TBD |
| 18 | Unitree H2 `unitree_h2` | 腿(2\*6)+腰(3)+臂(2\*7)=29 | ✅ | ✅ | TBD | TBD | TBD |
| 4 | Booster T1 `booster_t1` | TBD | ✅ | TBD | TBD | TBD |
| 5 | Booster T1 29dof `booster_t1_29dof` | TBD | ✅ | ✅ | TBD | TBD |
| 6 | Booster K1 `booster_k1` | 颈(2)+臂(2\*4)+腿(2\*6)=22 | ✅ | TBD | TBD | TBD |
| 7 | Stanford ToddlerBot `stanford_toddy` | TBD | ✅ | ✅ | TBD | TBD |
| 8 | Fourier N1 `fourier_n1` | TBD | ✅ | ✅ | TBD | TBD |
| 9 | ENGINEAI PM01 `engineai_pm01` | TBD | ✅ | ✅ | TBD | TBD |
| 10 | HighTorque Hi `hightorque_hi` | 头(2)+臂(2\*5)+腰(1)+腿(2\*6)=25 | ✅ | TBD | TBD | TBD |
| 11 | Galaxea R1 Pro `galaxea_r1pro`（轮式双臂） | 平面底盘(3)+舵轮(6)+躯干(4)+臂(2\*7)=27 | ✅ | TBD | TBD | TBD |
| 21 | Galbot One Golf `galbot_one_golf`（轮式双臂） | 底盘(3)+腿腰(5)+臂(2\*7)+头(2)=24 | ✅ | TBD | TBD | TBD |
| 12 | Kuavo `kuavo_s45` | 头(2)+臂(2\*7)+腿(2\*6)=28 | ✅ | TBD | TBD | TBD |
| 13 | Berkeley Humanoid Lite `berkeley_humanoid_lite`（需进一步调参） | 腿(2\*6)+臂(2\*5)=22 | ✅ | TBD | TBD | TBD |
| 14 | PND Adam Lite `pnd_adam_lite` | 腿(2\*6)+腰(3)+臂(2\*5)=25 | ✅ | TBD | TBD | TBD |
| 15 | Tienkung `tienkung` | 腿(2\*6)+臂(2\*4)=20 | ✅ | TBD | TBD | TBD |
| 16 | PAL Robotics Talos `pal_talos` | 头(2)+臂(2\*7)+腰(2)+腿(2\*6)=30 | ✅ | TBD | TBD | TBD |
| 17 | Fourier GR3 `fourier_gr3` | 头(2)+臂(2\*7)+腰(3)+腿(2\*6)=31 | ✅ | TBD | TBD | TBD |
| 更多机器人敬请期待 |
| 19 | AgiBot A2 `agibot_a2` | TBD | TBD | TBD | TBD | TBD |
| 20 | OpenLoong `openloong` | TBD | TBD | TBD | TBD | TBD |




## 安装

> [!NOTE]
> 代码在 Ubuntu 22.04/20.04 上测试通过。

创建 conda 环境：

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
```

安装 GMR：

```bash
pip install -e .
```

若使用 SMPL-X 的 pkl 文件，安装 SMPLX 后请将 `smplx/body_models.py` 中的 `ext` 从 `npz` 改为 `pkl`。

解决可能的渲染问题：

```bash
conda install -c conda-forge libstdcxx-ng -y
```

## C++ 功能（实验性）

本仓库在 [`cpp/`](cpp/) 下提供基于 GMR 的实验性 C++ 重定向实现。

### 包含内容
- 后端解耦的 `Retargeter` 接口，输出重定向后的机器人 `qpos`。
- 两种后端实现：
  - `PinocchioRetargetBackend`
  - `MujocoRetargetBackend`
- 复用 `whole_body_control` 风格的 QP/HQP 求解栈（`qp_solver`、`hqp_solver`、`qp_data`）。
- 单帧重定向 CLI：`gmr_retarget_cli`。
- 带 YAML 配置的 MuJoCo viewer：`gmr_retarget_viewer`。

### C++ 依赖
- `Eigen3`
- `qpOASES`
- `pinocchio`
- `mujoco`
- `nlohmann_json` headers
- `yaml-cpp`

### 编译
```bash
cd /data/open_src_code/GMR_custom
cmake -S cpp -B cpp/build \
  -DGMR_THIRDPARTY_PREFIX=/opt/robot/devel/x86_64-Linux-GNU-9.4.0 \
  -DGMR_MUJOCO_PREFIX=/opt/robot/devel_control/x86_64-Linux-GNU-9.4.0
cmake --build cpp/build -j
```

### 快速运行
```bash
/data/open_src_code/GMR_custom/cpp/build/gmr_retarget_cli \
  --backend pin_ik \
  --gmr_root /data/open_src_code/GMR_custom \
  --robot unitree_g1 \
  --human_frame_json /data/open_src_code/GMR_custom/cpp/examples/human_frame_smplx_g1_example.json \
  --actual_human_height 1.7 \
  --damping 0.5 \
  --max_iter 10 \
  --use_velocity_limit \
  --out_json /data/open_src_code/GMR_custom/tmp/gmr_cpp_qpos.json
```

### Viewer 运行
```bash
/data/open_src_code/GMR_custom/cpp/build/gmr_retarget_viewer \
  --backend pin_ik \
  --config /data/open_src_code/GMR_custom/cpp/examples/retarget_viewer_config.yaml
```

### 后端名称
- `pin_ik`（别名：`pinocchio`、`pinocchio_ik`）
- `mujoco_se3`（别名：`mujoco`、`se3`）

完整说明见 [`cpp/README.md`](cpp/README.md)。

## 数据准备

[[SMPLX](https://github.com/vchoutas/smplx) 人体模型] 从 [SMPL-X](https://smpl-x.is.tue.mpg.de/) 下载模型到 `assets/body_models`，目录结构如下：
```bash
- assets/body_models/smplx/
-- SMPLX_NEUTRAL.pkl
-- SMPLX_FEMALE.pkl
-- SMPLX_MALE.pkl
```

[[AMASS](https://amass.is.tue.mpg.de/) 动作数据] 从 [AMASS](https://amass.is.tue.mpg.de/) 下载原始 SMPL-X 数据到任意目录。注意：不要下载 SMPL+H 数据。

[[OMOMO](https://github.com/lijiaman/omomo_release) 动作数据] 从 [Google Drive](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing) 下载，使用 `scripts/tools/convert_omomo_to_smplx.py` 转为 SMPL-X 格式。

[[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 动作数据] 从[官方仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)下载 BVH，即 [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)。


## 人体/机器人动作数据格式

**人体动作数据**每帧为字典：`{人体部位名: (全局平移, 全局旋转)}`。旋转默认四元数 **wxyz** 顺序（与 MuJoCo 对齐）。

**机器人动作数据**每帧可理解为：`(基座平移, 基座旋转, 关节角)`。

## 使用方法

### GUI 启动器（快速调试）

基于 **[Gradio](https://gradio.app)** 的 Web 界面：

```bash
conda activate gmr
cd /path/to/GMR_custom
pip install -e .
python scripts/gmr_gui.py
# 或: gmr-gui
# 或: python -m general_motion_retargeting.gui.app
```

浏览器打开 `http://127.0.0.1:7860`。GUI 代码在 `general_motion_retargeting/gui/`，脚本目录见 [`scripts/README.md`](scripts/README.md)。

### [NEW] PICO 流式遥操作（TWIST2）

安装 PICO SDK：
1. 在 PICO 设备上安装 PICO SDK，见[此处](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/)。
2. 在 PC 上：
    - 下载 [Ubuntu 22.04 deb 包](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb)，或从[源码](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)编译。
    - 安装命令：
        ```bash
        sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
        ```
        安装后可在应用列表看到 `xrobotoolkit-pc-service`，遥操作前请先启动。
    - 编译 PICO PC Service SDK 与 Python SDK：
        ```bash
        conda activate gmr

        git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git
        cd XRoboToolkit-PC-Service-Pybind

        mkdir -p tmp
        cd tmp
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
        cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
        bash build.sh
        cd ../../../..

        mkdir -p lib
        mkdir -p include
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
        cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/

        conda install -c conda-forge pybind11
        pip uninstall -y xrobotoolkit_sdk
        python setup.py install
        ```

安装完成后，可参考 [TWIST2 的 teleop.sh](https://github.com/amazon-far/TWIST2/blob/master/teleop.sh)：
```bash
bash teleop.sh
```
即可在 MuJoCo 窗口中看到重定向后的机器人动作。

### 从 SMPL-X（AMASS、OMOMO）重定向到机器人

> [!NOTE]
> 使用 SMPL-X pkl 文件时，安装后请将 `smplx/body_models.py` 中的 `ext` 从 `npz` 改为 `pkl`。

单条动作重定向：

```bash
python scripts/retarget/smplx_to_robot.py --smplx_file <smplx数据路径> --robot <机器人名> --save_path <输出.pkl> --rate_limit
```

默认会在 MuJoCo 窗口中可视化。录制视频请加 `--record_video` 和 `--video_path <视频路径.mp4>`。

- `--rate_limit`：按人体动作帧率限速播放；若追求最快速度可去掉该参数。

批量重定向：

```bash
python scripts/retarget/smplx_to_robot_dataset.py --src_folder <smplx目录> --tgt_folder <输出目录> --robot <机器人名>
```

批量模式默认不可视化。

### 从 GVHMR 重定向到机器人

先按 [GVHMR 官方说明](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) 安装，并运行单目视频 demo：

```bash
cd path/to/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

人体姿态保存在 `GVHMR/outputs/demo/tennis/hmr4d_results.pt`，然后执行：

```bash
python scripts/gvhmr/to_robot.py --gvhmr_pred_file <hmr4d_results.pt路径> --robot unitree_g1 --record_video
```

### 将 GMR pickle 转为 CSV

批量转换目录中的 `.pkl`，CSV 列顺序为 `root_pos + root_rot + dof_pos`：

```bash
python scripts/tools/batch_gmr_pkl_to_csv.py --folder <包含pkl的目录>
```

GVHMR 单目视频结果可能带有持续向上的 root 漂移，表现为机器人行走一段时间后双脚浮空。当前 GMR 的 `contact_ground` 会以已估计的地面偏移判断接触，并在短暂腾空时保持该偏移；重新运行 `gvhmr_to_robot.py` 即可在生成 pickle 前修正漂移。

对于修复前已经生成的旧 pickle，或重定向时关闭了 `contact_ground`，Unitree G1 的走路等始终有支撑脚的动作仍可在转换时启用兼容校正：

```bash
python scripts/tools/batch_gmr_pkl_to_csv.py \
  --folder <包含pkl的目录> \
  --ground-feet
```

`--ground-feet` 使用 G1 MuJoCo 模型逐帧计算左右脚高度，并只修正 CSV 中的 `root_pos.z`，使较低脚保持在地面接触高度；不会修改原始 pickle、root 水平轨迹、旋转或关节角。该选项会强制至少一只脚接地，不要用于包含真实跳跃或腾空阶段的动作。新数据优先在 GMR 层启用 `contact_ground`，保留真实腾空。



## 从 BVH（LAFAN1、Nokov）重定向到机器人

### 实时重定向（推荐）

`scripts/retarget/bvh_to_robot.py` 每帧运行 IK 并在 MuJoCo 中可视化。使用 `--rate_limit` 按实时速度播放，`--loop` 循环播放。

```bash
python scripts/retarget/bvh_to_robot.py \
  --bvh_file <bvh文件路径> \
  --robot unitree_h2 \
  --format lafan1 \
  --motion_fps 30 \
  --rate_limit \
  --loop
```

这与 `vis_robot_motion.py` 不同：后者仅**回放**已保存的 `.pkl`，**不会**重新运行 IK。

**Unitree H2 示例（LAFAN1）：**

```bash
python scripts/retarget/bvh_to_robot.py \
  --bvh_file /path/to/lafan1/walk1_subject5.bvh \
  --robot unitree_h2 \
  --format lafan1 \
  --rate_limit \
  --loop
```

显式开启接触/地面穿透修复（H2 IK 配置默认已启用）：

```bash
python scripts/retarget/bvh_to_robot.py \
  --bvh_file /path/to/lafan1/fallAndGetUp1_subject4.bvh \
  --robot unitree_h2 \
  --format lafan1 \
  --contact_ground \
  --fix_robot_penetration \
  --rate_limit \
  --loop
```

检查合并后的接触配置及脚/躯干碰撞体解析：

```bash
python scripts/analysis/inspect_contact_ground.py --robot unitree_h2 --src_human bvh_lafan1
```

### 重定向并保存为 pickle

```bash
python scripts/retarget/bvh_to_robot.py --bvh_file <bvh路径> --robot <机器人名> --save_path <输出.pkl> --rate_limit --format <格式>
```

默认会在 MuJoCo 窗口中可视化。
- `--rate_limit`：按人体动作帧率限速；追求最快速度可去掉。
- `--format`：BVH 格式，支持 `lafan1` 和 `nokov`。
- 可选接触/地面控制（可独立开关）：
  - `--contact_ground` / `--no-contact_ground`：人体脚接触对齐与脚锁定。
  - `--foot_ground_limit` / `--no-foot_ground_limit`：实验性 IK/QP 脚-地面不等式约束。
  - `--fix_robot_penetration` / `--no-fix_robot_penetration`：IK 后抬高 root 修复机器人穿地。
- 各机器人接触体 preset 见 [`docs/contact_ground.md`](docs/contact_ground.md)。历史模式对比见 [`docs/contact_modes_analysis.md`](docs/contact_modes_analysis.md)；录屏 A/B 用 `scripts/analysis/bvh_compare_contact_ground.py`。

接触模式分析示例：

```bash
conda run -n py310 python scripts/analysis/bvh_compare_contact_ground.py \
  --bvh_file /path/to/motion.bvh \
  --robot unitree_g1 \
  --format lafan1 \
  --motion_fps 30 \
  --out_dir analysis/contact_modes_lafan1_all
```


批量重定向：

```bash
python scripts/retarget/bvh_to_robot_dataset.py --src_folder <bvh目录> --tgt_folder <输出目录> --robot <机器人名>
```

批量模式默认不可视化。


### 从人体帧 JSON 重定向

若已通过 `scripts/tools/bvh_to_retargeting_frame.py` 等导出人体帧 JSON，可用 `scripts/retarget/human_json_to_robot.py` 做实时 IK + 可视化：

```bash
python scripts/retarget/human_json_to_robot.py \
  --human_frame_json <retarget_frame.json路径> \
  --robot unitree_h2 \
  --rate_limit \
  --loop
```

脚本会自动读取 JSON 元数据中的 `src_human` 和 `actual_human_height`。也支持接触/地面参数：

```bash
python scripts/retarget/human_json_to_robot.py \
  --human_frame_json <retarget_frame.json路径> \
  --robot unitree_h2 \
  --contact_ground \
  --fix_robot_penetration \
  --rate_limit \
  --loop
```



## 从 Xsens 重定向到机器人

### 离线：Xsens BVH 到机器人

#### 用 MuJoCo 可视化 Xsens BVH

安装 PyQt6：
```bash
pip install PyQt6 PyQt6-Qt6 PyQt6-sip
```

```bash
python general_motion_retargeting/utils/xsens_vendor/mujoco_xsens_bvh_view.py \
  --bvh_file <bvh路径> \
  --scale <位移缩放系数> \
  --reset_to_zero
```

示例：
```bash
python general_motion_retargeting/utils/xsens_vendor/mujoco_xsens_bvh_view.py \
  --scale 0.01 \
  --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
  --reset_to_zero
```

- `--start`：起始帧，默认从第 0 帧开始。
- `--end`：结束帧，默认处理到最后一帧。
- `--reset_to_zero`：将位移与绕 Z 轴旋转归零；与 `--start` 配合可更好对齐初始姿态。
- `--scale`：位移缩放，取决于 BVH 单位与米的换算关系。
- 运行前需安装 PyQt6。执行后会打开 UI，可调整各关节通道角度，点击「Apply and Preview」生成本地 `offset.json` 并 MuJoCo 预览。运行 `xsens_bvh_to_robot.py` 前需先执行此步骤。

#### 单条动作重定向
```bash
python scripts/retarget/xsens_bvh_to_robot.py \
  --bvh_file <bvh路径> \
  --robot <机器人名> \
  --save_path <输出.pkl> \
  --rate_limit \
  --start <起始帧> \
  --scale <位移缩放> \
  --reset_to_zero \
  --bvh_format <导出格式>
```

示例：
```bash
python scripts/retarget/xsens_bvh_to_robot.py  \
  --robot unitree_h1_2 \
  --scale 0.01 \
  --reset_to_zero \
  --bvh_format 3DSM \
  --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
  --save_path retargeting_data/h1/251021_04_boxing_120Hz_cm_3DsMax.pkl
```

默认 MuJoCo 可视化。`--rate_limit` 等同上文。`--bvh_format` 建议使用 3D Studio Max 格式。

导出的 pkl 中四元数为 `wxyz` 格式。

---

### 在线流式（Xsens MVN）

从 **Xsens MVN** 实时流式输入 GMR 进行重定向。

#### 1. 安装 Xsens MVN UDP 解析库

```bash
git clone https://github.com/jiminghe/xsens_mvn_robot_python.git
cd xsens_mvn_robot_python
pip install xsens_mvn_robot_python-*-cp310-*.whl
```

选择与 Python 版本匹配的 wheel（如 `cp310` 对应 Python 3.10）。

#### 2. 配置 Xsens MVN Network Streamer

在 Windows 或 Linux 上启动 **Xsens MVN**，可实时采集或回放 `.mvn` 文件。

| 步骤 | 操作 |
| ---- | ---- |
| 1 | 点击 **Options → Network Streamer** |
| 2 | 点击 **Add** 添加目标 |
| 3 | 设置 **Host Address**（见下表） |
| 4 | 仅勾选 **Position + Orientation (Quaternion)** |
| 5 | GMR 重定向无需其他数据源 |
| 6 | 点击 **OK**，确认流状态为绿色 |

**Host Address 参考：**

| 场景 | 地址 |
| ---- | ---- |
| MVN 与 GMR 在同一 Linux 机器 | `127.0.0.1` |
| Windows MVN → Ubuntu GMR（同局域网） | Ubuntu IP，如 `192.168.1.10` |

> Windows 推流到 Ubuntu 时，需在同一局域网，并对 MVN 放行 UDP 端口 `9763` 或关闭防火墙。

#### 3. 运行 GMR 实时流式脚本

```bash
conda activate gmr
python scripts/retarget/xsens_live_streaming.py
```

MuJoCo 窗口将实时显示重定向后的 Unitree G1 动作。

### 从 FBX（OptiTrack）重定向到机器人

#### 离线 FBX

1. 按[说明](https://github.com/nv-tlabs/ASE/tree/main/ase/poselib#importing-from-fbx)安装 `fbx_sdk`（建议独立 conda 环境）。
2. 提取 FBX 动作：

```bash
cd third_party
python poselib/fbx_importer.py --input <文件.fbx> --output <输出.pkl> --root-joint <根关节名> --fps <帧率>
```

3. 重定向到机器人：

```bash
conda activate gmr
python scripts/retarget/fbx_offline_to_robot.py --motion_file <动作.pkl> --robot <机器人名> --save_path <输出.pkl> --rate_limit
```

#### 在线流式

使用 OptiTrack MoCap 实时流式重定向。通常一台电脑运行 Motive（服务端），另一台运行 GMR（客户端）。

![OptiTrack Streaming](./assets/optitrack.png)

```bash
python scripts/retarget/optitrack_to_robot.py --server_ip <服务端IP> --client_ip <客户端IP> --use_multicast False --robot unitree_g1
```

### 可视化已保存的机器人动作

仅回放（不运行 IK）。叠加 IK 目标锚点时，传入对应的人体帧 JSON。默认显示**缩放后的 IK 目标**（与实时重定向一致），而非 JSON 原始坐标。

```bash
python scripts/viz/vis_robot_motion.py \
  --robot unitree_h2 \
  --robot_motion_path <机器人动作.pkl> \
  --human_frame_json <retarget_frame.json>
```

- `--show_human_body_name`：显示锚点部位名称。
- `--show_raw_human_targets`：对比显示未缩放的 JSON 原始目标。

无锚点回放：

```bash
python scripts/viz/vis_robot_motion.py --robot <机器人名> --robot_motion_path <动作.pkl>
```

录制视频请加 `--record_video` 和 `--video_path <视频路径.mp4>`。

批量可视化：

```bash
python scripts/viz/vis_robot_motion_dataset.py --robot <机器人名> --robot_motion_folder <动作目录>
```

MuJoCo 窗口快捷键：
* `[`：上一条动作
* `]`：下一条动作
* `space`：播放/暂停

### 重定向质量检查清单

目前尚无适用于所有机器人的单一自动评分。推荐流程：

1. **视觉检查**：用实时 `bvh_to_robot.py`，或 `vis_robot_motion.py --human_frame_json` 回放，观察脚锚点、是否交叉腿、地面接触。
2. **Pickle 统计**：从 `.pkl` 检查 `root_pos[:,2]`、膝关节均值、关节限位饱和比例。
3. **接触指标**：用 `scripts/analysis/bvh_compare_contact_ground.py` 分析脚滑与穿透。

H2 LAFAN1 的 IK 配置：`general_motion_retargeting/ik_configs/bvh_lafan1_to_h2.json`。机器人资产：`assets/unitree_h2/`。

## 速度基准

| CPU | 重定向速度 |
| --- | --- |
| AMD Ryzen Threadripper 7960X 24 核 | 60~70 FPS |
| 13 代 Intel Core i9-13900K 24 核 | 35~45 FPS |
| TBD | TBD |

## 引用

若本代码对您的研究有帮助，请引用：

```bibtex
@article{joao2025gmr,
  title={Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking},
  author= {Joao Pedro Araujo and Yanjie Ze and Pei Xu and Jiajun Wu and C. Karen Liu},
  year= {2025},
  journal= {arXiv preprint arXiv:2510.02252}
}
```

```bibtex
@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author= {Yanjie Ze and Zixuan Chen and João Pedro Araújo and Zi-ang Cao and Xue Bin Peng and Jiajun Wu and C. Karen Liu},
  year= {2025},
  journal= {arXiv preprint arXiv:2505.02833}
}
```

以及本仓库：

```bibtex
@software{ze2025gmr,
  title={GMR: General Motion Retargeting},
  author= {Yanjie Ze and João Pedro Araújo and Jiajun Wu and C. Karen Liu},
  year= {2025},
  url= {https://github.com/YanjieZe/GMR},
  note= {GitHub repository}
}
```

## 已知问题

为不同体型人体设计单一配置并非易事，部分动作重定向效果可能不佳。欢迎反馈！问题动作合集见 [TEST_MOTIONS.md](TEST_MOTIONS.md)。

## 致谢

IK 求解器基于 [mink](https://github.com/kevinzakka/mink) 与 [mujoco](https://github.com/google-deepmind/mujoco)。可视化基于 [mujoco](https://github.com/google-deepmind/mujoco)。人体动作数据包括 [AMASS](https://amass.is.tue.mpg.de/)、[OMOMO](https://github.com/lijiaman/omomo_release)、[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)。

机器人模型来源：

* [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets)：CC-BY-SA-4.0
* [Booster K1](https://www.boosterobotics.com/)
* [Booster T1](https://booster.feishu.cn/wiki/UvowwBes1iNvvUkoeeVc3p5wnUg)（[English](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f)）
* [EngineAI PM01](https://github.com/engineai-robotics/engineai_ros2_workspace)：[文件链接](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/src/simulation/mujoco/assets/resource)
* [Fourier N1](https://github.com/FFTAI/Wiki-GRx-Gym)：[文件链接](https://github.com/FFTAI/Wiki-GRx-Gym/tree/FourierN1/legged_gym/resources/robots/N1)
* [Galaxea R1 Pro](https://galaxea-dynamics.com/)：MIT license
* [HighTorque Hi](https://www.hightorquerobotics.com/hi/)
* [LEJU Kuavo S45](https://gitee.com/leju-robot/kuavo-ros-opensource/blob/master/LICENSE)：MIT license
* [PAL Robotics Talos](https://github.com/google-deepmind/mujoco_menagerie)：[文件链接](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_talos)
* [Toddlerbot](https://github.com/hshi74/toddlerbot)：[文件链接](https://github.com/hshi74/toddlerbot/tree/main/toddlerbot/descriptions/toddlerbot_active)
* [Unitree G1](https://github.com/unitreerobotics/unitree_ros)：[文件链接](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
* [Unitree H2](https://www.unitree.com/)：本仓库 `assets/unitree_h2/`
