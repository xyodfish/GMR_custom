# 重定向算法：可借鉴点与优化方向

> 记录人：AI 结对讨论整理
> 创建日期：2026-07-20
> 背景：学习 `robot_retargeter` 仓库后，对 GMR 重定向流水线的可迁移特性与更本质的优化方向做的一次梳理。
> 相关文档：[`docs/retarget_methods_comparison.md`](retarget_methods_comparison.md)

---

## 0. 已完成的迁移（现状）

| 来自 robot_retargeter 的机制 | GMR 现状 | 结论 |
|---|---|---|
| 接触检测 | 已有 `contact_ground` 流式检测 | 无需迁移 |
| 自适应地面高度 | 已有 `contact_ground` | 无需迁移 |
| 足端滑动抑制（脚锁） | 已有 foot-lock | 无需迁移 |
| 膝关节预弯（两段骨 IK） | 已试做 | **验证在 GMR 里价值低**（IK 用合适膝权重已能处理），不再投入 |
| 关节限位安全余量 | **已迁移**（`joint_limit_margin_deg`，Python + C++） | 见对比文档 §4b |
| 逐连杆缩放（per-link scaling） | 未迁移（GMR 用全局身高缩放） | **仍值得做**，见 A.1 |

---

## A. robot_retargeter 里仍值得借鉴的点

### A.1 逐连杆缩放（per-link scaling）—— 最值得做
- **现状**：GMR 用**全局身高缩放**（一个标量把整个人体缩放到机器人尺寸）。
- **robot_retargeter**：逐段连杆按 `L_robot / L_human` 缩放，且**保留原方向向量**（只改长度不改朝向），沿父→子拓扑重建关键点。
- **为什么有价值**：人体各肢体比例与机器人往往不一致（如机器人手臂相对腿更短）。全局缩放会让某些 link 的目标**系统性不可达** → IK 顶到限位。**这正是我们用 `joint_limit_margin_deg` 事后打补丁的根因之一。** 逐连杆缩放从源头减少不可达目标，比事后 clamp 更本质。
- **风险**：会改变 IK 目标的绝对位置，可能与 `contact_ground` 贴地逻辑冲突，需联调。

### A.2 按腿长比例缩放 root 位移 —— 小改良
- robot_retargeter 用"腿长比"缩放 root 的 xyz 帧间位移来匹配步幅；GMR 按身高缩放。
- 对腿身比例差异大的机器人（矮胖 / 长腿），腿长比更准。属于对现有缩放的增量优化。

### A.3 每只脚前后双接触点 —— 可选增量
- robot_retargeter 每只脚设脚跟 + 脚尖两个接触点。
- 若 GMR 现为单点，双点对 heel-toe 滚动步态的支撑/离地判定更稳，能减少接触相位抖动。价值中等，取决于 GMR 现有实现精度。

**取舍**：只推 A.1 认真做；A.2 / A.3 锦上添花。

---

## B. 自己的理解与优化方向（受启发，非照搬）

### 核心判断
当前所有实现（GMR / robot_retargeter）本质都是**运动学层面的多目标 IK/QP**，缺两件事：
1. 让目标先落到机器人**可行流形**上再优化；
2. 用**"可跟踪性"**而非"运动学平滑度"当真正的目标函数。

我们加的 margin、脚锁、jerk 惩罚，都是在事后修补这两个缺口。以下方向按投入产出排序。

### 方向 1（最看好）：把"缩放源"换成"目标投影到可行流形"
- 现范式：缩放人 → 硬跑 IK → 顶限位就 clamp/加 margin（**对抗式**）。
- 新思路：**离线一次性**算机器人各末端/连杆的可达工作空间（采样或几何），在线时把人体目标先**投影到最近可达点**再做 IK。IK 不再和不可达目标较劲，limit 饱和 / self-collision 从源头减少，margin 反可调小。
- per-link scaling（A.1）是本思路的粗糙特例（用"长度匹配"近似"可达"）；投影更通用，能同时处理**长度、限位、自碰撞**三种不可达。

### 方向 2（改动最小，建议先做）：相位驱动的连续代价权重
- 现状：接触为真→锁脚的**二值切换**，是 online_qp/smooth 在 gait 上 jerk 反而 +23% 的元凶（切换点产生冲击）。
- 新思路：用短窗算**接触置信度 ∈ [0,1]**，让脚位权重 / 上肢跟踪权重随相位**平滑过渡**；再利用 online_qp 已有的 lookahead **提前 ramp**（接触前几帧就开始加脚位权重）。
- 收益：把"硬切换"变"预测性软调度"，**同时降滑脚和降 jerk**，不用互相牺牲。

### 方向 3：优化目标从"运动学平滑"升级到"控制可行"
- 真实目标是"让 RL/PD 更容易 track"；jerk 只是**代理指标**。
- 新思路：代价里加**轻量逆动力学可行项**——用重力+惯性估计算近似关节力矩，惩罚**接近力矩/速度饱和**的轨迹。
- 与 `joint_limit_margin_deg` 同一哲学（"离边界留余量"），但从**位置边界推广到力矩/速度边界**。做出来对下游控制器友好度会有质变。
- **已实证，见 §B2。** 结论：只有"ID 力矩软约束 @ 前瞻优化器"这一种落法有效；逐帧限幅 / 静态重加权 / 速度感知重加权都无益甚至有害。

### 方向 4：分层求解（低频全局 + 高频局部）
- 现状：单体窗口 GN 同时解 root + 全身。
- 新思路：按频率拆变量——root 轨迹、接触时序、地面高度这些**低频、全局强耦合**量，用长时域便宜地全局求解一次；再**固定这套 root/contact 计划**，对高 DOF 四肢做快速逐帧 IK。
- 收益：拿到 Batch-TO 级全局一致性（无漂移、无滑脚），速度接近逐帧 IK。也解释了为何 online_batch 扁平窗口两头不讨好。

### 方向 5（长期）：学习式残差 warm-start
- online_qp 是**手调权重**。把逐帧 IK 当基座，离线蒸馏一个**小残差修正**（网络或按动作缓存的查找表），专修机器人特有的系统性误差（自碰撞、特定限位）。在线只花一次前向代价，逼近离线 Batch-TO 质量。把"手调约束"变"数据驱动约束"。

### 反向（减法）
- **膝预弯**：已验证 GMR 里价值低，不再投入。
- **不要继续堆事后 clamp 类补丁**：margin 是好用的止血，但方向 1（可行流形投影）才治本；补丁堆多会互相打架。

---

## B2. 方向 3 实证结果（2026-07-20）

在 `unitree_g1` 上对方向 3 做了完整验证：先建评估器测 headroom，再试 4 种"控制可行"落法。

### 评估器
[`scripts/analysis/benchmark_control_feasibility.py`](../scripts/analysis/benchmark_control_feasibility.py)：对输出轨迹用逆动力学(`mj_rne`，有限差分 q̇/q̈)算每关节 `|τ|/τmax`，报告**力矩峰值 / 饱和帧(>0.8) / 速度饱和 / FK 跟随代价 / jerk**。上半身(腰+臂)力矩与接触无关、可信；支撑腿力矩需地反力，仅作参考。

> 踩坑：`mj_inverse` 若不先 `mj_forward` 会返回垃圾力矩(出现 673% 的伪值)。改用 `mj_forward` + `mj_rne` 后正常。

### 基线 headroom（关键前提）
| clip | 重力力矩峰值 | 惯性力矩峰值 | 速度峰值 |
|---|---|---|---|
| walking | 0.50 | 2.06 | 17.3 rad/s |
| tennis | 0.47 | 2.73 | 21.2 rad/s |

- **重力(静态)力矩根本不吃紧**(峰值≤50% τmax)——逐帧 IK 能塑形的那部分本就可行。
- 表面上的"力矩/速度饱和"**主要是逐帧 IK 抖动的伪影**(baseline jerk≈1013；walking 肘不可能真甩到 17 rad/s)，叠加腿部无接触力使腿力矩不可信。真正的可行性信号是**抖动**。

### 四种落法对比（越界=坏）
| 方法 | 机制 | 位置 | 结果 |
|---|---|---|---|
| 逐帧力矩限幅 | 因果硬 clip 加速度 | `motion_retarget.py`（原生逐帧 IK） | ✗ 滞后/追赶，峰值反升 |
| 静态力矩加权平滑 | 无差别 `(M/τmax)²` 权重 | Batch TO | ✗ 饿死轻关节，峰值/jerk 更差 |
| 速度感知平滑 | 按实测速度动态加权 | Batch TO | ~ vpeak 略降，FK/力矩 net 更差 |
| **ID 力矩软约束** | **只惩罚越界部分 τ>κτmax** | **Batch TO（前瞻）** | ✓ **同 FK 下峰值 −10~13%** |

关键对照(Batch TO，固定 `w_accel=10`，scope=upper)：

```
walking            FKcost  ALLpeak            tennis            FKcost  ALLpeak
uniform             6.026    2.42             uniform            5.537    3.19
ID约束 w=10~40      5.998    2.10 ✓           ID约束 w=10        5.491    2.86 ✓
ID约束 w=250        6.096    2.41 (反噬)       ID约束 w=250       5.573    5.49 (反噬)
```

online_qp(preset=anti_slip，lookahead)上收益更大——因为它平滑较弱、基线力矩峰值更高、headroom 更多：

```
tennis             FKcost  ALLpeak            walking           FKcost  ALLpeak
baseline            5.525    3.95             baseline           6.007    2.14
ID约束 w=10         5.535    2.43 (−38%)✓     ID约束 w=10        6.015    2.10
ID约束 w=40         5.533    2.49             ID约束 w=40        6.022    2.48 (反噬)
```
（tennis：峰值 −38% 而 FK/jerk/速度几乎不动；walking 上半身 UBpeak 2.11→1.98，但 ALLpeak 受未约束的支撑腿主导，改善有限。）

### 为什么只有 ID 软约束有效
- **约束 vs 重加权**：barrier 只在真正越界的帧/关节上施力(外科手术式)，其余轨迹不动 → FK 不受损；**前瞻窗口**提供真实 q̈，不会有逐帧因果限幅的滞后/追赶。
- **必须轻推**：越界力矩里混了大量逐帧 IK 抖动，且 GN 用对角惯性线性化(忽略科氏/∂M)不精确；权重一大就把越界帧的尖峰"搬家"到邻帧，峰值反升。甜点 `weight≈10~40`。
- **腿部要谨慎**：支撑腿力矩无地反力不可信，默认 `scope="upper"` 只约束腰+臂。

### 实现（默认关闭）
- 逐帧限幅：`GeneralMotionRetargeting(control_feasibility=..., cf_mode=..., cf_margin=...)`（[`control_feasibility.py`](../general_motion_retargeting/control_feasibility.py)）。**实测无益，仅保留供复现负结论。**
- Batch TO：`BatchTrajectoryConfig` 新增 `torque_weighted_smoothing` / `vel_aware_smoothing` / `torque_limit_constraint`(+`torque_limit_margin`/`torque_limit_weight`/`torque_limit_scope`)。**推荐仅用 `torque_limit_constraint`**，配置 `scope="upper", margin=0.1, weight≈10~20, w_accel=10`。
- online_qp：`OnlineQpConfig` 同名字段，前瞻档下同样可开。
- **C++ 已移植**：`BatchTrajectoryConfig` / `OnlineQpConfig`(C++) 新增 `torqueLimitConstraint` / `torqueLimitMargin` / `torqueLimitWeight` / `torqueLimitScope`；GN 项在 `BatchTrajectoryRetargeter::accumulateWindowTorqueLimitGn`(RNE+`mj_fullM` 有限差 q̈)，同时供 `optimizeQpWindow`(online) 与 batch 复用。CLI:`gmr_online_qp_cli` / `gmr_batch_to_cli` 加 `--torque_limit_weight/--torque_limit_margin/--torque_limit_scope`。online_qp 开启后 ms/frame ≈ +1（7.4→8.3），仍实时@30。

### 多 case 实测（结论：clip 相关，默认关闭）
在 6 个 gvhmr clip 上对比 online_qp 基线 vs `tqLim(w=10, scope=upper, margin=0.1)`：

```
Python online_qp                       C++ online_qp
clip           ALLpeak base→tq         clip           ALLpeak base→tq
tennis          3.95→2.43 (−38%)✓      walking         2.63→1.89 (−28%)✓
walking         2.14→2.10 (−2%)        gait_track_14   3.21→3.01 (−6%)✓
gait_track_12   1.25→1.25 (no-op*)     gait_track_12   0.97→0.97 (no-op*)
gait_track_14   2.85→2.88 (+1%)        gait_track_4    3.17→3.17 (no-op**)
sports2d_walk  19.12→20.31 (+6%†)      tennis          3.07→3.43 (+12%‡)
                                       sports2d_walk  20.58→23.02 (+12%†)
```
`*` 上肢峰值已 <κτmax → barrier 不触发,安全 no-op；`**` 超限在腿(scope=upper 不管)；`†` 峰值 19~20x 的病态 clip(GVHMR 数据差/FK 崩)；`‡` 该端基线已较低(3.07)、headroom 小,轻推反把尖峰搬家。

要点：
- **收益取决于基线是否真有可平滑的越界 headroom**。C++/Python 的赢家不同,是因为两端 online_qp 基线轨迹不同(C++ walking 基线 2.63 高→大赢;C++ tennis 基线 3.07 低→反噬)。
- **同一份 C++ 代码 walking −28% / batch_to walking −9% 证明数学(符号/GN 项)正确**;正/负号随 clip 波动是"软惩罚 min 平方越界 ≠ min 峰值"在 3 帧因果窗口下的固有性质。
- 病态/已收紧的 clip 上可轻微反噬,高权重(w20~40)可缓解 tennis 反噬(+12%→+4%)。**故默认关闭,按 case 开启并先测 headroom。**

### 一句话
对 G1 这类中轻载全身动作：**"控制可行 ≈ 抖动可行"**，抖动最好由已有前瞻平滑处理；若要把力矩当主动约束，唯一有效的是**前瞻优化器里的 ID 力矩软约束(轻推)**，稳健收益约 10%。要有质变需重载/负重场景或纳入接触的全身动力学(属下游控制器范畴)。这与膝预弯同源：**先测 headroom，没 headroom 别硬加约束**。

---

## C. 一句话总结
`robot_retargeter` 给的启发不是某个具体 feature，而是"把重定向拆成 **缩放 → 接触 → 约束** 的显式流水线"这个视角。但下一步的杠杆不在再加几何 trick，而在：
1. **让目标先可行**（方向 1）；
2. **用可跟踪性当目标**（方向 3）；
3. **分层求解**（方向 4）。

这三点是 GMR 和 robot_retargeter 现在都没有、且能带来质变的地方。

## D. 建议的下一步
先做**方向 2（连续接触置信度）**：改动最小、能直接压 online_qp 的 gait jerk 尖峰，作为验证"预测性软调度"思路的原型。
