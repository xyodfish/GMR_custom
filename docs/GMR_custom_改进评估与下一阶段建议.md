# GMR_custom 改进评估与下一阶段建议

## 1. 总体判断

你的仓库已经不只是“给 GMR 加补丁”，而是形成了一个完整的增强型 retargeting 系统：

- GMR 预处理与 IK bootstrap
- 自碰撞约束
- 接触检测与脚锁定
- 地面穿透修复
- 因果轨迹优化
- 离线 Batch TO
- Python / C++ 双实现
- 性能、消融与 parity 验证

但当前方法仍主要停留在：

\[
	ext{Geometry}
+
	ext{Smoothness}
+
	ext{Contact Heuristics}
\]

因此继续增加 penalty、规则和权重，边际收益会越来越低。

## 2. 当前已完成的核心改进

### 2.1 自碰撞约束

你已经将手臂—躯干、左右手臂、手臂—腿、左右腿、左右脚、手臂—头部等碰撞对接入 Mink IK limit。

这属于合理的约束级增强，而不是简单后处理。

当前问题是碰撞对仍偏向 Unitree G1 特化。后续应考虑根据机器人运动树、SRDF 或 collision group 自动生成候选碰撞对，而不是继续人工枚举。

### 2.2 Contact-ground 流水线

当前流程为：

```text
人体接触检测
→ 人体地面对齐
→ 接触脚 EMA 锁定
→ GMR IK
→ 机器人穿透修复
```

你还将以下功能解耦成独立开关：

- `contact_ground`
- `foot_ground_limit`
- `fix_robot_penetration`

实验已经说明：

- `contact_ground` 可略微降低 foot slip，但可能增加全身穿透；
- `foot_ground_limit` 单独效果有限；
- `fix_robot_penetration` 可以稳定消除穿透，但会产生 root lift；
- 全部开启并不一定取得最佳 tracking 和 foot slip。

这说明当前 contact pipeline 更像几何修复系统，而不是统一的接触优化系统。

### 2.3 GMR + L-BFGS / L-BFGS-B

你的仓库已经构成了一个明确的 GMR + L-BFGS 案例。

因果模式中，你优化：

\[
q_t=
rg\min_q
L_{\mathrm{FK}}
+
w_v\|q-q_{t-1}\|^2
+
w_a\|q-2q_{t-1}+q_{t-2}\|^2
\]

并使用轻量 IK warm start。

Full 模式则在窗口内联合优化：

- FK 位置和姿态跟踪
- 速度平滑
- 加速度平滑
- 窗口 anchor
- 关节上下界

这已经不是单纯替换 GMR 求解器，而是完整的时序轨迹优化。

### 2.4 Batch TO

你的 Batch TO 流程是：

```text
人体序列
→ GMR 预处理
→ 逐帧 IK bootstrap
→ 接触 mask
→ 滑窗多帧 Gauss–Newton
→ overlap 融合
→ contact finalize
```

目标函数包括：

\[
J(Q)=
J_{\mathrm{FK}}
+
J_{\mathrm{velocity}}
+
J_{\mathrm{acceleration}}
+
J_{\mathrm{anchor}}
+
J_{\mathrm{foot}}
\]

foot penalty 又包含：

- foot height
- foot slip
- foot IK anchor
- root XY contact
- contact joint anchor

此外，你已经完成：

- Python / C++ 对齐
- dense / banded GN
- 多步 line search
- quality / fast 配置
- 约 300 FPS 的 C++ Batch TO

这部分工程基础已经非常扎实。

## 3. 当前方法的结构性瓶颈

### 3.1 Contact mask 依赖 GMR 输出

当前接触状态主要由 IK bootstrap 轨迹推断：

\[
Q_{\mathrm{GMR}}
ightarrow
C_{\mathrm{contact}}
ightarrow
Q_{\mathrm{optimized}}
\]

如果 GMR 脚高或脚速判断错误，后续 foot penalty 会强化错误接触。

更合理的方向是将接触作为软变量联合优化：

\[
c_{t,f}\in[0,1]
\]

并加入：

\[
c_{t,f}\|v_{f,t}^{xy}\|^2
\]

\[
c_{t,f}(z_{f,t}-z_g)^2
\]

\[
\lambda_b c_{t,f}(1-c_{t,f})
\]

### 3.2 Anchor 项会限制机器人重新分配姿态

当前较强的 foot IK anchor、root XY anchor 和 contact joint anchor，会让结果过度贴近 GMR bootstrap。

这会限制机器人根据自身构型重新分配髋、膝、踝、腰部和躯干姿态。

建议将 joint anchor 改成按关节分组的 trust region，而不是全局固定权重。

### 3.3 固定权重无法适配所有动作

走路、跳跃、跌倒、起身和舞蹈需要完全不同的约束优先级。

因此不存在一组固定的：

\[
w_{\mathrm{slip}},
w_{\mathrm{height}},
w_{\mathrm{anchor}},
w_{\mathrm{smooth}}
\]

能够适配所有动作。

继续做全局权重搜索，容易退化为自动化调参，而不是范式突破。

### 3.4 Root lift 是后处理，不是联合优化

当前穿透修复本质上是：

\[
q^{IK}
ightarrow
q^{IK}+[0,0,\Delta z_{\mathrm{root}}]
\]

它可以消除穿透，但可能破坏脚底接触、末端跟踪、root motion、低姿态动作和后续动力学可执行性。

更合理的是把 root height 与关节一起优化，让修正分布在髋、膝、踝和躯干之间。

### 3.5 Overlap averaging 可能破坏流形和接触一致性

滑窗独立优化后再欧氏融合，可能导致：

- floating-base rotation 不连续
- 四元数符号问题
- 重叠区脚底重新滑动
- 两个窗口分别满足接触，但融合后不满足

应考虑：

- SO(3) log/exp 融合
- quaternion sign alignment
- 全局 banded sparse solve
- 融合后的 contact projection

## 4. 为什么继续加 penalty 会进入瓶颈

当前系统主要回答的是：

> 如何生成更平滑、更少滑脚、更少穿透的几何轨迹？

但真正需要回答的是：

> 如何生成更容易被机器人控制策略稳定执行的动作？

目前还没有把以下量放进 retargeting 目标：

- tracking policy
- 力矩饱和
- 接触力
- 摩擦约束
- 扰动鲁棒性
- episode completion
- sim-to-real feasibility

因此问题不在 GN、L-BFGS 或 QP 谁更好，而在 objective 仍然是纯运动学的。

## 5. 建议保留的现有基础

以下部分应继续保留并作为研究基座：

1. GMR 预处理与 IK bootstrap
2. 人体尺度与 target 解析
3. MuJoCo FK / Jacobian
4. C++ dense / banded GN
5. Python / C++ parity 测试
6. 接触与穿透分析工具
7. 多格式统一输入
8. IK、Causal TO、Batch TO 的统一接口

你的现有系统已经足够成熟，不需要推倒重来。

## 6. 下一阶段：Trackability-Aware Retargeting

下一步建议在现有 Batch TO 上加入：

\[
J(Q)
=
J_{\mathrm{existing}}(Q)
+
\lambda_{\pi}J_{\mathrm{trackability}}(Q)
\]

其中：

\[
J_{\mathrm{trackability}}(Q)=C_\psi(Q)
\]

\(C_\psi\) 是通过 BeyondMimic rollout 训练出的可执行性 critic。

### 6.1 构造 rollout 数据

分别生成并评估：

- 原始 GMR
- GMR + contact_ground
- Causal TO
- Batch TO
- Batch TO without foot penalties

每条动作运行多次 rollout，记录：

\[
y=
[
	ext{success},
	ext{completion},
e_{\mathrm{body}},
e_{\mathrm{joint}},
e_{\mathrm{contact}},
e_{	au},
	ext{termination}
]
\]

### 6.2 先验证现有指标是否真正相关

分析：

\[
ho(	ext{jerk},	ext{success})
\]

\[
ho(	ext{foot slip},	ext{success})
\]

\[
ho(	ext{IK error},	ext{success})
\]

\[
ho(	ext{root lift},	ext{success})
\]

这一步可以验证当前几何指标是否真的能代表策略可执行性。

### 6.3 训练 Trackability Critic

输入窗口：

\[
z=
[
q,\dot q,\ddot q,
p_{\mathrm{foot}},
v_{\mathrm{foot}},
v_{\mathrm{root}},
\omega_{\mathrm{root}},
c_{\mathrm{foot}}
]
\]

输出：

\[
C_\psi(z)=
[
P_{\mathrm{failure}},
\hat e_{	au},
\hat e_{\mathrm{tracking}},
\hat e_{\mathrm{contact}}
]
\]

### 6.4 接入现有优化器

第一版直接接入 Python L-BFGS-B：

\[
Q^*
=
rg\min_Q
J_{\mathrm{FK}}
+
J_{\mathrm{smooth}}
+
J_{\mathrm{contact}}
+
\lambda C_\psi(Q)
\]

这样无需立即修改 C++ GN，也无需使用可微物理。

验证有效后，再考虑：

- ONNX / C++ critic
- critic Jacobian
- SQP / GN surrogate
- 策略闭环优化

## 7. 最终定位

### 已完成：GMR++ 工程增强平台

- collision-aware
- contact-aware
- ground-aware
- temporal-aware
- Python / C++ 双实现
- 实时与离线模式
- 定量评估与 parity

### 当前阶段：Kinematic trajectory optimization retargeting

\[
	ext{Geometry}
+
	ext{Smoothness}
+
	ext{Contact Heuristics}
\]

### 下一阶段：Trackability-aware retargeting

\[
	ext{Motion Semantics}
+
	ext{Robot Embodiment}
+
	ext{Policy Executability}
\]

## 结论

你的现有工作并不是无效的“添油”，而是在构建一个足够成熟的 retargeting 研究平台。

但继续增加 foot penalty、平滑项和规则的收益会快速下降。最合适的转折点是：

> 停止把 jerk、foot slip 和 penetration 当作最终目标，开始使用 BeyondMimic rollout 的真实可执行性反向指导 Batch TO。
