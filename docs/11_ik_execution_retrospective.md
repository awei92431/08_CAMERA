# 从 mocap/weld 到 DLS IK + 关节执行器：完整技术复盘

## 0. 复盘范围与证据规则

这次工作不是重新训练策略，而是把同一个旧 PPO checkpoint 从一种近似运动学式的仿真执行方式，零样本迁移到更接近真实机器人软件结构的执行链，并逐层修复迁移暴露的问题。

本文按三类证据区分结论：

- **代码事实**：当前或原项目代码直接显示的控制路径与参数；
- **实验结论**：CSV、JSON、视频与可复现实验脚本支持的观测；
- **合理推断**：由多组实验共同支持、但没有被单个传感量直接观测的解释。

`07_test` 没有 `.git`；`06_4c2/.git` 也是缺少 HEAD 的空目录，因此没有可用 commit history。所谓“修改记录”只能由 `06_4c2`/`07_test` 当前文件差异、编号 docs 和原始 results 重建，本文不虚构 Git 时间线。

主要证据入口：

- 原/新首轮对照：`results/mocap/evaluation.json`、`results/ik/evaluation.json`
- 目标积压：`docs/02_ik_controller_diagnosis.md`、`results/diagnostics/`
- lead limit：`docs/03_tcp_lead_limit_test.md`、`results/lead_limit/`
- IK/actuator误差拆分：`docs/04_steady_state_bias_test.md`、`results/steady_state_bias/`
- kp与重力静差：`docs/05_actuator_tracking_test.md`、`results/actuator_tracking/`
- Place控制与实现分层：`docs/07_place_control_diagnosis.md`、`docs/08_place_ik_realization_diagnosis.md`
- posture消融：`docs/09_nullspace_posture_test.md`、`results/nullspace_posture/`
- 最终验证：`docs/10_final_ik_baseline.md`、`results/final_ik_baseline/`

## 1. 原项目并不是真正通过 IK 控制

### 1.1 代码里有 IK，不等于控制链用了 IK

原项目确实存在 `_solve_ik()`：它基于 `pinch` site 的位置与 approach-axis 误差，计算 Jacobian DLS，阻尼 `1e-4`，最多 15 次迭代，每次关节增量限制在 ±0.04 rad，并带 `0.02*(home-q)` posture 项。单看这个函数，很容易把系统描述成“IK控制”。

但真正决定控制方式的是 action 到 MuJoCo 受力之间的可达调用链。原 `06_4c2/fourc2/envs/allstage.py` 的 `_apply_tcp_action()` 并不调用 `_solve_ik()`：

1. PPO 输出四维 action；前三维形成 Cartesian TCP 增量；
2. FSM 对 Reach/Grasp/Lift/Place 的位移进行缩放、安全裁剪和平滑；
3. 目标写入 `data.mocap_pos/quat`；
4. `scene*.xml` 中的 `mocap_tcp_weld` 把 mocap target 与 `pinch` site 软焊接；
5. 每个 substep 还把机械臂 actuator ctrl 设回当前 qpos，即 `_neutralize_arm_actuators()`，让 position servo 基本不主动驱动机械臂；
6. TCP 实际由 equality weld 的约束力拖向 mocap target。

因此原真实链路是：

```text
PPO Cartesian action
  → TCP target / FSM / safety / smoothing
  → mocap body
  → soft weld equality force
  → robot TCP
```

而不是：

```text
Cartesian target → IK → joint target → joint actuator
```

实验也直接证明这一点：mocap基线 seeds 0～9 为 10/10，但 `ik_solve_calls=0`、`weld_present=true`（`results/mocap/evaluation.json`）。这是本项目最重要的审计经验：不能根据“仓库中存在某个算法函数”判断实际控制方式，必须沿调用路径追到最终 actuator/constraint。

另一个需要准确表述的事实是：PPO 第四维没有直接学习夹爪开合。`_set_gripper_from_action()` 使用 `_scripted_gripper_normalized()`，夹爪由 FSM 脚本控制。因此本文只讨论旧策略的 Cartesian 移动输出在新执行层上的迁移。

## 2. 接入真实 IK 执行链

迁移后的结构保留原来的 observation、四维 action、FSM、reward、success、workspace safety、目标平滑和脚本夹爪，只替换机械臂执行层：

```text
旧 PPO action
  → 原 Cartesian target generation
  → 30 mm target lead bound
  → DLS IK on independent ik_data
  → six-joint q_target
  → six position actuator ctrl
  → MuJoCo dynamics / contacts / gravity
```

关键代码事实见当前 `fourc2/envs/allstage.py`：

- `mocap_target` 仍被写入，但 XML 中已经没有 `mocap_tcp_weld`，只用于可视化；
- `_apply_tcp_action()` 每个 control step 实际调用 `_solve_ik()`；
- 同一个六关节 `q_target` 在 `frame_skip=10` 个 MuJoCo substep 中持续写入 position actuators；
- action执行路径不直接写真实 arm qpos；候选关节值只写入独立的 `ik_data.qpos` 做 FK/Jacobian计算；
- 运行回归测试见 `tests/test_ik_execution.py`。

首轮切换很残酷：旧 mocap 为 10/10，新 IK 为 0/10，平均/最大 TCP error 32.97/283.96 mm，虽然 `_solve_ik()` 被调用 7,475 次且 weld 已移除（`results/ik/evaluation.json`）。这说明迁移不是把一个等价模块换个名字，而是把旧策略第一次暴露在有限关节带宽、重力静差、关节目标实现误差和接触动力学下。

## 3. 问题一：TCP目标持续积压

### 发现异常

新链路出现约 284 mm 的巨大 tracking error，直觉上像 IK 发散、坐标系错误或阶段切换跳变；完整任务 0/10。

### 假设集合

- 阶段切换重置了目标，造成突跳；
- Jacobian/DLS数值发散；
- 关节达到限位；
- 机器人碰桌或异常接触；
- actuator跟不上快速变化的 `q_target`；
- Cartesian target 在真实TCP落后时继续累加。

### 诊断设计与排除

第一步不是调阻尼，而是按 stage 记录 target、actual TCP、q_target、qpos、clip、contact和transition jump。结果显示：

- Reach mean/P95/max 为 151.89/273.68/283.96 mm；
- Place只有14.22/24.17/41.04 mm；
- 所有 episode 最大值都在 Reach step 29～40；
- Reach→Grasp target jump ≤2.4 mm，Grasp→Lift ≤3.6 mm，Lift→Place <0.015 mm；
- 最大误差处没有 joint-limit clip、robot-table contact；
- IK用满15轮且dq clip，`||q_target-qpos||=0.832 rad`；15轮×单轮0.04 rad允许一次solve将某些关节目标推远约0.60 rad。

固定目标18组 step response 又显示系统不是普遍数值发散：持有4 s时无持续振荡、无关节限位、无IK数值失败，稳态误差约8～10 mm。

### 根因

代码审计发现 Reach/Grasp/Lift 以旧 `tcp_target_pos` 为base继续叠加新action，而不是以 measured TCP 重建。当真实关节执行层落后时，下一次策略动作仍向前推数学目标，形成 backlog；IK一次又能生成很远的关节目标，有限actuator响应进一步放大差距。

### 单变量修复

在安全/平滑后、进入IK前，限制 `||tcp_target-actual_tcp||`。测试20/25/30 mm：

| lead limit | Reach/Grasp/Lift/Place entry | full |
|---|---:|---:|
| 无限制 | 10/8/8/8 | 0/10 |
| 20 mm | 2/1/1/1 | 1/10 |
| 25 mm | 7/7/7/7 | 1/10 |
| 30 mm | 10/10/10/10 | 1/10 |

20 mm虽然误差更小，却破坏了旧策略所依赖的响应时序；30 mm把Reach post-actuation最大误差压到29.77 mm，同时保留所有stage handoff。因此选择30 mm体现的是性能/稳定性权衡，而不是盲目追求最小tracking error。

## 4. 问题二：IK数学误差还是actuator跟踪误差

lead被控制后仍有8～10 mm固定目标偏差。可能原因包括axis权重、posture正则、IK残差、position actuator静差或force saturation。

关键诊断是把总误差拆成两个可相加向量：

```text
target - actual
= (target - FK(q_target))       # IK数学目标误差
+ (FK(q_target) - actual)       # actuator/动力学实现误差
```

在相同18个固定目标上：

- 当前axis 0.35/posture 0.02：IK误差均值0.615 mm；actuator误差9.020 mm；实际3D误差8.856 mm；
- 关闭posture：IK误差降到0.089 mm，但actual误差略变差到8.998 mm；
- axis减半：几乎没有变化；
- signed误差主要是Z方向+8.777 mm。

这排除了“局部8～10 mm静差主要由DLS精度或axis/posture权重造成”。需要注意，这只是home附近固定目标的局部结论；它没有证明raw posture在远离home的Place状态也安全。后续Place实验正是对这种局部结论的边界修正，而不是数据自相矛盾。

## 5. 问题三：关节伺服重力静差与kp

### 诊断

逐关节记录 `q_target-qpos`、actuator force、`qfrc_bias`、forcerange和局部FK贡献：

- shoulder lift：-8.96 mrad，约17.92 Nm；
- elbow：-8.75 mrad，约17.50 Nm；
- wrist 1：-3.42 mrad，约1.71 Nm；
- 最大力22.50 Nm，远低于肩/肘±150 Nm与腕部±28 Nm；
- steady actuator force与`qfrc_bias`相符，残差随kp近似反比下降。

### 根因

这是有限增益position servo为了平衡重力/负载而产生的静态位置误差，不是force saturation，也不是需要立刻加大forcerange。

### 单变量实验

实验脚本以当时XML值为1×，运行时测试1×/1.5×/2×：稳态误差8.856→5.907→4.438 mm，actuator误差9.020→6.004→4.499 mm，rise time 0.431→0.313→0.236 s；2×无振荡、碰桌或饱和。完整任务中Place mean/P95从14.08/17.32降到11.38/13.76 mm，但full仍只有1/10。

这里必须澄清历史命名：`scripts/actuator_tracking.py`、`evaluate_full_task_kp2.py` 的“2×”是对加载后模型gain再乘2；当前`06_4c2`和`07_test` XML都已经是绝对gain 2000/500，且当前正式baseline不再运行时倍乘。由于缺少Git历史，不能从当前文件证明这些值相对哪个更早版本是2×。旧实验仍能证明“提高kp会减少重力静差”，但最终60-seed结果必须按当前绝对gain 2000/500理解。

## 6. 问题四：Place明明命令朝目标，为什么物体反而远离

kp改善精度后，Reach/Grasp/Lift/Place entry均10/10，但9个Place失败。最初合理怀疑：

- PPO residual 与内置servo互相打架；
- `goal-object` 的world-frame符号写反；
- release FSM有隐藏计数器或条件重置；
- actuator tracking仍把正确q_target做反；
- 物体在夹持中滑动；
- IK中的姿态目标把小XY命令扭曲。

### 6.1 A/B/C/D先排除策略与servo符号

比较A policy+servo、B servo-only、C policy-only、D oracle。A/B/D的最终Cartesian命令在全部非零step都朝goal，但actual TCP朝goal比例仅1.4%/11.0%/6.1%，oracle仍只有1/10。于是：

- PPO与servo确实冲突，但不是决定性根因；
- `goal_xy-object_xy` 符号正确；
- 即使oracle target正确，错误仍发生在下游。

seed 9还把release条件审计到边界：XY在Place step 11～23有效，最低10.485 mm；step24高度首次有效时XY=24.002 mm，比24 mm阈值多2 µm，因此两个条件没有同时成立。没有隐藏confirm counter被清空。

### 6.2 四层分解定位第一处反向

只看 `tcp_target_error≈11 mm` 会误导，因为Place每步从当前TCP附近rebase，短lead不代表对全局goal有进展。于是把每一步分成：

```text
commanded Cartesian delta
  → FK(q_target) delta
  → actual TCP delta
  → object delta
```

5,687个Place step中：

- command朝goal、FK反向：5,564，97.84%；
- FK朝goal、actual反向：34，0.60%；
- actual朝goal、object反向：0；
- 四层均朝goal：69，1.21%。

这直接排除了actuator时序是主因和物体夹持滑动。实现审计又排除了pinch site混用、joint/qpos/dof/actuator顺序错误、缺少`mj_forward`、q_target被后续逻辑覆盖。

### 6.3 快照消融锁定raw posture

在Place entry、开始明显远离前、已经远离后三个状态，各施加朝goal的5/10/12 mm命令：

- 当前axis+raw posture：2/9方向正确；
- position-only：9/9；
- 保留axis、关闭posture：9/9；
- 关闭axis、保留posture：2/9。

根因因此不是Jacobian符号，也不是approach-axis，而是：

```python
dq = dq_task + 0.02 * (home_qpos - qpos)
```

raw joint-space correction未投影到主任务零空间。在远离home的Place状态，毫米级Cartesian task很小，回home增量足以压过它，令`FK(q_target)`直接反向。早先固定目标实验只测到其0.5 mm级局部bias，没有覆盖这种远离home、小命令状态。

## 7. raw、off、nullspace：公式正确不等于机制有效

三种模式保持其余变量不变：

- raw：直接叠加posture；
- off：仅DLS主任务；
- nullspace：使用与主任务完全相同的加权J和阻尼逆，`N=I-J_pinv J`，再投影posture。

快照结果：raw 2/9，off/nullspace均9/9；position residual分别16.227/0.122/0.128 mm。nullspace的raw posture norm均值0.04767 rad，投影后仅9.1e-5 rad，缩小约524倍；最大`||J dq_posture||=1.997e-5`。

完整seeds 0～9：

| 模式 | full | command→FK反向 | final goal XY | 最小限位余量 |
|---|---:|---:|---:|---:|
| raw | 1/10 | 92.998% | 288.14 mm | 1.213 rad |
| off | 10/10 | 1.546% | 4.09 mm | 1.190 rad |
| nullspace | 10/10 | 0.808% | 4.09 mm | 1.190 rad |

off/nullspace都没有翻肘、姿态突变、碰桌、掉落或撞飞。它们的dq clip约32%，高于raw约1.84%，但没有对应的限位/速度/force安全退化。

为什么最终选off而不是更“高级”的nullspace？因为这是6关节机械臂，任务又把position与approach-axis组成近满秩6×6 Jacobian，几乎没有有效零空间。C的q-home、轨迹与B几乎相同；nullspace公式工作正常，但投影后几乎什么都不剩。当前选off更简单、更诚实；nullspace保留给未来7-DoF冗余臂或降低任务维数的场景。

## 8. 最终执行层与关键参数

当前代码默认：

- Cartesian action scale与FSM：保持旧checkpoint接口；
- TCP target smoothing/workspace safety：保持；
- `max_tcp_lead=0.03 m`；
- DLS damping `1e-4`；
- position weight 1.0，approach-axis weight 0.35；
- posture mode默认`off`，raw/nullspace仍可选；
- 最多15次IK迭代；每轮dq clip ±0.04 rad；
- q_target关节/ctrl limit；
- `frame_skip=10`，同一q_target持续送入六个position actuator；
- 当前XML绝对gain：前三关节2000、腕部500；对应position bias为负同值，速度反馈项分别-400/-100；forcerange前三±150、腕部±28；
- 无TCP weld，不直接写真实arm qpos；
- 夹爪继续由FSM脚本控制。

## 9. 为什么旧checkpoint不需要重新训练

不是因为新旧动力学完全等价，而是因为迁移保持了策略接口和任务语义：

- observation维数与含义不变；
- action仍是同一四维输出，前三维仍表示Cartesian意图；
- stage router、目标生成、reward、success、reset与脚本夹爪未改；
- 修复都发生在策略之下：限制不合理target backlog、提高/固化关节执行精度、移除破坏一级任务的raw posture。

换句话说，checkpoint已经学到了“往哪里移动”的高层策略；此前失败主要是新执行层没有忠实实现这个意图。先修执行层，比让PPO重新学习补偿系统性控制缺陷更合理。最终结果证明这是**旧策略在新IK执行链上的零样本迁移**，不是在新链路下重新训练得到的checkpoint。

## 10. 最终验证、剩余失败与局限

当前正式baseline使用XML绝对gain 2000/500、lead 30 mm、axis 0.35、posture off。回归测试6项通过，验证无weld、调用IK、不直接写arm qpos、off correction为零、三种模式可选、旧checkpoint可加载。

60-seed零样本评估（0～9固定复验，10～59扩展随机初始物体/目标）：

- Full/Place：59/60，98.33%；
- Reach/Grasp/Lift/Place entry/XY/low/open-ready/open：均60/60；
- final XY mean/P95/max：7.23/8.93/11.57 mm；
- Place TCP error mean/P95/max：12.22/17.03/20.00 mm；
- mean command→FK反向率1.72%；mean dq clip 36.25%；
- 最小joint-limit margin 1.117 rad（elbow）；
- max joint velocity 0.597 rad/s；max actuator force 44.90；
- 最大pad-object penetration 0.986 mm；无掉到桌面以下。

唯一失败seed 31不是IK方向回归：Reach/Grasp/Lift、XY、下降、open-ready、释放均完成，final XY 11.57 mm且物体最终静止，但机器人持续有一个table contact；success要求`table_contact_count==0`，因此900步truncated。CSV还把该episode标成`object_fling=True`，因为过程中曾超过脚本的0.5 m/s阈值；不能把它抹掉，也不能据此说物体最终仍在飞。直接阻断success的证据是397个持续robot-table contact steps。

当前局限：

- 仍是MuJoCo仿真，不是真机部署，也不是完整Sim-to-Real；
- 60个seed不能覆盖全部工作空间、模型误差和接触变化；
- mean dq clip较高，虽未造成已观测安全退化，仍值得长期监控；
- 6×6任务没有实用姿态零空间，off模式不主动回home；
- seed31提示释放后几何接触仍有边界case；
- 没有真实电机延迟、传感噪声、通信抖动、标定误差和安全控制器验证。

## 11. 这次排查体现的调试方法

这项工作的价值不只是把成功率从0/10做到59/60，而是形成了一条可迁移的方法论：

1. **先审计真实调用链**：函数存在不代表在执行；一路追到constraint、ctrl和physics。
2. **冻结上层，单变量修改**：checkpoint、reward、FSM、阈值不动，依次验证lead、kp、posture。
3. **按stage定位，而不是只看episode总指标**：284 mm最大误差属于Reach，不属于Place。
4. **按物理层分解误差**：target→FK(qtarget)与FK→actual隔离数学求解和执行器问题。
5. **设计反事实与oracle**：固定目标、servo-only、policy-only、oracle排除策略与符号假设。
6. **保存状态做局部因果消融**：同一个Place快照只切axis/posture，避免轨迹分布混杂。
7. **检查predicate而不是猜FSM**：seed9定位到2 µm阈值错位，seed31定位到table-contact gate。
8. **尊重局部结论边界**：home附近posture只造成0.5 mm bias，不代表远离home时不会反转主任务。
9. **用回归测试固化结构事实**：无weld、IK被调用、无qpos直写、旧checkpoint兼容。
10. **不把训练当万能修复**：先消除执行层的系统性错误，再判断是否需要学习补偿。

从技术能力角度，这覆盖了代码控制流审计、MuJoCo equality/actuator动力学、Jacobian DLS、阻尼伪逆与零空间、关节伺服静差、分层实验设计、日志体系、失败predicate复现和零样本策略迁移。更重要的是，每个结论都尽量对应可复现证据，而不是靠视频直觉或成功率猜根因。
