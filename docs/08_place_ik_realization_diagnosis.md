# Place IK realization 分层诊断

## 测试范围

- 环境：`07_test`，未修改 `06_4c2`
- checkpoint：`checkpoints/best_full_flow_v22.zip`
- deterministic seeds：0～9
- 固定执行层：`max_tcp_lead=0.03`、机械臂 position-servo `kp=2.0×`
- 未训练 PPO；未修改 reward、success、FSM、夹爪、正式 IK/actuator 参数或成功阈值
- 正式 A 组完整任务记录 5,687 个 Place step。快照测试不运行 PPO，且配置只存在于独立诊断环境。

## 结论摘要

反向运动首先、且几乎总是发生在 **Cartesian command → FK(q_target)**。5,687 个 Place step 中，A 类 5,564 步（97.84%）；B 类 34 步（0.60%）；C 类 0 步；四层均朝目标的 D 类 69 步（1.21%）。因此主因不是 actuator 时序，也不是物体在夹持中反向滑动，而是当前完整 IK 目标生成的 `q_target` 在这些 Place 状态下，其 FK 位移与目标方向相反。

快照单变量测试进一步显示：关闭 posture 后 9/9 固定命令的 FK、真实 TCP 和物体均朝目标；position-only 同样为 9/9。仅关闭 approach-axis 仍只有 2/9 朝目标，与当前配置的 2/9 相同。因此造成反向的因素是当前 raw joint-space posture correction，而不是 approach-axis 约束。

## 每步分层分类

| 分类 | 条件 | 步数 | 比例 |
|---|---|---:|---:|
| A | cmd 朝目标，FK(q_target) 背离 | 5,564 | 97.84% |
| B | cmd、FK 朝目标，actual TCP 背离 | 34 | 0.60% |
| C | cmd、FK、actual 朝目标，object 背离 | 0 | 0.00% |
| D | 四层均朝目标 | 69 | 1.21% |
| 未分类/近零边界 | 不满足严格正负分类 | 20 | 0.35% |

seed 0、2、4、5、6、7 在 Place 第 1 步即出现 A；seed 1、8、9 首次 A 分别在第 22、21、17 步。seed 3 首次严格反向为第 33 步的 B，但其全段仍以 A 为主（606 个 A、1 个 B）。详细逐 seed 计数与首次反向位置见 `seed_summary.csv`。

这说明 Place 中笛卡尔目标本身不是主要符号错误：命令朝目标，而 `q_target` 对应的末端位移已反向。actuator 偶发 B 类误差存在，但占比很小；没有观测到“前三层正确、物体单独反向”的 C 类证据。

## 状态快照固定命令测试

从 seed 0 保存三个状态：刚进入 Place（Place step 1，物体距目标 65.1 mm）、开始明显远离前（step 40，114.3 mm）和明显远离后（step 586，360.7 mm）。每个状态分别施加朝目标的 5、10、12 mm XY 命令。

| 配置 | FK 朝目标 | actual 朝目标 | object 朝目标 | mean cos(goal, FK) | mean cos(goal, actual) | mean cos(goal, object) |
|---|---:|---:|---:|---:|---:|---:|
| 当前：axis 0.35 + posture 0.02 | 2/9 | 2/9 | 2/9 | -0.303 | -0.298 | -0.329 |
| position-only | 9/9 | 9/9 | 9/9 | 1.000 | 0.963 | 0.894 |
| 关闭 posture | 9/9 | 9/9 | 9/9 | 1.000 | 0.974 | 0.896 |
| 关闭 approach-axis | 2/9 | 2/9 | 2/9 | -0.324 | -0.257 | -0.275 |

在“开始明显远离前”快照，当前配置的 FK/actual/object 平均 cosine 分别为 -0.815/-0.806/-0.803；关闭 posture 后变为 1.000/0.979/0.931。approach-axis 保留时关闭 posture 已完全恢复方向，因此 axis 不是本次反向的必要原因。position-only 与无 posture 接近，说明保留 axis 并不会妨碍正确方向。

上述快照是局部因果诊断，不写回正式基线，也不代表关闭 posture 后的完整任务成功率。

## 实现细节审计

1. **site 一致**：实际 TCP、目标误差、`mj_jacSite` 和独立 `ik_data` 的 FK 都使用同一个 `pinch_site_id`，运行时名称为 `pinch`。
2. **关节与 actuator 映射一致**：六个 arm joint 的 joint/qpos/dof id 均按 0～5 排列；六个 actuator id 也为 0～5，`actuator_trnid` 逐项连接到对应的 shoulder_pan、shoulder_lift、elbow、wrist_1、wrist_2、wrist_3 joint。
3. **IK forward 正确调用**：每轮把候选关节值写入独立 `ik_data.qpos` 后立即调用 `mj_forward`，随后才读取 site 位姿和计算 Jacobian；最终 FK 前也再次调用 `mj_forward`。
4. **无 q_target 覆盖**：`_solve_ik()` 返回完整六关节目标；每个 MuJoCo substep 都把同一目标写入六个 arm actuator。夹爪使用独立 actuator id，未覆盖 arm ctrl。
5. **Place 目标重构**：未释放时每步以当前真实 TCP 为基准构造新目标，再经安全、平滑与 30 mm lead limit。它会重置目标基准，但不会解释 cmd 已朝目标而 FK 反向的观测。
6. **posture 项的作用方式**：DLS 解之后，每次迭代直接执行 `dq += 0.02 * (home_arm_qpos - qpos)`。该项没有投影到 Cartesian task 的 null space。Place 姿态通常离 home 较远，而单步目标只有毫米级，因此 raw joint-space 回拉可以压过位置任务，产生与目标相反的末端位移。

未发现 site 混用、关节索引错位、actuator 顺序错位、缺少 `mj_forward` 或 IK 后目标被覆盖的证据。

## 对六个问题的明确回答

1. **反向首次发生在哪层？** 主体是 cmd→FK：全体 Place steps 的 97.84% 属于 A。FK→actual 仅 0.60%，actual→object 为 0%。
2. **IK 的 q_target 是否真的朝错误方向？** 是。这里“错误方向”由独立 `ik_data` 对 q_target 做 FK 后直接确认，不是从 actuator 结果倒推。
3. **哪个诊断配置恢复方向？** position-only 和关闭 posture 都是 9/9；仅关闭 axis 仍为 2/9，因此 posture 是关键变量。
4. **是否有 site、索引、覆盖或 mj_forward 错误？** 本次代码和运行时映射审计均未发现。
5. **能否明确说 IK 实现有问题？** 可以明确说当前 **IK 目标组合/正则实现对 Place 小步命令存在功能性问题**；不能把它描述成 Jacobian 符号、site 或关节接线错误。具体问题是 posture 修正直接叠加在关节增量上、未作 null-space 投影，能够破坏主位置任务。
6. **下一步修哪层？** 优先修 IK 的 posture 处理，而不是 actuator 时序或抓取保持。最小后续实验应比较 Place 中关闭/显著降低 posture，或将 posture 改为 Jacobian null-space 投影；之后再按 seeds 0～9 完整评估。正式修改前不需要 PPO 微调。

## 产物

- `results/place_ik_realization/step_layer_decomposition.csv`
- `results/place_ik_realization/seed_summary.csv`
- `results/place_ik_realization/snapshot_tests.csv`
- `results/place_ik_realization/classification_summary.json`
- `scripts/place_ik_realization.py`

本轮对环境源码的唯一改动是增加执行前后状态的只读诊断字段；控制计算和物理执行路径未改变。
