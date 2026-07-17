# Phase 2：可部署任务监督接口

> **模型版本说明（2026-07-17）：** 本文的100-seed FSM数字来自CAD相机
> 最终安装前的历史模型。当前相机安装、负载和观察姿态已经更新；
> TaskSupervisor接口和特权依赖结论仍有效，但若汇报当前模型成功率，
> 应重新运行Phase 2 A/B，而不是直接沿用下文数字。

## 结论

本阶段已经把可部署的抓取确认和任务阶段判断封装为独立 `TaskSupervisor`，并建立 `privileged_fsm` / `deployable_fsm` 显式模式。新 supervisor 本身不导入 MuJoCo，只接收 `ObjectEstimate`、`RobotState`、`GripperState`、目标位置和时间。

100 个相同 seed 的结果为：原 privileged FSM 完整成功 99/100，deployable FSM 完整成功 7/100。下降集中发生在抓取确认：93 个 deployable episode 没有形成同时满足几何稳定和夹爪保持反馈的连续确认。关闭 simulated latch 的 10-seed 诊断为 0/10，说明当前 MuJoCo 夹爪纯接触物理还不能稳定承担物体保持。

这不是 PPO 性能结论，也不应通过训练或放宽阈值掩盖。下一步应先补齐可靠的夹爪反馈/物体保持机制与抓取期间的估计更新。

## 代码结构

- `fourc2/task_supervisor.py`
  - `RobotState`：TCP 位置、姿态、速度、垂直对齐、时间戳、有效性和来源。
  - `GripperState`：命令开度、实际开度、速度/运动状态、执行器 effort、电流/力代理、故障、抓取保持置信度、时间戳、有效性和来源。
  - `TaskSupervisor`：Reach、Grasp、Lift、Place，以及夹爪命令、抓取确认、释放判断。
- `fourc2/envs/allstage.py`
  - 新增显式 `fsm_mode`：`privileged_fsm` 或 `deployable_fsm`。
  - `deployable_robot_state()` 当前用 MuJoCo 关节状态经 FK 得到 TCP，接口可替换成 UR5e 编码器/FK。
  - `deployable_gripper_state()` 当前用夹爪关节位置、速度和 actuator force 形成传感器代理，接口可替换成夹爪 SDK。
  - `update_task_supervisor()` 只把统一 ObjectEstimate、RobotState、GripperState、已知 goal 交给 supervisor。
  - deployable 模式的 reward 不再驱动阶段切换；reward/success 仍可读取真值，但只用于仿真评价。
- `scripts/eval_phase2_task_supervisor.py`：Shadow 与 100-seed A/B，另做 latch-off 诊断。
- `scripts/test_phase2_supervisor_failures.py`：缺失、过期、故障、空夹、依赖边界和 latch 隔离测试。
- `scripts/analyze_phase2_results.py`：从已保存 CSV 生成失败分类和逐事件统计。

## 数据流与边界

```text
ObjectEstimate -----------+
UR5e encoder -> TCP FK ---+--> TaskSupervisor --> stage / gripper command / release
Gripper SDK --------------+          |
goal + timers ------------+          +--> diagnostics/events

PPO obs[35:39] --------------------------> PPO（本阶段保持原样）
MuJoCo truth/contact/latch -> reward/success/evaluator（不进入 supervisor）
simulated latch ----------> 仿真物理保持辅助（不进入 supervisor）
```

`TaskSupervisor` 的源码依赖扫描确认没有 `mujoco`、`site_xpos`、object qpos、contact、penetration、`is_grasp_latched`、reward 或 success 读取。RGB-D/ObjectEstimate 的 source consistency failure 为 0。

### 抓取确认

确认候选必须同时具备：

1. TCP 与 ObjectEstimate 的抓取几何关系满足现有阈值；
2. 夹爪闭合命令已经实际生效；
3. 实际开度显示夹爪在完全闭合前受阻，或 effort 达到反馈阈值；
4. 夹爪停止并且 hold confidence 有效；
5. 连续满足现有稳定步数。

缺失、过期、fault 输入抛出明确 `SupervisorInputUnavailable`，不会生成零向量或读取物体真值。空夹测试没有误确认。

### simulated latch 隔离

deployable 模式只有在 supervisor 已经独立确认抓取后，仿真 latch 才允许启用。latch 不触发 ObjectEstimate 的 TCP—物体传播，不是 supervisor 输入，也不作为抓取成功判据。它仍会直接写 object qpos/qvel 来提供仿真物理保持，因此仍是不可迁移的物理辅助。

## Shadow Mode

原 FSM 继续控制，supervisor 只观察。100 seed 中：

| 事件 | 匹配 | 漏判 | 延迟中位数（policy step） | 范围 |
|---|---:|---:|---:|---:|
| Reach→Grasp | 100 | 0 | 0 | -1..1 |
| 抓取确认 | 91 | 9 | 6 | 5..9 |
| Grasp→Lift | 91 | 9 | 6 | 5..9 |
| Lift→Place | 91 | 9 | -1 | -2..0 |
| 释放命令 | 91 | 9 | -1 | -1 |

Shadow 仅有 1 次 `release_complete` 旁路事件；原 episode 通常在真实 success 后、夹爪完全停止前结束，因此该事件不能直接用于精确完成时刻对齐。

抓取确认相对原特权判定平均晚 6.36 step，并漏掉 9 次原 FSM 抓取。没有空夹误确认。

## 100-seed A/B

| 指标 | privileged_fsm | deployable_fsm |
|---|---:|---:|
| RGB-D 有效 | 100% | 100% |
| Reach | 100% | 100% |
| Grasp（真值 evaluator） | 100% | 100% |
| supervisor 抓取确认 | 91%（shadow） | 7% |
| Lift | 100% | 7% |
| entered Place | 100% | 7% |
| Place | 99% | 7% |
| Full success | 99% | 7% |
| 平均 episode steps | 458.25 | 869.70 |

相同初始状态为 100/100；两组逐 seed 最终结果相同 8/100。两种模式的数据源没有混用，PPO/safety/servo/supervisor ObjectEstimate 一致性错误为 0。

deployable 的 93 个失败全部停在抓取确认之前，终止诊断均同时显示 `stable_geometry=false`、`sensor_hold=false`、`effort_contact=false`。原 FSM 唯一失败属于 Place/释放评价失败。

### latch-off 诊断

只对 10 seed 关闭 object qpos/qvel 写入：Reach 和真值 Grasp 都为 10/10，但 supervisor 确认、Lift、Place、Full 均为 0/10。该小样本明确表明当前纯接触模型会在闭合期间推动/丢失方块，且一次性的抓取前 ObjectEstimate 不能描述这种位移；它尚不能证明真实夹爪也会有相同行为。

## 故障注入

- GripperState 缺失：fail closed，原因 `missing`。
- GripperState 过期：fail closed，原因 `stale`。
- GripperState fault：fail closed，原因 `fault`。
- ObjectEstimate 缺失：fail closed，原因 `missing`。
- 空夹（完全闭合、无 effort、零 confidence）：不确认抓取。
- 静态依赖边界测试：未发现 supervisor 读取 MuJoCo object truth/contact/latch/reward/success。

测试结果保存在 `outputs/phase2_task_supervisor/failure_injection_results.json`。

## 仍存在的特权状态

1. PPO `obs[35:39]` 保持原值，其中接触、vertical alignment 和 grasp phase 仍包含仿真特权/旧 FSM 信息；这是用户指定留给下一项敏感性实验的范围。
2. reward、success、逐 seed evaluator 仍读取 MuJoCo object/contact 真值，但不影响 deployable supervisor 的动作和阶段输出。
3. simulated latch 仍直接写 object qpos/qvel，属于物理辅助；已隔离但尚不能移除。
4. 当前 GripperState 的 actual opening、velocity、effort 是 MuJoCo 传感器代理。真机必须由夹爪 SDK 提供对应量及故障状态。
5. RobotState 当前来自 MuJoCo 关节/FK 代理；其接口形状可直接由 UR5e 编码器与 FK 替换。

## 真机 SDK 需要提供的夹爪信号

- 当前实际开度或手指关节位置；
- 开合速度、运行中/停止/到位状态；
- 电机电流、估计夹持力或驱动 effort；
- 故障、过流、堵转、通信超时；
- 命令确认和可靠时间戳。

`grasp_hold_confidence` 应由这些原始反馈与连续性规则计算，不应由 SDK 的单一“抓住”布尔量直接替代而不验证。

## 下一步最小顺序

1. 校准仿真 GripperState proxy 与未来真机 SDK 信号的单位、符号、堵转/接触响应；修复当前 effort 几乎不响应的问题。
2. 在不读取物体真值的条件下，增加抓取闭合期间/闭合后的视觉重定位，或明确进入抓住后 TCP 相对传播的条件，避免方块被推动后估计滞后。
3. 改善仿真夹爪—方块接触保持，使 latch-off 能稳定抬升；在此之前保留但隔离 latch。
4. 重做 Shadow 和 100-seed deployable A/B。抓取确认召回率、空夹误判率和 latch-off 保持通过后，再做 PPO `obs[35:39]` 替换敏感性实验。
5. 当前不需要 PPO 微调。失败发生在 supervisor 抓取确认/物理保持层，PPO 在 privileged 模式仍为 99%。

## 输出

- `outputs/phase2_task_supervisor/summary.json`
- `outputs/phase2_task_supervisor/privileged_episodes.csv`
- `outputs/phase2_task_supervisor/deployable_episodes.csv`
- `outputs/phase2_task_supervisor/seed_comparison.csv`
- `outputs/phase2_task_supervisor/shadow_event_differences.csv`
- `outputs/phase2_task_supervisor/latch_disabled_episodes.csv`
- `outputs/phase2_task_supervisor/failure_injection_results.json`
