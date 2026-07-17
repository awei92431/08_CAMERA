# Final IK execution baseline

## 正式默认配置

- `max_tcp_lead = 0.03`
- arm position-servo：XML 中固化的 2× kp（shoulder/elbow 2000，wrist 500；对应 position bias 为负同值）
- `ik_posture_mode = "off"`
- `ik_axis_weight = 0.35`
- 不存在 TCP mocap weld；mocap body 仅作可视化 marker
- arm 只通过 position actuator `ctrl` 执行，action path 不直接写真实 arm qpos

`raw`、`off`、`nullspace` 仍是合法显式模式。默认由 `raw` 改为 `off`。未修改 observation、action、reward、success、FSM、释放阈值、PPO、forcerange、速度反馈、action scale、frame skip 或 reset 分布；未训练 PPO；未修改 `06_4c2`。

## 根因与修复

原实现把 `0.02 * (home_qpos - qpos)` 直接加到 DLS 关节增量，未经 task null-space 投影。该 raw posture correction 能压过毫米级 Cartesian 命令并破坏一级任务，先前 5,687 个 Place step 中 97.84% 的反向首先出现在 command→FK(q_target)。

A/B/C 快照与 10-seed 对比证据：

| 模式 | 快照方向正确 | Full/Place success | command→FK 反向率 |
|---|---:|---:|---:|
| A raw | 2/9 | 1/10 | 93.00% |
| B off | 9/9 | 10/10 | 1.55% |
| C nullspace | 9/9 | 10/10 | 0.81% |

C 的最大 `||J dq_posture||` 为 1.997e-5，但当前机械臂只有 6 个关节，完整 position+approach-axis 任务形成近满秩 6×6 Jacobian，几乎没有有效零空间。投影后的 posture 增量均值仅 9.1e-5 rad，C 在实际轨迹上几乎等同 B，不能声称同时保留了有效回-home约束。`nullspace` 模式保留给未来冗余机械臂或降维 Cartesian task；当前正式默认采用更直接的 `off`。

## 回归测试

`tests/test_ik_execution.py` 共 6 项，全部通过：

- 默认 action shape 与参数值；
- `mocap_tcp_weld` id 为 -1，`tcp_weld_enabled=False`；
- action step 的 `_solve_ik` 调用计数严格增加 1；
- 默认 `diag_ik.posture_mode=off`，posture correction 为零；
- AST 检查 action execution path 没有向 `self.data.qpos` 赋值，只写 arm actuator ctrl；
- raw/off/nullspace 均可显式构造；
- arm gain/bias 精确锁定为 XML 的已验证 2×值；
- 原 `best_full_flow_v22.zip` 可加载、确定性 predict 并正常 step。

固定 deterministic seeds 0～9 在正式默认配置下再次达到 **10/10 full 与 Place success**。

## 60-seed 零样本扩展评估

使用 seeds 0～59；0～9 为固定复验，10～59 为 50 个扩展的随机初始物体/目标样本。未训练或调参。

| 指标 | 结果 |
|---|---:|
| Full / Place | 59/60 (98.33%) |
| Reach / Grasp / Lift | 60/60 / 60/60 / 60/60 |
| entered Place / XY release / low height | 60/60 / 60/60 / 60/60 |
| place_open_ready / gripper opened | 60/60 / 60/60 |
| final goal XY mean / P95 / max | 7.23 / 8.93 / 11.57 mm |
| Place TCP error mean / P95 / max | 12.22 / 17.03 / 20.00 mm |
| mean command→FK reverse rate | 1.72% |
| mean dq clip rate | 36.25% |
| minimum joint-limit margin | 1.117 rad (elbow) |
| per-joint minimum margins | 2.370, 4.026, 1.117, 4.633, 4.679, 3.948 rad |
| maximum joint velocity | 0.597 rad/s |
| maximum actuator force | 44.90 |
| maximum pad/object penetration | 0.986 mm |
| object dropped below table | 0 episodes |
| mean / max episode steps | 465.95 / 900 |

### 唯一失败：seed 31

分类为 **Place terminal safety-contact failure**，不是 Reach、Grasp、Lift、XY transport、下降或开夹失败：

- Reach/Grasp/Lift 均成功；
- 已进入 Place、到达 XY release region、达到低高度；
- `place_open_ready=True`、`place_opened=True`，物体已释放；
- final goal XY = 11.57 mm，object XY speed 约 `1.27e-17`，boundary penalty=0；
- 但 `table_contact_count=1`，而 Place success 明确要求 `table_contact_count==0`；
- episode 因此在 900 steps truncated；60 个 episode 的 397 个 robot-table contact steps 全来自该失败轨迹。

seed 31 曾出现 object speed >0.5 的诊断标记，但最终物体静止、未掉落且目标误差合格；决定 terminal failure 的直接条件是持续 robot-table contact。按要求没有调整阈值、FSM、控制器或训练策略。

## 最终判断

`posture_mode="off"` 已作为正式默认执行层固化。它消除了 raw posture 对一级 Cartesian task 的系统性破坏，原 checkpoint 无需训练即可在固定 seeds 维持 10/10，并在 50 个额外随机样本上得到 49/50，总计 59/60。唯一扩展失败已隔离到释放后的 robot-table contact success gate，不是 IK 方向回归。

本轮不因该单例失败自动训练或继续调参；若后续处理，应单独审计 seed 31 的释放姿态与 robot-table contact geometry。

## 产物

- `results/final_ik_baseline/episodes.csv`
- `results/final_ik_baseline/summary.json`
- `scripts/evaluate_final_ik_baseline.py`
- `docs/10_final_ik_baseline.md`
