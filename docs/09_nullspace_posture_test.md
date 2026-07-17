# Null-space posture 对比测试

## 配置与门控

在 `07_test` 中比较：A=`raw`，B=`off`，C=`nullspace`。三组均固定 `max_tcp_lead=0.03`、arm kp 2×、approach-axis weight 0.35、posture weight 0.02、`best_full_flow_v22.zip`，其余任务和物理配置不变。未训练 PPO，未修改 `06_4c2`，未启用 mocap weld，也未直接写 arm qpos。

C 严格复用主任务的加权 Jacobian 与阻尼逆：`J_pinv = J.T @ solve(J@J.T + 1e-4 I, I)`，`N=I-J_pinv@J`。投影发生在 IK 内部、合成与 dq clip 之前；之后仍走原有 ±0.04 clip 和关节限位。

快照门槛为 C 至少 8/9 同时满足 FK、actual TCP、object 朝 goal。实测 C 为 **9/9**，因此按规则继续完整评估。

## 快照结果

| 模式 | 三层方向通过 | mean cos(goal, FK) | mean position residual | mean projected posture norm | max `||J dq_posture||` |
|---|---:|---:|---:|---:|---:|
| A raw | 2/9 | -0.3123 | 16.227 mm | 0.04723 rad | 2.142e-2 |
| B off | 9/9 | 0.99994 | 0.122 mm | 0 | 0 |
| C nullspace | 9/9 | 0.99994 | 0.128 mm | 9.1e-5 rad | 1.997e-5 |

C 的 raw posture norm 均值为 0.04767 rad，投影后仅 9.1e-5 rad，缩小约 524 倍；`||J dq_posture||` 最大值 1.997e-5。阻尼 projector 理论上并非严格正交 projector，所以该值不必为机器零；相对 raw 的 task-space posture 影响（约 1.6e-2）已缩小约三个数量级，且位置方向 9/9 正确，足以视为本测试尺度下接近零。

## 完整任务结果（deterministic seeds 0～9）

| 指标 | A raw | B off | C nullspace |
|---|---:|---:|---:|
| full / Place success | 1/10 | 10/10 | 10/10 |
| Reach / Grasp / Lift | 10/10 / 10/10 / 10/10 | 10/10 / 10/10 / 10/10 | 10/10 / 10/10 / 10/10 |
| entered Place | 10/10 | 10/10 | 10/10 |
| reached XY release region | 2/10 | 10/10 | 10/10 |
| reached low height | 3/10 | 10/10 | 10/10 |
| place_open_ready / gripper opened | 1/10 / 1/10 | 10/10 / 10/10 | 10/10 / 10/10 |
| command→FK 反向率（episode rate 均值） | 92.998% | 1.546% | 0.808% |
| mean minimum goal XY | 81.76 mm | 4.09 mm | 4.09 mm |
| mean final goal XY | 288.14 mm | 4.09 mm | 4.09 mm |
| Place TCP error mean / P95 / max | 11.35 / 13.74 / 17.26 mm | 8.37 / 13.16 / 14.82 mm | 8.36 / 13.16 / 14.82 mm |
| mean episode steps | 832.8 | 273.4 | 273.7 |
| mean final `||q-home||` | 1.822 rad | 2.585 rad | 2.584 rad |
| 全程最小关节限位余量 | 1.213 rad | 1.190 rad | 1.190 rad |
| mean dq clip rate | 1.84% | 32.20% | 32.14% |
| max joint velocity | 0.836 | 1.058 | 1.058 |
| max actuator force | 32.60 | 38.54 | 38.59 |

B/C 均无姿态突变：最大单 control-step 六关节变化范数分别为 0.03286/0.03284 rad，远低于诊断阈值 0.25 rad；0 个 episode 被标记为肘部翻转/姿态突变。每关节最小余量中最小的是 elbow：B 1.18975 rad、C 1.18974 rad；其余均大于 2.36 rad。没有 robot-table 异常接触、掉落到桌面以下或撞飞事件。

B/C 的 dq clip 率明显高于 A，但最终运动、速度、force、限位余量和碰撞指标均正常；这是去除会抵消任务增量的 raw posture 后，任务 DLS 更常触及原有单迭代 dq clip，不等同于关节限位或执行失稳。

## 结论

1. **C 是否消除 command→FK 反向？** 是。快照 9/9，完整 Place 反向率由 A 的 93.00% 降到 0.81%。剩余少量严格符号分类发生在微小/瞬态位移附近，不影响 10/10 成功。
2. **`J dq_posture` 是否足够接近零？** 是，相对本任务尺度足够小：最大 1.997e-5，较 raw task-space posture 影响约小三个数量级。
3. **完整成功率？** A=1/10，B=10/10，C=10/10。
4. **B 是否姿态恶化或靠近限位？** `||q-home||` 增大，说明不再回 home；但无翻转/突变，最小限位余量仍为 1.190 rad，没有实际限位风险证据。
5. **C 是否同时保留 B 的方向和 A 的姿态约束？** C 保留了 B 的方向，但**没有实质保留 A 的姿态约束**。当前加权完整任务 J 是 6×6 且在测试状态下几乎满秩，阻尼 null-space 只剩极小分量；C 的 `q-home`、限位和轨迹指标几乎与 B 相同。
6. **C 能否替代 raw 成为正式基线？** C 在功能与安全测试上可以替代 raw，且 raw 不应继续作为默认。但若目标是“同时保留 posture”，本轮不能证明 C 达成该目标；在当前非冗余六关节/六维任务下，B 更简单且表现等价。建议正式基线优先采用 B，或把 C 作为为未来冗余任务预留的实现。不要宣称 C 提供了有效回 home 约束。
7. **C 若仍失败，属于哪层？** 本轮 C 未失败。公式和阻尼泄漏均符合预期；Place 其他控制在 B/C 下均达到 10/10。
8. **PPO？** 当前仍不训练 PPO。

## 产物

- `results/nullspace_posture/snapshot_comparison.csv`
- `results/nullspace_posture/episode_comparison.csv`
- `results/nullspace_posture/posture_projection_metrics.csv`
- `results/nullspace_posture/summary.json`
- `scripts/nullspace_posture_test.py`

环境保留 A/B/C 三种可选模式，默认仍为 `raw`，因此本轮没有静默改变既有环境默认行为。正式切换默认值应作为下一项单独变更。
