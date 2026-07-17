# IK执行层迁移复盘（精简版）

这次工作不是训练新策略，而是把旧PPO checkpoint从mocap/weld执行方式零样本迁移到“DLS IK→关节position actuator”的真实调用链。

审计发现，原项目虽然存在`_solve_ik()`，正式action路径却从未调用：PPO前三维生成TCP目标，目标写入mocap body，XML的`mocap_tcp_weld`再用约束力拖动`pinch` site；arm actuator只保持当前qpos。因此原链是“Cartesian action→mocap→weld”，不能称为IK控制。夹爪也由FSM脚本开合，不是PPO第四维学会的。原模式10/10但`ik_solve_calls=0`（`results/mocap/evaluation.json`）。

迁移后删除weld，mocap仅作marker；每步调用DLS IK，将六关节`q_target`送入10个MuJoCo substep，不直接写真实arm qpos。首轮成功率从10/10跌至0/10，最大TCP误差283.96 mm。按stage记录后，大误差集中在Reach；阶段跳变、限位和碰撞均被排除。根因是Reach/Grasp/Lift在旧target上继续累加action，而有限带宽关节尚未跟上。20/25/30 mm消融中，20 mm破坏策略时序，30 mm既把Reach最大误差压到约29.8 mm，又保持各阶段10/10进入，因此选30 mm。

随后仍有8～10 mm稳态偏差。通过`target-FK(qtarget)`与`FK(qtarget)-actual`分解，IK误差仅0.615 mm，actuator误差约9.020 mm。逐关节数据表明shoulder lift、elbow各落后约9 mrad，servo force与`qfrc_bias`匹配且未饱和，根因是有限kp平衡重力/负载的静差。kp扫描把固定目标均值误差从8.856降至4.438 mm，但完整任务仍只有1/10，说明Place另有根因。

对Place先比较policy+servo、servo-only、policy-only和oracle。即使oracle命令朝goal，actual TCP仍大多反向，排除了PPO和servo符号是决定性原因。再把每步拆成command→FK→actual→object：5,687步中5,564步（97.84%）首先在command→FK反向；actuator层仅0.60%，物体层为0。site、索引、`mj_forward`和qtarget覆盖也均被排除。

三个Place快照分别施加5/10/12 mm命令后，raw配置仅2/9正确；position-only和关闭posture均9/9；仅关闭axis仍2/9。根因是`0.02*(home-q)`未经零空间投影直接叠加，在远离home、Cartesian命令很小时压过一级任务。

raw/off/nullspace完整对比中，raw为1/10，off和nullspace均10/10。当前6关节、近满秩6维任务几乎没有有效零空间，nullspace投影后姿态增量缩小约524倍，实际与off等价，因此默认选off，nullspace保留给冗余机械臂或降维任务。

最终默认是lead 30 mm、DLS damping `1e-4`、axis weight 0.35、posture off、15轮IK、单轮dq ±0.04 rad、frame_skip 10；当前XML绝对gain为前三轴2000、腕部500。observation、action、reward、FSM均未变，修复位于策略以下，所以旧checkpoint无需重训。

最终seeds 0～59为59/60（98.33%），Reach/Grasp/Lift及Place关键条件均60/60，final XY均值/P95为7.23/8.93 mm。唯一seed31已释放且最终静止，但机器人持续table contact，success要求contact为0，故900步超时；这不是IK方向回归（`results/final_ik_baseline/summary.json`）。当前成果仍限于MuJoCo仿真，不代表真机部署或完整Sim-to-Real。
