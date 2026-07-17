# IK执行层迁移：面试讲述版本

## 30秒版本

我做过一次机械臂强化学习控制链审计和迁移。项目里虽然有IK函数，但实际PPO输出是写给mocap target，再通过weld约束拖动末端，IK根本没被调用。我把它改成Cartesian target经过DLS IK生成六关节目标，再用position actuator执行。旧checkpoint零样本切换后成功率一度从10/10降到0/10。我没有马上重训，而是分层诊断：先用30 mm lead limit解决目标积压，再把IK误差和actuator误差拆开，定位重力静差；最后把Place运动拆成command、FK、actual TCP、object四层，发现97.84%的反向发生在command到FK，根因是未投影的raw posture压过Cartesian主任务。关闭posture后固定10个seed为10/10，60-seed最终59/60。整个过程没有改reward或重训PPO，是旧策略在新IK执行链上的零样本迁移。

## 2分钟版本

这个项目最初看起来已经有IK，因为代码里存在Jacobian DLS函数。但我沿action调用链追到MuJoCo后发现，正式路径并不调用IK：PPO输出Cartesian增量，更新mocap body，XML的soft weld把mocap和TCP连接；arm actuator只保持当前qpos。所以原系统是mocap约束控制，不是真正的“IK→关节执行器”。夹爪也由FSM脚本控制，不是PPO学出来的。

我删除TCP weld，让mocap只做marker；每步调用DLS IK，把六关节qtarget送给position actuator。观察和action接口都不变。但旧checkpoint从mocap下10/10变成IK下0/10，最大TCP误差约284 mm。

我先按stage拆日志，发现大误差都发生在Reach，不在Place；阶段跳变、限位、碰撞都不是原因。Reach会在旧target上持续累加action，而真实关节带宽有限，于是目标越跑越远。我单变量测试20、25、30 mm lead bound，30 mm既把误差限制住，又保留10/10的stage handoff。

之后仍有8～10 mm静差。我用FK(qtarget)做中间变量，把总误差拆成IK误差和actuator误差：前者只有0.615 mm，后者约9.02 mm。逐关节发现肩关节和肘关节各有约9 mrad残差，servo force等于重力bias且远未饱和，所以是有限kp的重力静差。提高kp把固定目标误差降到4.44 mm，但完整Place仍只有1/10，说明还有别的问题。

对Place我先做policy+servo、servo-only、policy-only和oracle。即使oracle命令朝目标，真实TCP仍大多反向，于是排除PPO和servo符号。再把一步拆成command→FK(qtarget)→actual TCP→object。5,687步中97.84%首次在command→FK反向，actuator层只有0.60%，物体层为0。相同Place快照做posture消融：保留axis关posture是9/9，保留posture关axis仍2/9。根因是`0.02*(home-q)`没有投影，远离home时压过毫米级主任务。

最后比较raw、off和nullspace。off/nullspace都是10/10，raw只有1/10；但6自由度机械臂做近满秩6维任务几乎没有零空间，nullspace投影后姿态项缩小约524倍，实际上等价off，所以默认选off。最终旧checkpoint无需重训，60个seed成功59个；唯一失败是释放后robot-table contact门控，不是IK方向问题。

## 5分钟版本

### 背景与目标

目标是验证一个已有PPO策略能否从仿真里的mocap执行迁移到更真实的关节执行链。限制很明确：不改observation、action、reward、success和FSM，不训练PPO，不直接写arm qpos。

### 第一步：审计而不是先改代码

项目中有一个DLS IK函数，所以表面上像IK控制。但我检查正式`_apply_tcp_action`后发现它只更新mocap pose；XML有`mocap_tcp_weld`，substep里还把arm ctrl设成当前qpos。也就是说真正施力的是equality weld，IK函数是未接入的辅助代码。原模式10/10且IK调用数为0，这给了很强的运行时证据。

我将执行层改为：旧Cartesian target→DLS IK→六关节qtarget→position actuators；移除weld，保留mocap marker可视化。DLS使用position和approach-axis组成的加权Jacobian，阻尼1e-4，最多15轮，每轮dq限制±0.04 rad。

### 第二步：处理284 mm目标积压

切换后0/10、最大误差284 mm。我列出可能原因：阶段跳变、IK发散、限位、碰撞、actuator跟不上、target积压。按stage和transition记录后，大误差全部集中在Reach step29～40；阶段跳变只有毫米级，无限位和碰桌。代码又显示Reach/Grasp/Lift在旧target上继续累加action。真实关节没跟上，数学目标仍继续前移；15轮IK一次还能把关节目标推远约0.60 rad。

我只加lead bound，测试20/25/30 mm。20 mm误差最小，但Reach只剩2/10，说明太保守会破坏旧策略的时序；30 mm让Reach/Grasp/Lift/Place entry都保持10/10，因此选30 mm。这体现了控制参数不能只按单一误差指标选。

### 第三步：分解IK与关节执行误差

固定目标仍有8～10 mm偏差。我没有继续盲调IK权重，而是计算两个量：target−FK(qtarget)和FK(qtarget)−actual。IK误差0.615 mm，actuator误差9.020 mm，Z偏差占主导。关闭posture可把IK误差降到0.089 mm，但实际误差没有改善；axis减半也没影响。

逐关节数据进一步显示shoulder lift和elbow各落后约9 mrad，actuator force与qfrc_bias吻合但远未饱和。这是position servo在重力下的静差。提高kp后固定目标均值从8.856降到4.438 mm，无振荡和饱和。但完整任务仍1/10，所以我没有把“精度改善”误判成“任务根因已解决”。

### 第四步：Place四层定位

Place中物体持续远离goal。第一批假设是policy和servo互相打架、servo符号错误、release FSM有隐藏条件。A/B/C/D消融中，servo-only和oracle的Cartesian命令都朝goal，但真实TCP仍反向，所以PPO不是决定性根因，world-frame符号也正确。

关键突破是引入FK(qtarget)作为中间观测，把一步分成command→FK→actual→object。5,687个Place step里，5,564步即97.84%在第一层反向；第二层只有0.60%，第三层为0。于是主因一定在IK目标生成，不在actuator或夹持滑动。随后审计site、joint/actuator索引、mj_forward和ctrl覆盖，全部正常。

我保存三个Place状态，每个施加5/10/12 mm相同方向命令。raw配置2/9正确；position-only和关posture都是9/9；只关axis还是2/9。根因锁定为raw posture直接叠加，没有null-space投影，在远离home时压过很小的Cartesian任务。

### 第五步：选择最终posture模式

我实现raw/off/nullspace三种模式。nullspace严格复用主任务同一个加权J和阻尼逆，避免伪逆定义不一致。off/nullspace完整10个seed都是10/10，raw是1/10。nullspace的`J dq_posture`已经很小，但当前是6关节对近满秩6维任务，零空间几乎为空，因此它的轨迹与off相同。最终默认选更简单的off，nullspace留给7自由度机械臂或降维任务。

### 结果与边界

最终不训练PPO，60个seed成功59个；Reach/Grasp/Lift和Place关键中间条件都是60/60。final XY均值7.23 mm，最小关节限位余量1.117 rad。唯一seed31已释放且最终静止，但机器人持续碰桌，success gate要求零table contact，所以超时。它说明系统仍有释放后几何边界case。

这不是完成真机部署或Sim-to-Real。项目证明的是：在MuJoCo里，通过控制链审计、分层诊断和单变量实验，可以把旧策略零样本迁移到真实调用IK与关节执行器的链路，而不是用重新训练掩盖执行层错误。

## 常见追问与参考回答

### Q1：项目里明明有IK函数，为什么说原来不是IK控制？

因为控制方式由正式调用链决定。原action path没有调用`_solve_ik`，只写mocap pose；XML的weld约束产生TCP运动，arm actuator还被设为当前qpos。运行时`ik_solve_calls=0`也是直接证据。

### Q2：为什么不用PPO重新训练来适应新动力学？

最初失败来自可解释的执行层系统误差：target无限积压、重力静差和raw posture破坏主任务。让PPO补偿这些问题会浪费样本、降低可解释性，还可能过拟合错误执行器。接口与高层任务语义没变，所以先让执行层忠实实现旧action更合理。

### Q3：DLS公式是什么？

主任务用`dq_task = Jᵀ(JJᵀ+λI)⁻¹e`，λ为`1e-4`。J由position Jacobian和权重0.35的rotation/approach-axis Jacobian拼成；最多15轮，每轮dq限制±0.04 rad，再做关节限位。

### Q4：如何证明284 mm不是IK发散？

最大误差集中在Reach且随target累加；阶段跳变、限位、碰撞都不存在。18个固定目标持有4 s时无持续振荡或数值失败，最终只剩8～10 mm稳定偏差。加30 mm lead bound后最大Reach误差立即降到约29.8 mm，说明主因是backlog。

### Q5：为什么20 mm lead误差更小却不用？

因为控制参数要看闭环任务。20 mm把Reach保留率降到2/10，旧策略无法按原时序完成handoff；30 mm同时约束积压并保持各阶段10/10，是更好的折中。

### Q6：如何区分IK误差和actuator误差？

用独立`ik_data`对qtarget做FK。总向量误差可精确拆为`target-FK(qtarget)`与`FK(qtarget)-actual`。前者约0.615 mm，后者约9.020 mm，因此局部稳态偏差主要来自执行器。

### Q7：为什么判断是重力静差而非力矩饱和？

肩和肘的q残差约9 mrad，servo force与`qfrc_bias`匹配；最大力22.5 Nm，远低于±150 Nm。提高kp后残差近似反比下降且force不饱和，这是有限增益position servo平衡重力的典型表现。

### Q8：为什么局部实验说posture不是主要静差，后来又说posture是Place根因？

两者状态分布不同。固定目标实验在局部稳态，posture只贡献约0.5 mm IK bias；Place远离home，而且每步Cartesian命令只有5～12 mm，raw回home项相对更强，甚至改变方向。局部结论不能无条件外推。

### Q9：如何排除Place servo符号错误？

servo-only和oracle模式的最终Cartesian command都是朝goal的，但actual TCP仍大多反向。再看FK(qtarget)已经反向，说明符号正确，错误发生在IK组合层。

### Q10：为什么nullspace没有保留姿态约束？

公式没有错。当前J是近满秩6×6，六个关节几乎全被六维任务占满。投影后的posture增量缩小约524倍，所以它数学上被正确投影，但工程上几乎不起作用。

### Q11：off模式会不会导致翻肘或撞限位？

10-seed对比中没有姿态突变或翻肘，最小限位余量约1.19 rad；最终60-seed最小余量1.117 rad。但这只是当前仿真分布的证据，不能保证任意工作空间都安全，后续仍应监控关节构型。

### Q12：最终为什么是59/60，不是100%？

seed31完成了运输、下降、打开和释放，final XY 11.57 mm且最终静止，但机器人持续有一个table contact，success要求contact为0，所以900步超时。这是释放后几何/接触边界case，不是IK方向回归。

### Q13：当前项目能否称为Sim-to-Real？

不能。它只在MuJoCo中把执行链从weld约束迁移到IK和关节actuator，并验证旧策略零样本兼容。真机还需要标定、通信时延、噪声、真实动力学、安全限幅、碰撞保护和硬件实验。

### Q14：这项工作最能体现什么能力？

不是“会调一个参数”，而是能从调用链和物理层建立可证伪假设；通过stage、FK中间量、固定状态快照、oracle和单变量消融逐层排除；最后用回归测试与扩展seed固化结果，同时明确证据边界。
