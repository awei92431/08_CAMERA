# 08_camera 项目完整讲解（汇报与代码走读版）

## 1. 一句话说明项目

本项目在 MuJoCo 中建立 UR5e + 4C2 夹爪 + 腕部 D435i 的抓取放置系统。相机先把 3 cm 红色方块定位到机器人坐标系，视觉位置被写入 PPO 的 39 维观测；PPO 输出 TCP 增量，DLS IK 将 TCP 目标转换为六关节位置命令，夹爪 FSM 完成闭合、保持和释放。

当前最诚实的表述是：

> 已完成“仿真中的 RGB-D 视觉闭环抓放”和可迁移接口的第一轮解耦；尚未完成真正 Sim-to-Real，因为真实 D435i 标定、UR5e 通信、夹爪 SDK 反馈、安全系统，以及摆脱 simulated latch 的可靠物理抓取仍未接入。

## 2. 项目来源与当前基线

- `08_camera` 从 `07_test` 的真实 IK/关节 actuator 基线迁移，不是直接从 `06_4c2` 复制。
- 机器人环境：`My4C2AllStageSinglePPOV22Cube3cm-v0`。
- PPO：`checkpoints/best_full_flow_v22.zip`。
- XML：`scene_cube3cm.xml` 包含 `ur5e_4c2.xml`。
- 物体：边长 0.03 m 的红色立方体。
- 执行基线：`max_tcp_lead=0.03`、`ik_posture_mode=off`、无 TCP mocap weld、策略 deterministic。

旧 `README.md` 开头写着“尚未加入视觉识别”，这已经过期；当前视觉闭环以本文件、Phase 1/2 文档和实际代码为准。

## 3. 目录结构：先记住这六组

### 3.1 物理模型

- `scene_cube3cm.xml`：桌面、方块、目标、home/camera_observe keyframe。
- `ur5e_4c2.xml`：UR5e、4C2、相机、关节、actuator、碰撞模型。
- `assets/`：机器人和夹爪网格。

### 3.2 环境与控制核心

- `fourc2/envs/allstage.py`：Gym 环境、观测、reward、阶段、TCP控制、DLS IK、夹爪与仿真 latch。
- `fourc2/__init__.py`：所有 Gym 环境 ID 和 3 cm 方块参数注册。

### 3.3 视觉

- `fourc2/camera_geometry.py`：相机内参、坐标轴、Depth→Color、Color→World/Base。
- `fourc2/rgbd_cube_localizer.py`：HSV 检测和 RGB-D 三维定位。
- `configs/hsv_cube_localization.json`：颜色、轮廓、形态学和深度过滤参数。

### 3.4 状态接口

- `fourc2/object_estimate.py`：统一的 `ObjectEstimate` 和抓取后 TCP 传播。
- `fourc2/visual_observation_adapter.py`：把视觉物体位置重建成 PPO 兼容观测。
- `fourc2/task_supervisor.py`：`RobotState`、`GripperState` 和可部署 supervisor。

### 3.5 运行入口

- `scripts/eval_full_visual_closed_loop.py`：当前最重要的完整视觉闭环入口。
- `scripts/eval_phase2_task_supervisor.py`：原特权 FSM 与 deployable FSM 的 A/B。
- `scripts/eval.py`：旧的一般 PPO eval，不包含当前完整视觉观察流程。

### 3.6 证据与文档

- `docs/full_visual_closed_loop_evaluation.md`
- `docs/phase1_unified_object_estimate.md`
- `docs/phase2_deployable_task_supervisor.md`
- `docs/sim2real_privileged_state_audit.md`
- `docs/sim2real_migration_plan.md`
- `outputs/full_visual_closed_loop/`
- `outputs/phase2_task_supervisor/`

### 3.7 关键调试参数快速定位

这一节用于被问到参数时快速跳到代码。下面的行号对应当前版本；后续代码
增删可能使行号移动，因此同时给出可以用 rg 搜索的稳定名称。

#### 先记住参数优先级

1. 正式环境是 **My4C2AllStageSinglePPOV22Cube3cm-v0**。它用
   [CUBE3CM_KWARGS](../fourc2/__init__.py#L44) 覆盖 allstage.py 构造函数
   的通用默认值。因此 3 cm 任务优先查看 fourc2/__init__.py:44，不能只看
   allstage.py:119。
2. 相机、机械装配、质量、关节 actuator 和 keyframe 属于物理模型参数，
   修改位置在 XML，不在视觉代码。
3. HSV/Depth 和 ArUco 检测参数以 configs 目录中的 JSON 为权威来源。
4. eval_full_visual_closed_loop.py 的 ArUco 命令行参数可以临时覆盖 JSON。
5. checkpoint 对应原训练分布。修改 home、观测、动作尺度、阶段阈值或控制
   结构后，不能默认原 PPO 仍保持原性能，必须重新回归验证。

#### A. 演示入口与界面

| 想调什么 | 当前值/入口 | 代码位置与搜索名 | 影响 |
|---|---|---|---|
| 正式环境 ID | My4C2AllStageSinglePPOV22Cube3cm-v0 | [eval 脚本:48](../scripts/eval_full_visual_closed_loop.py#L48)，搜 ENV_ID | 决定加载哪套任务注册参数 |
| PPO checkpoint | best_full_flow_v22.zip | [eval 脚本:49](../scripts/eval_full_visual_closed_loop.py#L49)，搜 CHECKPOINT | 更换后确认观测维度和训练环境一致 |
| Color/Depth 尺寸 | 640×360 | [eval 脚本:54](../scripts/eval_full_visual_closed_loop.py#L54)，搜 WIDTH, HEIGHT | 同时影响内参、速度和可视化 |
| 第三人称窗口 | 1280×900 | [eval 脚本:806](../scripts/eval_full_visual_closed_loop.py#L806)，参数 --viewer-width/--viewer-height | 只影响 Viewer 初始窗口 |
| 观察移动额外时长 | live 默认 2.0 s | [eval 脚本:360](../scripts/eval_full_visual_closed_loop.py#L360)，参数 --observe-move-extra-seconds | home 与观察位之间的演示速度 |
| 观察位停顿 | live 默认 1.5 s | [eval 脚本:364](../scripts/eval_full_visual_closed_loop.py#L364)，参数 --observe-hold-seconds | 拍摄前停顿，不改变姿态 |
| 相机图像窗口 | --show-camera | [eval 脚本:804](../scripts/eval_full_visual_closed_loop.py#L804) | 显示 Color/Depth 诊断 |
| 隐藏 TCP 小球 | --hide-pinch-site | [eval 脚本:810](../scripts/eval_full_visual_closed_loop.py#L810) | 只隐藏 site 外观，不删除 FK/IK 使用的 pinch |
| 独立位姿查看视角 | distance 1.05、azimuth 135°、elevation -25° | [查看脚本:71](../scripts/view_eye_in_hand_camera_pose.py#L71) | 只影响第三人称查看视角 |

演示节奏优先通过命令行调，不必修改源码：

~~~bash
--observe-move-extra-seconds 3.0 --observe-hold-seconds 2.0
~~~

#### B. 机械臂、相机和夹爪 XML

| 想调什么 | 当前值 | 代码位置与搜索名 | 注意 |
|---|---|---|---|
| PPO 初始姿态 | home 六轴 [-1.5708,-1.5708,1.5708,-1.5708,-1.5708,0] rad | [scene:54](../scene_cube3cm.xml#L54)，搜 key name="home" | 会改变策略初始分布 |
| 相机观察姿态 | [-3.50356411,-1.39121562,0.03083238,-0.38197572,-1.56229418,-0.34007296] rad | [scene:60](../scene_cube3cm.xml#L60)，搜 camera_observe | 只改 qpos/ctrl 前六项，不要覆盖随机物体 freejoint |
| 转接件相对腕部 | pos="0 0.092435 0"，quat="0 0 1 1" | [robot XML:175](../ur5e_4c2.xml#L175)，搜 d435i_adapter | 机械装配总变换，不等于光学外参 |
| 转接件质量 | 0.100 kg | [robot XML:182](../ur5e_4c2.xml#L182)，搜 d435i_adapter_visual | 当前 visual geom 不参与碰撞 |
| D435i 质量 | 0.075 kg | [robot XML:191](../ur5e_4c2.xml#L191)，搜 d435i_camera_visual | mesh 位移来自 CAD 导出坐标 |
| 光学安装 frame | pos="0.017500001 0.098872644 0.054924176"，quat="0 0 1 0" | [robot XML:210](../ur5e_4c2.xml#L210)，搜 d435i_mount_frame | 调相机整体位姿应从这里核对 CAD |
| Depth 相机 | pos="0 0 0"，fovy=58° | [robot XML:218](../ur5e_4c2.xml#L218)，搜 eye_in_hand_depth | 共同 frame 的基准原点 |
| Color 相机 | pos="-0.015 0 0"，fovy=42.5° | [robot XML:219](../ur5e_4c2.xml#L219)，搜 eye_in_hand_color | 15 mm 符号已按 optical 坐标验证 |
| UR5e 前三轴 actuator | gain 2000、速度 bias 400、force ±150 | [robot XML:8](../ur5e_4c2.xml#L8)，搜 gainprm="2000" | 最终控制基线，不为视觉问题随意调 |
| UR5e 腕部三轴 actuator | gain 500、速度 bias 100、force ±28 | [robot XML:15](../ur5e_4c2.xml#L15)，搜 gainprm="500" | 同上 |
| 夹爪 actuator | ctrl 0..0.9、kp=15、force ±3 | [robot XML:330](../ur5e_4c2.xml#L330)，搜 fingers_actuator | 0.9 是 closure 命令，不是毫米开度 |

#### C. RGB-D 与 ArUco 参数

| 参数组 | 当前关键值 | 位置 |
|---|---|---|
| 方块尺寸/颜色 | 0.03 m；RGBA [0.9,0.15,0.1,1] | [HSV 配置:2](../configs/hsv_cube_localization.json#L2) |
| 红色 HSV | [0,120,80]..[12,255,255] 和 [170,120,80]..[179,255,255] | [HSV 配置:7](../configs/hsv_cube_localization.json#L7) |
| 形态学 | open 3、close 5、erode 3，各 1 次 | [HSV 配置:12](../configs/hsv_cube_localization.json#L12) |
| 轮廓过滤 | area 40..50000 px、rectangularity≥0.45、aspect 0.45..2.2 | [HSV 配置:16](../configs/hsv_cube_localization.json#L16) |
| Depth 过滤 | Z 0.02..1.5 m、至少 8 点、MAD 3.5、quantile 5%..95% | [HSV 配置:24](../configs/hsv_cube_localization.json#L24) |
| ArUco 字典/ID/尺寸 | DICT_4X4_50、ID 0、marker 0.04 m、cube 0.03 m | [ArUco 配置:2](../configs/aruco_goal_localization.json#L2) |
| ArUco 多帧稳定 | 8/30 帧、position std≤0.003 m、重投影≤3 px | [ArUco 配置:6](../configs/aruco_goal_localization.json#L6) |
| ArUco 合法范围 | depth 0.1..1.5 m、Base workspace low/high | [ArUco 配置:10](../configs/aruco_goal_localization.json#L10) |
| 到位后才识别 | q error≤0.06 rad、qvel≤0.03 rad/s、稳定 10 步、settle 0.5 s | [ArUco 配置:16](../configs/aruco_goal_localization.json#L16) |

HSV 失败先看 hsv_mask、轮廓面积和 depth point 数；ArUco 误差先确认渲染
实体尺寸与 marker_size_m 一致，再讨论 PnP。不要先放宽阈值。

#### D. PPO→TCP→DLS IK

| 参数 | 当前值 | 位置与搜索名 | 作用 |
|---|---|---|---|
| policy frame skip | 10 | [allstage:122](../fourc2/envs/allstage.py#L122)，搜 frame_skip | 一个 PPO 动作对应的 MuJoCo 子步数 |
| TCP 动作尺度 | action_scale=0.05 m | [allstage:124](../fourc2/envs/allstage.py#L124) | action[0:3] 到 TCP 增量的总尺度 |
| 最大 TCP lead | 0.03 m | [allstage:125](../fourc2/envs/allstage.py#L125)，限制逻辑在 [allstage:1272](../fourc2/envs/allstage.py#L1272) | 限制目标领先实际 TCP |
| IK 方向权重 | 0.35 | [allstage:126](../fourc2/envs/allstage.py#L126)，搜 ik_axis_weight | 保持夹爪 approach axis |
| IK 姿态模式 | off | [allstage:128](../fourc2/envs/allstage.py#L128)，搜 ik_posture_mode | 最终基线，不加 home 关节修正 |
| IK 阻尼 | 1e-4 | [allstage:2419](../fourc2/envs/allstage.py#L2419)，搜 damping | DLS 稳定性 |
| IK 最大迭代 | 15 | [allstage:2443](../fourc2/envs/allstage.py#L2443)，搜 range(15) | 每个 policy step 的迭代上限 |
| IK 收敛 | position<1 mm，axis error<0.03 | [allstage:2454](../fourc2/envs/allstage.py#L2454) | 诊断 converged 的条件 |
| 单次 IK 关节增量 | ±0.04 rad | [allstage:2519](../fourc2/envs/allstage.py#L2519) | 防止求解跳变 |
| TCP workspace | low [-0.20,-0.60,0.20]，high [0.80,0.60,0.85] m | [allstage:339](../fourc2/envs/allstage.py#L339) | 安全裁剪 |
| 桌面安全高度 | table 0.30 m；pinch min = table+0.027+0.002 | [allstage:249](../fourc2/envs/allstage.py#L249) | 防止最低夹爪 link 撞桌 |

#### E. 3 cm 任务、Place servo 与 FSM

下表是正式 Cube3cm 环境的实际覆盖值，集中位于
[fourc2/__init__.py:44](../fourc2/__init__.py#L44)：

| 参数 | 当前值 | 影响 |
|---|---|---|
| object_half_size | 0.015 m | 方块中心高度和几何中心补偿 |
| pregrasp_height | 通用默认 0.07 m | 方块上方预抓取点；Cube3cm 未覆盖 |
| grasp_height_offset | 0.018 m | 夹取 TCP 相对方块中心高度 |
| lift_height | 0.05 m | Lift 成功/目标高度 |
| grasp_descend_xy_threshold | 0.018 m | XY 足够近后才允许下降 |
| grasp_xy_close_threshold | 0.012 m | 允许闭合的 XY 条件 |
| grasp_z_close_threshold | 0.022 m | 允许闭合的 Z 条件 |
| stable_grasp_xy_threshold | 0.006 m | 多步稳定抓取 XY 条件 |
| latch_grasp_xy_threshold | 0.004 m | strict/latch 质量阈值 |
| place_handoff_xy_threshold | 0.024 m | 进入 Place handoff |
| place_descent_xy_threshold | 0.028 m | 允许向目标下降 |
| place_xy_servo_gain | 0.55 | ObjectEstimate→Goal 的 XY 修正增益 |
| place_xy_servo_max_delta | 0.012 m/step | servo 单步最大 XY 修正 |
| release_open_xy_threshold | 0.024 m | 允许释放的 XY 条件 |
| release_success_lift | 0.016 m | 释放/落桌高度条件 |
| release_min_open_steps | 8 | 最少保持打开步数 |

相关执行位置：

- Grasp/Lift/Place 动作整形：[allstage:1148](../fourc2/envs/allstage.py#L1148)；
- Place XY servo：[allstage:1169](../fourc2/envs/allstage.py#L1169) 和
  [allstage:1212](../fourc2/envs/allstage.py#L1212)；
- TaskSupervisor 参数组装：[allstage:647](../fourc2/envs/allstage.py#L647)；
- supervisor 代理阈值：input age 1.0 s、stopped velocity 0.02、effort 0.10、
  fully closed 0.97，位于 [task_supervisor:92](../fourc2/task_supervisor.py#L92)；
- object/goal 数据源、FSM 和 latch 架构开关位于
  [allstage:167](../fourc2/envs/allstage.py#L167)。这些不是普通性能调参，
  A/B 测试中禁止混用。

#### F. 随机位置与覆盖范围

- 方块初始位置：X 0.35..0.65 m、Y -0.20..0.20 m，
  [allstage:2922](../fourc2/envs/allstage.py#L2922)；
- Full/Place 目标相对方块半径 0.05..0.13 m、角度 -π..π，
  [allstage:396](../fourc2/envs/allstage.py#L396) 和
  [allstage:2932](../fourc2/envs/allstage.py#L2932)；
- 桌面边界：X 0.20..1.10 m、Y -0.35..0.35 m，
  [allstage:337](../fourc2/envs/allstage.py#L337)。

#### G. 最快搜索命令

~~~bash
# 姿态、相机和夹爪
rg -n 'camera_observe|d435i_mount_frame|eye_in_hand_|fingers_actuator' \
  scene_cube3cm.xml ur5e_4c2.xml

# PPO、TCP、IK、Place servo
rg -n 'action_scale|max_tcp_lead|ik_posture|damping|place_xy_servo' \
  fourc2/envs/allstage.py fourc2/__init__.py

# 阶段、抓取、释放与 latch
rg -n 'grasp_.*threshold|release_|stable_required|simulated_latch' \
  fourc2/envs/allstage.py fourc2/__init__.py fourc2/task_supervisor.py

# 视觉配置
rg -n 'marker_size|HSV|contour|mad|quantile|required_valid_frames' \
  configs fourc2 scripts
~~~

一次只改一类参数，并保存 seed、配置 diff 和回归结果。相机机械位姿变化后至少
重跑视野覆盖和定位误差；控制/FSM 参数变化后至少重跑固定 seed 的完整闭环
对照。不要用放宽成功阈值掩盖视觉、IK 或抓取保持的根因。

## 4. 完整运行数据流

```text
reset到home
  ↓
关节actuator插值运动到camera_observe
  ↓
Color RGB + Depth optical-Z
  ↓
HSV找到红色方块mask
  ↓
Depth像素反投影 → Depth光学三维点
  ↓ Color_T_Depth
Color光学三维点 → 投影到Color图像
  ↓ 仅保留落入mask的最近深度点
稳健表面中心 → 估计方块几何中心
  ↓ T_base_color_optical / inverse(T_base_world)
Base坐标 → World坐标 → ObjectEstimate
  ↓
回到home
  ↓
重建PPO obs[15:35]
  ↓
PPO(39维观测) → 4维动作
  ↓
前三维动作 → TCP位置增量
  ↓ workspace/safety/Place servo/max lead
DLS IK → 6个关节目标
  ↓
MuJoCo position actuator → UR5e运动
  ↓
夹爪FSM控制开合
  ↓
Reach → Grasp → Lift → Place → Release
```

## 5. XML 模型怎么讲

### 5.1 场景

`scene_cube3cm.xml` 包含：

- `table`：桌面；桌面顶面 Z=0.30 m。
- `object`：带 freejoint 的红色 3 cm 方块，中心初始 Z=0.315 m。
- `goal`：mocap 目标方块，坐标在任务中已知。
- `home` keyframe：六关节角约 `[-90,-90,90,-90,-90,0]°`。
- `camera_observe` keyframe：专门为腕部相机观察设计的固定安全姿态。

当前 `camera_observe` 六关节角为：

```text
[-3.50356411, -1.39121562, 0.03083238,
 -0.38197572, -1.56229418, -0.34007296] rad
```

它是在转接件法兰分度修正后、原弯臂姿态的小邻域内选择的观察位，
肩到腕距离没有增加，不是依靠接近伸直的机械臂换取视野。

完整视觉流程不是从观察姿态直接开始执行策略，而是：home → camera_observe → 拍摄 → home → PPO。这避免为了视觉永久改变 PPO 原训练初态。

### 5.2 D435i 模型

当前安装链位于 `ur5e_4c2.xml` 的 `wrist_3_link` 下：

```text
wrist_3_link
└── d435i_adapter                 mass=0.100 kg
    ├── d435i_camera              mass=0.075 kg
    │   └── d435i_mount_frame
    │       ├── eye_in_hand_depth pos=[0,0,0], fovy=58°
    │       └── eye_in_hand_color pos=[-0.015,0,0], fovy=42.5°
    └── base_4c2
```

- adapter 相对 wrist：`pos=[0,0.092435,0]`，`quat=[0,0,1,1]`；该
  quaternion 会由 MuJoCo 自动归一化。
- mount 相对 camera/adapter：`pos=[0.017500001,0.098872644,0.054924176]`，
  `quat=[0,0,1,0]`。
- 两路分辨率由 renderer 设为 640×360。
- Depth 原点代表左红外/深度光学原点。
- Color 相对 Depth 名义水平基线为 15 mm。
- 两相机光轴平行，分别对准 CAD 的左红外/Depth 与 RGB 镜头中心。
- 两个光学原点均从 CAD 镜头外表面沿光轴向内 4.2 mm。
- CAD 外观与装配分度已集成；4.2 mm 内缩和 15 mm 基线仍是 nominal
  optical extrinsic，真机必须用 RealSense 标定与手眼标定替换。
- 没有仿真左右红外匹配，只模拟最终 Color 与 Depth 数据流。

### 5.3 4C2 夹爪

- 主驱动关节：`r_1_joint`。
- actuator：`fingers_actuator`，控制范围 0～0.9。
- 其余手指关节通过 equality mimic 同步。
- `pinch` site 是 TCP/FK/IK 使用的夹取中心；界面的蓝/灰小点只是 site 可视化，可隐藏但不能删除其计算作用。

## 6. 相机几何：最容易被追问的部分

### 6.1 坐标轴

MuJoCo camera frame：

```text
+X 右，+Y 上，-Z 向前
```

标准 optical frame：

```text
+X 右，+Y 下，+Z 向前
```

转换只在 `camera_geometry.py` 中定义：

```python
MUJOCO_FROM_OPTICAL = diag(1, -1, -1)
```

这样避免在各函数里零散取反坐标。

### 6.2 内参

MuJoCo 给出垂直视场角 `fovy`，程序按针孔模型计算：

```text
fy = (H/2) / tan(fovy/2)
fx = fy
cx = (W-1)/2
cy = (H-1)/2
```

Color 与 Depth 分别根据自己的 fovy 计算，因此不能用 resize/crop 或人工估算 15 mm 对应多少像素。

### 6.3 深度定义

`mujoco.Renderer` 返回的是米制、正向的 optical-Z，不是伪彩色数值，也不是相机到点的欧氏距离。界面上的彩色 Depth 只用于人眼显示；算法使用原始浮点深度数组。

### 6.4 Depth→Color

`project_depth_points_to_color()` 实现：

```text
(u_d,v_d,Z_d)
→ Depth optical XYZ
→ Color_T_Depth
→ Color optical XYZ
→ (u_c,v_c)
```

多个 Depth 点落到同一个 Color 像素时使用 z-buffer，只保留 Color optical Z 最小的最近点。缺失点用 NaN，不制造 `[0,0,0]`。

### 6.5 Color→Base/World

变换命名统一为 `T_A_B`：把 B 系中的点转换到 A 系。

```text
T_base_color_optical
= T_base_world @ T_world_color_optical
```

相机世界位姿直接从当前 `mjData.cam_xpos/cam_xmat` 读取，不手抄 XML。视觉定位得到 Base 坐标后，用 `inverse(T_base_world)` 得到 PPO 使用的 World 坐标。

### 6.6 安装姿态变化为什么不需要重写视觉算法

- Color/Depth 世界位姿每帧从 `cam_xpos/cam_xmat` 读取；
- `Color_T_Depth` 每次由两台相机的实际位姿计算；
- 桌面法向会转换到当前 Color optical frame，图像旋转 180°不会改变
  方块几何中心的计算方向；
- HSV、形态学、MAD和分位数过滤不依赖固定图像上下方向；
- `camera_observe` 只由 XML keyframe读取，没有复制进视觉模块。

因此本次转接件翻转、4.2 mm光学内缩和观察姿态更新不需要修改定位
算法。需要重新验证的是视野覆盖、像素面积阈值和误差统计；这些已经用
当前模型的100个固定seed回归通过。

当前模型中 base/world 的关系应由实际 body pose读取，即使数值重合也不靠名称猜测。

## 7. 方块识别与三维定位

代码入口是 `localize_cube_rgbd()`。

### 7.1 RGB 检测

1. RGB→HSV。
2. 红色跨越 HSV 0°，所以配置使用两段阈值：0～12 和 170～179。
3. 形态学开运算去噪，闭运算补洞。
4. 查找轮廓。
5. 用面积、长宽比、矩形度过滤。
6. 选取得分最高的、面积大且接近方形的轮廓。
7. 生成 mask、腐蚀 mask、bbox 和像素中心。

所有参数集中在 `configs/hsv_cube_localization.json`，不是散落硬编码。

### 7.2 RGB-D 融合

算法不是读取 bbox 中心的一个深度值，而是：

- 把整幅 Depth 投影到 Color；
- 只取落入腐蚀后 HSV mask 的三维点；
- 删除非有限、Z≤0 和范围外的点；
- 用深度中位数、MAD 和 5%～95% 分位过滤边缘/桌面离群点；
- 对剩余三维点取中位数，得到 `visible_surface_point_color`。

### 7.3 几何中心估计

当前假设方块正立、放在水平桌面、相机主要看到顶面。可见顶面中心沿支撑面法向反方向移动半个边长 15 mm，得到 `estimated_object_center_color`。

这个方法在抓取前很有效，100 个可视随机位置平均三维误差约 0.775 mm；但方块倾斜、翻转、侧面主导或严重遮挡后，固定 15 mm 模型不再严格成立。

## 8. 39维 PPO 观测必须能背出来

| 索引 | 内容 | 当前 RGB-D 闭环来源 |
|---|---|---|
| 0:6 | 六关节角 | MuJoCo qpos；真机对应编码器 |
| 6:12 | 六关节速度 | MuJoCo qvel；真机对应机器人状态 |
| 12:15 | TCP/pinch 世界位置 | 关节 FK |
| 15:18 | object position | ObjectEstimate |
| 18:21 | pregrasp position | estimate + pregrasp height |
| 21:24 | grasp position | estimate + grasp height offset |
| 24:27 | pinch→pregrasp | 派生量 |
| 27:30 | pinch→grasp | 派生量 |
| 30:33 | object→goal | Place阶段才打开，否则为零 |
| 33 | gripper state | 当前是归一化夹爪命令，不是可靠真机反馈 |
| 34 | object lift | estimate Z − 初始 estimate Z |
| 35:37 | 左右接触 | 仍是 MuJoCo 特权接触状态 |
| 37 | vertical alignment | TCP FK方向与期望竖直方向点积 |
| 38 | grasp phase | 旧 FSM 阶段，仍含仿真特权依赖 |

视觉接入不是只改 `obs[15:18]`。`obs[15:35]` 中所有物体派生量必须基于同一个 estimate 重建，否则 pregrasp、相对向量等会泄露真实位置。

## 9. PPO 怎么工作

- 算法：Stable-Baselines3 PPO。
- 输入：39维 float observation。
- 网络：actor 和 critic 各两层 64 单元 Tanh MLP。
- 输出：4维连续动作，范围 [-1,1]。
- checkpoint 内参数：`gamma=0.98`、`n_steps=256`、`batch_size=256`。
- 评估使用 `deterministic=True`。

动作含义：

- `action[0:3]`：TCP XYZ 增量方向，经 `action_scale` 缩放。
- `action[3]`：策略形式上输出夹爪动作，但当前 privileged FSM 实际使用脚本化夹爪开合，因此第四维不是当前夹爪的最终权威命令。

不同阶段还会对 PPO TCP 动作整形：Grasp 限制横向/下降，Lift 只允许向上，Place 组合 PPO XY 与基于 ObjectEstimate 的 servo。

## 10. PPO动作到真实关节：DLS IK

核心调用链：

```text
env.step(action)
→ _apply_tcp_action(action)
→ safe TCP target
→ max_tcp_lead 0.03 m
→ _solve_ik(target_pos)
→ arm actuator ctrl
→ mujoco.mj_step
```

DLS IK 每个 policy step 最多迭代 15 次：

1. 用单独 `ik_data` 做 FK，不直接改真实机械臂 qpos。
2. 误差包含 TCP 位置误差和夹爪 approach axis 方向误差。
3. `mj_jacSite` 求位置/旋转 Jacobian。
4. 阻尼伪逆：`Jᵀ(JJᵀ + λI)⁻¹`，λ=1e-4。
5. 每次关节增量限制在 ±0.04 rad。
6. 裁剪关节上下限。
7. 最终输出六关节位置目标给 position actuators。

当前 `ik_posture_mode=off`，不会用 home 姿态项干扰主笛卡尔任务。mocap target 只作目标标记，不通过 weld 拉动机器人；策略执行期间也不直接写 arm qpos。

## 11. 阶段、夹爪和 latch

任务阶段是：Reach→Grasp→Lift→Place。

### 11.1 原 privileged FSM

它使用 MuJoCo 中的物体真值、左右碰撞接触、penetration 和稳定步数完成阶段切换及夹爪开合。优点是仿真稳定；缺点是真机无法直接得到这些量。

### 11.2 Phase 1：ObjectEstimate

`ObjectEstimate` 至少包含 position、timestamp、valid、confidence、source、estimate_id。`ObjectEstimateAuthority` 保证：

- episode 固定为 `ground_truth` 或 `rgbd` 模式；
- RGB-D 模式拒绝 ground-truth source；
- 无效、过期、未来时间戳直接报错；
- 不能静默用零或真实 object site 回退；
- PPO、Place servo、下降/释放几何和物体相关安全限制使用同一 estimate ID。

抓取前使用最新视觉位置。确认抓取后，`TcpObjectTracker` 保存视觉初始化的 TCP—物体相对关系，之后用 TCP FK 传播物体估计，不读取真实物体位置进行修正。

### 11.3 Phase 2：TaskSupervisor

新接口只消费：

- `ObjectEstimate`；
- `RobotState`：TCP FK、速度、方向；
- `GripperState`：命令/实际开度、速度、effort、fault、hold confidence；
- 目标、计时器、任务状态。

它不直接读取 object qpos/site、MuJoCo contact、penetration、latch、reward 或 success。

抓取确认需要几何稳定、闭合命令生效、夹爪在完全闭合前停止或 effort 上升、hold confidence 有效，并连续满足若干步。输入缺失/过期/fault 时 fail closed。

### 11.4 simulated latch

MuJoCo 夹爪纯接触难以稳定保持方块，所以旧逻辑会在确认抓取后直接传播/写入物体 qpos/qvel，相当于仿真辅助“粘住”物体。

Phase 2 已把 latch 与 supervisor 输入、ObjectEstimate 传播触发、抓取成功判据隔离，但 latch 仍然改变仿真物理。因此它不是可迁移机制，也不能说系统已经完全摆脱仿真特权。

## 12. 结果怎么解释，避免讲错

### 12.1 视觉与统一估计

- 当前CAD安装、负载和自然观察姿态：100/100 RGB-D初始定位成功；三维
  误差 mean 0.920 mm、median 0.844 mm、P95 1.923 mm、max 2.817 mm；
  异常任务接触0，方块最大扰动0.029 mm。
- Phase 1历史闭环使用旧相机安装版本：初始定位100/100，mean 0.775 mm、
  median 0.794 mm、P95 1.364 mm、max 2.135 mm。
- Phase 1 对照：Ground Truth 98/100，RGB-D 99/100。
- PPO/servo/safety 的 estimate source consistency failure 为 0。

历史结果说明约0.8 mm视觉误差没有明显降低原抓放性能；当前安装版本的
初始定位已经复验，但若要汇报“当前模型100-seed完整抓放成功率”，必须
重新运行完整Ground Truth/RGB-D对照，不能直接沿用Phase 1数字。

### 12.2 Phase 2 supervisor

- privileged FSM：99/100。
- deployable FSM：7/100。
- latch-off 诊断：0/10。
- 93 个 deployable 失败集中在抓取确认未建立，而不是 Reach 或视觉失败。

不要把 7% 解释为 PPO 不行。相同 PPO 在 privileged FSM 下仍为 99%；下降来自仿真 GripperState effort proxy、抓取确认和纯接触保持。

不同报告中 98%/99% 的 privileged 数字来自不同阶段的评估脚本/代码时点，汇报时分别注明 Phase 1 与 Phase 2，不要混成同一次实验。

## 13. 当前实时演示脚本

```bash
cd /home/lenovo/mujoco_learning/08_camera
MUJOCO_GL=glfw \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/eval_full_visual_closed_loop.py \
  --mode rgbd --episodes 1 --seed-offset 0 \
  --live --show-camera \
  --viewer-width 1280 --viewer-height 900 \
  --hide-pinch-site
```

界面显示 Color/Depth、HSV检测框、Color optical XYZ、mask 像素数和有效深度点数。live 模式默认将观察移动额外延长 2 s，并在 camera_observe 停 1.5 s。

## 14. Leader 常见追问与建议回答

### “这是真正视觉闭环吗？”

是仿真视觉闭环：RGB-D 结果真正进入 PPO 的物体观测，并统一进入 Place servo/几何控制，不是只做离线误差统计。但 obs[35:39]、原 FSM 和 simulated latch 仍有仿真特权，因此不是完整真机闭环。

### “为什么用 HSV，不用深度学习？”

当前任务是单一、颜色已知的红色规则方块。HSV 可解释、低延迟、无需数据集，适合先验证几何和控制链。真实环境光照/背景复杂后可将 detector 换成学习方法，而 Depth→Color 和坐标变换模块仍可复用。

### “深度图颜色差不多，真的使用深度了吗？”

伪彩色只是全图范围可视化；算法直接使用米制 optical-Z。它不是取一个中心像素，而是使用 mask 内数百个三维深度点并做稳健过滤。

### “为什么抓住后不一直看相机？”

夹爪会遮挡方块，单帧视觉不稳定。因此初版在确认抓取时保存 TCP—物体相对关系，之后通过关节编码器/FK传播。真机需要可靠的夹爪确认作为传播切换条件，也可增加低频重识别修正。

### “PPO直接输出关节角吗？”

不是。PPO输出 TCP XYZ 增量，DLS IK每步求六关节目标，再通过关节 position actuator 执行。这样更接近真机笛卡尔控制链，也避免直接写 qpos。

### “Place 是纯 PPO 吗？”

不是。默认是 combined：PPO给出运动趋势，Place XY servo 用同一个 ObjectEstimate 修正目标方向，下降和释放由几何/FSM门控。这一点必须主动说明。

### “99%是否代表能直接上真机？”

不能。99%证明的是当前仿真模型和特权 FSM 下的闭环可行性。真机还缺标定、接口、真实夹爪反馈、安全层和无 latch 的物理抓取验证。

### “deployable FSM为什么只有7%？”

不是视觉定位失败，也不是 PPO突然退化。主要是 MuJoCo effort proxy 和纯接触模型无法稳定给出连续抓取保持证据；关闭 latch 后也无法稳定抬升。需要真实夹爪 SDK 信号或更可信的传感器/接触建模，而不是立即放宽阈值或重训 PPO。

## 15. 真机迁移还缺什么

1. D435i SDK：Color、Depth、硬件时间戳和真实 depth scale。
2. 真实内参和 Depth→Color 外参，替换 nominal 15 mm。
3. 手眼标定，得到真实 `T_tcp_camera` 或 `T_base_camera`。
4. UR5e 状态接口：关节位置/速度、TCP、机器人模式、protective stop。
5. UR5e 控制接口：低速笛卡尔或关节 servo/trajectory，明确频率和延迟。
6. 夹爪 SDK：命令开度、实际开度、速度/到位、电流/力、fault。
7. 真机 TaskSupervisor：抓取确认、超时、重试、掉落和释放确认。
8. 安全：工作空间、速度/加速度、桌面 keep-out、watchdog、急停和人工接管。
9. 分级验证：只读→标定→空载固定轨迹→假物体→单步 Reach→低速抓取。

## 16. 你应该优先读的代码顺序

不要从 2800 行的 `allstage.py` 第一行硬读。推荐：

1. `scripts/eval_full_visual_closed_loop.py`：先理解完整故事。
2. `fourc2/rgbd_cube_localizer.py`：理解“怎么看见方块”。
3. `fourc2/camera_geometry.py`：理解“像素怎么变成机器人坐标”。
4. `fourc2/object_estimate.py`：理解“谁是物体位置权威来源”。
5. `allstage.py::_get_obs`：记住39维观测。
6. `allstage.py::_apply_tcp_action`：看 PPO 动作如何变成 TCP 目标。
7. `allstage.py::_solve_ik`：看 TCP 如何变成关节目标。
8. `allstage.py::_update_grasp_latch`：理解仿真辅助及其局限。
9. `fourc2/task_supervisor.py`：理解真机替换方向。
10. Phase 1/2 和 Sim-to-Real 文档：最后整理实验结论。

## 17. 五分钟汇报稿

> 我的项目目标是让 UR5e 和 4C2 夹爪完成视觉引导的方块抓取放置。原系统已经有 PPO，但物体位置来自 MuJoCo 真值。我在真实 IK 和关节 actuator 基线上加入了腕部 D435i 双数据流模型，并建立了从 RGB-D 到 PPO 和控制层的完整数据链。
>
> 相机模型有 Color 和 Depth 两个光学原点，分别使用实际的分辨率和视场角。视觉首先在 Color 图像中用 HSV 和轮廓几何检测红色 3 cm 方块，然后把每个 Depth 像素反投影到 Depth optical frame，通过 Depth→Color 外参投影到 Color 图像，只保留落入 HSV mask 的最近三维点。经过中位数、MAD和分位数滤波后得到可见表面中心，再根据方块尺寸估计几何中心。之后从 Color optical 转到 UR5e Base，再转到 PPO 的 World frame。
>
> PPO输入是39维。我没有只替换物体坐标三维，而是用同一视觉估计重建 object、pregrasp、grasp、相对位移、object-to-goal和lift等所有物体派生字段。PPO输出4维动作，其中前三维形成TCP增量；经过安全限制和Place servo后，DLS IK计算六关节目标，最后由关节actuator执行。抓取后由于遮挡，我保存视觉初始化的TCP—物体相对关系，用编码器/FK继续传播物体估计。
>
> 100个随机位置的RGB-D定位成功率为100%，平均三维误差0.775毫米。统一ObjectEstimate后，Ground Truth和RGB-D完整成功率分别为98%和99%，说明该视觉误差没有明显影响原控制性能，而且PPO、servo和安全层每一步使用的是同一个estimate ID。
>
> 同时我做了Sim-to-Real真值审计，并建立RobotState、GripperState和独立TaskSupervisor。它不读取MuJoCo物体真值、接触、reward或latch。但当前deployable FSM只有7%，而原FSM是99%，问题集中在夹爪抓取确认和纯接触物理保持；关闭仿真latch后也无法稳定抬升。这说明下一步不是继续调PPO，而是接入真实D435i标定、UR5e状态/控制接口和夹爪开度、电流、故障反馈，并逐级做低速安全测试。

## 18. 最后必须守住的表述边界

可以说：

- 仿真 RGB-D 定位已完成；
- 视觉估计真正进入 PPO 和控制消费者；
- PPO→TCP→DLS IK→关节 actuator 已跑通；
- 已建立可替换的 ObjectEstimate/RobotState/GripperState/TaskSupervisor 接口；
- 已明确真机迁移阻碍和测试路线。

暂时不要说：

- 已经完全摆脱 MuJoCo 真值；
- deployable FSM 已达到可用性能；
- simulated latch 等价于真实抓取；
- 仿真 0.8 mm 误差代表 D435i 真机也能达到同样精度；
- 当前系统可以未经标定与安全验证直接上 UR5e。
