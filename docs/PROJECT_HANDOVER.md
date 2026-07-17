# UR5e + 4C2 + D435i 视觉抓放项目交接文档

## 1. 文档目的

本文档用于交接 `/home/lenovo/mujoco_learning/08_camera` 工程，使后续负责人无需阅读开发过程、阶段性测试记录或历史调参文档，也能完成以下工作：

- 理解系统目标、边界与完整运行链；
- 安装并启动当前仿真视觉闭环；
- 定位核心模型、配置和业务代码；
- 修改相机、视觉、PPO、IK或任务监督模块而不破坏接口；
- 将仿真接口逐步替换为D435i、UR5e和真实夹爪接口；
- 识别当前仍属于仿真专用的机制，避免将其误用于真机。

本文档描述的是当前可维护系统，不记录逐次实验、失败尝试和临时诊断脚本。

---

## 2. 项目概述

本项目在MuJoCo中构建了一套UR5e机械臂、4C2夹爪和腕部D435i相机的视觉抓取放置系统。系统处理3 cm红色立方体，任务流程为：

```text
home
→ camera_observe
→ RGB-D定位物体
→ 返回home
→ PPO生成TCP动作
→ DLS IK生成六关节目标
→ 关节执行器驱动机械臂
→ FSM控制夹爪和任务阶段
→ 抓取、抬升、搬运、放置和释放
```

当前工程已经形成仿真中的RGB-D视觉闭环。视觉估计不只是用于显示或离线计算误差，而是进入PPO物体观测、Place XY servo、下降/释放几何判断和物体相关安全限制。

当前工程还不能等同于完整真机系统。真实相机标定、UR5e通信、夹爪SDK反馈、硬件安全层和不依赖simulated latch的可靠抓取仍需接入。

### 2.1 当前固定运行基线

| 项目 | 当前值 |
|---|---|
| 工程根目录 | `/home/lenovo/mujoco_learning/08_camera` |
| Gym环境 | `My4C2AllStageSinglePPOV22Cube3cm-v0` |
| PPO checkpoint | `checkpoints/best_full_flow_v22.zip` |
| 任务场景 | `scene_cube3cm.xml` |
| 机器人模型 | `ur5e_4c2.xml` |
| 物体 | 0.03 m红色立方体 |
| PPO观测维度 | 39 |
| PPO动作维度 | 4 |
| 策略推理 | deterministic |
| TCP最大超前量 | `max_tcp_lead=0.03 m` |
| IK姿态模式 | `ik_posture_mode=off` |
| TCP mocap weld | 禁用；mocap仅可作为目标可视化 |
| 转接件/相机质量 | 0.100 kg / 0.075 kg |
| 当前相机模型状态 | CAD外观和安装分度已集成；光学原点仍待真机标定 |

工程由`07_test`的真实IK和关节执行器基线迁移而来，不以`06_4c2`作为当前代码基线。

---

## 3. 系统边界

### 3.1 当前系统负责

- MuJoCo中的UR5e、4C2、桌面、方块和目标建模；
- 腕部Color/Depth双相机数据渲染；
- HSV方块检测；
- Depth→Color几何投影和三维定位；
- 相机、World和UR5e Base坐标转换；
- 统一物体状态发布与失效处理；
- 39维PPO观测构造；
- PPO确定性推理；
- TCP目标生成、安全裁剪和Place XY修正；
- DLS IK与关节位置执行器；
- 仿真夹爪控制、任务阶段和成功评价；
- 面向真机替换的RobotState、GripperState和TaskSupervisor接口。

### 3.2 当前系统不负责

- D435i真实设备采集与硬件同步；
- 真机内参、Depth→Color外参和手眼标定；
- UR5e RTDE、servo或trajectory通信；
- 真实夹爪SDK；
- 真实夹持力、电流、堵转和到位反馈；
- 工业级碰撞检测、protective stop和急停；
- 任意物体、任意颜色或复杂背景下的通用识别；
- 方块六自由度姿态估计；
- 无simulated latch的稳定仿真物体保持。

---

## 4. 核心工程结构

交接维护只需优先关注以下文件。

```text
08_camera/
├── scene_cube3cm.xml
├── ur5e_4c2.xml
├── assets/
├── checkpoints/
│   └── best_full_flow_v22.zip
├── configs/
│   └── hsv_cube_localization.json
├── fourc2/
│   ├── __init__.py
│   ├── envs/
│   │   └── allstage.py
│   ├── camera_geometry.py
│   ├── rgbd_cube_localizer.py
│   ├── object_estimate.py
│   ├── visual_observation_adapter.py
│   └── task_supervisor.py
└── scripts/
    ├── eval_full_visual_closed_loop.py
    ├── eval.py
    └── trainenv.py
```

### 4.1 文件职责

| 文件 | 职责 | 修改风险 |
|---|---|---|
| `scene_cube3cm.xml` | 桌面、物体、目标、home和camera_observe | 改动会影响初态、视野、碰撞和训练分布 |
| `ur5e_4c2.xml` | UR5e、4C2、相机、mesh、关节和actuator | 改动会影响FK、IK、碰撞和相机外参 |
| `fourc2/__init__.py` | 注册Gym环境及3 cm任务参数 | 改动可能使checkpoint与环境不兼容 |
| `allstage.py` | 观测、控制、IK、FSM、reward和success | 核心高风险文件 |
| `camera_geometry.py` | 相机模型和全部刚体变换 | 坐标方向必须集中维护 |
| `rgbd_cube_localizer.py` | HSV与RGB-D三维定位 | 不应读取仿真物体真值 |
| `object_estimate.py` | 实时物体状态权威来源 | 不允许RGB-D模式静默真值回退 |
| `visual_observation_adapter.py` | 重建物体相关PPO观测 | 所有派生字段必须内部一致 |
| `task_supervisor.py` | 可部署任务阶段与夹爪接口 | 不允许读取MuJoCo object/contact真值 |
| `hsv_cube_localization.json` | 视觉阈值和过滤参数 | 真机环境需要重新标定 |
| `eval_full_visual_closed_loop.py` | 当前完整视觉闭环与实时演示入口 | 生产演示的首选入口 |

`docs/`和`outputs/`中的阶段报告、CSV、JSON及诊断结果属于开发证据，不是运行时依赖。删除这些记录不会改变控制行为，但正式归档时建议保留。

---

## 5. 完整运行架构

```text
┌─────────────────────────────────────────────────────────────┐
│                       感知层                                │
│ Color RGB ──HSV/轮廓──┐                                    │
│ Depth Z ──反投影──────┼→ Depth→Color投影 → mask内三维点     │
│ 相机位姿/内参─────────┘                    ↓                │
│                                  方块Color光学坐标          │
└────────────────────────────────────────────┬────────────────┘
                                             ↓
┌─────────────────────────────────────────────────────────────┐
│                     坐标与状态层                            │
│ Color optical → UR5e Base → MuJoCo World                    │
│                        ↓                                    │
│                 ObjectEstimateAuthority                     │
│         ┌──────────────┼───────────────┐                    │
│         ↓              ↓               ↓                    │
│      PPO观测       Place servo      安全/释放几何           │
└─────────┬───────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────┐
│                       决策层                                │
│ 39维observation → PPO → 4维action                           │
│ ObjectEstimate + RobotState + GripperState → TaskSupervisor │
└─────────┬───────────────────────────────────────┬───────────┘
          ↓                                       ↓
┌─────────────────────────────────────────────────────────────┐
│                       执行层                                │
│ TCP增量 → 阶段整形 → workspace/max lead → DLS IK            │
│ → 六关节目标 → position actuators                           │
│ TaskSupervisor/FSM → 夹爪开度命令                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. MuJoCo模型

### 6.1 场景模型

`scene_cube3cm.xml`通过`<include file="ur5e_4c2.xml"/>`加载机器人，并定义：

- 桌面顶面高度：Z=0.30 m；
- 物体：带freejoint的红色立方体；
- 物体边长：0.03 m，初始中心高度0.315 m；
- goal：已知任务目标，使用mocap body表示；
- `home` keyframe；
- `camera_observe` keyframe。

当前六关节关键姿态：

```text
home:
[-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]

camera_observe:
[-3.50356411, -1.39121562, 0.03083238,
 -0.38197572, -1.56229418, -0.34007296]
```

视觉采集使用固定过程：home平滑运动到camera_observe，采集RGB-D，再返回home。这样不会改变PPO原本使用的策略初态。

当前观察姿态是在原弯臂姿态的小邻域内调整得到的，六轴变化范数约
0.26 rad、单轴最大变化约0.16 rad，肩到腕距离略有减小，不属于依靠
机械臂过度伸直获得视野的方案。

### 6.2 UR5e和4C2

`ur5e_4c2.xml`包含六个UR5e关节、六个关节位置执行器、4C2夹爪和碰撞几何。

关键名称：

| 对象 | 名称 |
|---|---|
| TCP site | `pinch` |
| 末端安装site | `attachment_site` |
| 夹爪主关节 | `r_1_joint` |
| 夹爪执行器 | `fingers_actuator` |
| 夹爪控制范围 | 0～0.9 |

4C2的其余手指关节通过MuJoCo equality mimic与主关节同步。视觉mesh和碰撞proxy可能不同，修改外观mesh时不要误删碰撞几何。

### 6.3 当前相机模型

当前CAD安装链为：

```text
wrist_3_link
└── d435i_adapter                  mass=0.100 kg
    ├── d435i_camera               mass=0.075 kg
    │   └── d435i_mount_frame
    │       ├── eye_in_hand_depth
    │       └── eye_in_hand_color
    └── base_4c2
```

当前名义参数：

| 参数 | Depth | Color |
|---|---:|---:|
| 分辨率 | 640×360 | 640×360 |
| fovy | 58° | 42.5° |
| local position | `[0,0,0]` | `[-0.015,0,0]` m |
| local rotation | 单位旋转 | 单位旋转 |

关键固定变换：

```text
d435i_adapter relative to wrist_3_link:
pos  = [0, 0.092435, 0] m
quat = [0, 0, 1, 1]        # MuJoCo自动归一化

d435i_mount_frame relative to d435i_camera/adapter:
pos  = [0.017500001, 0.098872644, 0.054924176] m
quat = [0, 0, 1, 0]
```

- 转接件绕UR工具法兰轴的分度已按当前实物意图修正180°；
- `base_4c2`作了对应补偿，夹爪和`pinch` TCP世界位姿未改变；
- Depth对应左红外/Depth镜头，Color对应RGB镜头；
- 两个光学原点分别位于各自CAD镜头中心向内4.2 mm处；
- Color相对Depth保持名义+15 mm optical-X基线；
- CAD外形和机械安装关系已经集成，真机运行前仍要用设备标定替换
  4.2 mm内缩、15 mm基线和手眼外参。

当前安装版本的观察回归（seeds 0–99）：RGB-D定位100/100，三维误差
mean 0.920 mm、median 0.844 mm、P95 1.923 mm、max 2.817 mm；
异常任务接触0，方块最大扰动0.029 mm。该结果只验证观察与初始定位，
旧Phase 1/2文档中的完整任务成功率属于更新CAD前的历史模型版本。

### 6.4 CAD更新约定

当前`simple_model`中的转接件和相机CAD已经完成一次集成。以下约定用于
后续制造版本或CAD再次更新，不代表当前仍未集成。

CAD输入建议包含：

- `camera_adapter.step`：新转接件；
- `d435i_camera.step`：简化相机；
- `camera_mount_assembly.step`：只包含4C2安装基准、新转接件和D435i的小装配体；
- 可选的两个简化STL用于MuJoCo直接加载。

更新原则：

1. 保留现有UR5e和4C2运动/碰撞结构；
2. 只替换adapter视觉mesh并加入D435i视觉mesh；
3. 相机外观默认`contype=0, conaffinity=0`，避免未经验证改变碰撞；
4. 从小装配体读取真实相对位置和旋转，不凭截图猜符号；
5. CAD通常以毫米导出，MuJoCo使用米，必须显式缩放；
6. 光学camera元素必须放在真实镜头光学原点，而不是外壳几何中心；
7. 更新后重新确认视野、机器人自碰撞、桌面间隙和抓取覆盖范围。

---

## 7. 相机坐标与几何

### 7.1 坐标轴约定

MuJoCo camera frame：

```text
+X 向右，+Y 向上，-Z 向前
```

标准optical frame：

```text
+X 向右，+Y 向下，+Z 向前
```

转换集中在`camera_geometry.py`：

```python
MUJOCO_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])
```

禁止在其他模块中零散手动取反坐标。

### 7.2 变换命名

统一使用：

```text
T_A_B：把B坐标系中的点转换到A坐标系
```

例如：

```text
p_base = T_base_color_optical · p_color_optical
T_base_color_optical = T_base_world · T_world_color_optical
```

Base→World使用：

```text
T_world_base = inverse(T_base_world)
```

相机世界位姿必须从当前MuJoCo状态的`cam_xpos/cam_xmat`读取，不应手工复制XML数值到视觉代码。

### 7.3 安装变化与视觉代码依赖

当前视觉链没有硬编码相机world pose或`camera_observe`关节角：

- Color和Depth世界位姿每帧读取`cam_xpos/cam_xmat`；
- `Color_T_Depth`由两台相机当前位姿计算；
- 支撑面法向动态转换到Color optical frame；
- HSV、形态学和稳健深度过滤不依赖固定图像上下方向；
- 图像因安装分度旋转180°时，两路相机与坐标变换同步旋转。

因此更换安装位姿通常不需要改定位算法，但必须重新运行双相机几何、
Depth→Color、HSV定位、视野覆盖和Base坐标回归测试。固定15 mm断言仅
代表当前名义双流外参；真机接入时应改为读取设备标定值。

### 7.4 相机内参

由分辨率和垂直视场角计算：

```text
fy = 0.5H / tan(0.5fovy)
fx = fy
cx = (W-1)/2
cy = (H-1)/2
```

当前假设方形像素，主点位于图像中心。真机应直接读取RealSense SDK标定内参。

### 7.5 MuJoCo深度

`mujoco.Renderer`输出的Depth为正向、米制optical-Z，即标准光学坐标中的Z，不是彩色预览像素，也不是相机到空间点的欧氏距离。

界面中的Depth伪彩色只供人眼观察。定位算法使用原始浮点深度数组。

### 7.6 Depth→Color对齐

算法为：

```text
Depth像素(u,v,Z)
→ Depth optical XYZ
→ T_color_depth
→ Color optical XYZ
→ Color像素(u,v)
```

多个Depth点投影到同一个Color像素时使用z-buffer，只保留Color optical Z最小的最近点。没有对应深度的Color像素保持NaN，不使用零向量伪造空间点。

---

## 8. RGB-D物体定位

生产定位入口为`rgbd_cube_localizer.py`中的`localize_cube_rgbd()`。

### 8.1 RGB检测流程

```text
RGB
→ HSV
→ 双红色区间inRange
→ 形态学开运算
→ 形态学闭运算
→ 轮廓查找
→ 面积/长宽比/矩形度过滤
→ 最优轮廓
→ mask、腐蚀mask、bbox、像素中心
```

参数全部集中在`configs/hsv_cube_localization.json`。当前目标颜色来自XML：

```text
rgba = [0.9, 0.15, 0.1, 1.0]
```

当前HSV范围：

```text
H 0～12，S 120～255，V 80～255
H 170～179，S 120～255，V 80～255
```

红色跨越OpenCV HSV的0/179边界，因此必须使用两段范围。

### 8.2 三维点筛选

定位不读取bbox中心单像素深度。流程为：

1. 将整幅Depth投影到Color图像；
2. 只保留投影像素落入腐蚀后HSV mask的点；
3. 去除非有限点、Z≤0和配置深度范围外的点；
4. 多点落到同一Color像素时保留最近点；
5. 对Z使用中位数、MAD和分位数过滤；
6. 对保留的Color optical XYZ取中位数。

输出明确区分：

- `visible_surface_point_color`：当前可见表面的稳健中心；
- `estimated_object_center_color`：方块几何中心估计。

检测失败返回`valid=False`和`failure_reason`，不允许返回`[0,0,0]`。

### 8.3 几何中心假设

当前对象模型假设：

- 方块边长0.03 m；
- 方块正立；
- 方块位于水平支撑面；
- 相机主要看到顶面。

算法从可见表面中心沿支撑面法向反方向偏移半个边长0.015 m，得到几何中心。

方块倾斜、翻转、主要看到侧面或被夹爪严重遮挡后，该假设不再成立。扩展到一般物体时应改为六自由度姿态估计、模型拟合或多视角重建。

---

## 9. ObjectEstimate：物体状态权威来源

`object_estimate.py`定义统一实时物体状态：

```text
ObjectEstimate
├── position
├── orientation_wxyz（可选）
├── timestamp
├── valid
├── confidence
├── source
└── estimate_id
```

`ObjectEstimateAuthority`管理两种显式模式：

- `ground_truth`：仅用于仿真对照；
- `rgbd`：正式视觉输入模式。

关键规则：

- RGB-D模式拒绝`ground_truth`来源；
- Ground Truth模式拒绝视觉来源；
- 位置必须有限；
- confidence必须在[0,1]；
- 无效、缺失、过期或未来时间戳均抛出明确异常；
- 不允许用零向量代替失败结果；
- 不允许RGB-D模式静默读取MuJoCo物体真值；
- 每次消费记录consumer、control step、estimate ID、timestamp和source。

实时消费者包括：

- PPO观测；
- Reach/Grasp物体相关安全目标；
- Place XY servo；
- Place下降判断；
- Place释放几何判断；
- TaskSupervisor。

同一control step中的消费者应使用同一个`estimate_id + timestamp`。

### 9.1 抓取后传播

抓取前使用视觉位置。确认抓取后，`TcpObjectTracker`记录：

```text
p_tcp_object = R_world_tcpᵀ · (p_world_object - p_world_tcp)
```

之后利用TCP FK传播：

```text
p_world_object = p_world_tcp + R_world_tcp · p_tcp_object
```

该方法适合夹爪遮挡物体后的短期状态传播。真机中传播切换必须由可靠夹爪反馈确认，不能由simulated latch触发。

---

## 10. PPO接口

### 10.1 策略结构

- Stable-Baselines3 PPO；
- 39维输入；
- actor和critic各为两层64单元Tanh MLP；
- 4维连续动作；
- 动作范围[-1,1]；
- checkpoint参数包括`gamma=0.98`、`n_steps=256`、`batch_size=256`；
- 正式评估使用`deterministic=True`。

### 10.2 39维观测

| 索引 | 字段 | 当前来源 | 真机来源 |
|---|---|---|---|
| 0:6 | 六关节角 | MuJoCo qpos | UR5e编码器 |
| 6:12 | 六关节速度 | MuJoCo qvel | UR5e状态接口 |
| 12:15 | TCP/pinch位置 | FK | UR5e FK |
| 15:18 | object position | ObjectEstimate | RGB-D/跟踪器 |
| 18:21 | pregrasp position | estimate派生 | estimate派生 |
| 21:24 | grasp position | estimate派生 | estimate派生 |
| 24:27 | pinch→pregrasp | estimate+FK | estimate+FK |
| 27:30 | pinch→grasp | estimate+FK | estimate+FK |
| 30:33 | object→goal | Place门控+estimate | 已知goal+estimate |
| 33 | gripper state | 归一化夹爪命令 | 应改为SDK实际开度 |
| 34 | object lift | estimate Z差值 | estimate/传播Z差值 |
| 35:37 | 左右接触 | MuJoCo特权接触 | 不能直接获得，需替代 |
| 37:38 | vertical alignment | TCP FK | UR5e FK |
| 38:39 | grasp phase | 当前FSM状态 | 可部署supervisor状态 |

不能只替换`obs[15:18]`。pregrasp、grasp、相对位移、object-to-goal和lift都直接或间接依赖物体位置，必须基于同一个估计重新构造。

### 10.3 object_to_goal阶段门控

非Place阶段：

```text
obs[30:33] = [0,0,0]
```

Place阶段：

```text
obs[30:33] = goal_position - estimated_object_position
```

视觉适配器必须保留该门控，否则会改变策略训练时的数据分布。

### 10.4 PPO动作

```text
action[0:3]：TCP XYZ增量方向
action[3]：策略形式上的夹爪动作
```

当前privileged FSM使用脚本化夹爪命令，因此`action[3]`不是实际夹爪控制的唯一权威来源。阶段逻辑会对前三维动作再次整形：

- Reach：接近pregrasp；
- Grasp：限制XY，控制对齐和下降；
- Lift：只允许向上运动；
- Place：组合PPO XY和ObjectEstimate驱动的Place servo。

---

## 11. TCP控制与DLS IK

控制调用链：

```text
model.predict(observation)
→ env.step(action)
→ _apply_tcp_action(action)
→ TCP增量缩放和阶段整形
→ Place servo
→ workspace/table安全裁剪
→ target smoothing
→ max_tcp_lead裁剪
→ _solve_ik(target_pos)
→ 六关节位置目标
→ position actuators
→ mujoco.mj_step
```

### 11.1 Place XY servo

默认`place_xy_control_mode="combined"`：

```text
final_xy = policy_xy + servo_xy
servo_xy = gain · (goal_xy - estimated_object_xy)
```

servo有最大增量限制。Place不是纯PPO控制，维护和汇报时必须明确说明。

### 11.2 TCP lead限制

目标TCP与当前TCP的距离不得超过0.03 m：

```text
||target_tcp - actual_tcp|| <= 0.03
```

这用于防止PPO连续动作使未实现的TCP目标无限超前。

### 11.3 DLS IK

`_solve_ik()`使用单独`ik_data`迭代，不直接写真实arm qpos。

每次求解：

1. 从当前六关节角开始；
2. FK获得pinch位置和approach axis；
3. 计算位置误差和方向轴误差；
4. 使用`mj_jacSite`得到位置、旋转Jacobian；
5. 构造阻尼伪逆：

```text
J⁺ = Jᵀ(JJᵀ + λI)⁻¹
λ = 1e-4
```

6. 每次迭代关节增量裁剪到±0.04 rad；
7. 裁剪关节上下限；
8. 最多迭代15次；
9. 输出六关节位置目标给actuator。

收敛条件包括TCP位置误差和approach axis误差。当前`ik_posture_mode=off`，不加入home姿态项。

策略执行期间不直接写arm qpos，也不启用mocap weld拉动机械臂。

---

## 12. 任务阶段与夹爪

### 12.1 任务阶段

```text
Reach
→ Grasp: ALIGN → DESCEND → CLOSE → CONFIRM
→ Lift
→ Place
→ Release
```

### 12.2 当前privileged FSM

当前稳定演示默认仍使用原仿真FSM。它会读取：

- MuJoCo物体真实位置；
- 左右pad/object接触；
- contact penetration；
- 物体高度、速度和漂移；
- simulated grasp latch。

这些输入使仿真任务稳定，但无法直接从真机获得。因此该FSM属于仿真控制基线，不是最终部署方案。

### 12.3 可部署状态接口

`task_supervisor.py`定义：

```text
RobotState
├── tcp_position
├── world_from_tcp
├── tcp_velocity
├── vertical_alignment
├── timestamp
├── valid
└── source

GripperState
├── commanded_opening
├── actual_opening
├── velocity / motion_status
├── actuator_effort
├── fault
├── grasp_hold_confidence
├── timestamp
├── valid
└── source
```

`TaskSupervisor`只允许消费：

- ObjectEstimate；
- RobotState/TCP FK；
- GripperState；
- goal；
- 内部计时器和任务状态。

它不应读取object qpos/site、MuJoCo contact、penetration、simulated latch、reward或success。

### 12.4 可部署抓取确认

抓取确认综合：

- 实际夹爪开度；
- 闭合是否停止；
- actuator current/effort；
- TCP与ObjectEstimate几何关系；
- hold confidence；
- 连续多步一致性；
- 超时和fault。

缺失、过期或fault输入必须fail closed，不得使用零向量或仿真真值继续动作。

### 12.5 simulated latch

MuJoCo中的simulated latch会在满足仿真抓取条件后辅助保持物体，并可能直接更新object freejoint qpos/qvel。

必须牢记：

- latch不是传感器；
- latch不是真机抓取确认；
- latch不能作为ObjectEstimate传播的部署触发器；
- latch不能作为真机成功判据；
- 真实物体应由物理夹持保持，不存在软件写物体位姿的机制。

当前仿真纯接触保持尚不够稳定，因此latch仍保留为仿真辅助。

---

## 13. 配置管理

### 13.1 视觉配置

`configs/hsv_cube_localization.json`包含：

- 物体边长和放置假设；
- HSV上下界；
- 开闭运算kernel和迭代数；
- 轮廓面积范围；
- 长宽比和矩形度；
- mask腐蚀参数；
- 合法深度范围；
- 最少深度点数；
- MAD和分位数过滤参数。

不得在localizer或演示脚本中复制另一套视觉阈值。

### 13.2 任务参数

3 cm方块的任务阈值集中注册在`fourc2/__init__.py`的`CUBE3CM_KWARGS`。包括：

- `object_half_size=0.015`；
- pregrasp/grasp/lift高度；
- 对齐、闭合和稳定抓取阈值；
- Place servo增益与最大增量；
- Place下降和释放阈值。

修改这些参数会改变checkpoint所处的环境分布。除非计划重新验证或重新训练，不应只为提高成功率随意放宽阈值。

### 13.3 Checkpoint兼容性

加载checkpoint时必须确认：

- observation shape为39；
- action shape为4；
- 环境ID与V22 3 cm任务对应；
- 字段顺序、单位和阶段门控不变；
- 输入未额外归一化或改变VecNormalize行为。

---

## 14. 安装与启动

### 14.1 Python环境

当前常用解释器：

```bash
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python
```

项目提供：

- `environment.yml`；
- `requirements.txt`；
- `requirements-lock.txt`。

迁移到新机器后应创建独立环境，不要依赖硬编码的旧机器Conda路径。

### 14.2 实时完整视觉闭环

```bash
cd /home/lenovo/mujoco_learning/08_camera

MUJOCO_GL=glfw \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/eval_full_visual_closed_loop.py \
  --mode rgbd \
  --episodes 1 \
  --seed-offset 0 \
  --live \
  --show-camera \
  --viewer-width 1280 \
  --viewer-height 900 \
  --hide-pinch-site
```

窗口包括：

- MuJoCo第三人称Viewer；
- D435i Color/Depth实时窗口；
- RGB检测框；
- 方块检测状态；
- Color optical三维坐标；
- mask像素数和有效深度点数。

### 14.3 显示参数

```text
--viewer-width / --viewer-height
    修改第三人称窗口初始尺寸。

--hide-pinch-site
    仅隐藏TCP site外观，不影响FK、IK和控制。

--observe-move-extra-seconds
    增加home↔camera_observe移动时长。

--observe-hold-seconds
    设置camera_observe处停顿时间。
```

live模式和无GUI批量运行使用不同OpenGL后端：

- GUI：`MUJOCO_GL=glfw`；
- headless：通常使用`MUJOCO_GL=egl`。

### 14.4 Ground Truth模式

Ground Truth只用于仿真对照：

```bash
... scripts/eval_full_visual_closed_loop.py \
  --mode ground_truth \
  --episodes 1
```

不要把Ground Truth模式用于真机部署入口，也不要与RGB-D模式在同一episode混用。

---

## 15. 失败处理

### 15.1 视觉失败

localizer返回：

```text
valid=False
failure_reason=<明确原因>
```

控制入口应停止、保持或进入诊断失败，不应：

- 使用`[0,0,0]`；
- 使用上一帧但伪造新timestamp；
- 自动读取MuJoCo object site；
- 无限制继续发送PPO动作。

若未来增加“保持上一帧”，必须显式定义：

- 最大允许年龄；
- confidence衰减；
- 最大连续丢帧；
- 超时后的安全动作。

### 15.2 状态过期

ObjectEstimate、RobotState和GripperState必须带timestamp。过期状态不得继续用于任务监督。

### 15.3 IK失败

运行时应记录：

- 是否收敛；
- 迭代次数；
- position error；
- approach axis error；
- dq clip；
- joint limit clip；
- TCP tracking error。

真机接口中若IK失败，应停止或保持当前目标，不应发送未验证的关节跳变。

### 15.4 夹爪故障

真实GripperState出现以下情况必须fail closed：

- 通信中断；
- timestamp过期；
- fault/overcurrent；
- 命令无响应；
- 开合位置非法；
- 抓取确认超时。

---

## 16. 已知限制

### 16.1 视觉

- HSV只适合颜色已知的红色方块；
- 真机光照、曝光和背景会改变HSV分布；
- 当前只估计位置，不估计完整姿态；
- 固定15 mm中心偏移依赖正立方块和顶面可见；
- 抓取后遮挡会使实时重识别不稳定；
- 仿真相机没有RealSense噪声、空洞、反光和硬件同步问题。

### 16.2 相机模型

- 当前CAD相机、转接件、安装分度和质量已经集成；
- 当前4.2 mm镜头内缩和15 mm基线仍是名义光学外参；
- CAD镜头面不等于真机标定得到的光学原点；
- 真机必须读取SDK内参并完成手眼标定。

### 16.3 PPO观测

- `obs[35:37]`仍为MuJoCo左右接触特权状态；
- `obs[38]`仍依赖当前FSM阶段；
- `obs[33]`主要是夹爪命令而非可靠实际开度；
- 替换这些字段前需做策略敏感性评估，可能需要微调PPO。

### 16.4 FSM和夹爪

- 稳定演示默认仍使用privileged FSM；
- deployable TaskSupervisor接口已建立，但当前反馈代理不能完全代表真实夹爪；
- simulated latch仍是仿真保持辅助；
- 纯MuJoCo接触模型暂不能稳定替代latch。

### 16.5 安全

- MuJoCo workspace和接触限制不等同于工业机器人安全；
- 尚未接入UR protective stop、安全状态、速度缩放和急停；
- 真机运行前必须增加独立watchdog和硬件级安全流程。

---

## 17. 真机接口设计

### 17.1 D435i接口

输入应至少包含：

```text
Color image
Depth image / depth scale
Color intrinsics
Depth intrinsics
T_color_depth
hardware timestamp
frame validity
```

真实帧应转换为当前localizer期望的RGB和米制optical-Z。不得使用屏幕伪彩图作为深度输入。

### 17.2 UR5e状态接口

应提供：

```text
joint positions
joint velocities
TCP pose / FK
robot mode
safety mode
protective stop
timestamp
valid
```

### 17.3 UR5e控制接口

必须明确：

- 使用笛卡尔servo还是关节servo/trajectory；
- 控制频率；
- 速度和加速度限制；
- 命令超时；
- 网络延迟；
- 停止行为；
- watchdog；
- 上电和恢复流程。

不要将MuJoCo position actuator命令直接等价为真机接口。

### 17.4 夹爪SDK

至少需要：

```text
commanded opening
actual opening
opening velocity / moving / stopped
motor current or effort
command reached
fault / overcurrent / communication status
timestamp
```

`grasp_hold_confidence`应由这些原始量和连续时间规则计算，不应只信任单一“已抓住”布尔值。

### 17.5 手眼标定

当前是eye-in-hand安装。真机需要确定相机optical frame相对TCP或法兰的刚体变换：

```text
T_tcp_color_optical
T_tcp_depth_optical
```

标定完成后应验证：

- 旋转矩阵正交且det=+1；
- 齐次矩阵最后一行为`[0,0,0,1]`；
- Base→Camera→Base往返误差；
- 不同机械臂姿态观察同一固定点时Base坐标稳定；
- RGB-D估计与独立测量真值的一致性。

---

## 18. 真机迁移顺序

必须按低风险顺序推进。

### 阶段1：CAD和相机模型

- 已集成新转接件和D435i外观mesh；
- 已从CAD和实物装配意图确定当前安装关系与法兰分度；
- 已设置转接件100 g、相机75 g；
- 已确认夹爪/TCP不受分度修正影响，观察往返无异常任务接触；
- 已用自然camera_observe完成100/100可视位置复验。

### 阶段2：D435i只读

- 接入Color/Depth；
- 读取SDK内外参；
- 验证深度单位、时间戳和帧同步；
- 不发送机器人运动。

### 阶段3：手眼标定

- 标定相机相对TCP；
- 验证Base坐标稳定性；
- 测量真实定位误差和失效率。

### 阶段4：UR5e只读

- 读取关节和机器人状态；
- 用FK重建TCP；
- 与机器人控制器报告的TCP对比；
- 验证坐标系、单位和时间戳。

### 阶段5：低速空载运动

- home→camera_observe→home固定轨迹；
- 低速、低加速度；
- 人工确认；
- 急停和watchdog可用；
- 不抓物体。

### 阶段6：夹爪接口

- 只做开合测试；
- 标定开度、电流、堵转和到位状态；
- 构造真实GripperState；
- 不执行自主抓取。

### 阶段7：Shadow Mode

- 视觉估计和TaskSupervisor旁路运行；
- PPO仍不控制真机或只显示建议TCP；
- 比较人工/传感器判断；
- 验证失效和超时行为。

### 阶段8：分步抓取

- 假物体；
- 人工确认后执行Reach；
- 人工确认后下降；
- 低力闭合；
- 小高度抬升；
- 最后再启用完整抓放。

---

## 19. 维护规则

### 19.1 修改相机模型

必须同时检查：

- 外观mesh位置；
- optical origin；
- optical axis；
- Color/Depth相对变换；
- fovy/分辨率；
- 相机视野覆盖率；
- camera_observe姿态；
- 机械干涉和桌面间隙。

### 19.2 修改视觉算法

必须保持输入/输出契约：

```text
输入：RGB、米制Depth、内参、T_color_depth、支撑面方向、配置
输出：valid、failure_reason、表面点、几何中心、诊断信息
```

detector和localizer不得读取object site真值。真值只能进入仿真评价器。

### 19.3 修改PPO观测

- 不得改变39维顺序而继续使用旧checkpoint；
- 不得只改object_position而保留真实派生字段；
- 保持object_to_goal阶段门控；
- 保持单位为m、rad、s体系；
- 观测来源变化需重新做策略敏感性评估。

### 19.4 修改控制

- 不要绕过DLS IK直接写arm qpos；
- 不要启用mocap weld代替关节执行；
- 不要取消TCP lead和关节限位；
- Place servo修改必须确认使用统一ObjectEstimate；
- reward/success不应驱动未来真机TaskSupervisor。

### 19.5 修改FSM

- deployable supervisor不得读取MuJoCo contact/object truth；
- 抓取确认必须使用可在真机获得的量；
- 输入无效时fail closed；
- simulated latch必须继续隔离；
- 不要通过放宽任务成功阈值掩盖抓取确认问题。

---

## 20. 常见问题定位

| 现象 | 优先检查 |
|---|---|
| Viewer不刷新 | GLFW context是否在RGB-D离屏渲染后恢复 |
| Color看不到方块 | camera_observe、mount外参、fovy、物体采样范围 |
| HSV没有轮廓 | 色彩空间、曝光、HSV配置、面积/矩形度 |
| 有轮廓但定位失败 | mask腐蚀、有效Depth点数、深度范围 |
| Depth图颜色接近 | 这是全图伪彩色范围；检查原始米制Depth统计 |
| 三维坐标方向错误 | optical轴、T_A_B方向、Color_T_Depth符号 |
| Base坐标随姿态变化 | 手眼外参、矩阵方向、相机world pose读取 |
| PPO动作异常 | 39维索引、派生字段一致性、阶段门控 |
| TCP目标越积越远 | max_tcp_lead、tracking error、actuator跟踪 |
| IK跳动或不收敛 | Jacobian、关节限位、dq clip、目标不可达 |
| 能接近但不能闭合 | Grasp阶段、XY/Z对齐、夹爪FSM |
| 能闭合但不能抬升 | 抓取确认、夹爪反馈、物体保持、latch |
| Place偏移 | ObjectEstimate传播、goal坐标、Place servo |
| 释放失败 | 下降高度、goal XY、夹爪实际开度、释放计时 |

---

## 21. 接手检查清单

### 工程可运行

- [ ] 能加载`scene_cube3cm.xml`；
- [ ] 能加载V22 checkpoint；
- [ ] observation shape为39；
- [ ] action shape为4；
- [ ] live Viewer和RGB-D窗口正常刷新；
- [ ] home→camera_observe→home正常；
- [ ] RGB-D结果进入`ObjectEstimate`；
- [ ] PPO、Place servo和几何判断使用统一estimate；
- [ ] DLS IK通过关节actuator执行；
- [ ] 未启用mocap weld；
- [ ] 未直接写arm qpos。

### CAD更新

- [x] 获取简化转接件和相机STEP/STL；
- [x] 获取安装装配体并解析固定相对关系；
- [x] 确认毫米/米单位；
- [x] 确认安装面、CAD原点和法兰轴；
- [x] 确认Color、左IR/Depth名义光学原点；
- [x] 更新mesh、质量和mount位姿；
- [x] 检查观察往返干涉与100-seed视野覆盖；
- [ ] 用制造版CAD复核最终孔位和外壳间隙；
- [ ] 用RealSense设备标定替换名义光学外参。

### 真机前

- [ ] D435i SDK输入有效；
- [ ] 内参和Depth→Color外参来自设备；
- [ ] 手眼标定完成；
- [ ] UR5e状态时间戳和坐标系确认；
- [ ] 夹爪actual opening/current/fault可读；
- [ ] 限速、workspace、watchdog和急停可用；
- [ ] Shadow Mode通过；
- [ ] 分级低速测试方案审批完成。

---

## 22. 交接结论

当前工程已经具备清晰的仿真视觉闭环和真机接口替换方向：

- RGB-D视觉链已经模块化；
- 相机几何和坐标约定集中管理；
- 物体状态通过ObjectEstimate统一；
- PPO接口和39维观测已固定；
- PPO动作通过TCP目标、DLS IK和关节actuator执行；
- TaskSupervisor为真机夹爪和阶段判断预留了接口；
- 当前特权FSM和simulated latch的边界已经明确。

接手后的近期主线不是重新训练PPO，而是：

```text
更新真实CAD安装
→ 接入D435i与标定
→ 接入UR5e状态/低速控制
→ 接入夹爪SDK反馈
→ 完善deployable TaskSupervisor
→ 建立真机安全层
→ 分阶段完成真实抓取
```

在以上接口和安全验证完成前，项目应被描述为“仿真中的完整RGB-D视觉闭环抓放系统及其Sim-to-Real接口原型”，而不是已经完成部署的真实机器人系统。
