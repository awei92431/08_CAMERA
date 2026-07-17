# 08_camera：UR5e + 4C2 完整视觉闭环仿真

本项目由 `07_test` 的真实 IK/关节执行器基线迁移而来，现已包含腕部
D435i Color/Depth 双流模型、RGB-D 方块定位、ArUco 目标定位、冻结式
`ObjectEstimate`/`GoalEstimate`、PPO TCP 输出、DLS IK、关节 actuator、
夹爪 FSM，以及 Ground Truth / 视觉闭环对照评估。

当前 ArUco 实体边长为 40 mm，方块边长为 30 mm。项目仍是仿真验证，
真机部署前还需要 D435i 实际内外参、手眼标定、UR5e/夹爪 SDK 与安全层。
整体结构见 [项目交接文档](docs/PROJECT_HANDOVER.md)，视觉与迁移边界见
[Sim-to-Real 审计](docs/sim2real_privileged_state_audit.md)。

UR5e cube grasp/place project with the original Robotiq 2F85 gripper replaced by the 4C2 gripper.

## Reproduce on another device

The repository includes runtime meshes, formal PPO checkpoints, TensorBoard logs,
the final 100-episode evidence package, and an exact Python dependency lock.
Follow [PROJECT_HANDOVER.md](docs/PROJECT_HANDOVER.md), then run:

```bash
conda env create -f environment.yml
conda activate fourc2
python -B scripts/verify_portable_install.py --episodes 1
```

The original CAD/assembly source files are retained for future mechanical
changes. Runtime loading uses the converted MuJoCo meshes under `assets/`;
SolidWorks/STEP/Z3 source files are not loaded by the simulator.

## Baseline

- Code baseline: `07_test` real-IK execution path
- Robot: UR5e with the 4C2 gripper
- Policy checkpoint: `checkpoints/best_full_flow_v22.zip`
- Default task: `My4C2AllStageSinglePPOV22Cube3cm-v0`

## Main Files

- `ur5e_4c2.xml`: UR5e arm plus mounted 4C2 gripper.
- `scene_cube3cm.xml`: default 3 cm cube task scene, now including `ur5e_4c2.xml`.
- `scene.xml`: larger cube scene, also switched to `ur5e_4c2.xml`.
- `fourc2/envs/allstage.py`: 0706 staged training environment, with 4C2 stage IDs registered from `fourc2`.
- `scripts/trainenv.py`: PPO training entrypoint.
- `scripts/eval.py`: evaluation entrypoint.
- `scripts/eval_full_visual_closed_loop.py`: interactive/full visual loop.
- `scripts/eval_xy_error_statistics.py`: 40 mm ArUco / 30 mm cube XY audit.

The MuJoCo task interface is intentionally kept compatible with the 0706 code:

- Gripper actuator name: `fingers_actuator`
- TCP site name: `pinch`
- Attachment site name: `attachment_site`
- Contact pad geoms: `left_pad1`, `left_pad2`, `l_3_geom`, `l_2_geom`, `right_pad1`, `right_pad2`, `r_3_geom`, `r_2_geom`

## Verified

Run from this project root:

```bash
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python - <<'PY'
import gymnasium as gym
import numpy as np
import fourc2

env = gym.make("My4C2AllStageCube3cm-v0")
obs, info = env.reset(seed=0)
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
assert np.isfinite(obs).all()
env.close()
print("06_4c2 smoke pass")
PY
```

Additional checks already passed:

- `scene_cube3cm.xml`: `nq=19`, `nv=18`, `nu=7`, `neq=6`
- `ur5e_4c2.xml`: 4C2 gripper actuator is `fingers_actuator`, `ctrlrange="0 0.9"`, `kp="25"`
- 4C2 is modeled as one electric cylinder: `r_1_joint`, `l_1_joint`, `r_2_joint`, `l_2_joint`, `r_3_joint`, and `l_3_joint` are synchronized by joint mimic equalities.
- Cube3cm grasp target keeps the 4C2 lowest finger link just above the table: `grasp_height_offset=0.018`, with `pinch_min_z_over_table=0.333`.
- The visual STL fingers/pads are visual-only for cube contact; grasp physics and eval contact metrics use only the invisible box collision proxies (`*_pad*_collision`).
- The `pinch` TCP is aligned near the nominal center of the tactile block proxies, and the blue mocap target sphere is reduced to avoid hiding contact geometry.
- Stage reset/step smoke passed for reach, grasp, lift, place, and full cube3cm envs.
- Tiny SB3 training smoke passed:

```bash
CUDA_VISIBLE_DEVICES= /home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python -B scripts/trainenv.py \
  --env-id My4C2ReachStageCube3cm-v0 \
  --total-timesteps 64 \
  --n-envs 1 \
  --n-steps 32 \
  --batch-size 32 \
  --eval-freq 32 \
  --eval-episodes 1 \
  --run-name smoke_4c2_reach \
  --terminal-log-interval 1 \
  --log-interval 1
```
