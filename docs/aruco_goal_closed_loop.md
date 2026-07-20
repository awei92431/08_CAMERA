# ArUco episode goal closed loop

## Scope

This feature adds a visual goal source without changing the red-cube RGB-D
localizer, PPO checkpoint, IK, controller gains, reward thresholds, gripper
FSM, or observation dimension. The legacy default remains
`goal_source=ground_truth`; ArUco is enabled explicitly.

## Runtime sequence and data flow

```text
home
  -> actuator interpolation to camera_observe
  -> WAIT_OBSERVE_STABLE (no formal ArUco capture)
  -> CAPTURE_GOAL (the only state that accepts detections)
  -> 8 valid frames, median position, per-axis std <= 3 mm
  -> GOAL_FROZEN in Base, then converted once to World
  -> existing RGB-D cube localization at camera_observe
  -> RETURN_HOME (goal cannot be updated)
  -> POLICY_EXECUTION
       frozen GoalEstimate -> PPO obs object_to_goal
                           -> Place XY servo
                           -> Place descent/release geometry
                           -> task metrics / supervisor
```

The coordinate chain is:

```text
ArUco corners
 -> solvePnP: T_color_optical_marker
 -> dynamic camera pose: T_base_color_optical
 -> marker center in Base
 -> + [0, 0, cube_size/2] in Base table-normal direction
 -> goal center in Base
 -> inv(T_base_world)
 -> frozen goal center in MuJoCo World
```

`T_A_B` maps a point in B into A. `T_base_color_optical` and
`T_base_world` are read from current MuJoCo state through the existing
`camera_geometry.py`; no camera XML pose is copied into the detector. The
named `base` body is yawed about 180 degrees relative to World in this model,
so the table has negative Base X. This is reflected in the configured Base
workspace bounds.

The 30 mm cube-center correction is implemented as `0.5 * cube_size_m` in
`fourc2/aruco_goal_localizer.py`. With the default cube this is exactly 15 mm
along Base table +Z. The marker face is at table z + 0.2 mm to avoid
z-fighting.

## Timing gate

The gate is implemented by `GoalCaptureSession` in
`fourc2/aruco_goal_localizer.py` and executed by
`capture_aruco_episode_goal()` in `scripts/eval_full_visual_closed_loop.py`.
The transition to `CAPTURE_GOAL` requires all of:

- maximum six-joint error <= 0.06 rad;
- maximum six-joint speed <= 0.03 rad/s;
- 10 consecutive stable simulation steps;
- 0.5 s continuous settle time.

The 0.06 rad position tolerance is intentionally above the measured steady
actuator residual (about 0.046 rad) of this model. A 0.02 rad threshold would
never open the capture window even when the physical arm had stopped.
Interpolation completion by itself is not considered stable.

`submit_detection()` immediately ignores calls in every state except
`CAPTURE_GOAL`. After `GOAL_FROZEN`, `GoalEstimateAuthority` rejects all later
publications until episode reset. The live D435i panel stays visually quiet
outside valid detections. During capture, a valid marker receives a purple
outline and the label “GOAL DETECTED”; there is no dedicated ArUco window.

## Detection and validation

- OpenCV `DICT_4X4_50`, marker ID 0, physical edge 40 mm.
- `ArucoDetector` with subpixel corner refinement.
- `SOLVEPNP_IPPE_SQUARE` using the real marker edge length.
- Expected ID only; pixel area, optical depth, finite pose, reprojection error,
  Base workspace and goal height are validated.
- At least 8 valid frames out of at most 30 are required.
- The frozen position is the per-axis median; maximum per-axis standard
  deviation must be <= 3 mm.
- Invalid estimates contain NaN and an explicit reason, never `[0, 0, 0]`.
- Failure terminates before PPO. ArUco mode rejects ground-truth publication.

The scene marker is a UV-mapped, non-colliding mesh. The original `goal_site`
remains a small diagnostic site and is read only by the ground-truth source or
explicitly named evaluator fields.

## Shared frozen source

`GoalEstimateAuthority` is the single runtime source. In ArUco mode the direct
MuJoCo goal reads remaining in `allstage.py` are limited to:

- the explicit ground-truth publisher, which rejects use outside
  `goal_observation_mode=ground_truth`;
- `goal_truth_position_evaluator_only` and localization error diagnostics.

Observation, Place servo, descent/release, task metrics and TaskSupervisor all
call `_control_goal_estimate()` / `_control_goal_position()`. Per-step usage
logs contain consumer, `estimate_id`, timestamp and source. In the executed
closed-loop episode, all 456 checked policy steps used one goal ID/timestamp;
consistency failures were 0.

Reward/success still use MuJoCo object/contact state as part of the existing
simulation evaluator/FSM. Their goal distances now use the frozen ArUco goal,
not `goal_site`. Removing the remaining object/contact privileges is outside
this goal-localization change.

## Actual tests

### Gate/freeze/failure tests

`scripts/test_aruco_goal_localization.py` passed all assertions:

- detection outside the observation state is ignored;
- nonzero qvel prevents capture even at the target q;
- only stable multi-frame capture freezes the goal;
- a frozen goal cannot be overwritten;
- 30 invalid frames fail closed;
- an empty ArUco authority raises instead of reading goal truth;
- an ArUco authority rejects a ground-truth estimate.

### 40 mm marker, 100-seed XY evaluation

- ArUco goal detected: 83/100. The other 17 episodes failed closed before
  object localization or PPO.
- ArUco goal XY error on 83 valid estimates: mean 1.708 mm, median 1.638 mm,
  P95 4.330 mm, max 5.184 mm.
- RGB-D object XY error: mean 0.971 mm, median 0.871 mm, P95 2.189 mm,
  max 2.450 mm.
- Final object center to true goal center XY error: mean 8.361 mm,
  median 7.857 mm, P95 14.976 mm, max 18.693 mm.
- All 83 episodes that entered policy control completed the simulator's task;
  overall success was 83/100 because all 17 failures were ArUco visibility or
  decode failures.
- Goal-error/final-placement-error correlation was 0.128 and
  object-error/final-placement-error correlation was 0.027. The final Place
  residual is therefore dominated by Place approach/release behavior rather
  than the initial visual XY estimates.
- The simple XY “inside goal region” rule uses center distance
  <= (40 mm - 30 mm)/2 = 5 mm; 13/83 executed episodes met it. This scalar
  proxy ignores cube yaw and is not an exact polygon-containment proof.

Results: `outputs/aruco4cm_xy_error_100seeds/`.

### Complete visual closed loop

Final geometry, seed 0:

- ArUco goal valid: 8/8 frames; 3D goal error 2.608 mm (evaluator only).
- RGB-D cube error: 0.458 mm.
- Reach, Grasp, Lift, Place, full success: all true.
- Episode policy steps: 456.
- Frozen-goal source consistency failures: 0.

Results are in `outputs/aruco_goal_closed_loop/summary.json`,
`aruco_rgbd_episodes.csv`, and `aruco_rgbd_steps.csv`.

## Reproduction

```bash
cd /home/lenovo/mujoco_learning/08_camera

MUJOCO_GL=egl \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/test_aruco_goal_localization.py --episodes 20

MUJOCO_GL=egl \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/eval_aruco_goal_closed_loop.py --episodes 1 --seed-offset 0

# Interactive proof of the gated capture window:
MUJOCO_GL=glfw \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/eval_full_visual_closed_loop.py --mode rgbd --episodes 1 \
  --goal-source aruco --live --show-camera
```

## Current limitations

- The current fixed `camera_observe` yields only 83% complete-marker detection
  coverage for a 40 mm marker over seeds 0-99. The system correctly fails
  closed, but the observation pose/FOV must be improved before deployment.
- The simulation has ideal lens distortion and a generated marker. Real D435i
  deployment still needs measured Color intrinsics/distortion, CAD/hand-eye
  extrinsics, printed-marker size verification, exposure testing and a real
  table/base transform.
- One full task episode demonstrates integration, not a statistically strong
  task-success estimate. A larger closed-loop run should follow after deciding
  whether to adjust the observation pose for the lower image boundary.
- This is a complete visual goal loop in simulation, not a completed real-robot
  ArUco interface.
