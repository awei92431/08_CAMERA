# 07_test: mocap weld vs actuator IK zero-shot test

This directory is an independent copy of the runtime environment. The source
`06_4c2` was treated as read-only. No policy was trained.

## Frozen checkpoint and protocol

- checkpoint: `checkpoints/best_full_flow_v22.zip`
- source: `06_4c2/runs/v22_4c2_lift_place_transition_200k/models/best_handoff_model.zip`
- SHA256: `4d3ed31379804a2a769e4eaa19d31b295b4415b23544a7cea2cd9a9003036d8f`
- deterministic policy, identical reset seeds 0--9, 10 episodes per execution layer
- environment: `My4C2AllStageSinglePPOV22Cube3cm-v0`

## Results

| execution | full success | Reach ever | Grasp ever | Lift ever | Place ever | mean TCP tracking error | max TCP tracking error |
|---|---:|---:|---:|---:|---:|---:|---:|
| mocap + soft weld | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 21.83 mm | 72.82 mm |
| DLS IK + joint actuators | 0/10 | 10/10 | 8/10 | 8/10 | 0/10 | 32.97 mm | 283.96 mm |

IK called the existing `_solve_ik()` 7,475 times. The IK XML contains no
`mocap_tcp_weld`; the mocap body remains only as a target marker. Arm qpos is
not assigned in the action path. `_solve_ik()` returns clipped joint targets,
which are written to the six existing position actuators on every substep.

The main failure is Place (8/10 episodes reached Place and timed out there).
Seeds 1 and 8 failed in Grasp. Thus the old policy retains useful Reach and
substantial Grasp/Lift behaviour, but no complete-task success under the
untuned actuator layer.

Recommendation: tune the IK/actuator tracking layer first, especially target
rate/lag and Place tracking. A short fine-tune is premature while mean error is
33 mm and transient maximum error is 284 mm. After controller-only tuning is
validated with the same seeds, use a short fine-tune only if Place still fails.

Structured results, per-episode CSVs, and two videos per mode are under
`results/mocap` and `results/ik`. Regression tests are in
`tests/test_ik_execution.py`.
