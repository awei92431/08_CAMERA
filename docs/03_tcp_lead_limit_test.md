# TCP lead-limit test

This experiment changes exactly one controller item: after the existing action
scaling, target smoothing and workspace/table target processing, and before
DLS IK, the Cartesian target is limited to a configurable radius around the
measured TCP. `max_tcp_lead=None` preserves the old baseline. No observation,
action, reward, success, FSM, gripper, IK parameter, actuator parameter, PPO
checkpoint or weld setting was changed. Arm qpos is never assigned directly.

Protocol: frozen `best_full_flow_v22.zip`, deterministic policy, seeds 0--9,
10 episodes per configuration.

## Comparison

| config | full | Reach | Grasp | Lift | entered Place | Place success | mean steps | dq clips | joint clips | lead clips | raw lead max | final lead max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0/10 | 10/10 | 8/10 | 8/10 | 8/10 | 0/10 | 747.5 | 1106 | 0 | 0 | 307.62 mm | 307.62 mm |
| 20 mm | 1/10 | 2/10 | 1/10 | 1/10 | 1/10 | 1/10 | 875.6 | 2519 | 0 | 8663 | 51.32 mm | 20.00 mm |
| 25 mm | 1/10 | 7/10 | 7/10 | 7/10 | 7/10 | 1/10 | 857.9 | 2867 | 0 | 7029 | 50.38 mm | 25.00 mm |
| 30 mm | 1/10 | 10/10 | 10/10 | 10/10 | 10/10 | 1/10 | 849.3 | 3342 | 0 | 5001 | 53.75 mm | 30.00 mm |

`raw lead max` is sampled after normal target formation but before the new
limit. `final lead max` is the target passed to IK. The earlier 283.96 mm
number was measured after the ten MuJoCo actuator substeps; baseline raw lead
is 307.62 mm because it is measured immediately before those substeps.

## Per-stage TCP error

Values below are mean / P95 / max.

| config | Reach | Grasp | Lift | Place |
|---|---|---|---|---|
| baseline | 151.89 / 273.68 / 283.96 mm | 21.00 / 36.92 / 44.53 mm | 40.36 / 58.42 / 62.06 mm | 14.22 / 24.17 / 41.04 mm |
| 20 mm | 19.22 / 20.08 / 20.23 mm | 17.82 / 19.79 / 19.88 mm | 18.76 / 19.11 / 19.13 mm | 12.55 / 15.57 / 19.14 mm |
| 25 mm | 23.86 / 25.04 / 25.06 mm | 22.11 / 24.72 / 24.83 mm | 23.16 / 24.11 / 24.57 mm | 13.79 / 17.24 / 24.17 mm |
| 30 mm | 28.26 / 29.68 / 29.77 mm | 24.91 / 29.57 / 29.87 mm | 27.43 / 28.98 / 29.41 mm | 14.08 / 17.32 / 29.04 mm |

## Interpretation

All three limits solve the 284 mm runaway lead. The 30 mm configuration caps
post-actuation Reach error below 29.8 mm while retaining Reach, Grasp, Lift and
Place entry on all ten seeds. It is therefore the best stability/response
trade-off tested.

20 mm is too restrictive for the frozen policy/controller timing: it clips on
nearly every step, leaves most episodes in Reach, and reduces Grasp/Lift and
Place entry to 1/10 despite one successful seed. 25 mm is usable but still
loses three Reach handoffs. These degradations explain why the smallest error
number is not the best controller setting.

30 mm does not degrade Grasp or Lift; both improve from baseline 8/10 to 10/10.
It also changes Place from 8/10 entries and 0 successes to 10/10 entries and
1 success. Seed 8 succeeds for all three limits, with completion at 665, 479
and 393 steps for 20, 25 and 30 mm respectively, showing the response-speed
advantage of 30 mm.

The total dq-clip count increases with the limits. This is not a joint-limit
problem (all groups have zero joint clips); the bounded Cartesian target is
continually re-requested while DLS uses its unchanged per-iteration dq cap.
This round intentionally does not tune that behavior.

## Recommendation

Use 30 mm as the lead-bound candidate for the next isolated experiment. Do not
fine-tune PPO yet: 9/10 Place episodes still fail even though all enter Place,
so controller precision remains the dominant unresolved issue. The next
single-variable investigation should address the previously measured 8--10 mm
fixed-target steady-state/off-axis bias (position versus approach-axis/posture
objective and actuator equilibrium), while keeping this 30 mm bound fixed.
Only after Place becomes repeatable under deterministic seeds should a short
PPO fine-tune be considered.
