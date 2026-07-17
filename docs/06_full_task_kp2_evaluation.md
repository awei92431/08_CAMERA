# Full-task zero-shot evaluation: 30 mm lead + 2x arm kp

This is evaluation only. The frozen checkpoint, deterministic policy, seeds
0--9, observation/action, reward, success, FSM, gripper, IK, dq clip,
forcerange, velocity feedback, action scale, frame skip and reset distribution
are unchanged. TCP weld remains absent and arm qpos is never assigned directly.
Only the six arm position gain and matching position-bias coefficient are 2x.

## Outcome versus 30 mm lead + 1x kp

| metric | 1x kp | 2x kp |
|---|---:|---:|
| full / Place success | 1/10 | 1/10 |
| Reach | 10/10 | 10/10 |
| Grasp | 10/10 | 10/10 |
| Lift | 10/10 | 10/10 |
| entered Place | 10/10 | 10/10 |
| mean episode steps | 849.3 | 832.8 |
| successful seed 8 steps | 393 | 228 |

The complete-task success rate is unchanged, but the successful trajectory is
165 steps faster.

## Stage TCP tracking error

| stage | 1x mean / P95 / max | 2x mean / P95 / max |
|---|---|---|
| Reach | 28.26 / 29.68 / 29.77 mm | 26.36 / 28.71 / 29.13 mm |
| Grasp | 24.91 / 29.57 / 29.87 mm | 22.46 / 28.60 / 29.13 mm |
| Lift | 27.43 / 28.98 / 29.41 mm | 24.76 / 27.60 / 28.59 mm |
| Place | 14.08 / 17.32 / 29.04 mm | 11.38 / 13.76 / 27.80 mm |

Place mean error improves by about 19%, and P95 by about 21%. This confirms the
fixed-target kp improvement carries into policy execution, but it does not by
itself solve Place.

## Place and failure analysis

- 8/9 failures: never reached the XY release region.
- seed 9: reached XY and low-height predicates at some point, but never formed
  the full `place_open_ready` condition and never opened.
- seed 8: reached the region, descended, triggered release, opened and
  succeeded at 19.59 mm final goal XY and 9.47 mm height error.
- Only seed 8 opened/released. The episode terminates immediately on success,
  so long post-release stability cannot be observed; the success predicate did
  verify the required low XY speed at termination. No drop below the table,
  fling or >15 mm post-release bounce was detected.

Across all seeds the mean final goal XY is 288.1 mm and mean final height error
is 35.5 mm, dominated by the eight policies that keep moving away from the
release region. Thus the dominant remaining problem is Place XY command/FSM
behavior, not millimetre-scale actuator sag.

Grasp remains centered: the worst per-episode minimum grasp XY error is
3.90 mm, and maximum grasp-stage horizontal object drift is 7.93 mm. There is
no evidence that 2x kp introduced systematic off-center grasping.

## Dynamics and safety

- maximum arm joint speed: 0.836 rad/s;
- maximum actuator force: 32.60 Nm;
- actuator saturation: 0%;
- dq-clipped IK iterations: 1,941 / 124,080 = 1.564%;
- joint-limit clips: 0;
- lead-limit triggers: 2,404;
- robot/gripper table-contact steps: 0;
- detected drops below table, flings, or significant rebounds: 0.

The saved videos show no new persistent oscillation or abnormal impact. These
metrics do not indicate that 2x kp caused collision, force saturation, jitter
or grasp damage.

## Recommendation

2x kp is suitable as the new IK execution-layer baseline: it improves every
stage's TCP tracking, preserves 10/10 Reach/Grasp/Lift/Place entry, halves the
isolated static bias, and introduces no measured safety regression. It does
not improve success beyond 1/10, so kp was not the main Place bottleneck.

Next, diagnose Place control/FSM using the eight `未到达释放区域` trajectories:
object/goal XY, servo correction, policy residual and target motion over time.
A 50k PPO fine-tune is premature because failures are concentrated in a
deterministic Place execution mode with an observable control/FSM mismatch.
Resolve or bound that behavior first; fine-tune only after the fixed Place
controller reliably reaches its release region.
