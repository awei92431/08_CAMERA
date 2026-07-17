# Phase 1: unified runtime ObjectEstimate

> **模型版本说明（2026-07-17）：** 本文的100-seed任务数字来自CAD相机
> 最终安装前的历史模型。当前已更新转接件法兰分度、负载、4.2 mm光学
> 内缩和`camera_observe`。ObjectEstimate架构结论仍有效，但当前模型的
> 完整Ground Truth/RGB-D成功率必须重新评估，不能直接引用本文数字。

## Scope

Phase 1 changes only the runtime source of object position. It does not change
PPO weights or dimensions, gains, thresholds, reward, success, contact logic,
grasp-latch mechanics, gripper FSM states or `obs[35:39]`.

The fixed comparison uses seeds 0–99, deterministic PPO,
`best_full_flow_v22.zip`, `max_tcp_lead=0.03`, posture mode `off`, no mocap
weld and the existing XML actuator gains.

## Implemented architecture

`fourc2/object_estimate.py` defines:

- `ObjectEstimate`
  - `position` and optional `orientation_wxyz`;
  - `timestamp`;
  - `valid`;
  - `confidence`;
  - `source`;
  - `estimate_id`.
- `ObjectEstimateAuthority`
  - fixed episode mode: `ground_truth` or `rgbd`;
  - rejects cross-mode publication;
  - rejects invalid, non-finite, stale or future estimates;
  - records consumer, control step, ID, timestamp and source;
  - never substitutes zero or simulator truth in RGB-D mode.
- `TcpObjectTracker`
  - captures the visually initialized TCP–object rigid offset;
  - propagates object position using TCP FK after grasp confirmation.

The environment receives `object_observation_mode` at construction. It cannot
switch or combine modes during an episode.

```text
RGB-D -> world ObjectEstimate ----+
                                   +-> one ObjectEstimateAuthority
TCP FK propagation after latch ---+          |
                                              +-> PPO obs[15:35]
                                              +-> Reach safety clamp
                                              +-> Grasp safety clamp
                                              +-> Place XY servo
                                              +-> Place descent geometry
                                              +-> Place release geometry
```

Ground-truth mode explicitly publishes a `source=ground_truth_simulation`
estimate. RGB-D mode rejects that source. Ground truth remains separately
available to reward, success and evaluator code.

## Modified runtime consumers

| Consumer | Before | After |
|---|---|---|
| PPO `obs[15:35]` | environment truth or external shadow adapter | authority estimate and coherent derived fields |
| Place XY servo | `site_xpos[object_site_id]` | `ObjectEstimate.position` |
| Place descent entry | true object-to-goal XY | authority object-to-goal XY |
| Place low/release geometry | true object Z/XY | authority position and authority initial Z |
| Reach Z safety | true object Z | authority object Z |
| Grasp XY/Z safety | true object XYZ | authority object XYZ |

The goal site remains a known/fixed task target. TCP and vertical alignment
remain robot-FK quantities.

## Per-step source consistency

The formal test records every use by these consumers:

- `ppo_observation`;
- `reach_safety`;
- `grasp_safety`;
- `place_servo`;
- `place_descent`;
- `place_release`.

For each policy control step, all used consumers must have one identical pair
`(estimate_id, timestamp)`. Results:

| Mode | Episodes | Checked episode consistency failures | Consistent episodes |
|---|---:|---:|---:|
| Ground Truth | 100 | 0 | 100/100 |
| RGB-D | 100 | 0 | 100/100 |

Aggregate consumption counts:

| Mode | PPO | Reach safety | Grasp safety | Place servo | Descent | Release |
|---|---:|---:|---:|---:|---:|---:|
| Ground Truth | 45,801 | 58,374 | 6,404 | 10,488 | 22,759 | 3,641 |
| RGB-D | 45,825 | 58,378 | 6,308 | 10,528 | 22,800 | 3,890 |

RGB-D consumer reads by source were 97,029 `rgbd_visual_hold` and 50,700
`tcp_fk_propagated`; no `ground_truth_simulation` source entered RGB-D mode.

## 100-seed task results

| Metric | Ground Truth | RGB-D |
|---|---:|---:|
| Initial RGB-D localization | 100/100 | 100/100 |
| Reach | 100/100 | 100/100 |
| Grasp | 99/100 | 100/100 |
| Lift | 99/100 | 100/100 |
| Entered Place | 99/100 | 100/100 |
| Place | 98/100 | 99/100 |
| Full success | **98/100** | **99/100** |

Initial RGB-D 3-D error was unchanged: mean 0.775 mm, median 0.794 mm,
P95 1.364 mm and maximum 2.135 mm.

Paired outcomes were identical for 99/100 seeds and randomized object/goal
initial states matched for 100/100.

### Failure interpretation

- Seed 31 failed in both modes after Reach, Grasp, Lift and Place entry. Both
  opened the gripper but did not satisfy final Place success. This remains the
  known Place/release boundary failure and is not a source-consistency failure.
- Seed 79 failed only in Ground Truth mode during Grasp. Its object horizontal
  drift reached 27.0 mm and the existing stage failure terminated the episode.
  RGB-D succeeded. Source consistency was zero-failure in both runs. The visual
  error was 0.608 mm; that small input difference led to a different boundary
  trajectory rather than a hidden truth fallback.

No gain, threshold or success condition was adjusted to change either outcome.

## Invalid, dropout and stale behavior

`scripts/test_phase1_object_estimate.py` verifies:

| Test | Result |
|---|---|
| no estimate after RGB-D-mode reset | `ObjectEstimateUnavailable(reason="missing")` |
| explicit localization dropout | `reason="rgbd_dropout"` |
| timestamp older than configured max age | `reason="stale"` |
| ground-truth source published into RGB-D mode | rejected with `ValueError` |
| zero/non-finite fallback | never produced |
| physics during failure | MuJoCo time and qpos unchanged |

The caller therefore stops before `model.predict` or `env.step` and records a
diagnostic failure. There is no silent hold unless the caller explicitly
publishes a valid `rgbd_visual_hold` estimate with a current timestamp.

## Post-grasp propagation test

The rigid propagation unit test captures a one-metre TCP-local offset, applies
a 90-degree TCP rotation and two-metre translation, and obtains the expected
world point with 0 m numerical error. It has no object-truth input.

The 100 RGB-D episodes also recorded 50,700 control-consumer reads from
`tcp_fk_propagated`, proving the propagated estimate was used by PPO and the
modified control consumers, rather than only calculated for evaluation.

Phase 1 deliberately keeps `env.is_grasp_latched` as the propagation handoff
trigger. Replacing that trigger is Phase 2.

## Direct truth-read audit after Phase 1

In the modified target consumers, object pose reaches control only through
`_control_object_estimate()` / `_control_object_position()`. The remaining
`object_site` reads inside `_apply_tcp_action` are `diag_object_before/after`
logging only and do not change action.

Truth is still read elsewhere for the explicitly deferred systems:

1. `_update_grasp_phase`, `_pregrasp_handoff_ready`, `_grasp_xy_aligned` and
   close predicates use true object-derived geometry;
2. left/right MuJoCo contact and penetration feed grasp confirmation;
3. `_update_grasp_latch` reads/writes object qpos and simulates attachment;
4. Reach/Grasp/Lift/Place stage success and stage failure use true pose,
   velocity, contact and boundary values;
5. reward and evaluator metrics use truth;
6. PPO `obs[35:39]` retains contact/alignment/FSM state as requested.

Using the previous audit's functional-group counting, Phase 1 removed six
runtime privileged groups (old RT-03 through RT-08) from the specified motion
consumers. **Thirteen runtime privileged groups remain**, concentrated in FSM,
contact/latch, stage transition/success and safety termination. Reward-only and
evaluator-only reads are not included in thirteen.

## Files and outputs

- `fourc2/object_estimate.py`
- `fourc2/envs/allstage.py`
- `scripts/eval_full_visual_closed_loop.py`
- `scripts/test_phase1_object_estimate.py`
- `outputs/full_visual_closed_loop/ground_truth_episodes.csv`
- `outputs/full_visual_closed_loop/rgbd_episodes.csv`
- `outputs/full_visual_closed_loop/seed_comparison.csv`
- `outputs/full_visual_closed_loop/summary.json`
- `outputs/phase1_unified_object_estimate/fail_closed_and_propagation_tests.json`

## Phase 1 conclusion

There is now one authoritative object-position source for PPO, Place servo,
Place descent/release geometry and Reach/Grasp target safety. RGB-D mode is
fail-closed and cannot silently access simulator object truth.

The system is not yet truth-free: contact/grasp FSM, simulated latch, stage
success/failure and `obs[35:39]` remain privileged and are the Phase 2 scope.
