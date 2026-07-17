# Place XY command and release-condition diagnosis

Evaluation only: frozen checkpoint, deterministic seeds 0--9, 30 mm lead,
2x arm position kp, unchanged Z/FSM/gripper/reward/success/IK/safety. The four
diagnostic modes change only Place XY composition:

- A: policy residual + existing servo;
- B: existing servo only;
- C: policy XY only;
- D: oracle `0.30*(goal-object)`, clipped to the original 12 mm step limit.

## A/B/C/D results

| mode | full | entered | reached XY | reached low | open ready | opened | mean min XY | mean final XY | Place TCP mean/P95/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A current | 1/10 | 10/10 | 2/10 | 3/10 | 1/10 | 1/10 | 81.8 mm | 288.1 mm | 11.34/13.16/17.26 mm |
| B servo only | 1/10 | 10/10 | 2/10 | 3/10 | 2/10 | 2/10 | 81.0 mm | 211.2 mm | 13.65/14.58/21.67 mm |
| C policy only | 0/10 | 10/10 | 1/10 | 1/10 | 0/10 | 0/10 | 89.0 mm | 438.0 mm | 5.97/6.93/15.49 mm |
| D oracle | 1/10 | 10/10 | 2/10 | 2/10 | 1/10 | 1/10 | 82.0 mm | 238.9 mm | 12.79/13.74/15.49 mm |

D succeeds only on seed 8 and does not reliably reach the release region. B
lets seed 9 reach/open/release, but the object moves back outside the 24 mm
success radius before terminal success and the episode times out.

## Direction decomposition

| mode | policy toward | servo toward | final command toward | policy/servo opposed | actual TCP toward | actual TCP mean cosine | object toward | distance-increase steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.3% | 100% | 100% | 94.4% | 1.4% | -0.791 | 1.4% | 98.3% |
| B | 4.4%* | 100% | 100% | 84.6%* | 11.0% | -0.615 | 6.2% | 84.7% |
| C | 81.5% | 100%* | 81.5% | 18.2%* | 5.2% | -0.535 | 0.9% | 34.2% |
| D | 0.5%* | 100% | 100% | 94.2%* | 6.1% | -0.696 | 6.1% | 93.7% |

`*` marks a component computed for diagnostics but disabled from the final
command in that mode.

The existing policy and servo strongly conflict in A, and policy-only C is the
worst success configuration. However, conflict is not the decisive failure:
the servo is correctly signed and large enough that A's final target command
is toward the goal on every nonzero step. B and D also command toward the goal
on every step. Nevertheless, measured TCP and latched object movement are
predominantly opposite. Therefore there is no world-frame/sign error in the
Place servo formula `goal_xy-object_xy`.

The low 11 mm `tcp_target_error` is misleading here because each Place target
is rebased close to the current TCP. It measures the short target lead, not
whether the TCP is progressing toward the global goal. The data show the
downstream IK/joint target can move the actual TCP away while the newly rebased
target remains only about one servo step away. Given the solver's home-posture
regularization, a Place-specific IK realization/posture pull is the leading
hypothesis and should be directly logged next (`FK(q_target)` displacement
cosine versus commanded displacement). This test does not change it.

## Exact release chain and seed 9

The actual code contains no hidden TCP-error, contact, speed, or resettable
confirm counter in `place_open_ready`. It is exactly:

1. `place_xy_ready`: object-goal XY < 24 mm;
2. `place_low_ready`: object lift <= 16 mm;
3. `release_steps >= 8`.

The opening predicate uses the same XY, height and delay conditions. Actual
`place_opened` additionally requires gripper command <0.20 and latch released.
Final `place_success` then requires open-ready, opened, object XY speed <0.04,
zero robot-table contact and no table-boundary penalty.

For A seed 9, delay is already satisfied from Place step 8. XY is valid for 13
consecutive steps, 11--23, with minimum 10.485 mm at step 15. Height becomes
valid at step 24 and stays valid. At that exact step XY is 24.002 mm, just
2 micrometres outside the 24 mm threshold. Thus XY and height never overlap;
`place_open_ready` is never true. No counter is being cleared—there is no such
counter in this predicate.

## Failure classification and recommendation

A: 8 `未到达释放区域`, 1 `未触发释放条件`, 1 success.
B: 8 `未到达释放区域`, 1 post-release timeout, 1 success.
C: 9 `未到达释放区域`, 1 `未触发释放条件`, 0 success.
D: 8 `未到达释放区域`, 1 `未触发释放条件`, 1 success.

No mode introduces robot-table contact, detected drop or fling. Oracle D cannot
stably solve Place, so PPO is not the next remedy. The formal servo sign and
release FSM should not be changed yet. First diagnose and correct Place command
realization after target generation: log commanded XY versus `FK(q_target)`
and the posture contribution at Place states, then isolate Place posture weight
or use an absolute/goal-progress-aware Cartesian target test. Only after an
oracle command actually moves the TCP/object toward the goal should policy
fine-tuning be considered.
