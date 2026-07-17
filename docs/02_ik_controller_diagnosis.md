# DLS IK + joint actuator control diagnosis

No PPO training or controller tuning was performed. Reward, success criteria,
FSM, checkpoint and all control outputs were kept unchanged. Diagnostic fields
were added only to expose the existing computation.

## 1. Stage TCP tracking error (seeds 0--9)

| Stage | mean | median | P95 | max | samples |
|---|---:|---:|---:|---:|---:|
| Reach | 151.89 mm | 158.68 mm | 273.68 mm | 283.96 mm | 612 |
| Grasp | 21.00 mm | 20.27 mm | 36.92 mm | 44.53 mm | 476 |
| Lift | 40.36 mm | 42.44 mm | 58.42 mm | 62.06 mm | 135 |
| Place | 14.22 mm | 13.00 mm | 24.17 mm | 41.04 mm | 6252 |

The earlier episode-wide 284 mm maximum is a Reach transient, not a Place
error. Every episode maximum occurred in Reach at steps 29--40.

## 2. Maximum-error condition

The global maximum was seed 3, step 33, Reach. Actual TCP was
`[0.302710, 0.159214, 0.389070] m`; target was
`[0.441244, -0.088622, 0.384971] m`. Error was
`[+138.53, -247.84, -4.10] mm`, norm 283.96 mm. This is a two-axis XY error;
Z was normal, so it is not a simultaneous three-axis divergence.

At all ten per-episode maxima: the IK used all 15 iterations, did not meet its
convergence predicate, and triggered the per-iteration `dq` clip. No joint
limit clip and no robot/table contact occurred. The reported four contacts are
normal gripper/internal model contacts, not table collisions. Workspace safety
clipping was active, primarily because Reach constrains target Z; it did not
create the large XY separation. At the global maximum, `||q_target-qpos||` was
0.832 rad and the returned total joint change contained two approximately
0.60 rad components. The position actuator therefore received a target far
ahead of current qpos.

Stage transitions do not introduce a large Cartesian jump. Across the ten
episodes, Reach→Grasp and Grasp→Lift target jumps were at most about 2.4 mm and
3.6 mm; Lift→Place jumps were below 0.015 mm.

## 3. Independent step responses

Eighteen fixed-target tests (+/- X/Y/Z, 10/30/50 mm) used no PPO. Each target
was held for 4 s. Twelve 30/50 mm cases triggered internal dq clipping; none
hit joint limits, oscillated, or produced an IK numerical failure.

- Rise time for responses that crossed 90% was about 0.14--0.58 s.
- Mean steady-state 3D error across all cases was 8.86 mm; worst was 10.67 mm.
- 10 mm cases did not dq-clip; 30/50 mm cases did.
- No sustained oscillation was observed. Overshoot was negligible in X and
  positive Y, below 1 mm in negative Y, but negative-Z tests passed the target
  by about 8 mm.
- Only 7/18 met the strict 1 mm commanded-axis endpoint criterion. The common
  8--10 mm 3D residual is largely an off-axis/Z bias.

The consistent off-axis residual indicates that the position objective is
competing with the approach-axis constraint and posture term (and actuator
equilibrium), even when a fixed Cartesian target is not accumulating.

## 4. Target accumulation

The current code does contain target accumulation. In Reach, Grasp and Lift,
`target_base_pos` is the previous `tcp_target_pos`, then the scaled action is
added and smoothed. It does not first rebase on the measured TCP. When the arm
lags, subsequent policy actions keep moving the target. This produced the
large Reach lead: policy actions of roughly 50 mm per control step accumulated
while qpos could not follow the successive q targets.

Place is different: before release, its target base is the current measured
TCP, so normal Place motion does not accumulate an unlimited backlog. After
`release_has_opened`, Place again holds the previous target. The measured Place
error decreases over long episodes rather than growing, and its mean is only
14.2 mm. Therefore the 284 mm event is definitely cumulative Reach lead, while
the Place timeout is not explained by an ever-growing Place TCP target.

## 5. IK, q_target and actuator bottleneck

`_solve_ik()` is not one Jacobian update. It performs up to 15 Jacobian DLS
iterations (`damping=1e-4`), and returns a complete six-joint `q_target`, not a
joint increment. Each internal iteration clips every joint's dq to +/-0.04 rad,
so one call can still move a joint target by about 0.60 rad. The actuator gets
that complete q_target for ten MuJoCo substeps. At large Reach errors the next
policy command arrives while actual qpos is still far behind; thus target
accumulation and the large per-call q_target jump reinforce each other.

The primary source of the 284 mm maximum is target-generation backlog. The
second bottleneck is the IK-to-actuator rate mismatch (up to 0.60 rad target
change per policy step versus finite actuator response). Joint limits and
collisions are not responsible. The fixed-target 8--10 mm bias additionally
shows a smaller IK objective/actuator equilibrium issue involving axis and
posture constraints.

## 6. Why Place fails

Eight episodes enter Place and all eight time out; two fail earlier in Grasp.
Place has the lowest mean error after Grasp and no accumulating lead before
release. Its failure is therefore a precision/FSM timing retention problem,
not the source of the global tracking spike: the old policy/FSM expects the
near-kinematic mocap response and exact release/descent geometry, while the
actuator layer retains roughly 13 mm median tracking error and 8--10 mm
fixed-target bias. Those errors are small compared with the Reach spike but
large enough to miss Place release/success geometry repeatedly.

The corrected metrics are `entered_place=8/10` and `place_success=0/10`.

## 7. First priority and minimal proposed repair

The first priority is target logic, not PPO or actuator gain. Add a Cartesian
lead bound before IK, for example cap
`tcp_target_pos - actual_tcp` to a small norm (initial diagnostic candidate:
20--30 mm), or pause further accumulation whenever tracking error exceeds that
threshold. Preserve all workspace and stage safety clipping after this bound.
This is the smallest change that directly removes the demonstrated 284 mm
backlog without altering observations, actions, reward, success or FSM.

After that change, rerun the same diagnostics before considering damping,
axis/posture weights or actuator gains. Fine-tuning PPO now would train the
policy to compensate for an avoidable execution-layer backlog and would likely
overfit transient actuator lag. Controller behavior should first be bounded
and repeatable; only then is a short policy fine-tune justified.
