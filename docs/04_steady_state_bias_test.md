# Fixed-target steady-state bias decomposition

All tests use `max_tcp_lead=30 mm`. No PPO training was performed. Observation,
action, reward, success, FSM, checkpoint, action scale, gripper, actuator,
damping, iteration count and dq clip were unchanged.

Three isolated configurations were tested on the same 18 targets (+/- X/Y/Z,
10/30/50 mm), held for 4 s each:

- A current: approach-axis weight 0.35, posture weight 0.02.
- B no posture: approach-axis weight 0.35, posture weight 0.
- C half axis: approach-axis weight 0.175, posture weight 0.02.

## Aggregate results

| config | steady 3D mean / P95 / max | signed mean XYZ error | IK solve error | actuator tracking error | approach-axis error | `||q_target-qpos||` | dq-clip rate | <=1/3/5 mm |
|---|---|---|---:|---:|---:|---:|---:|---|
| A current | 8.856 / 9.891 / 10.667 mm | -0.000 / +1.030 / +8.777 mm | 0.615 mm | 9.020 mm | 0.02078 | 0.01300 rad | 1.833% | 0/0/0 |
| B no posture | 8.998 / 10.031 / 10.625 mm | +0.001 / +1.082 / +8.922 mm | 0.089 mm | 9.021 mm | 0.02113 | 0.01300 rad | 1.963% | 0/0/0 |
| C half axis | 8.856 / 9.891 / 10.667 mm | -0.000 / +1.030 / +8.777 mm | 0.615 mm | 9.020 mm | 0.02079 | 0.01300 rad | 1.807% | 0/0/0 |

All 18 steady targets in every configuration met the existing IK convergence
condition. At steady state the solver generally needed only 2--3 iterations.

## Error decomposition

For A, `target_tcp - FK(q_target)` averages only 0.615 mm, while
`FK(q_target) - actual_tcp` averages 9.020 mm. The observed TCP bias is thus
actuator tracking/equilibrium error. Its signed average is dominated by Z:
the actual TCP sits about 8.78 mm below target, with about 1.03 mm Y error and
negligible X bias. A roughly 0.013 rad joint-space residual is sufficient to
produce this Cartesian offset in the current configuration.

The two vector components do not add by their scalar norms, but their signed
vectors reconstruct `target_tcp - actual_tcp` exactly at every sample.

## Posture and approach-axis tests

Removing posture regularization makes the mathematical IK target more accurate
(0.615 mm to 0.089 mm), proving that posture contributes roughly half a
millimetre of IK bias. However, actual steady error becomes slightly worse
(8.856 mm to 8.998 mm) because the unchanged actuator error remains about
9.02 mm. Posture is therefore measurable inside IK but is not the source of
the 8--10 mm real bias, and disabling it is not beneficial end-to-end.

Halving approach-axis weight changes actual error by only about 0.000006 mm
and does not materially change axis error, IK error, q gap or convergence.
There is no evidence that the current axis weight is too strong in these local
fixed-target tests.

## Conclusions and next step

1. The 8--10 mm bias is primarily actuator tracking/equilibrium error, not IK
   solve error.
2. Posture regularization adds a small IK bias but does not explain the actual
   TCP bias; turning it off slightly worsens end-to-end accuracy.
3. Approach-axis weight 0.35 is not over-constraining position in this test.
4. A is numerically the best actual-position configuration, though A and C are
   effectively identical.
5. A is also the best position/attitude compromise because it preserves the
   existing attitude objective and posture behavior without an accuracy cost.
6. The next isolated work should investigate actuator equilibrium/tracking
   (joint-by-joint q residual, gravity/load, position-servo gain/force limits),
   not further IK weight tuning. No actuator parameter was changed here.
7. Re-running seeds 0--9 with B or C is not worthwhile: neither improves
   actual TCP accuracy over A. The existing 30 mm full-task evaluation already
   represents the recommended A configuration. Diagnose actuator tracking
   first, then repeat the full-task evaluation after a validated isolated fix.

PPO fine-tuning remains premature because the dominant bias is below the
policy, in the actuator execution layer.
