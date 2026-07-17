# Joint actuator tracking and kp sweep

This test keeps `max_tcp_lead=0.03`, IK configuration A, all PPO/task fields,
force limits, damping, dq clip, frame skip and action scale unchanged. Only the
six arm position-servo kp terms were scaled by 1.0, 1.5 and 2.0. The velocity
feedback coefficient was not changed.

## Baseline joint diagnosis

| joint | kp | force range | mean residual | max residual | mean abs force | max abs force | saturation | mean TCP-Z contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder pan | 2000 | +/-150 | ~0 rad | 0.0000001 rad | ~0 Nm | 0.000005 Nm | 0% | ~0 mm |
| shoulder lift | 2000 | +/-150 | -0.00896 rad | 0.01125 rad | 17.92 Nm | 22.50 Nm | 0% | +4.38 mm |
| elbow | 2000 | +/-150 | -0.00875 rad | 0.00876 rad | 17.50 Nm | 17.52 Nm | 0% | +4.23 mm |
| wrist 1 | 500 | +/-28 | -0.00342 rad | 0.00342 rad | 1.71 Nm | 1.71 Nm | 0% | +0.31 mm |
| wrist 2 | 500 | +/-28 | +0.000003 rad | 0.000003 rad | 0.0014 Nm | 0.0014 Nm | 0% | ~0 mm |
| wrist 3 | 500 | +/-28 | ~0 rad | 0.000013 rad | ~0 Nm | 0.00013 Nm | 0% | ~0 mm |

Shoulder lift and elbow dominate the Z bias, contributing about 4.38 and
4.23 mm respectively; wrist 1 contributes about 0.31 mm. Their steady actuator
forces closely match `qfrc_bias`, and residual magnitude falls approximately
inversely with kp. This is the characteristic static error of a finite-gain
position servo balancing gravity/load, not force saturation.

The largest baseline force is 22.50 Nm versus a +/-150 Nm range. Wrist force is
also far below +/-28 Nm. No actuator force sample reached 99% of its limit.

## kp comparison

| kp scale | steady TCP mean / P95 / max | signed mean XYZ | actuator error | `||qtarget-qpos||` | mean rise | max joint speed | max force | saturation | dq-clip rate | oscillating tests | robot-table contact |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 8.856 / 9.891 / 10.667 mm | -0.000 / +1.030 / +8.777 mm | 9.020 mm | 0.01300 rad | 0.431 s | 0.589 rad/s | 22.50 Nm | 0% | 1.833% | 0/18 | 0/18 |
| 1.5x | 5.907 / 6.844 / 7.129 mm | -0.001 / +0.658 / +5.846 mm | 6.004 mm | 0.00864 rad | 0.313 s | 0.844 rad/s | 22.36 Nm | 0% | 1.228% | 0/18 | 0/18 |
| 2.0x | 4.438 / 5.342 / 5.377 mm | -0.001 / +0.481 / +4.381 mm | 4.499 mm | 0.00647 rad | 0.236 s | 1.102 rad/s | 22.29 Nm | 0% | 0.974% | 0/18 | 0/18 |

At 2.0x, 16/18 tests are within 5 mm; none are within 3 mm. At 1.5x only
1/18 is within 5 mm. Increasing kp reduces the static residual, TCP error,
rise time and apparent Z overshoot monotonically. Peak joint speed rises from
0.59 to 1.10 rad/s, but no sustained oscillation, robot-table contact, force
saturation or abnormal contact-count increase was observed.

The per-joint counterfactual FK Z contributions above are local diagnostic
estimates. Joint effects are coupled and must not be interpreted as exactly
additive.

The model always contains normal object-table and gripper/internal contacts.
The collision metric explicitly excludes object-table support contact; all
reported robot/table-contact counts are zero.

## Conclusions

1. Shoulder lift and elbow fail to reach q_target by about 9 mrad each at the
   current kp; wrist 1 has a smaller 3.4 mrad residual. Other joints track.
2. Actuators are not saturated. Maximum arm force is only 22.5 Nm against the
   relevant +/-150 Nm limit.
3. The bottleneck is finite kp under gravity/load, not forcerange. The equality
   between steady servo force and qfrc_bias is direct evidence of static
   position-servo sag.
4. 1.5x reduces mean TCP bias from 8.86 to 5.91 mm; 2.0x reduces it to 4.44 mm.
5. 2.0x is the best tested accuracy/stability trade-off. It is faster and more
   accurate without observed oscillation, impact, collision or saturation,
   though its higher peak joint speed should be monitored in full tasks.
6. 2.0x reaches the requested 3--5 mm band on average and in 16/18 cases at
   the 5 mm threshold, but does not yet achieve 3 mm.
7. A deterministic seeds 0--9 zero-shot full-task evaluation with 2.0x kp is
   justified as the next isolated validation, keeping 30 mm lead and all other
   settings fixed. It was intentionally not run in this round.
8. PPO fine-tuning is still unnecessary: a single execution-layer parameter
   reduced the dominant error by half, showing that policy learning is not the
   appropriate remedy yet.

No forcerange change or gravity compensation was applied.
