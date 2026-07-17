# 40 mm ArUco / 30 mm cube XY error evaluation

## Configuration

- Seeds: 0-99, deterministic policy.
- ArUco: DICT_4X4_50 ID 0, rendered and solved as 40 mm physical edge.
- Cube: 30 mm edge.
- Errors in this report use X/Y only; Z is excluded.
- Object source: existing RGB-D cube localizer.
- Goal source: stable-window, multi-frame frozen ArUco estimate.
- Final placement error: MuJoCo final cube center XY versus MuJoCo true goal
  center XY. Truth is evaluator-only.
- A failed ArUco capture terminates before RGB-D object localization and PPO.

The planar PnP implementation evaluates every IPPE square candidate and
rejects candidates outside the known Base workspace/table-height range. This
removes planar mirror solutions without using the true goal coordinate.

## Results

| Metric | Count | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| ArUco goal XY error | 83 | 1.708 mm | 1.638 mm | 4.330 mm | 5.184 mm |
| RGB-D object XY error | 83 | 0.971 mm | 0.871 mm | 2.189 mm | 2.450 mm |
| Final object to true goal XY | 83 | 8.361 mm | 7.857 mm | 14.976 mm | 18.693 mm |
| Final object to estimated goal XY | 83 | 8.967 mm | 8.552 mm | 15.899 mm | 20.314 mm |

- ArUco capture: 83/100 = 83%.
- RGB-D object detection after valid goal: 83/83 = 100%.
- Full task success after control started: 83/83 = 100%.
- Full success over all requested seeds: 83/100 = 83%.
- Fail closed at ArUco capture: 17.
- Object-localization failures after valid goal: 0.
- Task failures after both visual estimates were available: 0.
- True placement error <= 15 mm: 78/83 = 93.98%; five episodes exceeded
  15 mm.
- Final absolute X error: mean 4.313 mm, P95 10.106 mm, max 14.382 mm.
- Final absolute Y error: mean 6.386 mm, P95 11.358 mm, max 16.825 mm.
- Full 30 mm cube footprint inside the 40 mm goal square requires
  `abs(dx) <= 5 mm` and `abs(dy) <= 5 mm`; only 18/83 = 21.69% met this
  stricter physical criterion.

Failed ArUco seeds:

```text
1, 11, 15, 28, 31, 41, 44, 45, 48, 50, 53, 59, 68, 71, 87, 93, 95
```

## Interpretation

The ArUco and RGB-D XY errors are not the main source of the final 8.36 mm
mean placement residual:

- correlation(goal XY error, final placement XY error): 0.128;
- correlation(object XY error, final placement XY error): 0.027;
- correlation(goal + object XY error, final placement XY error): 0.122.

The object finishes about 9 mm from even the *estimated* goal on average. This
means the dominant residual comes after perception: Place servo convergence,
descent geometry, release timing and post-release object motion. The current
24 mm simulator success threshold labels all 83 controlled episodes as
successful, but that should not be interpreted as sub-centimeter placement
guaranteed on a real 30 mm cube.

For a 30 mm cube, a 15 mm center error reaches the edge of the nominal target
footprint. P95 is essentially 15 mm and max is 18.7 mm, so the present Place
accuracy is marginal if the real requirement is full footprint overlap.

The larger immediate problem is ArUco coverage: reducing the marker from 60 mm
to 40 mm reduced reliable full-marker decoding, and 17% of sampled targets did
not enter control. The next change should first improve the fixed observation
pose/FOV or use a deliberate second observation pose. After coverage is fixed,
Place residuals should be measured and optimized independently; loosening the
success threshold would hide the problem.

## Files and reproduction

- `outputs/aruco4cm_xy_error_100seeds/summary.json`
- `outputs/aruco4cm_xy_error_100seeds/episodes.csv`
- `outputs/aruco4cm_xy_error_100seeds/three_xy_errors_boxplot.png`
- `outputs/aruco4cm_xy_error_100seeds/vision_vs_placement_xy_error.png`

```bash
cd /home/lenovo/mujoco_learning/08_camera
MUJOCO_GL=egl \
/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python \
scripts/eval_xy_error_statistics.py --episodes 100 --seed-offset 0
```
