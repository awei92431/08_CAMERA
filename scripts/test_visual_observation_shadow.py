"""Stage 6A: compare RGB-D shadow observations with truth observations."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fourc2  # noqa: E402,F401
from fourc2.camera_geometry import (  # noqa: E402
    base_from_color_optical_transform,
    base_world_transform,
    camera_world_pose_optical,
    intrinsics_from_fovy,
    invert_transform,
    relative_optical_transform,
    transform_point,
    validate_rigid_transform,
)
from fourc2.rgbd_cube_localizer import (  # noqa: E402
    load_localization_config,
    localize_cube_rgbd,
)
from fourc2.visual_observation_adapter import (  # noqa: E402
    FIELD_SLICES,
    VisualObservationAdapter,
    internal_consistency_residual,
)


ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
CHECKPOINT = ROOT / "checkpoints" / "best_full_flow_v22.zip"
CONFIG_PATH = ROOT / "configs" / "hsv_cube_localization.json"
OUTPUT_DIR = ROOT / "outputs" / "visual_observation_shadow"
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"
BASE_BODY_NAME = "base"
WIDTH, HEIGHT = 640, 360
SAMPLES = 20
COMPARE_FIELDS = [
    "object_position", "pregrasp_position", "grasp_position",
    "pinch_to_pregrasp", "pinch_to_grasp", "object_to_goal", "object_lift",
]


def set_object_position_test_only(env, position):
    """Test fixture only; never called by detector or adapter."""
    env.data.qpos[env.object_qpos_slice] = np.array(
        [*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    env.data.qvel[env.object_qvel_slice] = 0.0
    env.object_initial_position = np.asarray(position, dtype=np.float64).copy()
    env.object_initial_z = float(position[2])
    mujoco.mj_forward(env.model, env.data)


def render_rgb_depth(renderer, data, color_id, depth_id):
    renderer.update_scene(data, camera=color_id)
    color = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=depth_id)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return color, depth


def visual_estimate_base(env, renderer, color_id, depth_id, base_id,
                         color_k, depth_k, config):
    """Vision pipeline: deliberately has no object site/truth argument."""
    color_pose = camera_world_pose_optical(env.data, color_id)
    depth_pose = camera_world_pose_optical(env.data, depth_id)
    color_from_depth = relative_optical_transform(color_pose, depth_pose)
    support_up_color = color_pose[1].T @ np.array([0.0, 0.0, 1.0])
    color, depth = render_rgb_depth(
        renderer, env.data, color_id, depth_id
    )
    localization = localize_cube_rgbd(
        color, depth, depth_k, color_k, color_from_depth,
        support_up_color, config,
    )
    if not localization.valid:
        return None, localization.failure_reason
    t_base_color = base_from_color_optical_transform(
        env.data, color_id, base_id
    )
    estimate_base = transform_point(
        t_base_color, localization.estimated_object_center_color
    )
    return estimate_base, None


def truth_evaluator_test_only(env, shadow_observation, truth_observation):
    """Only this evaluator reads object-site truth."""
    truth_world = env.data.site_xpos[env.object_site_id].copy()
    if not np.allclose(
        truth_world, truth_observation[FIELD_SLICES["object_position"]],
        atol=1e-7,
    ):
        raise AssertionError("truth evaluator/environment observation mismatch")
    errors = {}
    for name in COMPARE_FIELDS:
        delta = (
            shadow_observation[FIELD_SLICES[name]]
            - truth_observation[FIELD_SLICES[name]]
        )
        errors[name] = np.asarray(delta, dtype=np.float64)
    return truth_world, errors


def summarize(records):
    valid = [record for record in records if record["valid"]]
    summary = {
        "samples": len(records),
        "visual_localization_success_rate": len(valid) / len(records),
        "failure_reason_counts": dict(Counter(
            record["failure_reason"] for record in records
            if record["failure_reason"] is not None
        )),
    }
    for name in COMPARE_FIELDS:
        values = np.asarray([
            record["field_error_vectors_m"][name] for record in valid
        ], dtype=np.float64)
        norms = np.linalg.norm(values, axis=1) if len(values) else np.array([])
        summary[name] = {
            "xyz_mae_m": None if not len(values) else np.mean(
                np.abs(values), axis=0
            ).tolist(),
            "norm_mean_m": None if not len(norms) else float(np.mean(norms)),
            "norm_median_m": None if not len(norms) else float(np.median(norms)),
            "norm_max_m": None if not len(norms) else float(np.max(norms)),
        }
    summary["full_39d_max_abs_difference"] = (
        None if not valid else float(max(
            record["full_39d_max_abs_difference"] for record in valid
        ))
    )
    summary["maximum_internal_consistency_residual"] = (
        None if not valid else float(max(
            record["internal_consistency_residual"] for record in valid
        ))
    )
    summary["unchanged_obs_35_39_max_abs_difference"] = (
        None if not valid else float(max(
            record["unchanged_obs_35_39_max_abs_difference"]
            for record in valid
        ))
    )
    return summary


def main():
    config = load_localization_config(CONFIG_PATH)
    wrapped = gym.make(ENV_ID)
    env = wrapped.unwrapped
    model = PPO.load(CHECKPOINT, device="cpu")
    if model.observation_space.shape != (39,):
        raise RuntimeError("checkpoint does not consume 39-D observations")
    initial_obs, _ = wrapped.reset(seed=0)
    color_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, COLOR_NAME
    )
    depth_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_NAME
    )
    base_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME
    )
    if min(color_id, depth_id, base_id) < 0:
        raise RuntimeError("camera/base body missing")
    color_k = intrinsics_from_fovy(
        WIDTH, HEIGHT, env.model.cam_fovy[color_id]
    )
    depth_k = intrinsics_from_fovy(
        WIDTH, HEIGHT, env.model.cam_fovy[depth_id]
    )
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    rng = np.random.default_rng(20260714)
    records = []
    t_base_world_reference = None
    transform_direction_residuals = []
    adapter_contract_audit = None
    try:
        for index in range(SAMPLES):
            truth_observation, _ = wrapped.reset(seed=index)
            camera_position, world_from_color = camera_world_pose_optical(
                env.data, color_id
            )
            center_z = env.table_top_z + env.object_half_size
            distance = (
                (center_z - camera_position[2]) / world_from_color[2, 2]
            )
            point_color = np.array([
                rng.uniform(-0.035, -0.005),
                rng.uniform(-0.010, 0.010),
                distance,
            ])
            point_world = camera_position + world_from_color @ point_color
            point_world[2] = center_z
            set_object_position_test_only(env, point_world)
            truth_observation = env._get_obs().astype(np.float32)

            t_base_world = base_world_transform(env.data, base_id)
            validate_rigid_transform(t_base_world)
            if t_base_world_reference is None:
                t_base_world_reference = t_base_world.copy()
            estimate_base, failure_reason = visual_estimate_base(
                env, renderer, color_id, depth_id, base_id,
                color_k, depth_k, config,
            )
            adapter = VisualObservationAdapter(
                env.pregrasp_height, env.grasp_height_offset
            )
            adapter.reset()
            goal_world = env.data.site_xpos[env.goal_site_id].copy()
            shadow = adapter.build(
                truth_observation, estimate_base, t_base_world, goal_world
            )
            if not shadow.valid:
                records.append({
                    "index": index, "valid": False,
                    "failure_reason": failure_reason or shadow.failure_reason,
                })
                continue

            # Direction check independent of object truth: Base -> World -> Base.
            round_trip_base = transform_point(
                t_base_world,
                transform_point(invert_transform(t_base_world), estimate_base),
            )
            direction_residual = float(np.linalg.norm(
                round_trip_base - estimate_base
            ))
            transform_direction_residuals.append(direction_residual)
            consistency = internal_consistency_residual(
                shadow.observation, env.pregrasp_height,
                env.grasp_height_offset, shadow.estimated_initial_object_z,
                goal_world,
            )
            if consistency >= 1e-7:
                raise AssertionError(f"consistency residual {consistency}")

            # Shadow contract: action is computed exclusively from truth obs.
            truth_action, _ = model.predict(
                truth_observation, deterministic=True
            )
            truth_world, field_errors = truth_evaluator_test_only(
                env, shadow.observation, truth_observation
            )
            records.append({
                "index": index,
                "valid": True,
                "failure_reason": None,
                "estimated_object_center_base": estimate_base.tolist(),
                "estimated_object_center_world": (
                    shadow.estimated_object_world.tolist()
                ),
                "truth_object_center_world_evaluator_only": truth_world.tolist(),
                "field_error_vectors_m": {
                    name: value.tolist() for name, value in field_errors.items()
                },
                "full_39d_max_abs_difference": float(np.max(np.abs(
                    shadow.observation.astype(np.float64)
                    - truth_observation.astype(np.float64)
                ))),
                "internal_consistency_residual": consistency,
                "unchanged_obs_35_39_max_abs_difference": float(np.max(
                    np.abs(
                        shadow.observation[35:39].astype(np.float64)
                        - truth_observation[35:39].astype(np.float64)
                    )
                )),
                "base_world_round_trip_residual_m": direction_residual,
                "policy_input": "original_truth_observation",
                "truth_policy_action": np.asarray(truth_action).tolist(),
            })

        # Pure adapter contract checks, separate from the visual/truth error
        # evaluator: explicit Place gate and explicit failed localization.
        audit_adapter = VisualObservationAdapter(
            env.pregrasp_height, env.grasp_height_offset
        )
        audit_observation = initial_obs.copy()
        audit_observation[FIELD_SLICES["object_to_goal"]] = [1.0, 1.0, 1.0]
        audit_world = np.array([0.40, -0.10, 0.32])
        audit_goal = np.array([0.55, 0.00, 0.40])
        audit_base = transform_point(t_base_world_reference, audit_world)
        place_result = audit_adapter.build(
            audit_observation, audit_base, t_base_world_reference, audit_goal
        )
        place_residual = float(np.max(np.abs(
            place_result.observation[FIELD_SLICES["object_to_goal"]]
            - (audit_goal - audit_world)
        )))
        failed_result = audit_adapter.build(
            audit_observation, None, t_base_world_reference, audit_goal
        )
        if place_residual >= 1e-7:
            raise AssertionError("Place object_to_goal gate is inconsistent")
        if failed_result.valid or failed_result.observation is not None:
            raise AssertionError("failed vision fabricated an observation")
        adapter_contract_audit = {
            "place_gate_object_to_goal_residual": place_residual,
            "failed_visual_localization_valid": failed_result.valid,
            "failed_visual_localization_reason": failed_result.failure_reason,
            "failed_visual_localization_observation": None,
        }
    finally:
        renderer.close()
        wrapped.close()

    summary = summarize(records)
    summary["base_world_round_trip_max_m"] = float(max(
        transform_direction_residuals, default=0.0
    ))
    summary["policy_used_shadow_observation"] = False
    payload = {
        "frame_convention": "T_A_B maps B-frame points into frame A",
        "base_to_world_transform": "T_world_base = inverse(T_base_world)",
        "T_base_world": t_base_world_reference.tolist(),
        "T_world_base": invert_transform(t_base_world_reference).tolist(),
        "pregrasp_height_from_environment_m": float(env.pregrasp_height),
        "grasp_height_offset_from_environment_m": float(
            env.grasp_height_offset
        ),
        "vision_truth_separation": {
            "detector_reads_object_site_truth": False,
            "adapter_reads_object_site_truth": False,
            "truth_only_in_function": "truth_evaluator_test_only",
            "policy_input": "original_truth_observation",
        },
        "adapter_contract_audit": adapter_contract_audit,
        "summary": summary,
        "records": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "visual_observation_shadow_results.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    csv_path = OUTPUT_DIR / "visual_observation_shadow_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "index", "valid", "failure_reason",
            "full_39d_max_abs_difference",
            "internal_consistency_residual",
            "base_world_round_trip_residual_m",
        ] + [f"{name}_error_m" for name in COMPARE_FIELDS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {name: record.get(name) for name in fields}
            if record["valid"]:
                for name in COMPARE_FIELDS:
                    row[f"{name}_error_m"] = json.dumps(
                        record["field_error_vectors_m"][name]
                    )
            writer.writerow(row)
    print("T_base_world:\n", t_base_world_reference)
    print("T_world_base:\n", invert_transform(t_base_world_reference))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT_DIR}")
    if summary["maximum_internal_consistency_residual"] >= 1e-7:
        raise SystemExit("FAIL: inconsistent derived observation")
    print("PASS: shadow observation built; PPO still used truth observation")


if __name__ == "__main__":
    main()
