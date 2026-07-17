"""Validate Color-optical cube estimates in the actual UR5e base frame."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fourc2  # noqa: E402,F401
from fourc2.camera_geometry import (  # noqa: E402
    base_from_color_optical_transform,
    base_world_transform,
    camera_world_pose_optical,
    invert_transform,
    intrinsics_from_fovy,
    relative_optical_transform,
    transform_point,
    validate_rigid_transform,
    world_from_color_optical_transform,
)
from fourc2.rgbd_cube_localizer import (  # noqa: E402
    load_localization_config,
    localize_cube_rgbd,
)


WIDTH, HEIGHT = 640, 360
RANDOM_SAMPLES = 20
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"
BASE_BODY_NAME = "base"
CONFIG_PATH = ROOT / "configs" / "hsv_cube_localization.json"
OUTPUT_DIR = ROOT / "outputs" / "color_to_base_localization"


def object_position_set(env, position):
    env.data.qpos[env.object_qpos_slice] = np.array(
        [*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    env.data.qvel[env.object_qvel_slice] = 0.0
    mujoco.mj_forward(env.model, env.data)


def render(renderer, data, color_id, depth_id):
    renderer.update_scene(data, camera=color_id)
    color = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=depth_id)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return color, depth


def localize_current(env, renderer, color_id, depth_id, base_id, config,
                     color_k, depth_k):
    color_pose = camera_world_pose_optical(env.data, color_id)
    depth_pose = camera_world_pose_optical(env.data, depth_id)
    color_from_depth = relative_optical_transform(color_pose, depth_pose)
    support_up_color = color_pose[1].T @ np.array([0.0, 0.0, 1.0])
    color, depth = render(renderer, env.data, color_id, depth_id)
    result = localize_cube_rgbd(
        color, depth, depth_k, color_k, color_from_depth,
        support_up_color, config
    )
    t_world_color = world_from_color_optical_transform(env.data, color_id)
    t_base_world = base_world_transform(env.data, base_id)
    t_base_color = base_from_color_optical_transform(
        env.data, color_id, base_id
    )
    for transform in (t_world_color, t_base_world, t_base_color):
        validate_rigid_transform(transform)
    estimated_base = None
    if result.valid:
        estimated_base = transform_point(
            t_base_color, result.estimated_object_center_color
        )
        round_trip = transform_point(
            invert_transform(t_base_color), estimated_base
        )
        if np.linalg.norm(
            round_trip - result.estimated_object_center_color
        ) > 1e-10:
            raise AssertionError("base -> color -> base round trip failed")
    return {
        "color": color,
        "result": result,
        "estimated_base": estimated_base,
        "T_world_color_optical": t_world_color,
        "T_base_world": t_base_world,
        "T_base_color_optical": t_base_color,
    }


def truth_evaluation(env, base_id, estimate):
    truth_world = env.data.site_xpos[env.object_site_id].copy()
    truth_base = transform_point(
        base_world_transform(env.data, base_id), truth_world
    )
    error_vector = None
    error_norm = None
    if estimate is not None:
        error_vector = estimate - truth_base
        error_norm = float(np.linalg.norm(error_vector))
    return truth_world, truth_base, error_vector, error_norm


def matrix_list(value):
    return np.asarray(value).tolist()


def record_result(group, index, localized, truth):
    truth_world, truth_base, error_vector, error_norm = truth
    result = localized["result"]
    camera_transform = localized["T_world_color_optical"]
    return {
        "group": group,
        "index": index,
        "valid": result.valid,
        "hsv_detected": result.hsv_detected,
        "failure_reason": result.failure_reason,
        "camera_world_position": camera_transform[:3, 3].tolist(),
        "camera_world_rotation_optical": matrix_list(camera_transform[:3, :3]),
        "estimated_object_center_color": (
            None if result.estimated_object_center_color is None
            else result.estimated_object_center_color.tolist()
        ),
        "estimated_object_center_base": (
            None if localized["estimated_base"] is None
            else localized["estimated_base"].tolist()
        ),
        "truth_object_center_world_test_only": truth_world.tolist(),
        "truth_object_center_base_test_only": truth_base.tolist(),
        "error_xyz_base_m_test_only": (
            None if error_vector is None else error_vector.tolist()
        ),
        "error_3d_m_test_only": error_norm,
    }


def summary_for_records(records):
    detected = sum(record["hsv_detected"] for record in records)
    valid = sum(record["valid"] for record in records)
    error_vectors = np.asarray([
        record["error_xyz_base_m_test_only"] for record in records
        if record["error_xyz_base_m_test_only"] is not None
    ], dtype=np.float64)
    norms = np.linalg.norm(error_vectors, axis=1) if error_vectors.size else np.array([])
    failures = Counter(
        record["failure_reason"] for record in records
        if record["failure_reason"] is not None
    )
    return {
        "samples": len(records),
        "hsv_detection_success_rate": detected / len(records),
        "localization_success_rate": valid / len(records),
        "xyz_mae_m": None if not norms.size else np.mean(np.abs(error_vectors), axis=0).tolist(),
        "error_3d_mean_m": None if not norms.size else float(np.mean(norms)),
        "error_3d_median_m": None if not norms.size else float(np.median(norms)),
        "error_3d_max_m": None if not norms.size else float(np.max(norms)),
        "failure_reason_counts": dict(failures),
    }


def save_visualization(color, record, path):
    image = color.copy()
    estimate_color = record["estimated_object_center_color"]
    estimate_base = record["estimated_object_center_base"]
    error = record["error_3d_m_test_only"]
    lines = [
        f"Color: {np.array(estimate_color)}",
        f"Base:  {np.array(estimate_base)}",
        f"Truth: {np.array(record['truth_object_center_base_test_only'])}",
        f"3D error: {1000.0 * error:.3f} mm",
    ]
    panel_height = 23 * len(lines) + 10
    cv2.rectangle(image, (0, 0), (WIDTH - 1, panel_height), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(
            image, line, (8, 22 + 23 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
            cv2.LINE_AA
        )
    Image.fromarray(image).save(path)


def main():
    config = load_localization_config(CONFIG_PATH)
    wrapped = gym.make("My4C2GraspStageCube3cm-v0")
    env = wrapped.unwrapped
    env.reset(seed=17)
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
        raise RuntimeError("required camera/base frame missing")
    color_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[color_id])
    depth_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[depth_id])
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    reference_qpos = env.data.qpos.copy()
    reference_qvel = env.data.qvel.copy()
    reference_ctrl = env.data.ctrl.copy()
    reference_tcp = env.data.site_xpos[env.pinch_site_id].copy()
    reference_color_pose = camera_world_pose_optical(env.data, color_id)
    rng = np.random.default_rng(20260714)
    random_records = []
    representative = None
    try:
        # Group 1: 20 random positions inside the current unmodified camera FOV.
        for index in range(RANDOM_SAMPLES):
            camera_position, world_from_color = camera_world_pose_optical(
                env.data, color_id
            )
            center_z = env.table_top_z + env.object_half_size
            distance = (center_z - camera_position[2]) / world_from_color[2, 2]
            point_color = np.array([
                rng.uniform(-0.035, -0.005),
                rng.uniform(-0.010, 0.010),
                distance,
            ])
            point_world = camera_position + world_from_color @ point_color
            point_world[2] = center_z
            object_position_set(env, point_world)
            localized = localize_current(
                env, renderer, color_id, depth_id, base_id, config,
                color_k, depth_k
            )
            record = record_result(
                "random_position", index, localized,
                truth_evaluation(env, base_id, localized["estimated_base"])
            )
            random_records.append(record)
            if representative is None and record["valid"]:
                representative = (localized["color"].copy(), record)

        # Group 2: one fixed cube, five small existing-IK observation offsets.
        env.data.qpos[:] = reference_qpos
        env.data.qvel[:] = reference_qvel
        env.data.ctrl[:] = reference_ctrl
        mujoco.mj_forward(env.model, env.data)
        camera_position, world_from_color = reference_color_pose
        center_z = env.table_top_z + env.object_half_size
        distance = (center_z - camera_position[2]) / world_from_color[2, 2]
        fixed_color = np.array([-0.020, 0.0, distance])
        fixed_world = camera_position + world_from_color @ fixed_color
        fixed_world[2] = center_z
        object_position_set(env, fixed_world)
        observation_offsets = [
            [0.0, 0.0, 0.0],
            [0.008, 0.0, 0.0],
            [-0.008, 0.0, 0.0],
            [0.0, 0.008, 0.0],
            [0.0, -0.008, 0.0],
        ]
        pose_records = []
        transformed_fixed_base = []
        for index, offset in enumerate(observation_offsets):
            env.data.qpos[:] = reference_qpos
            env.data.qvel[:] = reference_qvel
            env.data.ctrl[:] = reference_ctrl
            mujoco.mj_forward(env.model, env.data)
            target = reference_tcp + np.asarray(offset, dtype=np.float64)
            joint_target = env._solve_ik(target)
            env.data.qpos[env.arm_qpos_ids] = joint_target
            env.data.qvel[env.arm_qvel_ids] = 0.0
            env.data.ctrl[env.arm_actuator_ids] = joint_target
            object_position_set(env, fixed_world)
            localized = localize_current(
                env, renderer, color_id, depth_id, base_id, config,
                color_k, depth_k
            )
            truth = truth_evaluation(env, base_id, localized["estimated_base"])
            record = record_result("multi_pose", index, localized, truth)
            record["tcp_world_offset_m"] = offset
            record["ik_position_residual_m"] = float(
                np.linalg.norm(
                    env.data.site_xpos[env.pinch_site_id] - target
                )
            )
            pose_records.append(record)

            # Pure transform invariance: the same fixed world point converted
            # world->color->base must equal direct world->base at every pose.
            t_world_color = localized["T_world_color_optical"]
            fixed_in_color = transform_point(
                invert_transform(t_world_color), fixed_world
            )
            fixed_in_base = transform_point(
                localized["T_base_color_optical"], fixed_in_color
            )
            direct_base = transform_point(
                localized["T_base_world"], fixed_world
            )
            if np.linalg.norm(fixed_in_base - direct_base) > 1e-10:
                raise AssertionError("same world point is inconsistent across poses")
            transformed_fixed_base.append(fixed_in_base)
    finally:
        renderer.close()
        wrapped.close()

    random_summary = summary_for_records(random_records)
    pose_summary = summary_for_records(pose_records)
    pose_estimates = np.asarray([
        record["estimated_object_center_base"] for record in pose_records
        if record["estimated_object_center_base"] is not None
    ], dtype=np.float64)
    if pose_estimates.shape[0] >= 2:
        pairwise = np.linalg.norm(
            pose_estimates[:, None, :] - pose_estimates[None, :, :], axis=2
        )
        pose_summary["estimate_std_xyz_m"] = np.std(pose_estimates, axis=0).tolist()
        pose_summary["max_inter_pose_distance_m"] = float(np.max(pairwise))
    else:
        pose_summary["estimate_std_xyz_m"] = None
        pose_summary["max_inter_pose_distance_m"] = None
    transformed_fixed_base = np.asarray(transformed_fixed_base)
    pose_summary["pure_transform_consistency_max_m"] = float(np.max(
        np.linalg.norm(
            transformed_fixed_base - transformed_fixed_base[0], axis=1
        )
    ))

    # The base origin equals world, but its runtime rotation is 180 deg about Z.
    t_base_world = base_world_transform(env.data, base_id)
    world_base_coincident = bool(np.allclose(t_base_world, np.eye(4), atol=1e-9))
    payload = {
        "frame_convention": "T_A_B maps B-frame points into frame A",
        "world_and_base_frames_coincident": world_base_coincident,
        "T_base_world": matrix_list(t_base_world),
        "random_position_summary": random_summary,
        "multi_pose_summary": pose_summary,
        "random_position_records": random_records,
        "multi_pose_records": pose_records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "base_localization_results.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    all_records = random_records + pose_records
    csv_fields = [
        "group", "index", "valid", "hsv_detected", "failure_reason",
        "camera_world_position", "estimated_object_center_color",
        "estimated_object_center_base", "truth_object_center_base_test_only",
        "error_xyz_base_m_test_only", "error_3d_m_test_only",
    ]
    with (OUTPUT_DIR / "base_localization_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for record in all_records:
            writer.writerow({
                field: (
                    json.dumps(record.get(field))
                    if isinstance(record.get(field), (list, dict))
                    else record.get(field)
                ) for field in csv_fields
            })
    if representative is not None:
        save_visualization(
            representative[0], representative[1],
            OUTPUT_DIR / "base_localization_overlay.png"
        )

    print("T_base_world:\n", t_base_world)
    print("random position summary:\n", json.dumps(random_summary, indent=2))
    print("multi-pose records:")
    for record in pose_records:
        print(json.dumps(record, ensure_ascii=False))
    print("multi-pose summary:\n", json.dumps(pose_summary, indent=2))
    print(f"saved: {OUTPUT_DIR}")
    if random_summary["localization_success_rate"] < 1.0:
        raise SystemExit("FAIL: random localization incomplete")
    if pose_summary["localization_success_rate"] < 1.0:
        raise SystemExit("FAIL: multi-pose localization incomplete")
    print("PASS: Color optical -> UR5e base localization")


if __name__ == "__main__":
    main()
