"""Test HSV + projected RGB-D cube localization at 20 visible positions."""

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
    camera_world_pose_optical,
    intrinsics_from_fovy,
    relative_optical_transform,
)
from fourc2.rgbd_cube_localizer import (  # noqa: E402
    load_localization_config,
    localize_cube_rgbd,
)


WIDTH, HEIGHT = 640, 360
EPISODES = 20
SEED = 20260714
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"
CONFIG_PATH = ROOT / "configs" / "hsv_cube_localization.json"
OUTPUT_DIR = ROOT / "outputs" / "hsv_rgbd_localization"


def camera_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if value < 0:
        raise RuntimeError(f"camera not found: {name}")
    return value


def render_rgb(renderer, data, cam_id):
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


def render_depth(renderer, data, cam_id):
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=cam_id)
        return renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()


def set_object_position(env, position):
    env.data.qpos[env.object_qpos_slice] = np.array(
        [*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    env.data.qvel[env.object_qvel_slice] = 0.0
    mujoco.mj_forward(env.model, env.data)


def detection_overlay(color, result):
    overlay = color.copy()
    if result.bbox_xywh is not None:
        x, y, w, h = result.bbox_xywh
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    if result.pixel_center_uv is not None:
        u, v = np.rint(result.pixel_center_uv).astype(int)
        cv2.drawMarker(overlay, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 13, 2)
    return overlay


def projected_points_overlay(color, result):
    overlay = color.copy()
    if result.projected_point_map.shape[:2] != color.shape[:2]:
        return overlay
    projected = np.isfinite(result.projected_point_map).all(axis=2)
    inside = projected & (result.eroded_mask > 0)
    overlay[projected] = (
        0.75 * overlay[projected] + 0.25 * np.array([0, 120, 255])
    ).astype(np.uint8)
    overlay[inside] = np.array([0, 255, 0], dtype=np.uint8)
    return overlay


def main():
    config = load_localization_config(CONFIG_PATH)
    # Existing Grasp reset supplies a realistic fixed wrist approach posture.
    # Subsequent cube placement is test-scene generation only; no truth enters
    # localize_cube_rgbd().
    wrapped = gym.make("My4C2GraspStageCube3cm-v0")
    env = wrapped.unwrapped
    env.reset(seed=17)
    color_id = camera_id(env.model, COLOR_NAME)
    depth_id = camera_id(env.model, DEPTH_NAME)
    color_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[color_id])
    depth_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[depth_id])
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    rng = np.random.default_rng(SEED)
    rows = []
    errors = []
    failure_counts = Counter()
    representative = None
    try:
        for index in range(EPISODES):
            color_pose = camera_world_pose_optical(env.data, color_id)
            depth_pose = camera_world_pose_optical(env.data, depth_id)
            color_from_depth = relative_optical_transform(color_pose, depth_pose)
            camera_position, world_from_color = color_pose
            # Intersect a small random Color-optical ray patch with the known
            # horizontal support height. This samples positions that the fixed,
            # unmodified wrist camera can actually observe.
            center_z = env.table_top_z + env.object_half_size
            forward_world = world_from_color[:, 2]
            distance = (center_z - camera_position[2]) / forward_world[2]
            optical_position = np.array([
                rng.uniform(-0.035, -0.005),
                rng.uniform(-0.010, 0.010),
                distance,
            ])
            world_position = camera_position + world_from_color @ optical_position
            world_position[2] = center_z
            set_object_position(env, world_position)

            # Camera poses are recomputed after mj_forward; no assumption that
            # the fixed robot posture yields an exact identity world rotation.
            color_pose = camera_world_pose_optical(env.data, color_id)
            depth_pose = camera_world_pose_optical(env.data, depth_id)
            color_from_depth = relative_optical_transform(color_pose, depth_pose)
            support_up_color = color_pose[1].T @ np.array([0.0, 0.0, 1.0])
            color = render_rgb(renderer, env.data, color_id)
            depth = render_depth(renderer, env.data, depth_id)
            result = localize_cube_rgbd(
                color, depth, depth_k, color_k, color_from_depth,
                support_up_color, config
            )

            truth_world = env.data.site_xpos[env.object_site_id].copy()
            truth_color = color_pose[1].T @ (truth_world - color_pose[0])
            error = None
            if result.valid:
                error = float(np.linalg.norm(
                    result.estimated_object_center_color - truth_color
                ))
                errors.append(error)
            else:
                failure_counts[result.failure_reason] += 1
            record = result.to_dict()
            record.update({
                "sample_index": index,
                "truth_object_center_color_test_only": truth_color.tolist(),
                "center_error_m_test_only": error,
            })
            rows.append(record)
            if representative is None or (result.valid and not representative[2].valid):
                representative = (color.copy(), depth.copy(), result, record)
    finally:
        renderer.close()
        wrapped.close()

    hsv_successes = sum(row["hsv_detected"] for row in rows)
    localization_successes = sum(row["valid"] for row in rows)
    error_array = np.asarray(errors, dtype=np.float64)
    summary = {
        "samples": EPISODES,
        "hsv_detection_successes": int(hsv_successes),
        "hsv_detection_success_rate": float(hsv_successes / EPISODES),
        "localization_successes": int(localization_successes),
        "localization_success_rate": float(localization_successes / EPISODES),
        "center_error_m": {
            "mean": None if not errors else float(np.mean(error_array)),
            "median": None if not errors else float(np.median(error_array)),
            "max": None if not errors else float(np.max(error_array)),
        },
        "failure_reason_counts": dict(failure_counts),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    color, _, result, representative_record = representative
    Image.fromarray(color).save(OUTPUT_DIR / "color.png")
    Image.fromarray(result.mask).save(OUTPUT_DIR / "hsv_mask.png")
    Image.fromarray(result.eroded_mask).save(OUTPUT_DIR / "eroded_mask.png")
    Image.fromarray(detection_overlay(color, result)).save(
        OUTPUT_DIR / "color_detection_overlay.png"
    )
    Image.fromarray(projected_points_overlay(color, result)).save(
        OUTPUT_DIR / "projected_depth_points_overlay.png"
    )
    with (OUTPUT_DIR / "localization_result.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"config": config, "representative": representative_record,
             "summary": summary, "samples": rows},
            stream, indent=2, ensure_ascii=False
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("representative diagnostics:")
    print(json.dumps(representative_record, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT_DIR}")
    if localization_successes == 0:
        raise SystemExit("FAIL: no valid 3D localizations")
    print("PASS: HSV projected RGB-D cube localization")


if __name__ == "__main__":
    main()
