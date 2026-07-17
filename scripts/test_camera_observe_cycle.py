"""Validate the pre-policy home -> camera_observe -> home acquisition cycle."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np

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

ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"
WIDTH, HEIGHT = 640, 360
OUTPUT_DIR = ROOT / "outputs" / "camera_observe_cycle"
CONFIG_PATH = ROOT / "configs" / "hsv_cube_localization.json"


def obj_id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"missing MuJoCo object: {name}")
    return value


def move_arm(env, target, interpolation_steps, substeps):
    start = env.data.qpos[env.arm_qpos_ids].copy()
    for index in range(1, interpolation_steps + 1):
        t = index / interpolation_steps
        blend = t * t * (3.0 - 2.0 * t)
        command = start + blend * (target - start)
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            command, env.arm_ctrl_low, env.arm_ctrl_high
        )
        for _ in range(substeps):
            mujoco.mj_step(env.model, env.data)
    return env.data.qpos[env.arm_qpos_ids].copy()


def hold_arm(env, target, steps):
    for _ in range(steps):
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            target, env.arm_ctrl_low, env.arm_ctrl_high
        )
        mujoco.mj_step(env.model, env.data)


def render_pair(renderer, data, color_id, depth_id):
    renderer.update_scene(data, camera=color_id)
    color = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=depth_id)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return color, depth


def active_bad_contacts(env):
    names = []
    table = env.table_geom_id
    obj = env.object_geom_id
    for contact in env.data.contact:
        if contact.dist >= 0:
            continue
        pair = {int(contact.geom1), int(contact.geom2)}
        # The cube resting on the tabletop is expected.
        if pair == {table, obj}:
            continue
        if table in pair or obj in pair:
            names.append([
                mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            ])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--interpolation-steps", type=int, default=180)
    parser.add_argument("--substeps", type=int, default=4)
    args = parser.parse_args()

    wrapped = gym.make(ENV_ID)
    env = wrapped.unwrapped
    model = env.model
    observe_key = obj_id(model, mujoco.mjtObj.mjOBJ_KEY, "camera_observe")
    color_id = obj_id(model, mujoco.mjtObj.mjOBJ_CAMERA, COLOR_NAME)
    depth_id = obj_id(model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_NAME)
    observe_q = model.key_qpos[observe_key, env.arm_qpos_ids].copy()
    config = load_localization_config(CONFIG_PATH)
    color_k = intrinsics_from_fovy(WIDTH, HEIGHT, model.cam_fovy[color_id])
    depth_k = intrinsics_from_fovy(WIDTH, HEIGHT, model.cam_fovy[depth_id])
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    rows = []
    failures = Counter()
    try:
        for sample in range(args.seeds):
            seed = args.seed_offset + sample
            env.reset(seed=seed)
            home_q = env.data.qpos[env.arm_qpos_ids].copy()
            object_before = env.data.site_xpos[env.object_site_id].copy()

            reached_q = move_arm(
                env, observe_q, args.interpolation_steps, args.substeps
            )
            hold_arm(env, observe_q, 120)
            observe_joint_error = float(np.max(np.abs(
                env.data.qpos[env.arm_qpos_ids] - observe_q
            )))
            contacts_at_observe = active_bad_contacts(env)

            color_pose = camera_world_pose_optical(env.data, color_id)
            depth_pose = camera_world_pose_optical(env.data, depth_id)
            color_from_depth = relative_optical_transform(color_pose, depth_pose)
            support_up_color = color_pose[1].T @ np.array([0.0, 0.0, 1.0])
            color, depth = render_pair(renderer, env.data, color_id, depth_id)
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
            else:
                failures[result.failure_reason] += 1

            move_arm(env, home_q, args.interpolation_steps, args.substeps)
            hold_arm(env, home_q, 120)
            env._sync_mocap_target_to_tcp()
            home_joint_error = float(np.max(np.abs(
                env.data.qpos[env.arm_qpos_ids] - home_q
            )))
            contacts_at_home = active_bad_contacts(env)
            object_after = env.data.site_xpos[env.object_site_id].copy()
            displacement = float(np.linalg.norm(object_after - object_before))
            rows.append({
                "seed": seed,
                "vision_valid": bool(result.valid),
                "failure_reason": result.failure_reason,
                "center_error_m": error,
                "mask_pixels": int(result.mask_pixel_count),
                "valid_depth_points": int(result.mask_depth_point_count),
                "observe_joint_error_max_rad": observe_joint_error,
                "home_joint_error_max_rad": home_joint_error,
                "object_displacement_m": displacement,
                "bad_contacts_at_observe": contacts_at_observe,
                "bad_contacts_at_home": contacts_at_home,
                "camera_world_position": color_pose[0].tolist(),
                "reached_joint_position": reached_q.tolist(),
            })
            print(
                f"seed={seed:03d} valid={result.valid} "
                f"error_mm={None if error is None else round(error*1000, 3)} "
                f"object_shift_mm={displacement*1000:.4f}"
            )
    finally:
        renderer.close()
        wrapped.close()

    errors = np.array([
        row["center_error_m"] for row in rows
        if row["center_error_m"] is not None
    ])
    success_count = sum(row["vision_valid"] for row in rows)
    bad_contact_count = sum(bool(row["bad_contacts_at_observe"] or row["bad_contacts_at_home"])
                            for row in rows)
    summary = {
        "environment": ENV_ID,
        "samples": args.seeds,
        "seed_range": [args.seed_offset, args.seed_offset + args.seeds - 1],
        "home_arm_qpos_rad": rows and home_q.tolist(),
        "camera_observe_arm_qpos_rad": observe_q.tolist(),
        "vision_success_count": int(success_count),
        "vision_success_rate": success_count / args.seeds,
        "center_error_mm": None if not len(errors) else {
            "mean": float(errors.mean() * 1000),
            "median": float(np.median(errors) * 1000),
            "p95": float(np.percentile(errors, 95) * 1000),
            "max": float(errors.max() * 1000),
        },
        "failure_reason_counts": dict(failures),
        "episodes_with_bad_task_contacts": int(bad_contact_count),
        "max_object_displacement_mm": float(max(
            row["object_displacement_m"] for row in rows
        ) * 1000),
        "max_observe_joint_error_rad": float(max(
            row["observe_joint_error_max_rad"] for row in rows
        )),
        "max_home_joint_error_rad": float(max(
            row["home_joint_error_max_rad"] for row in rows
        )),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "camera_observe_cycle_results.json"
    output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2),
                      encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output}")
    if success_count != args.seeds or bad_contact_count:
        raise SystemExit("FAIL: vision failure or unsafe task contact detected")
    print("PASS: home -> camera_observe -> RGB-D -> home")


if __name__ == "__main__":
    main()
