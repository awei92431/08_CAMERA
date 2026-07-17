"""Stage 6B: full-episode dynamic RGB-D visual observation shadow mode."""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fourc2  # noqa: E402,F401
from fourc2.camera_geometry import (  # noqa: E402
    base_from_color_optical_transform, base_world_transform,
    camera_world_pose_optical, intrinsics_from_fovy, invert_transform,
    project_points, relative_optical_transform, transform_point,
)
from fourc2.envs.allstage import (  # noqa: E402
    STAGE_GRASP, STAGE_LIFT, STAGE_PLACE, STAGE_REACH,
)
from fourc2.rgbd_cube_localizer import (  # noqa: E402
    load_localization_config, localize_cube_rgbd,
)
from fourc2.visual_observation_adapter import (  # noqa: E402
    FIELD_SLICES, VisualObservationAdapter, internal_consistency_residual,
)

ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
CHECKPOINT = ROOT / "checkpoints" / "best_full_flow_v22.zip"
CONFIG = ROOT / "configs" / "hsv_cube_localization.json"
OUTPUT = ROOT / "outputs" / "dynamic_visual_shadow"
WIDTH, HEIGHT = 640, 360
STAGE_LABELS = {STAGE_REACH: "Reach", STAGE_GRASP: "Grasp",
                STAGE_LIFT: "Lift", STAGE_PLACE: "Place"}
FIELDS = ["object_position", "pregrasp_position", "grasp_position",
          "pinch_to_pregrasp", "pinch_to_grasp", "object_to_goal",
          "object_lift"]


def render_pair(renderer, data, color_id, depth_id):
    renderer.update_scene(data, camera=color_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=depth_id)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return rgb, depth


def process_label(env, info):
    stage = env.stage
    any_contact = bool(info["has_any_contact"])
    if stage == STAGE_REACH:
        return "pre_grasp"
    if stage == STAGE_GRASP:
        return "near_first_contact" if any_contact else "pre_grasp"
    if stage == STAGE_LIFT:
        return "post_latch" if info["object_lift"] < 0.010 else "lifting"
    if stage == STAGE_PLACE:
        if env.release_has_opened or env.place_descent_active or env._place_release_phase():
            return "place_release"
        return "transport"
    return "other"


def truth_visibility_and_points(env, color_pose, color_k):
    """Evaluator only: truth centre/top surface and whether centre is in FOV."""
    truth_world = env.data.site_xpos[env.object_site_id].copy()
    cam_pos, world_from_color = color_pose
    center_color = world_from_color.T @ (truth_world - cam_pos)
    uv, z = project_points(center_color[None, :], color_k)
    u, v = uv[0]
    in_view = bool(z[0] > 0 and 0 <= u < WIDTH and 0 <= v < HEIGHT)
    support_up_color = world_from_color.T @ np.array([0.0, 0.0, 1.0])
    top_surface_color = center_color + env.object_half_size * support_up_color
    return truth_world, center_color, top_surface_color, in_view, [float(u), float(v)]


def longest_failure_streak(records):
    longest = current = 0
    previous_episode = None
    for record in records:
        if previous_episode is not None and record["episode"] != previous_episode:
            current = 0
        previous_episode = record["episode"]
        if record["visual_valid"]:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def stats_for(records):
    valid = [r for r in records if r["visual_valid"]]
    out = {
        "steps": len(records),
        "visual_success_rate": len(valid) / len(records) if records else None,
        "longest_consecutive_failure_steps": longest_failure_streak(records),
        "failure_reasons": dict(Counter(r["failure_reason"] for r in records
                                        if not r["visual_valid"])),
        "cube_out_of_color_fov_steps": sum(not r["truth_center_in_color_fov"]
                                             for r in records),
        "gripper_occlusion_failure_steps": sum(r["gripper_occlusion_failure"]
                                                for r in records),
        "mask_area_px": {}, "valid_depth_points": {}, "fields": {},
    }
    for source, key in [(records, "mask_area_px"), (records, "valid_depth_points")]:
        values = np.asarray([r[key] for r in source if r[key] is not None], float)
        out[key] = None if not len(values) else {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
        }
    for field in FIELDS:
        arrays = np.asarray([r["field_error_vectors"][field] for r in valid], float)
        norms = np.linalg.norm(arrays, axis=1) if len(arrays) else np.array([])
        out["fields"][field] = None if not len(norms) else {
            "xyz_mae_m": np.mean(np.abs(arrays), axis=0).tolist(),
            "mean_m": float(np.mean(norms)), "median_m": float(np.median(norms)),
            "p95_m": float(np.quantile(norms, .95)), "max_m": float(np.max(norms)),
        }
    for key in ["center_error_m", "visible_surface_error_m"]:
        values = np.asarray([r[key] for r in valid], float)
        out[key] = None if not len(values) else {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values)),
        }
    return out


def make_plots(records, stage_summary):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stages = list(STAGE_LABELS.values())
    rates = [100 * stage_summary[s]["visual_success_rate"]
             if s in stage_summary and stage_summary[s]["visual_success_rate"] is not None
             else 0 for s in stages]
    plt.figure(figsize=(7, 4)); plt.bar(stages, rates); plt.ylim(0, 105)
    plt.ylabel("Visual localization success (%)"); plt.tight_layout()
    plt.savefig(OUTPUT / "visual_success_rate_by_stage.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 4))
    offset = 0
    for episode in sorted(set(r["episode"] for r in records)):
        rs = [r for r in records if r["episode"] == episode]
        y = [1000 * r["center_error_m"] if r["center_error_m"] is not None else np.nan
             for r in rs]
        x = np.arange(len(rs)) + offset
        plt.plot(x, y, linewidth=.65)
        offset += len(rs)
    plt.xlabel("Policy step (episodes concatenated)"); plt.ylabel("Center error (mm)")
    plt.tight_layout(); plt.savefig(OUTPUT / "localization_error_per_step.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    config = load_localization_config(CONFIG)
    wrapped = gym.make(ENV_ID)
    env = wrapped.unwrapped
    model = PPO.load(CHECKPOINT, device="cpu")
    color_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                 "eye_in_hand_color")
    depth_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                 "eye_in_hand_depth")
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    color_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[color_id])
    depth_k = intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[depth_id])
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    records, episode_summaries = [], []
    max_consistency = 0.0
    try:
        for episode in range(args.episodes):
            observation, _ = wrapped.reset(seed=episode)
            adapter = VisualObservationAdapter(env.pregrasp_height,
                                               env.grasp_height_offset)
            adapter.reset(); done = False; step = 0; ep_return = 0.0
            while not done:
                # Current diagnostics and vision are sampled from exactly the
                # state represented by this truth observation.
                info_now = env._get_info()
                stage = STAGE_LABELS.get(env.stage, str(env.stage))
                process = process_label(env, info_now)
                color_pose = camera_world_pose_optical(env.data, color_id)
                depth_pose = camera_world_pose_optical(env.data, depth_id)
                rgb, depth = render_pair(renderer, env.data, color_id, depth_id)
                support_up_color = color_pose[1].T @ np.array([0., 0., 1.])
                localized = localize_cube_rgbd(
                    rgb, depth, depth_k, color_k,
                    relative_optical_transform(color_pose, depth_pose),
                    support_up_color, config,
                )
                truth_world, truth_color, truth_surface, in_fov, truth_uv = (
                    truth_visibility_and_points(env, color_pose, color_k)
                )
                t_base_world = base_world_transform(env.data, base_id)
                estimate_base = None
                if localized.valid:
                    estimate_base = transform_point(
                        base_from_color_optical_transform(env.data, color_id, base_id),
                        localized.estimated_object_center_color,
                    )
                shadow = adapter.build(
                    observation, estimate_base, t_base_world,
                    env.data.site_xpos[env.goal_site_id].copy(),
                )
                field_errors = None; center_error = surface_error = None
                if shadow.valid:
                    field_errors = {}
                    for field in FIELDS:
                        delta = (shadow.observation[FIELD_SLICES[field]].astype(float)
                                 - observation[FIELD_SLICES[field]].astype(float))
                        field_errors[field] = delta.tolist()
                    center_error = float(np.linalg.norm(
                        shadow.estimated_object_world - truth_world))
                    surface_error = float(np.linalg.norm(
                        localized.visible_surface_point_color - truth_surface))
                    residual = internal_consistency_residual(
                        shadow.observation, env.pregrasp_height,
                        env.grasp_height_offset, shadow.estimated_initial_object_z,
                        env.data.site_xpos[env.goal_site_id].copy())
                    max_consistency = max(max_consistency, residual)
                    if residual >= 1e-7:
                        raise AssertionError(f"adapter residual {residual}")
                near_gripper = info_now["pinch_to_object_distance"] < 0.08
                occlusion = bool(not localized.valid and in_fov and
                                 (near_gripper or info_now["has_any_contact"]
                                  or info_now["is_grasp_latched"]))
                record = {
                    "episode": episode, "step": step, "stage": stage,
                    "process": process, "visual_valid": bool(localized.valid),
                    "failure_reason": localized.failure_reason,
                    "truth_center_in_color_fov": in_fov,
                    "truth_center_uv": truth_uv,
                    "gripper_occlusion_failure": occlusion,
                    "has_any_contact": bool(info_now["has_any_contact"]),
                    "is_grasp_latched": bool(info_now["is_grasp_latched"]),
                    "release_has_opened": bool(env.release_has_opened),
                    "mask_area_px": localized.contour_area_px,
                    "valid_depth_points": localized.filtered_depth_point_count,
                    "center_error_m": center_error,
                    "visible_surface_error_m": surface_error,
                    "field_error_vectors": field_errors,
                    "estimated_object_world": None if not shadow.valid else
                        shadow.estimated_object_world.tolist(),
                    "truth_object_world_evaluator_only": truth_world.tolist(),
                    "policy_input": "original_truth_observation",
                }
                records.append(record)

                # Critical Shadow contract: visual/shadow is never referenced
                # by model.predict or env.step.
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = wrapped.step(action)
                ep_return += float(reward); step += 1
                done = bool(terminated or truncated)
            episode_summaries.append({
                "episode": episode, "steps": step, "return": ep_return,
                "task_success": bool(info.get("is_success", False)),
                "visual_success_rate": sum(r["visual_valid"] for r in records
                    if r["episode"] == episode) / step,
            })
            print(f"episode={episode:02d} steps={step} task_success={info.get('is_success')} "
                  f"visual={episode_summaries[-1]['visual_success_rate']:.3f}")
    finally:
        renderer.close(); wrapped.close()

    by_stage = {name: stats_for([r for r in records if r["stage"] == name])
                for name in STAGE_LABELS.values()}
    process_names = ["pre_grasp", "near_first_contact", "post_latch",
                     "lifting", "transport", "place_release"]
    by_process = {name: stats_for([r for r in records if r["process"] == name])
                  for name in process_names}
    overall = stats_for(records)
    overall["maximum_adapter_consistency_residual"] = max_consistency
    overall["task_success_rate"] = float(np.mean(
        [e["task_success"] for e in episode_summaries]))
    payload = {
        "checkpoint": str(CHECKPOINT), "environment": ENV_ID,
        "policy_input": "original truth observation only",
        "shadow_input_affects_action": False,
        "episodes": episode_summaries, "overall": overall,
        "by_stage": by_stage, "by_process": by_process,
        "records": records,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "dynamic_visual_shadow_results.json").open("w") as f:
        json.dump(payload, f, indent=2)
    columns = ["episode", "step", "stage", "process", "visual_valid",
               "failure_reason", "truth_center_in_color_fov",
               "gripper_occlusion_failure", "has_any_contact",
               "is_grasp_latched", "release_has_opened", "mask_area_px",
               "valid_depth_points", "center_error_m", "visible_surface_error_m"]
    with (OUTPUT / "dynamic_visual_shadow_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in records)
    typical = [r for r in records if r["episode"] == 0]
    with (OUTPUT / "typical_episode_000_timeseries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in typical)
    make_plots(records, by_stage)
    print(json.dumps({"overall": overall, "by_stage": by_stage,
                      "by_process": by_process}, indent=2))
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
