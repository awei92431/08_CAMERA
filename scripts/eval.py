import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MUJOCO_GL", "glfw")

from stable_baselines3 import PPO

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fourc2


STAGE_NAMES = {
    0: "reach",
    1: "grasp",
    2: "lift",
    3: "full",
    4: "reach_grasp",
    5: "place",
}


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def max_or_zero(values):
    return float(np.max(values)) if values else 0.0


def fmt_first_step(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return "none"
    return f"{np.mean(valid):.1f} ({len(valid)}/{len(values)})"


def resolve_model_path(model_path_arg):
    model_path = Path(model_path_arg)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="评估训练好的 4C2 夹爪 PPO 模型。")
    parser.add_argument("--env-id", default="My4C2AllStageCube3cm-v0")
    parser.add_argument(
        "--model-path",
        required=True,
        help="要加载的 SB3 模型路径，例如 runs/ppo_10k/models/best_model.zip。",
    )
    parser.add_argument("--episodes", type=int, default=5, help="评估 episode 数量。")
    parser.add_argument("--seed", type=int, default=123, help="评估随机种子。")
    parser.add_argument("--render", action="store_true", help="打开 MuJoCo viewer。")
    parser.add_argument("--sleep", type=float, default=0.03, help="渲染时每步暂停时间。")
    parser.add_argument(
        "--success-hold",
        type=float,
        default=1.0,
        help="渲染时每个成功 episode 结束后停留秒数，方便观察夹住状态。",
    )
    parser.add_argument("--show-safety", action="store_true", help="额外打印安全/碰撞指标。")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)
    model = PPO.load(str(model_path))

    print("env_id:", args.env_id)
    print("model_path:", model_path)
    print("episodes:", args.episodes)
    print("render:", args.render)
    if args.render:
        print("success_hold:", args.success_hold)

    episode_rewards = []
    final_distances = []
    successes = []
    stage_successes = []
    reach_successes = []
    reach_centered = []
    reach_xy_centered = []
    reach_z_centered = []
    reach_tcp_tracked = []
    grasp_successes = []
    coarse_grasp_successes = []
    stable_grasp_successes = []
    strict_grasp_successes = []
    grasp_stable_counts = []
    grasp_reason_gripper_closed = []
    grasp_reason_bilateral_contact = []
    grasp_reason_distance_ok = []
    grasp_reason_pose_ok = []
    grasp_reason_strict_pose_ok = []
    grasp_reason_penetration_ok = []
    lift_successes = []
    place_successes = []
    place_xy_ready = []
    place_low_ready = []
    place_open_ready = []
    place_opened = []
    place_has_opened = []
    place_height_ok = []
    lifts = []
    lift_distances = []
    object_to_goal_distances = []
    object_to_goal_xy_distances = []
    object_goal_z_errors = []
    pad_object_penetrations = []
    max_pad_object_penetrations = []
    contact_penetration_ok = []
    object_uprights = []
    object_upright_ok = []
    object_still_near_start = []
    xy_drifts = []
    xy_steps = []
    object_xy_speeds = []
    gripper_states = []
    latched = []
    ever_latched = []
    bilateral = []
    ever_bilateral = []
    unilateral = []
    any_contact = []
    ever_any_contact = []
    raw_bilateral = []
    ever_raw_bilateral = []
    raw_any_contact = []
    ever_raw_any_contact = []
    final_stages = []
    grasp_phases = []
    pregrasp_distances = []
    pregrasp_xy_errors = []
    pregrasp_z_errors = []
    grasp_distances = []
    best_pregrasp_distances = []
    best_grasp_distances = []
    max_lifts = []
    max_xy_drifts = []
    tcp_errors = []
    max_tcp_errors = []
    close_allowed = []
    ever_close_allowed = []
    fine_close_allowed = []
    ever_fine_close_allowed = []
    xy_aligned = []
    ever_xy_aligned = []
    descent_allowed = []
    ever_descent_allowed = []
    stage_failures = []
    ever_stage_failures = []
    grasp_xy_errors = []
    grasp_z_errors = []
    table_contacts = []
    boundary_penalties = []
    first_success_steps = []
    first_stage_success_steps = []
    first_reach_success_steps = []
    first_grasp_success_steps = []
    first_lift_success_steps = []
    first_place_success_steps = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        final_info = info
        step_count = 0
        episode_min_pregrasp = float("inf")
        episode_min_grasp = float("inf")
        episode_max_lift = 0.0
        episode_max_xy_drift = 0.0
        episode_max_tcp_error = 0.0
        episode_max_pad_penetration = 0.0
        episode_ever_latched = False
        episode_ever_bilateral = False
        episode_ever_any_contact = False
        episode_ever_raw_bilateral = False
        episode_ever_raw_any_contact = False
        episode_ever_close_allowed = False
        episode_ever_fine_close_allowed = False
        episode_ever_xy_aligned = False
        episode_ever_descent_allowed = False
        episode_ever_stage_failure = False
        first_success_step = None
        first_stage_success_step = None
        first_reach_success_step = None
        first_grasp_success_step = None
        first_lift_success_step = None
        first_place_success_step = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            final_info = info
            step_count += 1
            done = terminated or truncated

            episode_min_pregrasp = min(
                episode_min_pregrasp,
                float(info.get("pinch_to_pregrasp_distance", float("inf"))),
            )
            episode_min_grasp = min(
                episode_min_grasp,
                float(info.get("pinch_to_grasp_distance", float("inf"))),
            )
            episode_max_lift = max(
                episode_max_lift,
                float(info.get("object_lift", 0.0)),
            )
            episode_max_xy_drift = max(
                episode_max_xy_drift,
                float(info.get("object_horizontal_drift", 0.0)),
            )
            episode_max_tcp_error = max(
                episode_max_tcp_error,
                float(info.get("tcp_target_error", 0.0)),
            )
            episode_max_pad_penetration = max(
                episode_max_pad_penetration,
                float(info.get("pad_object_penetration", 0.0)),
                float(info.get("max_pad_object_penetration", 0.0)),
            )
            episode_ever_latched = episode_ever_latched or bool(
                info.get("is_grasp_latched", False)
            )
            episode_ever_bilateral = episode_ever_bilateral or bool(
                info.get("has_bilateral_contact", False)
            )
            episode_ever_any_contact = episode_ever_any_contact or bool(
                info.get("has_any_contact", False)
            )
            episode_ever_raw_bilateral = episode_ever_raw_bilateral or bool(
                info.get("has_raw_bilateral_contact", False)
            )
            episode_ever_raw_any_contact = episode_ever_raw_any_contact or bool(
                info.get("has_raw_any_contact", False)
            )
            episode_ever_close_allowed = episode_ever_close_allowed or bool(
                info.get("grasp_close_allowed", False)
            )
            episode_ever_fine_close_allowed = episode_ever_fine_close_allowed or bool(
                info.get("fine_grasp_close_allowed", False)
            )
            episode_ever_xy_aligned = episode_ever_xy_aligned or bool(
                info.get("grasp_xy_aligned", False)
            )
            episode_ever_descent_allowed = episode_ever_descent_allowed or bool(
                info.get("grasp_descent_allowed", False)
            )
            episode_ever_stage_failure = episode_ever_stage_failure or bool(
                info.get("stage_failure", False)
            )

            if first_success_step is None and bool(info.get("is_success", False)):
                first_success_step = step_count
            if first_stage_success_step is None and bool(info.get("stage_success", False)):
                first_stage_success_step = step_count
            if first_reach_success_step is None and bool(info.get("reach_success", False)):
                first_reach_success_step = step_count
            if first_grasp_success_step is None and bool(info.get("grasp_success", False)):
                first_grasp_success_step = step_count
            if first_lift_success_step is None and bool(info.get("lift_success", False)):
                first_lift_success_step = step_count
            if first_place_success_step is None and bool(info.get("place_success", False)):
                first_place_success_step = step_count

            if args.render and args.sleep > 0:
                time.sleep(args.sleep)

        if args.render and bool(final_info.get("is_success", False)) and args.success_hold > 0:
            hold_until = time.time() + args.success_hold
            while time.time() < hold_until:
                env.render()
                time.sleep(min(args.sleep if args.sleep > 0 else 0.03, 0.05))

        final_distance = float(final_info["distance"])
        is_success = float(final_info["is_success"])
        stage_id = int(final_info.get("active_stage", final_info.get("stage", 0)))

        episode_rewards.append(episode_reward)
        final_distances.append(final_distance)
        successes.append(is_success)
        stage_successes.append(float(final_info.get("stage_success", is_success)))
        reach_successes.append(float(final_info.get("reach_success", False)))
        reach_centered.append(float(final_info.get("reach_centered", False)))
        reach_xy_centered.append(float(final_info.get("reach_xy_centered", False)))
        reach_z_centered.append(float(final_info.get("reach_z_centered", False)))
        reach_tcp_tracked.append(float(final_info.get("reach_tcp_tracked", False)))
        grasp_successes.append(float(final_info.get("grasp_success", False)))
        coarse_grasp_successes.append(
            float(final_info.get("coarse_grasp_success", final_info.get("grasp_success", False)))
        )
        stable_grasp_successes.append(float(final_info.get("stable_grasp_success", False)))
        strict_grasp_successes.append(float(final_info.get("strict_grasp_success", False)))
        grasp_stable_counts.append(float(final_info.get("grasp_stable_count", 0)))
        grasp_reason_gripper_closed.append(
            float(final_info.get("grasp_success_gripper_closed", False))
        )
        grasp_reason_bilateral_contact.append(
            float(final_info.get("grasp_success_bilateral_contact", False))
        )
        grasp_reason_distance_ok.append(
            float(final_info.get("grasp_success_distance_ok", False))
        )
        grasp_reason_pose_ok.append(
            float(final_info.get("grasp_success_pose_ok", False))
        )
        grasp_reason_strict_pose_ok.append(
            float(final_info.get("grasp_success_strict_pose_ok", False))
        )
        grasp_reason_penetration_ok.append(
            float(final_info.get("grasp_success_penetration_ok", True))
        )
        lift_successes.append(float(final_info.get("lift_success", False)))
        place_successes.append(float(final_info.get("place_success", False)))
        place_xy_ready.append(float(final_info.get("place_xy_ready", False)))
        place_low_ready.append(float(final_info.get("place_low_ready", False)))
        place_open_ready.append(float(final_info.get("place_open_ready", False)))
        place_opened.append(
            float(final_info.get("place_opened", final_info.get("release_opened", False)))
        )
        place_has_opened.append(float(final_info.get("place_has_opened", False)))
        place_height_ok.append(float(final_info.get("place_height_ok", False)))
        lifts.append(float(final_info.get("object_lift", 0.0)))
        lift_distances.append(float(final_info.get("lift_distance", 0.0)))
        object_to_goal_distances.append(
            float(final_info.get("object_to_goal_distance", 0.0))
        )
        object_to_goal_xy_distances.append(
            float(final_info.get("object_to_goal_xy_distance", 0.0))
        )
        object_goal_z_errors.append(abs(float(final_info.get("object_goal_z_error", 0.0))))
        pad_object_penetrations.append(float(final_info.get("pad_object_penetration", 0.0)))
        max_pad_object_penetrations.append(
            max(
                episode_max_pad_penetration,
                float(final_info.get("max_pad_object_penetration", 0.0)),
            )
        )
        contact_penetration_ok.append(
            float(final_info.get("contact_penetration_ok", True))
        )
        object_uprights.append(float(final_info.get("object_upright", 1.0)))
        object_upright_ok.append(float(final_info.get("object_upright_ok", True)))
        object_still_near_start.append(
            float(final_info.get("object_still_near_start", True))
        )
        xy_drifts.append(float(final_info.get("object_horizontal_drift", 0.0)))
        xy_steps.append(float(final_info.get("object_xy_step", 0.0)))
        object_xy_speeds.append(float(final_info.get("object_xy_speed", 0.0)))
        gripper_states.append(float(final_info.get("gripper_state", 0.0)))
        latched.append(float(final_info.get("is_grasp_latched", False)))
        ever_latched.append(float(episode_ever_latched))
        bilateral.append(float(final_info.get("has_bilateral_contact", False)))
        ever_bilateral.append(float(episode_ever_bilateral))
        unilateral.append(float(final_info.get("has_unilateral_contact", False)))
        any_contact.append(float(final_info.get("has_any_contact", False)))
        ever_any_contact.append(float(episode_ever_any_contact))
        raw_bilateral.append(float(final_info.get("has_raw_bilateral_contact", False)))
        ever_raw_bilateral.append(float(episode_ever_raw_bilateral))
        raw_any_contact.append(float(final_info.get("has_raw_any_contact", False)))
        ever_raw_any_contact.append(float(episode_ever_raw_any_contact))
        final_stages.append(stage_id)
        grasp_phases.append(int(final_info.get("grasp_phase", 0)))
        pregrasp_distances.append(
            float(final_info.get("pinch_to_pregrasp_distance", 0.0))
        )
        pregrasp_xy_errors.append(float(final_info.get("pregrasp_xy_error", 0.0)))
        pregrasp_z_errors.append(abs(float(final_info.get("pregrasp_z_error", 0.0))))
        grasp_distances.append(float(final_info.get("pinch_to_grasp_distance", 0.0)))
        best_pregrasp_distances.append(
            0.0 if np.isinf(episode_min_pregrasp) else episode_min_pregrasp
        )
        best_grasp_distances.append(
            0.0 if np.isinf(episode_min_grasp) else episode_min_grasp
        )
        max_lifts.append(episode_max_lift)
        max_xy_drifts.append(episode_max_xy_drift)
        tcp_errors.append(float(final_info.get("tcp_target_error", 0.0)))
        max_tcp_errors.append(episode_max_tcp_error)
        close_allowed.append(float(final_info.get("grasp_close_allowed", False)))
        ever_close_allowed.append(float(episode_ever_close_allowed))
        fine_close_allowed.append(float(final_info.get("fine_grasp_close_allowed", False)))
        ever_fine_close_allowed.append(float(episode_ever_fine_close_allowed))
        xy_aligned.append(float(final_info.get("grasp_xy_aligned", False)))
        ever_xy_aligned.append(float(episode_ever_xy_aligned))
        descent_allowed.append(float(final_info.get("grasp_descent_allowed", False)))
        ever_descent_allowed.append(float(episode_ever_descent_allowed))
        stage_failures.append(float(final_info.get("stage_failure", False)))
        ever_stage_failures.append(float(episode_ever_stage_failure))
        grasp_xy_errors.append(float(final_info.get("grasp_xy_error", 0.0)))
        grasp_z_errors.append(abs(float(final_info.get("grasp_z_error", 0.0))))
        table_contacts.append(float(final_info.get("table_contact_count", 0)))
        boundary_penalties.append(
            float(final_info.get("object_table_boundary_penalty", 0.0))
        )
        first_success_steps.append(first_success_step)
        first_stage_success_steps.append(first_stage_success_step)
        first_reach_success_steps.append(first_reach_success_step)
        first_grasp_success_steps.append(first_grasp_success_step)
        first_lift_success_steps.append(first_lift_success_step)
        first_place_success_steps.append(first_place_success_step)

        line = (
            f"episode={episode} "
            f"steps={step_count} "
            f"reward={episode_reward:.3f} "
            f"stage={STAGE_NAMES.get(stage_id, str(stage_id))} "
            f"phase={int(final_info.get('grasp_phase', 0))} "
            f"ok={is_success:.0f} "
            f"stage_ok={float(final_info.get('stage_success', is_success)):.0f} "
            f"reach_center={int(final_info.get('reach_centered', False))} "
            f"pre={float(final_info.get('pinch_to_pregrasp_distance', 0.0)):.3f} "
            f"pre_xy={float(final_info.get('pregrasp_xy_error', 0.0)):.3f} "
            f"pre_z={float(final_info.get('pregrasp_z_error', 0.0)):.3f} "
            f"grasp_d={float(final_info.get('pinch_to_grasp_distance', 0.0)):.3f} "
            f"lift={float(final_info.get('object_lift', 0.0)):.4f} "
            f"lift_dist={float(final_info.get('lift_distance', 0.0)):.4f} "
            f"goal_xy={float(final_info.get('object_to_goal_xy_distance', 0.0)):.3f} "
            f"xy_drift={float(final_info.get('object_horizontal_drift', 0.0)):.4f} "
            f"best_pre={best_pregrasp_distances[-1]:.3f} "
            f"best_grasp={best_grasp_distances[-1]:.3f} "
            f"first_ok={first_stage_success_step if first_stage_success_step is not None else '-'} "
            f"latched={int(final_info.get('is_grasp_latched', False))} "
            f"any={int(final_info.get('has_any_contact', False))} "
            f"bilateral={int(final_info.get('has_bilateral_contact', False))} "
            f"unilateral={int(final_info.get('has_unilateral_contact', False))} "
            f"gripper={float(final_info.get('gripper_state', 0.0)):.2f}"
        )
        if args.show_safety:
            line += (
                f" table_contacts={int(final_info.get('table_contact_count', 0))} "
                f"boundary_penalty={float(final_info.get('object_table_boundary_penalty', 0.0)):.4f} "
                f"clearance_penalty={float(final_info.get('table_clearance_penalty', 0.0)):.4f} "
                f"side_penalty={float(final_info.get('table_side_penalty', 0.0)):.4f} "
                f"low_penalty={float(final_info.get('low_away_from_object_penalty', 0.0)):.4f} "
                f"obj_v={float(final_info.get('object_xy_speed', 0.0)):.4f} "
                f"tcp_err={float(final_info.get('tcp_target_error', 0.0)):.4f} "
                f"fail={int(final_info.get('stage_failure', False))} "
                f"raw_any={int(final_info.get('has_raw_any_contact', False))} "
                f"raw_bi={int(final_info.get('has_raw_bilateral_contact', False))} "
                f"pen={float(final_info.get('pad_object_penetration', 0.0)):.4f} "
                f"pen_max={max_pad_object_penetrations[-1]:.4f} "
                f"pen_ok={int(final_info.get('contact_penetration_ok', True))} "
                f"upright={float(final_info.get('object_upright', 1.0)):.3f} "
                f"upright_ok={int(final_info.get('object_upright_ok', True))} "
                f"near_start={int(final_info.get('object_still_near_start', True))} "
                f"close_allowed={int(final_info.get('grasp_close_allowed', False))} "
                f"fine_close={int(final_info.get('fine_grasp_close_allowed', False))} "
                f"align={int(final_info.get('grasp_xy_aligned', False))} "
                f"desc={int(final_info.get('grasp_descent_allowed', False))} "
                f"grasp_xy={float(final_info.get('grasp_xy_error', 0.0)):.4f} "
                f"grasp_z={float(final_info.get('grasp_z_error', 0.0)):.4f} "
                f"coarse={int(final_info.get('coarse_grasp_success', False))} "
                f"stable={int(final_info.get('stable_grasp_success', False))} "
                f"stable_count={int(final_info.get('grasp_stable_count', 0))} "
                f"g_closed={int(final_info.get('grasp_success_gripper_closed', False))} "
                f"g_bi={int(final_info.get('grasp_success_bilateral_contact', False))} "
                f"g_dist={int(final_info.get('grasp_success_distance_ok', False))} "
                f"g_pose={int(final_info.get('grasp_success_pose_ok', False))} "
                f"g_strict_pose={int(final_info.get('grasp_success_strict_pose_ok', False))} "
                f"g_pen={int(final_info.get('grasp_success_penetration_ok', True))}"
            )
        print(line)

    print("\n评估汇总")
    print(
        "score:",
        f"reward={mean_or_zero(episode_rewards):.3f}",
        f"success={mean_or_zero(successes):.3f}",
        f"stage_success={mean_or_zero(stage_successes):.3f}",
    )
    print(
        "stage:",
        f"reach={mean_or_zero(reach_successes):.3f}",
        f"grasp={mean_or_zero(grasp_successes):.3f}",
        f"coarse={mean_or_zero(coarse_grasp_successes):.3f}",
        f"stable={mean_or_zero(stable_grasp_successes):.3f}",
        f"strict={mean_or_zero(strict_grasp_successes):.3f}",
        f"lift={mean_or_zero(lift_successes):.3f}",
        f"place={mean_or_zero(place_successes):.3f}",
    )
    print(
        "reach_detail:",
        f"centered={mean_or_zero(reach_centered):.3f}",
    )
    print()
    print(
        "distance:",
        f"task_final={mean_or_zero(final_distances):.4f}",
        f"pre_xy={mean_or_zero(pregrasp_xy_errors):.4f}",
        f"pre_abs_z={mean_or_zero(pregrasp_z_errors):.4f}",
        f"grasp_xy={mean_or_zero(grasp_xy_errors):.4f}",
        f"grasp_abs_z={mean_or_zero(grasp_z_errors):.4f}",
        f"goal_xy={mean_or_zero(object_to_goal_xy_distances):.4f}",
    )
    print()
    print(
        "grasp:",
        f"align_final={mean_or_zero(xy_aligned):.3f}",
        f"latched_final={mean_or_zero(latched):.3f}",
        f"latched_ever={mean_or_zero(ever_latched):.3f}",
        f"bilateral_final={mean_or_zero(bilateral):.3f}",
        f"upright_ok={mean_or_zero(object_upright_ok):.3f}",
        f"gripper={mean_or_zero(gripper_states):.3f}",
    )
    print(
        "grasp_reason:",
        f"closed={mean_or_zero(grasp_reason_gripper_closed):.3f}",
        f"bilateral={mean_or_zero(grasp_reason_bilateral_contact):.3f}",
        f"distance={mean_or_zero(grasp_reason_distance_ok):.3f}",
        f"pose15={mean_or_zero(grasp_reason_pose_ok):.3f}",
        f"strict_pose12={mean_or_zero(grasp_reason_strict_pose_ok):.3f}",
        f"penetration={mean_or_zero(grasp_reason_penetration_ok):.3f}",
        f"stable_count={mean_or_zero(grasp_stable_counts):.1f}",
        f"stable_count_max={max_or_zero(grasp_stable_counts):.0f}",
    )
    print(
        "grasp_quality:",
        f"upright={mean_or_zero(object_uprights):.3f}",
        f"near_start={mean_or_zero(object_still_near_start):.3f}",
        f"pen_ok={mean_or_zero(contact_penetration_ok):.3f}",
    )
    print()
    print(
        "contact:",
        f"raw_any_final={mean_or_zero(raw_any_contact):.3f}",
        f"raw_bi_final={mean_or_zero(raw_bilateral):.3f}",
        f"raw_bi_ever={mean_or_zero(ever_raw_bilateral):.3f}",
    )
    print(
        "lift:",
        f"final={mean_or_zero(lifts):.4f}",
        f"max={max_or_zero(max_lifts):.4f}",
    )
    print(
        "place:",
        f"success={mean_or_zero(place_successes):.3f}",
        f"height_ok={mean_or_zero(place_height_ok):.3f}",
        f"goal_xy={mean_or_zero(object_to_goal_xy_distances):.4f}",
        f"goal_abs_z={mean_or_zero(object_goal_z_errors):.4f}",
    )
    print(
        "place_detail:",
        f"xy_ready={mean_or_zero(place_xy_ready):.3f}",
        f"low_ready={mean_or_zero(place_low_ready):.3f}",
        f"open_ready={mean_or_zero(place_open_ready):.3f}",
        f"opened={mean_or_zero(place_opened):.3f}",
        f"opened_ever={mean_or_zero(place_has_opened):.3f}",
    )
    print(
        "safety:",
        f"fail_final={mean_or_zero(stage_failures):.3f}",
        f"xy_drift={mean_or_zero(xy_drifts):.4f}",
        f"obj_xy_speed={mean_or_zero(object_xy_speeds):.4f}",
        f"table_contacts={mean_or_zero(table_contacts):.3f}",
        f"pad_pen={mean_or_zero(pad_object_penetrations):.4f}",
        f"pad_pen_ep={mean_or_zero(max_pad_object_penetrations):.4f}",
        f"pad_pen_max={max_or_zero(max_pad_object_penetrations):.4f}",
    )
    print(
        "first_success_step:",
        f"stage={fmt_first_step(first_stage_success_steps)}",
        f"grasp={fmt_first_step(first_grasp_success_steps)}",
        f"lift={fmt_first_step(first_lift_success_steps)}",
        f"place={fmt_first_step(first_place_success_steps)}",
    )
    stage_counts = {
        STAGE_NAMES.get(stage, str(stage)): final_stages.count(stage)
        for stage in sorted(set(final_stages))
    }
    print("final_stage_counts:", stage_counts)
    phase_counts = {
        str(phase): grasp_phases.count(phase)
        for phase in sorted(set(grasp_phases))
    }
    print("final_grasp_phase_counts:", phase_counts)

    env.close()


if __name__ == "__main__":
    main()
