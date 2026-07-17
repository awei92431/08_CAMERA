import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "glfw")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fourc2  # noqa: F401
from fourc2.envs.allstage import (
    GRASP_PHASE_ALIGN,
    GRASP_PHASE_CLOSE,
    GRASP_PHASE_CONFIRM,
    GRASP_PHASE_DESCEND,
    STAGE_GRASP,
    STAGE_LIFT,
)


def set_gripper(env, normalized):
    normalized = float(np.clip(normalized, 0.0, 1.0))
    env.data.ctrl[env.gripper_actuator_id] = env.gripper_ctrl_low + normalized * (
        env.gripper_ctrl_high - env.gripper_ctrl_low
    )


def set_tcp_target(env, target_position):
    target_position = np.asarray(target_position, dtype=np.float64)
    target_position = env._safe_tcp_target_pos(target_position)
    env.tcp_target_pos = target_position.copy()
    env.data.mocap_pos[env.tcp_mocap_id] = env.tcp_target_pos
    env.data.mocap_quat[env.tcp_mocap_id] = env.tcp_target_quat


def empty_stats():
    return {
        "max_lift": 0.0,
        "max_penetration": 0.0,
        "max_table_contacts": 0,
        "min_upright": 1.0,
        "ever_raw_bilateral": False,
        "ever_raw_any": False,
        "ever_latched": False,
        "table_pairs": set(),
        "object_pairs": set(),
    }


def geom_name(env, geom_id):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    if name is None:
        return f"geom#{int(geom_id)}"
    return name


def update_table_pairs(env, stats):
    if env.table_geom_id < 0:
        return

    for contact_id in range(env.data.ncon):
        contact = env.data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if env.table_geom_id not in (geom1, geom2):
            continue

        other_geom = geom2 if geom1 == env.table_geom_id else geom1
        if other_geom == env.object_geom_id:
            continue
        stats["table_pairs"].add(
            f"{geom_name(env, env.table_geom_id)}:{geom_name(env, other_geom)}"
        )


def update_object_pairs(env, stats):
    if env.object_geom_id < 0:
        return

    for contact_id in range(env.data.ncon):
        contact = env.data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if env.object_geom_id not in (geom1, geom2):
            continue

        other_geom = geom2 if geom1 == env.object_geom_id else geom1
        stats["object_pairs"].add(
            f"{geom_name(env, env.object_geom_id)}:{geom_name(env, other_geom)}"
        )


def update_stats(env, stats, info):
    stats["max_lift"] = max(stats["max_lift"], float(info.get("object_lift", 0.0)))
    stats["max_penetration"] = max(
        stats["max_penetration"],
        float(info.get("pad_object_penetration", 0.0)),
    )
    stats["max_table_contacts"] = max(
        stats["max_table_contacts"],
        int(info.get("table_contact_count", 0)),
    )
    stats["min_upright"] = min(
        stats["min_upright"],
        float(info.get("object_upright", 1.0)),
    )
    stats["ever_raw_bilateral"] = stats["ever_raw_bilateral"] or bool(
        info.get("has_raw_bilateral_contact", False)
    )
    stats["ever_raw_any"] = stats["ever_raw_any"] or bool(
        info.get("has_raw_any_contact", False)
    )
    stats["ever_latched"] = stats["ever_latched"] or bool(
        info.get("is_grasp_latched", False)
    )
    if int(info.get("table_contact_count", 0)) > 0:
        update_table_pairs(env, stats)
    update_object_pairs(env, stats)


def raw_steps(env, steps, gripper_norm, args, stats):
    for step_id in range(int(steps)):
        set_gripper(env, gripper_norm)
        env._neutralize_arm_actuators()
        mujoco.mj_step(env.model, env.data)
        if args.virtual_latch:
            env._update_grasp_latch(
                update_counter=(step_id + 1) % env.frame_skip == 0
            )

        info = env._get_info()
        update_stats(env, stats, info)

        if args.render and step_id % args.render_every == 0:
            env.render()
            if args.sleep > 0.0:
                time.sleep(args.sleep)


def move_to_target(env, target_position, max_steps, tolerance, gripper_norm, args, stats):
    target_position = np.asarray(target_position, dtype=np.float64)
    reached = False
    for _ in range(int(max_steps)):
        pinch_position = env.data.site_xpos[env.pinch_site_id].copy()
        delta = target_position - pinch_position
        distance = float(np.linalg.norm(delta))
        if distance <= tolerance:
            reached = True
            break

        step_delta = delta.copy()
        step_norm = float(np.linalg.norm(step_delta))
        if step_norm > args.servo_step:
            step_delta *= args.servo_step / step_norm
        set_tcp_target(env, env.tcp_target_pos + step_delta)
        raw_steps(env, env.frame_skip, gripper_norm, args, stats)

    return reached


def print_phase(name, env, stats, show_safety):
    info = env._get_info()
    message = (
        f"{name}: "
        f"grasp_d={float(info.get('pinch_to_grasp_distance', 0.0)):.4f} "
        f"lift={float(info.get('object_lift', 0.0)):.4f} "
        f"raw_bi={int(info.get('has_raw_bilateral_contact', False))} "
        f"raw_any={int(info.get('has_raw_any_contact', False))} "
        f"latched={int(info.get('is_grasp_latched', False))} "
        f"upright={float(info.get('object_upright', 1.0)):.3f} "
        f"pen={float(info.get('pad_object_penetration', 0.0)):.4f}"
    )
    if show_safety:
        message += (
            f" max_lift={stats['max_lift']:.4f}"
            f" max_pen={stats['max_penetration']:.4f}"
            f" max_table={stats['max_table_contacts']}"
            f" min_upright={stats['min_upright']:.3f}"
            f" xy_drift={float(info.get('object_horizontal_drift', 0.0)):.4f}"
        )
        if stats["table_pairs"]:
            pairs = ",".join(sorted(stats["table_pairs"])[:4])
            message += f" table_pairs={pairs}"
        if stats["object_pairs"]:
            pairs = ",".join(sorted(stats["object_pairs"])[:6])
            message += f" object_pairs={pairs}"
    print(message)


def run_episode(env, episode, args):
    _, _ = env.reset(seed=args.seed + episode)
    env.stage = STAGE_GRASP
    env.is_grasp_latched = False
    env.bilateral_contact_steps = 0
    env._set_grasp_phase(GRASP_PHASE_ALIGN)
    # Keep the reset controller's downward-facing target orientation.  Syncing
    # the complete pose here would lock in any transient wrist tilt left by the
    # reset move, which can make an open finger strike the cube while descending.
    target_quat = env.tcp_target_quat.copy()
    env._sync_mocap_target_to_tcp()
    env.tcp_target_quat = target_quat
    env.data.mocap_quat[env.tcp_mocap_id] = target_quat
    set_gripper(env, 0.0)
    mujoco.mj_forward(env.model, env.data)

    object_start = env.data.site_xpos[env.object_site_id].copy()
    pregrasp_target = object_start + np.array(
        [0.0, 0.0, env.pregrasp_height],
        dtype=np.float64,
    )
    grasp_target = object_start + np.array(
        [0.0, 0.0, env.grasp_height_offset],
        dtype=np.float64,
    )

    stats = empty_stats()
    print(f"\nepisode={episode}")

    pregrasp_reached = move_to_target(
        env,
        pregrasp_target,
        args.pregrasp_steps,
        args.pregrasp_tolerance,
        0.0,
        args,
        stats,
    )
    pregrasp_settle_steps = int(
        round(args.pregrasp_settle_seconds / env.model.opt.timestep)
    )
    raw_steps(env, pregrasp_settle_steps, 0.0, args, stats)
    print_phase(
        f"pregrasp reached={int(pregrasp_reached)}",
        env,
        stats,
        args.show_safety,
    )

    env._set_grasp_phase(GRASP_PHASE_DESCEND)
    descend_reached = move_to_target(
        env,
        grasp_target,
        args.descend_steps,
        args.descend_tolerance,
        0.0,
        args,
        stats,
    )
    print_phase(
        f"descend reached={int(descend_reached)}",
        env,
        stats,
        args.show_safety,
    )

    env._set_grasp_phase(GRASP_PHASE_CLOSE)
    for step_id in range(args.close_steps):
        gripper = (step_id + 1) / max(1, args.close_steps)
        raw_steps(env, env.frame_skip, gripper, args, stats)
    print_phase("close", env, stats, args.show_safety)

    env._set_grasp_phase(GRASP_PHASE_CONFIRM)
    hold_raw_steps = int(round(args.hold_seconds / env.model.opt.timestep))
    raw_steps(env, hold_raw_steps, 1.0, args, stats)
    hold_info = env._get_info()
    print_phase("hold", env, stats, args.show_safety)

    env.stage = STAGE_LIFT
    lift_target = env.data.site_xpos[env.pinch_site_id].copy()
    lift_target[2] += args.lift_height
    lift_reached = move_to_target(
        env,
        lift_target,
        args.lift_steps,
        args.lift_tolerance,
        1.0,
        args,
        stats,
    )
    print_phase(
        f"lift reached={int(lift_reached)}",
        env,
        stats,
        args.show_safety,
    )

    min_lift = args.min_lift
    if min_lift is None:
        min_lift = env.lift_height
    penetration_limit = env.grasp_success_penetration_tolerance
    lift_ok = stats["max_lift"] >= min_lift
    penetration_ok = stats["max_penetration"] <= penetration_limit
    table_ok = stats["max_table_contacts"] == 0
    upright_ok = stats["min_upright"] >= env.grasp_upright_threshold
    hold_raw_bilateral = bool(hold_info.get("has_raw_bilateral_contact", False))
    oracle_ok = (
        lift_ok
        and penetration_ok
        and table_ok
        and upright_ok
        and hold_raw_bilateral
    )

    print(
        "episode_result:",
        f"ok={int(oracle_ok)}",
        f"lift_ok={int(lift_ok)}",
        f"hold_raw_bi={int(hold_raw_bilateral)}",
        f"raw_bi_ever={int(stats['ever_raw_bilateral'])}",
        f"max_lift={stats['max_lift']:.4f}",
        f"min_lift={min_lift:.4f}",
        f"max_pen={stats['max_penetration']:.4f}",
        f"pen_limit={penetration_limit:.4f}",
        f"max_table={stats['max_table_contacts']}",
        f"min_upright={stats['min_upright']:.3f}",
        f"hold_strict_grasp={int(hold_info.get('strict_grasp_success', False))}",
    )
    if stats["table_pairs"]:
        print("table_contact_pairs:", ", ".join(sorted(stats["table_pairs"])))
    if stats["object_pairs"]:
        print("object_contact_pairs:", ", ".join(sorted(stats["object_pairs"])))
    return oracle_ok, stats


def main():
    parser = argparse.ArgumentParser(
        description="Scripted oracle test for 4C2 grasp contacts without RL.",
    )
    parser.add_argument("--env-id", default="My4C2GraspStageCube3cm-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--show-safety", action="store_true")
    parser.add_argument(
        "--virtual-latch",
        action="store_true",
        help="Also enable the env's artificial object latch. Off by default.",
    )
    parser.add_argument("--servo-step", type=float, default=0.010)
    parser.add_argument("--pregrasp-steps", type=int, default=220)
    parser.add_argument("--descend-steps", type=int, default=180)
    parser.add_argument("--close-steps", type=int, default=80)
    parser.add_argument("--pregrasp-settle-seconds", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--min-lift", type=float, default=None)
    parser.add_argument("--pregrasp-tolerance", type=float, default=0.010)
    parser.add_argument("--descend-tolerance", type=float, default=0.006)
    parser.add_argument("--lift-tolerance", type=float, default=0.010)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        render_mode="human" if args.render else None,
    ).unwrapped

    print("env_id:", args.env_id)
    print("episodes:", args.episodes)
    print("virtual_latch:", args.virtual_latch)
    print("render:", args.render)
    print("pregrasp_settle_seconds:", args.pregrasp_settle_seconds)
    print("hold_seconds:", args.hold_seconds)

    results = []
    max_lifts = []
    max_penetrations = []
    max_table_contacts = []
    min_uprights = []

    try:
        for episode in range(args.episodes):
            ok, stats = run_episode(env, episode, args)
            results.append(float(ok))
            max_lifts.append(stats["max_lift"])
            max_penetrations.append(stats["max_penetration"])
            max_table_contacts.append(stats["max_table_contacts"])
            min_uprights.append(stats["min_upright"])
    finally:
        env.close()

    print("\noracle_summary")
    print(
        "score:",
        f"ok={float(np.mean(results)) if results else 0.0:.3f}",
        f"max_lift_mean={float(np.mean(max_lifts)) if max_lifts else 0.0:.4f}",
        f"max_lift_max={float(np.max(max_lifts)) if max_lifts else 0.0:.4f}",
    )
    print(
        "safety:",
        f"max_pen_mean={float(np.mean(max_penetrations)) if max_penetrations else 0.0:.4f}",
        f"max_pen_max={float(np.max(max_penetrations)) if max_penetrations else 0.0:.4f}",
        f"table_max={int(np.max(max_table_contacts)) if max_table_contacts else 0}",
        f"upright_min={float(np.min(min_uprights)) if min_uprights else 1.0:.3f}",
    )


if __name__ == "__main__":
    main()
