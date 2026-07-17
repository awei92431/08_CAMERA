"""Phase 2 shadow and A/B evaluation of deployable task supervision."""

import argparse
import ast
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import fourc2  # noqa: E402,F401
from eval_full_visual_closed_loop import (  # noqa: E402
    CHECKPOINT, CONFIG, ENV_ID, HEIGHT, WIDTH, acquire_rgbd, hold,
    load_localization_config, mid, move_arm,
)
from fourc2.object_estimate import ObjectEstimate, TcpObjectTracker  # noqa: E402

OUTPUT = ROOT / "outputs" / "phase2_task_supervisor"


def tracker_update(tracker, env, confirmed):
    return tracker.update(
        env.data.site_xpos[env.pinch_site_id],
        env.data.site_xmat[env.pinch_site_id].reshape(3, 3), confirmed)


def event_steps(supervisor, seen, step):
    for index, event in enumerate(supervisor.events):
        if index not in seen:
            seen[index] = step
    out = {}
    for index, at in seen.items():
        out.setdefault(supervisor.events[index]["event"], at)
    return out


def run_mode(mode, seeds, model, latch_physics=True):
    wrapped = gym.make(
        ENV_ID, object_observation_mode="rgbd", fsm_mode=mode,
        simulated_latch_physics=latch_physics, max_tcp_lead=.03,
        ik_posture_mode="off", disable_env_checker=True)
    env = wrapped.unwrapped
    observe_key = mid(mujoco.mjtObj.mjOBJ_KEY, env.model, "camera_observe")
    observe_q = env.model.key_qpos[observe_key, env.arm_qpos_ids].copy()
    ids = (mid(mujoco.mjtObj.mjOBJ_CAMERA, env.model, "eye_in_hand_color"),
           mid(mujoco.mjtObj.mjOBJ_CAMERA, env.model, "eye_in_hand_depth"),
           mid(mujoco.mjtObj.mjOBJ_BODY, env.model, "base"))
    from fourc2.camera_geometry import intrinsics_from_fovy
    intrinsics = (intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[ids[0]]),
                  intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[ids[1]]))
    config = load_localization_config(CONFIG)
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    rows = []; shadow_rows = []
    try:
        for seed in seeds:
            wrapped.reset(seed=seed); home = env.data.qpos[env.arm_qpos_ids].copy()
            initial_object = env.data.site_xpos[env.object_site_id].copy()
            initial_goal = env.data.site_xpos[env.goal_site_id].copy()
            move_arm(env, observe_q); hold(env, observe_q)
            localized, _, estimate_world, _ = acquire_rgbd(
                env, renderer, ids, intrinsics, config)
            move_arm(env, home); hold(env, home); env._sync_mocap_target_to_tcp()
            tracker = None if estimate_world is None else TcpObjectTracker(estimate_world)
            seq = 0; done = False; step = 0; total_return = 0.
            reach = grasp = lift = place = entered = False
            latch_ever = False; latch_first = None; release_first = None
            privileged_events = {}; previous_stage = int(env.stage)
            supervisor_seen = {}; supervisor_events = {}
            consistency_failures = 0; empty_grasp = False; dropped = slipped = False
            max_hold_error = 0.; final_info = env._get_info()
            while step < 900 and not done:
                if tracker is None:
                    break
                # Propagation handoff is supervisor confirmation, never sim latch.
                propagation_handoff = (
                    int(env.stage) >= 2 if mode == "privileged_fsm"
                    else env.task_supervisor.grasp_confirmed
                )
                position = tracker_update(tracker, env, propagation_handoff)
                seq += 1
                source = ("tcp_fk_propagated" if tracker.tcp_local_offset is not None
                          else "rgbd_visual_hold")
                estimate = ObjectEstimate(
                    position, env.data.time, True, 1.0, source,
                    f"{mode}-seed-{seed:03d}-{seq:06d}")
                env.publish_object_estimate(estimate)
                authority_step = env.control_step_index
                try:
                    env.update_task_supervisor()
                except Exception as exc:
                    env.task_supervisor.failed = True
                    env.task_supervisor.failure_reason = f"{type(exc).__name__}:{exc}"
                    break
                supervisor_events = event_steps(
                    env.task_supervisor, supervisor_seen, step)
                if mode == "deployable_fsm" and env.task_supervisor.task_complete:
                    final_info = env._get_info(); break
                observation = env.control_observation()
                action, _ = model.predict(observation, deterministic=True)
                stage_before = int(env.stage)
                _, reward, terminated, truncated, info = wrapped.step(action)
                step += 1; total_return += float(reward)
                done = bool(terminated or truncated); final_info = info
                current_stage = int(env.stage)
                if current_stage != previous_stage:
                    labels = {(0,1): "reach_to_grasp", (1,2): "grasp_to_lift",
                              (2,5): "lift_to_place"}
                    name = labels.get((previous_stage, current_stage))
                    if name is not None: privileged_events.setdefault(name, step)
                previous_stage = current_stage
                if env.is_grasp_latched and latch_first is None: latch_first = step
                if env.release_has_opened and release_first is None: release_first = step
                latch_ever |= bool(env.is_grasp_latched)
                reach |= bool(info.get("reach_success")); grasp |= bool(info.get("grasp_success"))
                lift |= bool(info.get("lift_success")); place |= bool(info.get("place_success"))
                entered |= stage_before == 5 or current_stage == 5
                truth_obj = env.data.site_xpos[env.object_site_id].copy()
                hold_error = float(np.linalg.norm(tracker.world_position - truth_obj))
                max_hold_error = max(max_hold_error, hold_error)
                dropped |= bool(truth_obj[2] < env.table_top_z + env.object_half_size - .01)
                slipped |= bool(env.task_supervisor.grasp_confirmed
                                and not env.task_supervisor.release_commanded
                                and not env.is_grasp_latched)
                uses = [u for u in env.object_estimate_authority.uses_for_step(
                    authority_step) if u["consumer"] in {
                        "task_supervisor", "ppo_observation", "reach_safety",
                        "grasp_safety", "place_servo", "place_descent", "place_release"}]
                keys = {(u["estimate_id"], u["timestamp"]) for u in uses}
                consistency_failures += int(len(keys) > 1)
            supervisor_events = event_steps(env.task_supervisor, supervisor_seen, step)
            truth_now = env._get_info(); final_info = truth_now
            full = bool(truth_now.get("place_success", False))
            if mode == "privileged_fsm":
                full = bool(truth_now.get("is_success", False) or place)
            confirmed = bool(env.task_supervisor.grasp_confirmed)
            empty_grasp = bool(confirmed and not latch_ever)
            row = {
                "seed": seed, "mode": mode, "latch_physics": latch_physics,
                "rgbd_valid": bool(localized.valid), "episode_steps": step,
                "episode_return": total_return, "full_success": full,
                "reach_success": reach, "grasp_success": grasp,
                "lift_success": lift, "entered_place": entered,
                "place_success": place or bool(truth_now.get("place_success")),
                "supervisor_grasp_confirmed": confirmed,
                "supervisor_task_complete": bool(env.task_supervisor.task_complete),
                "supervisor_failed": bool(env.task_supervisor.failed),
                "supervisor_failure_reason": env.task_supervisor.failure_reason,
                "sim_latch_ever": latch_ever, "empty_grasp_confirmation": empty_grasp,
                "object_dropped": dropped, "hold_lost_or_unlatched": slipped,
                "max_tracker_truth_error_m_evaluator_only": max_hold_error,
                "source_consistency_failures": consistency_failures,
                "supervisor_events": supervisor_events,
                "privileged_events": privileged_events,
                "sim_latch_first_step": latch_first,
                "privileged_release_first_step": release_first,
                "initial_object_xyz": initial_object.tolist(),
                "initial_goal_xyz": initial_goal.tolist(),
                "final_goal_xy_m": float(truth_now["object_to_goal_xy_distance"]),
                "supervisor_stage": int(env.task_supervisor.stage),
                "supervisor_grasp_phase": int(env.task_supervisor.grasp_phase),
                "supervisor_last_diagnostics": env.task_supervisor.last_diagnostics,
                "final_gripper_state": env.deployable_gripper_state().__dict__,
            }
            rows.append(row)
            if mode == "privileged_fsm":
                for event in ["reach_to_grasp", "grasp_confirmed", "grasp_to_lift",
                              "lift_to_place", "release_commanded", "release_complete"]:
                    old = (latch_first if event in ("grasp_confirmed", "grasp_to_lift")
                           else release_first if event == "release_commanded"
                           else privileged_events.get(event))
                    new = supervisor_events.get(event)
                    shadow_rows.append({"seed": seed, "event": event,
                        "privileged_step": old, "supervisor_step": new,
                        "delay_steps": None if old is None or new is None else new-old,
                        "missed": old is not None and new is None,
                        "false_positive": old is None and new is not None})
            print(mode, seed, full, confirmed, env.task_supervisor.task_complete,
                  env.task_supervisor.failure_reason, flush=True)
    finally:
        renderer.close(); wrapped.close()
    return rows, shadow_rows


def summarize(rows):
    n = len(rows)
    keys = ["rgbd_valid", "full_success", "reach_success", "grasp_success",
            "lift_success", "entered_place", "place_success",
            "supervisor_grasp_confirmed", "supervisor_task_complete"]
    out = {key: {"count": int(sum(r[key] for r in rows)),
                 "rate": float(np.mean([r[key] for r in rows]))} for key in keys}
    out.update({"episodes": n,
        "source_consistency_failures": int(sum(r["source_consistency_failures"] for r in rows)),
        "empty_grasp_confirmations": int(sum(r["empty_grasp_confirmation"] for r in rows)),
        "drop_episodes": int(sum(r["object_dropped"] for r in rows)),
        "hold_lost_or_unlatched_episodes": int(sum(r["hold_lost_or_unlatched"] for r in rows)),
        "supervisor_failure_reasons": dict(Counter(r["supervisor_failure_reason"] for r in rows
                                                    if r["supervisor_failure_reason"])),
        "steps_mean": float(np.mean([r["episode_steps"] for r in rows]))})
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--latch-off-episodes", type=int, default=10); args = parser.parse_args()
    seeds = list(range(args.episodes)); model = PPO.load(CHECKPOINT, device="cpu")
    privileged, shadow = run_mode("privileged_fsm", seeds, model)
    deployable, _ = run_mode("deployable_fsm", seeds, model)
    latch_off, _ = run_mode("deployable_fsm", list(range(args.latch_off_episodes)),
                            model, latch_physics=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "privileged_episodes.csv", privileged)
    write_csv(OUTPUT / "deployable_episodes.csv", deployable)
    write_csv(OUTPUT / "shadow_event_differences.csv", shadow)
    write_csv(OUTPUT / "latch_disabled_episodes.csv", latch_off)
    comparison=[]
    for a,b in zip(privileged,deployable):
        comparison.append({"seed":a["seed"],"same_initial_object":a["initial_object_xyz"]==b["initial_object_xyz"],
            "same_initial_goal":a["initial_goal_xyz"]==b["initial_goal_xyz"],
            "privileged_success":a["full_success"],"deployable_success":b["full_success"],
            "outcome_identical":a["full_success"]==b["full_success"],
            "privileged_steps":a["episode_steps"],"deployable_steps":b["episode_steps"]})
    write_csv(OUTPUT / "seed_comparison.csv", comparison)
    delays=[r["delay_steps"] for r in shadow if r["delay_steps"] is not None]
    payload={"configuration":{"seeds":seeds,"checkpoint":str(CHECKPOINT)},
        "privileged_fsm":summarize(privileged),"deployable_fsm":summarize(deployable),
        "shadow":{"rows":len(shadow),"matched_events":len(delays),
          "delay_steps_mean":None if not delays else float(np.mean(delays)),
          "delay_steps_median":None if not delays else float(np.median(delays)),
          "missed_events":int(sum(r["missed"] for r in shadow)),
          "false_positive_events":int(sum(r["false_positive"] for r in shadow))},
        "latch_disabled":summarize(latch_off),
        "seed_comparison":{"identical":int(sum(r["outcome_identical"] for r in comparison)),
                           "same_initial_states":int(sum(r["same_initial_object"] and r["same_initial_goal"] for r in comparison))}}
    (OUTPUT / "summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__": main()
