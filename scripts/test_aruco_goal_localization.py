#!/usr/bin/env python3
"""Timing, freeze, fail-closed and coordinate tests for visual episode goals."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import fourc2  # noqa: E402,F401
from eval_full_visual_closed_loop import (  # noqa: E402
    capture_aruco_episode_goal, mid, move_arm,
)
from fourc2.aruco_goal_localizer import (  # noqa: E402
    GoalCaptureSession, GoalCaptureState, load_aruco_goal_config,
)
from fourc2.camera_geometry import intrinsics_from_fovy  # noqa: E402
from fourc2.goal_estimate import (  # noqa: E402
    GoalEstimate, GoalEstimateAuthority, GoalEstimateUnavailable,
)

ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
CONFIG = ROOT / "configs" / "aruco_goal_localization.json"
OUTPUT = ROOT / "outputs" / "aruco_goal_closed_loop"


def fake_estimate(position, valid=True, marker_id=0, reason=None):
    return GoalEstimate(
        position=np.asarray(position if valid else [np.nan]*3),
        timestamp=1.0, valid=valid, confidence=1.0 if valid else 0.0,
        source="aruco_test_candidate", estimate_id="candidate",
        marker_id=marker_id, frame="base", failure_reason=reason,
    )


def state_machine_tests(config):
    candidate = fake_estimate([-0.45, 0.0, 0.315])

    ignored = GoalCaptureSession(config)
    ignored.submit_detection(candidate, 0.0)
    assert not ignored.goal_is_frozen and not ignored.valid_estimates

    moving = GoalCaptureSession(config)
    moving.reached_observe_command()
    for step in range(config.observe_stable_steps + 20):
        moving.update_stability(np.zeros(6), np.ones(6), np.zeros(6), step*.01)
    assert moving.state == GoalCaptureState.WAIT_OBSERVE_STABLE

    capture = GoalCaptureSession(config)
    capture.reached_observe_command()
    for step in range(200):
        capture.update_stability(np.zeros(6), np.zeros(6), np.zeros(6), step*.01)
        if capture.state == GoalCaptureState.CAPTURE_GOAL:
            break
    assert capture.state == GoalCaptureState.CAPTURE_GOAL
    for frame in range(config.required_valid_frames):
        capture.submit_detection(candidate, 2.0 + frame*.01)
    assert capture.goal_is_frozen
    frozen = capture.frozen_estimate.position.copy()
    capture.submit_detection(fake_estimate([-0.20, 0.20, 0.330]), 4.0)
    assert np.array_equal(capture.frozen_estimate.position, frozen)

    failed = GoalCaptureSession(config)
    failed.reached_observe_command()
    failed.state = GoalCaptureState.CAPTURE_GOAL
    for frame in range(config.maximum_capture_frames):
        failed.submit_detection(
            fake_estimate([0, 0, 0], valid=False, reason="marker_not_detected"),
            frame*.01)
    assert failed.state == GoalCaptureState.FAILED
    assert not failed.goal_is_frozen
    authority = GoalEstimateAuthority("aruco")
    no_fallback = False
    try:
        authority.require(0.0, "ppo_observation", 0)
    except GoalEstimateUnavailable as exc:
        no_fallback = exc.reason == "missing"
    assert no_fallback
    truth_rejected = False
    try:
        authority.publish(GoalEstimate(
            position=[0.45, 0.0, 0.315], timestamp=0.0, valid=True,
            confidence=1.0, source="ground_truth_simulation",
            estimate_id="forbidden", frame="world"))
    except ValueError:
        truth_rejected = True
    assert truth_rejected
    return {
        "non_observe_detection_ignored": True,
        "moving_arm_cannot_enter_capture": True,
        "stable_multiframe_freeze": True,
        "frozen_goal_cannot_be_overwritten": True,
        "missing_marker_fails_closed": True,
        "aruco_authority_has_no_truth_fallback": True,
    }


def run_render_tests(episodes, seed_offset, config):
    wrapped = gym.make(
        ENV_ID, object_observation_mode="ground_truth",
        goal_observation_mode="aruco", disable_env_checker=True)
    env = wrapped.unwrapped
    observe_key = mid(mujoco.mjtObj.mjOBJ_KEY, env.model, "camera_observe")
    observe_q = env.model.key_qpos[observe_key, env.arm_qpos_ids].copy()
    color_id = mid(mujoco.mjtObj.mjOBJ_CAMERA, env.model, "eye_in_hand_color")
    base_id = mid(mujoco.mjtObj.mjOBJ_BODY, env.model, "base")
    intrinsics = intrinsics_from_fovy(640, 360, env.model.cam_fovy[color_id])
    renderer = mujoco.Renderer(env.model, width=640, height=360)
    rows = []
    try:
        for index, seed in enumerate(range(seed_offset, seed_offset + episodes)):
            wrapped.reset(seed=seed)
            home_q = env.data.qpos[env.arm_qpos_ids].copy()
            truth_world = env.data.site_xpos[env.goal_site_id].copy()
            move_arm(env, observe_q, interpolation_steps=180, substeps=4)
            session, estimate = capture_aruco_episode_goal(
                env, renderer, color_id, base_id, intrinsics, observe_q, config,
                estimate_id=f"seed-{seed:03d}-goal-000001")
            valid = estimate is not None
            error = None if not valid else float(
                np.linalg.norm(estimate.position - truth_world))
            xy_error = None if not valid else float(
                np.linalg.norm((estimate.position - truth_world)[:2]))
            frozen_before = None if not valid else estimate.position.copy()
            if valid:
                session.begin_return_home()
                move_arm(env, home_q, interpolation_steps=180, substeps=4)
                session.begin_policy_execution()
                assert np.array_equal(
                    env.goal_estimate_authority.current.position, frozen_before)
                overwrite_rejected = False
                try:
                    env.publish_goal_estimate(GoalEstimate(
                        position=truth_world + .1, timestamp=env.data.time,
                        valid=True, confidence=1.0, source="aruco_test_overwrite",
                        estimate_id="forbidden-overwrite", frame="world"))
                except RuntimeError:
                    overwrite_rejected = True
                assert overwrite_rejected
                used = [env._control_goal_estimate(name) for name in (
                    "ppo_observation", "place_servo", "place_descent",
                    "place_release")]
                assert len({(item.estimate_id, item.timestamp) for item in used}) == 1
            rows.append({
                "seed": seed, "valid": valid,
                "failure_reason": session.failure_reason,
                "capture_frames": session.capture_frames,
                "valid_frames": len(session.valid_estimates),
                "error_m_evaluator_only": error,
                "xy_error_m_evaluator_only": xy_error,
                "truth_goal_world_evaluator_only": truth_world.tolist(),
                "estimated_goal_world": None if not valid else estimate.position.tolist(),
            })
            print(f"seed={seed:03d} valid={valid} error_mm="
                  f"{None if error is None else 1000*error:.3f}" if valid else
                  f"seed={seed:03d} valid=False reason={session.failure_reason}")
    finally:
        renderer.close()
        wrapped.close()
    errors = np.asarray([
        row["error_m_evaluator_only"] for row in rows
        if row["error_m_evaluator_only"] is not None], dtype=float)
    xy_errors = np.asarray([
        row["xy_error_m_evaluator_only"] for row in rows
        if row["xy_error_m_evaluator_only"] is not None], dtype=float)
    stats = {
        "episodes": episodes,
        "valid_count": int(sum(row["valid"] for row in rows)),
        "valid_rate": float(np.mean([row["valid"] for row in rows])),
        "error_m": None if not len(errors) else {
            "mean": float(errors.mean()), "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)), "max": float(errors.max())},
        "xy_error_m": None if not len(xy_errors) else {
            "mean": float(xy_errors.mean()), "median": float(np.median(xy_errors)),
            "p95": float(np.percentile(xy_errors, 95)), "max": float(xy_errors.max())},
    }
    return rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    config = load_aruco_goal_config(CONFIG)
    state_results = state_machine_tests(config)
    rows, stats = run_render_tests(args.episodes, args.seed_offset, config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"state_machine_tests": state_results,
              "localization": stats, "episodes": rows}
    (OUTPUT / "localization_test_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"state_machine_tests": state_results,
                      "localization": stats}, indent=2))


if __name__ == "__main__":
    main()
