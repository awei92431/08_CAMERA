"""A/B evaluation of truth versus RGB-D-derived PPO object observations.

The MuJoCo environment, PPO, checkpoint, reward, FSM and execution layer are
not modified.  RGB-D mode replaces every object-position-derived policy field
immediately before model.predict.  MuJoCo object truth is evaluator-only, apart
from the unchanged environment's own privileged execution/FSM implementation.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco.glfw import glfw
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourc2  # noqa: E402,F401
from fourc2.camera_geometry import (  # noqa: E402
    base_from_color_optical_transform, base_world_transform,
    camera_world_pose_optical, intrinsics_from_fovy,
    relative_optical_transform, transform_point,
)
from fourc2.aruco_goal_localizer import (  # noqa: E402
    GoalCaptureSession, GoalCaptureState, detect_aruco_goal,
    load_aruco_goal_config,
)
from fourc2.goal_estimate import GoalEstimate, GoalEstimateUnavailable  # noqa: E402
from fourc2.envs.allstage import STAGE_GRASP, STAGE_LIFT, STAGE_PLACE, STAGE_REACH
from fourc2.object_estimate import (
    ObjectEstimate, ObjectEstimateUnavailable, TcpObjectTracker,
)
from fourc2.rgbd_cube_localizer import load_localization_config, localize_cube_rgbd

ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
CHECKPOINT = ROOT / "checkpoints" / "best_full_flow_v22.zip"
CONFIG = ROOT / "configs" / "hsv_cube_localization.json"
ARUCO_CONFIG = ROOT / "configs" / "aruco_goal_localization.json"
OUTPUT = ROOT / "outputs" / "full_visual_closed_loop"
DOC = ROOT / "docs" / "full_visual_closed_loop_evaluation.md"
WIDTH, HEIGHT = 640, 360
STAGES = {STAGE_REACH: "Reach", STAGE_GRASP: "Grasp",
          STAGE_LIFT: "Lift", STAGE_PLACE: "Place"}


def mid(kind, model, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"missing MuJoCo object {name}")
    return value


def set_simulated_aruco_marker_size(model, marker_size_m):
    """Idempotently synchronize the rendered marker with the PnP size.

    The current physical size is read from MuJoCo's compiled mesh vertices;
    there is no hard-coded 40/60 mm source-size assumption. MuJoCo may reorder
    mesh principal axes during compilation, so the two planar spans are found
    dynamically. Only this non-colliding marker visual is affected.
    """
    mesh_id = mid(mujoco.mjtObj.mjOBJ_MESH, model, "aruco_goal_plane")
    first = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[first:first + count]
    spans = np.ptp(vertices, axis=0)
    planar_axes = np.argsort(spans)[-2:]
    compiled_size_m = float(np.mean(spans[planar_axes]))
    if compiled_size_m <= 0.0:
        raise RuntimeError("compiled ArUco marker has invalid planar size")
    ratio = float(marker_size_m) / compiled_size_m
    center = np.mean(vertices, axis=0)
    for axis in planar_axes:
        vertices[:, axis] = center[axis] + ratio * (
            vertices[:, axis] - center[axis])
    return {
        "mesh_id": int(mesh_id), "ratio": ratio,
        "compiled_size_before_m": compiled_size_m,
        "requested_size_m": float(marker_size_m),
        "planar_axes": planar_axes.astype(int).tolist(),
        "resulting_spans_m": np.ptp(vertices, axis=0).tolist(),
    }


def move_arm(env, target, interpolation_steps=180, substeps=4, frame_callback=None):
    start = env.data.qpos[env.arm_qpos_ids].copy()
    for i in range(1, interpolation_steps + 1):
        x = i / interpolation_steps
        s = x * x * (3 - 2 * x)
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            start + s * (target - start), env.arm_ctrl_low, env.arm_ctrl_high)
        for _ in range(substeps):
            mujoco.mj_step(env.model, env.data)
        if frame_callback is not None and i % 6 == 0:
            frame_callback()


def hold(env, target, steps=120, frame_callback=None):
    for index in range(steps):
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            target, env.arm_ctrl_low, env.arm_ctrl_high)
        mujoco.mj_step(env.model, env.data)
        if frame_callback is not None and (index + 1) % 10 == 0:
            frame_callback()


def render_pair(renderer, env, color_id, depth_id):
    renderer.update_scene(env.data, camera=color_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(env.data, camera=depth_id)
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return rgb, depth


def acquire_rgbd(env, renderer, ids, intrinsics, config):
    color_id, depth_id, base_id = ids
    color_k, depth_k = intrinsics
    color_pose = camera_world_pose_optical(env.data, color_id)
    depth_pose = camera_world_pose_optical(env.data, depth_id)
    rgb, depth = render_pair(renderer, env, color_id, depth_id)
    result = localize_cube_rgbd(
        rgb, depth, depth_k, color_k,
        relative_optical_transform(color_pose, depth_pose),
        color_pose[1].T @ np.array([0., 0., 1.]), config)
    estimate_base = estimate_world = None
    if result.valid:
        estimate_base = transform_point(
            base_from_color_optical_transform(env.data, color_id, base_id),
            result.estimated_object_center_color)
        t_base_world = base_world_transform(env.data, base_id)
        t_world_base = np.linalg.inv(t_base_world)
        estimate_world = transform_point(t_world_base, estimate_base)
    return result, estimate_base, estimate_world, rgb


def capture_aruco_episode_goal(env, renderer, color_id, base_id,
                               color_intrinsics, observe_q, config,
                               frame_callback=None, show_camera=False,
                               estimate_id="aruco-goal-000001",
                               status_callback=None,
                               truth_goal_world_evaluator_only=None):
    """Wait for actual arm stability, then detect only in CAPTURE_GOAL."""
    session = GoalCaptureSession(config)
    session.reached_observe_command()
    if status_callback is not None:
        status_callback("WAIT_OBSERVE_STABLE")
    wait_limit = int(np.ceil(
        (config.observe_settle_seconds + 5.0) / env.model.opt.timestep))
    for step in range(wait_limit):
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            observe_q, env.arm_ctrl_low, env.arm_ctrl_high)
        mujoco.mj_step(env.model, env.data)
        session.update_stability(
            env.data.qpos[env.arm_qpos_ids],
            env.data.qvel[env.arm_qvel_ids], observe_q, env.data.time)
        if frame_callback is not None and (step + 1) % 10 == 0:
            frame_callback()
        if session.state == GoalCaptureState.CAPTURE_GOAL:
            break
    if session.state != GoalCaptureState.CAPTURE_GOAL:
        session.state = GoalCaptureState.FAILED
        session.failure_reason = (
            "camera_observe_not_stable:"
            f"qerr={np.max(np.abs(env.data.qpos[env.arm_qpos_ids]-observe_q)):.6f},"
            f"qvel={np.max(np.abs(env.data.qvel[env.arm_qvel_ids])):.6f}")
        return session, None

    if status_callback is not None:
        status_callback("CAPTURING GOAL AT CAMERA_OBSERVE")

    last_rgb = None
    while session.state == GoalCaptureState.CAPTURE_GOAL:
        env.data.ctrl[env.arm_actuator_ids] = np.clip(
            observe_q, env.arm_ctrl_low, env.arm_ctrl_high)
        mujoco.mj_step(env.model, env.data)
        renderer.update_scene(env.data, camera=color_id)
        last_rgb = renderer.render().copy()
        detection = detect_aruco_goal(
            last_rgb, color_intrinsics, config.marker_size_m,
            config.marker_id,
            base_from_color_optical_transform(env.data, color_id, base_id),
            config.cube_size_m, dictionary_name=config.dictionary,
            timestamp=env.data.time,
            max_reprojection_error_px=config.max_reprojection_error_px,
            min_marker_area_px=config.min_marker_area_px,
            min_depth_m=config.min_depth_m, max_depth_m=config.max_depth_m,
            workspace_base_low=config.workspace_base_low,
            workspace_base_high=config.workspace_base_high,
            table_normal_base=config.table_normal_base,
        )
        frozen_base = session.submit_detection(detection, env.data.time)
        if show_camera:
            panel = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
            if detection.diagnostics is not None:
                corners = np.rint(detection.diagnostics["corners"]).astype(int)
                cv2.polylines(panel, [corners], True, (40, 220, 40), 2)
            lines = [
                "CAPTURING GOAL AT CAMERA_OBSERVE",
                f"state: {session.state.value}",
                (f"valid: {len(session.valid_estimates)}/"
                 f"{config.required_valid_frames}  frame: {session.capture_frames}/"
                 f"{config.maximum_capture_frames}"),
                (f"ID: {config.marker_id}  "
                 f"reason: {detection.failure_reason or 'valid'}"),
            ]
            if detection.valid:
                p = detection.position
                lines.append(f"goal Base: {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f} m")
            if session.valid_estimates:
                std = np.std(np.stack([
                    item.position for item in session.valid_estimates]), axis=0)
                lines.append("Base std: " + " ".join(f"{v*1000:.3f}" for v in std)
                             + " mm")
            if (detection.valid
                    and truth_goal_world_evaluator_only is not None):
                truth_base = transform_point(
                    base_world_transform(env.data, base_id),
                    truth_goal_world_evaluator_only)
                lines.append(
                    "truth error (eval only): "
                    f"{1000*np.linalg.norm(detection.position-truth_base):.3f} mm")
            if frozen_base is not None:
                lines.append("GOAL FROZEN")
            for index, line in enumerate(lines):
                cv2.putText(panel, line, (12, 25 + 22*index),
                            cv2.FONT_HERSHEY_SIMPLEX, .52,
                            (40, 220, 40) if detection.valid else (40, 40, 240),
                            1, cv2.LINE_AA)
            cv2.imshow("ArUco goal capture (camera_observe only)", panel)
            cv2.waitKey(1)
        if frame_callback is not None:
            frame_callback()

    if show_camera:
        cv2.destroyWindow("ArUco goal capture (camera_observe only)")
    if not session.goal_is_frozen:
        if status_callback is not None:
            status_callback("ARUCO GOAL FAILED CLOSED")
        return session, None
    t_world_base = np.linalg.inv(base_world_transform(env.data, base_id))
    frozen_world = transform_point(t_world_base, session.frozen_estimate.position)
    estimate = GoalEstimate(
        position=frozen_world, rotation=None,
        timestamp=float(session.frozen_estimate.timestamp), valid=True,
        confidence=float(session.frozen_estimate.confidence),
        source="aruco_color_multiframe_frozen", estimate_id=str(estimate_id),
        marker_id=config.marker_id, frame="world",
        diagnostics={
            "goal_position_base": session.frozen_estimate.position.copy(),
            "capture_frames": session.capture_frames,
            "valid_frames": len(session.valid_estimates),
        },
    )
    env.publish_goal_estimate(estimate)
    if status_callback is not None:
        status_callback("GOAL FROZEN")
    return session, estimate


def update_tracker(tracker, env):
    return tracker.update(
        env.data.site_xpos[env.pinch_site_id],
        env.data.site_xmat[env.pinch_site_id].reshape(3, 3),
        env.is_grasp_latched,
    )


def abnormal_contacts(env):
    count = 0
    for contact in env.data.contact:
        if contact.dist >= 0:
            continue
        pair = {int(contact.geom1), int(contact.geom2)}
        if env.table_geom_id in pair and env.object_geom_id not in pair:
            count += 1
    return count


def failure_class(row):
    if row["full_success"]:
        return "success"
    if not row["rgbd_valid"]:
        return "RGB-D未定位"
    if row["visual_error_m"] is not None and row["visual_error_m"] > .02:
        return "坐标转换错误"
    if not row["reach_success"]:
        return "Reach偏差"
    if not row["grasp_success"]:
        return "未形成稳定抓取" if row["min_grasp_xy_error_m"] < .012 else "抓取偏心"
    if not row["lift_success"]:
        return "Lift失败"
    if not row["entered_place"]:
        return "Place定位失败"
    if row["gripper_opened"] and not row["place_success"]:
        return "释放失败"
    if row["robot_table_contact_steps"]:
        return "安全条件失败"
    return "其他"


def evaluate_mode(mode, seeds, model, save_steps=False, video_path=None,
                  live=False, show_camera=False, viewer_size=(1280, 900),
                  hide_pinch_site=False, observe_move_extra_seconds=None,
                  observe_hold_seconds=None, goal_source="ground_truth",
                  aruco_config=None):
    wrapped = gym.make(ENV_ID, max_tcp_lead=.03, ik_posture_mode="off",
                       object_observation_mode=mode,
                       goal_observation_mode=goal_source,
                       render_mode=("human" if live else
                                    "rgb_array" if video_path else None),
                       disable_env_checker=True)
    env = wrapped.unwrapped
    if live:
        env.mujoco_renderer.width, env.mujoco_renderer.height = viewer_size
    if hide_pinch_site:
        # Visual-only toggle. The site pose remains available to FK/control.
        env.model.site_rgba[env.pinch_site_id, 3] = 0.0
    if video_path is not None:
        env.mujoco_renderer.width = 960
        env.mujoco_renderer.height = 720
    if env.tcp_weld_enabled:
        raise RuntimeError("evaluation requires the mocap weld to be disabled")
    if env.max_tcp_lead != .03 or env.ik_posture_mode != "off":
        raise RuntimeError("execution-layer configuration mismatch")
    observe_key = mid(mujoco.mjtObj.mjOBJ_KEY, env.model, "camera_observe")
    observe_q = env.model.key_qpos[observe_key, env.arm_qpos_ids].copy()
    ids = (mid(mujoco.mjtObj.mjOBJ_CAMERA, env.model, "eye_in_hand_color"),
           mid(mujoco.mjtObj.mjOBJ_CAMERA, env.model, "eye_in_hand_depth"),
           mid(mujoco.mjtObj.mjOBJ_BODY, env.model, "base"))
    intrinsics = (
        intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[ids[0]]),
        intrinsics_from_fovy(WIDTH, HEIGHT, env.model.cam_fovy[ids[1]]))
    config = load_localization_config(CONFIG)
    if aruco_config is None:
        aruco_config = load_aruco_goal_config(ARUCO_CONFIG)
    marker_resize = None
    if goal_source == "aruco":
        marker_resize = set_simulated_aruco_marker_size(
            env.model, aruco_config.marker_size_m)
    renderer = mujoco.Renderer(env.model, width=WIDTH, height=HEIGHT)
    rows, step_rows = [], []
    video = None
    live_clock = {"wall": None, "sim": None}
    move_substeps = 4
    original_move_seconds = 180 * move_substeps * env.model.opt.timestep
    if observe_move_extra_seconds is None:
        observe_move_extra_seconds = 2.0 if live else 0.0
    if observe_hold_seconds is None:
        observe_hold_seconds = 1.5 if live else 120 * env.model.opt.timestep
    move_steps = max(1, int(round(
        (original_move_seconds + observe_move_extra_seconds)
        / (move_substeps * env.model.opt.timestep)
    )))
    observe_hold_steps = max(0, int(round(
        observe_hold_seconds / env.model.opt.timestep
    )))
    goal_ui_state = {"text": (
        "DETECTION IGNORED: NOT AT CAMERA_OBSERVE"
        if goal_source == "aruco" else "")}

    def camera_panel():
        rgb, depth = render_pair(renderer, env, ids[0], ids[1])
        color_pose = camera_world_pose_optical(env.data, ids[0])
        depth_pose = camera_world_pose_optical(env.data, ids[1])
        detection = localize_cube_rgbd(
            rgb, depth, intrinsics[1], intrinsics[0],
            relative_optical_transform(color_pose, depth_pose),
            color_pose[1].T @ np.array([0.0, 0.0, 1.0]), config,
        )
        valid = np.isfinite(depth) & (depth > 0.0)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            near, far = np.percentile(depth[valid], [2.0, 98.0])
            scale = max(float(far - near), 1e-6)
            preview[valid] = np.clip(
                255.0 * (far - depth[valid]) / scale, 0.0, 255.0
            ).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(rgb_bgr, "eye_in_hand_color", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(depth_bgr, "eye_in_hand_depth", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if goal_ui_state["text"]:
            cv2.putText(rgb_bgr, goal_ui_state["text"], (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 180, 255), 1,
                        cv2.LINE_AA)
        if detection.bbox_xywh is not None:
            x, y, w, h = detection.bbox_xywh
            color = (40, 220, 40) if detection.valid else (0, 165, 255)
            cv2.rectangle(rgb_bgr, (x, y), (x + w, y + h), color, 2)
            if detection.pixel_center_uv is not None:
                u, v = (int(round(value)) for value in detection.pixel_center_uv)
                cv2.drawMarker(rgb_bgr, (u, v), color,
                               cv2.MARKER_CROSS, 14, 2)
        if detection.valid:
            center = detection.estimated_object_center_color
            lines = [
                "CUBE DETECTED (HSV + RGB-D)",
                (f"Color optical XYZ: {center[0]:+.3f}, "
                 f"{center[1]:+.3f}, {center[2]:.3f} m"),
                (f"mask: {detection.mask_pixel_count} px  "
                 f"depth points: {detection.filtered_depth_point_count}"),
            ]
            text_color = (40, 220, 40)
        else:
            lines = ["CUBE NOT DETECTED",
                     f"reason: {detection.failure_reason}"]
            text_color = (40, 40, 240)
        panel_top = HEIGHT - 76
        cv2.rectangle(rgb_bgr, (0, panel_top - 8), (WIDTH, HEIGHT),
                      (15, 15, 15), -1)
        for index, line in enumerate(lines):
            cv2.putText(rgb_bgr, line, (12, panel_top + 21 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, text_color, 1,
                        cv2.LINE_AA)
        cv2.imshow("D435i live RGB-D (Color | Depth)",
                   np.hstack((rgb_bgr, depth_bgr)))
        cv2.waitKey(1)

    def capture():
        nonlocal video
        if show_camera:
            camera_panel()
        if live:
            # The offscreen RGB-D renderer makes its own OpenGL context
            # current. Restore Gymnasium's GLFW viewer context explicitly;
            # otherwise the simulation advances while the third-person
            # window remains stuck on its first frame.
            viewer = env.mujoco_renderer._get_viewer(render_mode="human")
            glfw.make_context_current(viewer.window)
            viewer.render()
        if live and live_clock["wall"] is not None:
            deadline = (live_clock["wall"] + float(env.data.time)
                        - live_clock["sim"])
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        if video_path is None:
            return
        frame = wrapped.render()
        if frame is None:
            return
        if video is None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                (frame.shape[1], frame.shape[0]))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    try:
        for number, seed in enumerate(seeds):
            _, _ = wrapped.reset(seed=seed)
            live_clock.update(wall=time.monotonic(), sim=float(env.data.time))
            home_q = env.data.qpos[env.arm_qpos_ids].copy()
            initial_object_preobserve = env.data.site_xpos[env.object_site_id].copy()
            initial_goal = env.data.site_xpos[env.goal_site_id].copy()
            capture()
            move_arm(env, observe_q, interpolation_steps=move_steps,
                     substeps=move_substeps, frame_callback=capture)
            goal_session = None
            goal_estimate = None
            if goal_source == "aruco":
                goal_session, goal_estimate = capture_aruco_episode_goal(
                    env, renderer, ids[0], ids[2], intrinsics[0], observe_q,
                    aruco_config, frame_callback=capture,
                    show_camera=show_camera,
                    estimate_id=f"seed-{seed:03d}-goal-000001",
                    status_callback=lambda text: goal_ui_state.update(text=text),
                    truth_goal_world_evaluator_only=initial_goal,
                )
                if goal_estimate is None:
                    env.invalidate_goal_estimate(
                        goal_session.failure_reason or "aruco_capture_failed")
                    raise RuntimeError(
                        "ArUco goal capture failed closed before PPO: "
                        f"seed={seed}; reason={goal_session.failure_reason}; "
                        f"failures={dict(goal_session.failure_counts)}")
            else:
                hold(env, observe_q, steps=observe_hold_steps,
                     frame_callback=capture)
            capture()
            localized, estimate_base, estimate_world, _ = acquire_rgbd(
                env, renderer, ids, intrinsics, config)
            visual_truth_at_capture = env.data.site_xpos[env.object_site_id].copy()
            visual_error = (None if estimate_world is None else float(
                np.linalg.norm(estimate_world - visual_truth_at_capture)))
            visual_xy_error = (None if estimate_world is None else float(
                np.linalg.norm((estimate_world - visual_truth_at_capture)[:2])))
            if goal_session is not None:
                goal_session.begin_return_home()
            move_arm(env, home_q, interpolation_steps=move_steps,
                     substeps=move_substeps, frame_callback=capture)
            hold(env, home_q, frame_callback=capture)
            env._sync_mocap_target_to_tcp(); capture()
            if goal_session is not None:
                goal_session.begin_policy_execution()
            postcycle_object = env.data.site_xpos[env.object_site_id].copy()
            tracker = None if estimate_world is None else TcpObjectTracker(estimate_world)
            estimate_sequence = 0
            if mode == "ground_truth":
                env._publish_ground_truth_object_estimate()
            elif tracker is not None:
                estimate_sequence += 1
                env.publish_object_estimate(ObjectEstimate(
                    tracker.world_position, env.data.time, True, 1.0, "rgbd_visual",
                    f"seed-{seed:03d}-estimate-{estimate_sequence:06d}"))
            # Reset/bootstrap observations are not consumed by PPO. Start the
            # formal per-control-step source audit after camera acquisition.
            env.object_estimate_authority.uses.clear()
            env.goal_estimate_authority.uses.clear()

            def policy_observation():
                nonlocal estimate_sequence
                if mode == "ground_truth":
                    return env.control_observation()
                if tracker is None:
                    return None
                estimated_world_now = update_tracker(tracker, env)
                estimate_sequence += 1
                source = ("tcp_fk_propagated" if tracker.tcp_local_offset is not None
                          else "rgbd_visual_hold")
                env.publish_object_estimate(ObjectEstimate(
                    estimated_world_now, env.data.time, True, 1.0, source,
                    f"seed-{seed:03d}-estimate-{estimate_sequence:06d}"))
                return env.control_observation()

            done = False; steps = 0; total_return = 0.; reach = grasp = lift = entered = False
            place_success = opened = False; min_grasp_xy = np.inf
            grasp_drift = 0.; initial_grasp_object = None; grasp_visual_xy = None
            final_info = env._get_info(); tcp_errors = defaultdict(list)
            ik_fail = ik_calls = dq_clip = dq_iterations = 0
            max_qvel = max_force = 0.; table_steps = 0; dropped = flung = False
            max_visual_runtime_error = 0.; first_action_delta = None
            consistency_failures = 0; checked_control_steps = 0
            consumer_use_counts = Counter()
            estimate_source_counts = Counter()
            goal_consistency_failures = 0
            goal_checked_control_steps = 0
            goal_consumer_use_counts = Counter()
            while not done and steps < 2000:
                try:
                    policy_obs = policy_observation()
                except ObjectEstimateUnavailable:
                    policy_obs = None
                if policy_obs is None:
                    break
                action, _ = model.predict(policy_obs, deterministic=True)
                stage_before = int(env.stage)
                authority_step = env.control_step_index
                truth_before = env.data.site_xpos[env.object_site_id].copy()
                estimate_before = None if tracker is None else tracker.world_position.copy()
                _, reward, terminated, truncated, info = wrapped.step(action)
                runtime_consumers = {
                    "ppo_observation", "place_servo", "place_descent",
                    "place_release", "reach_safety", "grasp_safety",
                }
                uses = [u for u in env.object_estimate_authority.uses_for_step(
                    authority_step) if u["consumer"] in runtime_consumers]
                for use in uses:
                    consumer_use_counts[use["consumer"]] += 1
                    estimate_source_counts[use["source"]] += 1
                if uses:
                    checked_control_steps += 1
                    keys = {(u["estimate_id"], u["timestamp"]) for u in uses}
                    consistency_failures += int(len(keys) != 1)
                goal_runtime_consumers = {
                    "ppo_observation", "post_step_observation",
                    "place_servo", "place_descent", "place_release",
                    "task_metrics", "task_supervisor",
                }
                goal_uses = [u for u in env.goal_estimate_authority.uses_for_step(
                    authority_step) if u["consumer"] in goal_runtime_consumers]
                if goal_uses:
                    goal_checked_control_steps += 1
                    for use in goal_uses:
                        goal_consumer_use_counts[use["consumer"]] += 1
                    goal_keys = {(u["estimate_id"], u["timestamp"])
                                 for u in goal_uses}
                    goal_consistency_failures += int(len(goal_keys) != 1)
                capture()
                done = bool(terminated or truncated); steps += 1; total_return += float(reward)
                final_info = info
                reach |= bool(info.get("reach_success")); grasp |= bool(info.get("grasp_success"))
                lift |= bool(info.get("lift_success")); place_success |= bool(info.get("place_success"))
                entered |= stage_before == STAGE_PLACE or int(env.stage) == STAGE_PLACE
                opened |= bool(info.get("place_opened") or info.get("place_has_opened"))
                stage_name = STAGES.get(stage_before, str(stage_before))
                tcp_errors[stage_name].append(float(info.get("tcp_target_error", np.nan)))
                if stage_before == STAGE_GRASP:
                    truth_obj = env.data.site_xpos[env.object_site_id].copy()
                    pinch = env.data.site_xpos[env.pinch_site_id].copy()
                    min_grasp_xy = min(min_grasp_xy, float(np.linalg.norm((truth_obj-pinch)[:2])))
                    if initial_grasp_object is None: initial_grasp_object = truth_obj.copy()
                    grasp_drift = max(grasp_drift, float(np.linalg.norm(
                        (truth_obj-initial_grasp_object)[:2])))
                    if env.is_grasp_latched and grasp_visual_xy is None and tracker is not None:
                        grasp_visual_xy = float(np.linalg.norm((tracker.world_position-truth_obj)[:2]))
                diag = env.diag_ik
                ik_calls += 1; ik_fail += int(not diag["converged"])
                dq_clip += int(diag["dq_clip_iterations"]); dq_iterations += int(diag["iterations"])
                max_qvel = max(max_qvel, float(np.max(np.abs(env.data.qvel[env.arm_qvel_ids]))))
                max_force = max(max_force, float(np.max(np.abs(env.data.actuator_force[env.arm_actuator_ids]))))
                table_steps += int(abnormal_contacts(env) > 0)
                dropped |= bool(info["object_position"][2] < env.table_top_z + env.object_half_size - .01)
                flung |= bool(info.get("object_speed", 0.) > .5)
                if tracker is not None:
                    update_tracker(tracker, env)
                    runtime_error = float(np.linalg.norm(
                        tracker.world_position - env.data.site_xpos[env.object_site_id]))
                    max_visual_runtime_error = max(max_visual_runtime_error, runtime_error)
                if save_steps:
                    step_rows.append({"mode": mode, "seed": seed, "step": steps,
                        "stage": stage_name, "reward": reward,
                        "visual_error_m_evaluator_only": None if tracker is None else runtime_error,
                        "truth_object_x_evaluator_only": truth_before[0],
                        "estimated_object_x": None if estimate_before is None else estimate_before[0]})
            final_goal_xy = float(final_info.get("object_to_goal_xy_distance", np.nan))
            row = {"seed": seed, "mode": mode, "rgbd_valid": bool(localized.valid),
                "rgbd_failure_reason": localized.failure_reason,
                "full_success": bool(final_info.get("is_success", False)),
                "reach_success": reach, "grasp_success": grasp, "lift_success": lift,
                "entered_place": entered, "place_success": place_success,
                "episode_steps": steps, "episode_return": total_return,
                "visual_error_m": visual_error, "visual_xy_error_m": visual_xy_error,
                "capture_distance_m": None if estimate_world is None else float(np.linalg.norm(
                    estimate_world-camera_world_pose_optical(env.data, ids[0])[0])),
                "object_cycle_displacement_m": float(np.linalg.norm(
                    postcycle_object-initial_object_preobserve)),
                "grasp_visual_xy_error_m": grasp_visual_xy,
                "min_grasp_xy_error_m": None if not np.isfinite(min_grasp_xy) else min_grasp_xy,
                "grasp_horizontal_drift_m": grasp_drift,
                "final_goal_xy_error_m": final_goal_xy,
                "ik_nonconverged_calls": ik_fail, "ik_calls": ik_calls,
                "dq_clip_rate": dq_clip/max(dq_iterations, 1),
                "max_joint_velocity_rad_s": max_qvel, "max_actuator_force_n": max_force,
                "robot_table_contact_steps": table_steps, "object_dropped": dropped,
                "object_fling": flung, "gripper_opened": opened,
                "release_failed": bool(entered and not place_success),
                "max_runtime_estimator_error_m_evaluator_only": max_visual_runtime_error,
                "first_action_delta_vs_truth": first_action_delta,
                "object_source_checked_steps": checked_control_steps,
                "object_source_consistency_failures": consistency_failures,
                "object_source_consistent": consistency_failures == 0,
                "object_source_consumer_counts": dict(consumer_use_counts),
                "object_estimate_source_counts": dict(estimate_source_counts),
                "goal_source": goal_source,
                "simulated_marker_size_m": aruco_config.marker_size_m,
                "marker_resize_diagnostics": marker_resize,
                "goal_estimate_id": (None if goal_estimate is None else
                                     goal_estimate.estimate_id),
                "goal_frozen_at_state": (None if goal_session is None else
                                         GoalCaptureState.GOAL_FROZEN.value),
                "goal_frozen_at_timestamp": (None if goal_estimate is None else
                                              goal_estimate.timestamp),
                "goal_capture_frames": (None if goal_session is None else
                                        goal_session.capture_frames),
                "goal_valid_frames": (None if goal_session is None else
                                      len(goal_session.valid_estimates)),
                "goal_visual_error_m_evaluator_only": (None if goal_estimate is None else
                    float(np.linalg.norm(goal_estimate.position - initial_goal))),
                "goal_visual_xy_error_m_evaluator_only": (
                    None if goal_estimate is None else float(np.linalg.norm(
                        (goal_estimate.position - initial_goal)[:2]))),
                "final_true_goal_xy_error_m_evaluator_only": float(np.linalg.norm(
                    (np.asarray(final_info["object_position"]) - initial_goal)[:2])),
                "final_object_xyz_evaluator_only": np.asarray(
                    final_info["object_position"]).tolist(),
                "goal_source_checked_steps": goal_checked_control_steps,
                "goal_source_consistency_failures": goal_consistency_failures,
                "goal_source_consistent": goal_consistency_failures == 0,
                "goal_source_consumer_counts": dict(goal_consumer_use_counts),
                "initial_object_xyz": initial_object_preobserve.tolist(),
                "initial_goal_xyz": initial_goal.tolist()}
            for name in STAGES.values():
                values = np.asarray(tcp_errors[name], float)
                row[f"tcp_tracking_{name.lower()}_mean_m"] = (
                    None if not len(values) else float(np.nanmean(values)))
                row[f"tcp_tracking_{name.lower()}_max_m"] = (
                    None if not len(values) else float(np.nanmax(values)))
            row["failure_class"] = failure_class(row) if mode == "rgbd" else (
                "success" if row["full_success"] else "task_failure")
            rows.append(row)
            print(f"{mode} seed={seed:03d} success={row['full_success']} "
                  f"vision={row['rgbd_valid']} steps={steps} fail={row['failure_class']}", flush=True)
    finally:
        if video is not None:
            video.release()
        if show_camera:
            cv2.destroyWindow("D435i live RGB-D (Color | Depth)")
        renderer.close(); wrapped.close()
    return rows, step_rows


def scalar_stats(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return None if not len(a) else {"mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def summarize(rows):
    n = len(rows)
    result = {"episodes": n}
    for key in ["rgbd_valid", "full_success", "reach_success", "grasp_success",
                "lift_success", "entered_place", "place_success"]:
        result[key] = {"count": int(sum(bool(r[key]) for r in rows)),
                       "rate": float(np.mean([bool(r[key]) for r in rows]))}
    for key in ["episode_steps", "episode_return", "visual_error_m", "visual_xy_error_m",
                "grasp_visual_xy_error_m", "min_grasp_xy_error_m",
                "grasp_horizontal_drift_m", "final_goal_xy_error_m",
                "dq_clip_rate", "max_joint_velocity_rad_s", "max_actuator_force_n",
                "max_runtime_estimator_error_m_evaluator_only"]:
        result[key] = scalar_stats([r[key] for r in rows])
    result["ik_nonconverged_calls"] = int(sum(r["ik_nonconverged_calls"] for r in rows))
    result["ik_calls"] = int(sum(r["ik_calls"] for r in rows))
    result["robot_table_contact_steps"] = int(sum(r["robot_table_contact_steps"] for r in rows))
    result["drop_episodes"] = int(sum(r["object_dropped"] for r in rows))
    result["fling_episodes"] = int(sum(r["object_fling"] for r in rows))
    result["object_source_consistency_failures"] = int(sum(
        r["object_source_consistency_failures"] for r in rows))
    result["object_source_consistent_episodes"] = int(sum(
        r["object_source_consistent"] for r in rows))
    result["failure_classes"] = dict(Counter(r["failure_class"] for r in rows))
    result["tcp_tracking"] = {stage: scalar_stats([
        r[f"tcp_tracking_{stage.lower()}_mean_m"] for r in rows]) for stage in STAGES.values()}
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_outputs(gt, rgbd, summary):
    OUTPUT.mkdir(parents=True, exist_ok=True); DOC.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "ground_truth_episodes.csv", gt)
    write_csv(OUTPUT / "rgbd_episodes.csv", rgbd)
    pairs = []
    for a, b in zip(gt, rgbd):
        pairs.append({"seed": a["seed"], "same_initial_object": a["initial_object_xyz"] == b["initial_object_xyz"],
            "same_initial_goal": a["initial_goal_xyz"] == b["initial_goal_xyz"],
            "ground_truth_success": a["full_success"], "rgbd_success": b["full_success"],
            "success_identical": a["full_success"] == b["full_success"],
            "ground_truth_steps": a["episode_steps"], "rgbd_steps": b["episode_steps"],
            "rgbd_visual_error_m": b["visual_error_m"], "rgbd_failure_class": b["failure_class"]})
    write_csv(OUTPUT / "seed_comparison.csv", pairs)
    summary["seed_comparison"] = {"identical_outcome_count": int(sum(p["success_identical"] for p in pairs)),
        "identical_outcome_rate": float(np.mean([p["success_identical"] for p in pairs])),
        "same_initial_state_count": int(sum(p["same_initial_object"] and p["same_initial_goal"] for p in pairs))}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    errors = np.array([r["visual_error_m"]*1000 for r in rgbd if r["visual_error_m"] is not None])
    outcomes = np.array([r["full_success"] for r in rgbd if r["visual_error_m"] is not None])
    plt.figure(figsize=(7,4)); plt.scatter(errors, outcomes.astype(int), alpha=.7)
    plt.yticks([0,1],["failure","success"]); plt.xlabel("Initial RGB-D 3D error (mm)")
    plt.ylabel("Task outcome"); plt.tight_layout()
    plt.savefig(OUTPUT / "visual_error_vs_task_success.png", dpi=170); plt.close()
    doc = "# Full visual closed-loop evaluation\n\n"
    doc += f"- Checkpoint: `{CHECKPOINT}`\n- Environment: `{ENV_ID}`\n"
    doc += "- Policy: deterministic PPO; observations are passed directly to `model.predict` (no VecNormalize wrapper).\n"
    doc += "- Flow: home → camera_observe → RGB-D → home → PPO → DLS IK → joint actuators → existing FSM.\n"
    doc += "- RGB-D policy source: all object-derived fields `obs[15:35]` are coherently rebuilt.\n"
    doc += "- After latch: object estimate is propagated from TCP FK and the visually initialized TCP-object transform; no object truth correction.\n\n"
    doc += "## Important privilege audit\n\n"
    doc += "`obs[35:39]` remains simulated contact/alignment/FSM state. The unchanged environment also uses object truth internally for workspace safety, grasp alignment/closing predicates, Place XY servo, reward and success. Therefore this test proves real RGB-D injection into PPO, but is not yet a truth-free real-robot controller.\n\n"
    doc += "## Summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    DOC.write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--mode", choices=["both", "ground_truth", "rgbd"], default="both")
    parser.add_argument("--goal-source", choices=["ground_truth", "aruco"],
                        default="ground_truth")
    parser.add_argument("--aruco-marker-id", type=int, default=None)
    parser.add_argument("--aruco-marker-size", type=float, default=None)
    parser.add_argument("--aruco-dictionary", default=None)
    parser.add_argument("--aruco-required-valid-frames", type=int, default=None)
    parser.add_argument("--aruco-max-capture-frames", type=int, default=None)
    parser.add_argument("--aruco-position-std-threshold", type=float, default=None)
    parser.add_argument("--video", type=Path, default=None,
                        help="Record one single-mode/single-seed deterministic replay")
    parser.add_argument("--live", action="store_true",
                        help="Open the live MuJoCo third-person viewer")
    parser.add_argument("--show-camera", action="store_true",
                        help="Show live Color and Depth camera images")
    parser.add_argument("--viewer-width", type=int, default=1280,
                        help="Initial MuJoCo viewer width (default: 1280)")
    parser.add_argument("--viewer-height", type=int, default=900,
                        help="Initial MuJoCo viewer height (default: 900)")
    parser.add_argument("--hide-pinch-site", action="store_true",
                        help="Hide the small pinch/TCP site marker visually")
    parser.add_argument("--observe-move-extra-seconds", type=float, default=None,
                        help="Extra duration added to each observe/home move "
                             "(live default: 2.0 s)")
    parser.add_argument("--observe-hold-seconds", type=float, default=None,
                        help="Pause at camera_observe before RGB-D capture "
                             "(live default: 1.5 s)")
    args = parser.parse_args()
    aruco_config = load_aruco_goal_config(ARUCO_CONFIG)
    overrides = {
        "marker_id": args.aruco_marker_id,
        "marker_size_m": args.aruco_marker_size,
        "dictionary": args.aruco_dictionary,
        "required_valid_frames": args.aruco_required_valid_frames,
        "maximum_capture_frames": args.aruco_max_capture_frames,
        "position_std_threshold_m": args.aruco_position_std_threshold,
    }
    aruco_config = replace(
        aruco_config, **{key: value for key, value in overrides.items()
                         if value is not None})
    if (args.live or args.show_camera or args.video) and args.mode == "both":
        parser.error("live/camera/video viewing requires --mode ground_truth or --mode rgbd")
    if args.live and args.video is not None:
        parser.error("--live is an interactive view; do not combine it with --video")
    model = PPO.load(CHECKPOINT, device="cpu")
    if tuple(model.observation_space.shape) != (39,):
        raise RuntimeError(f"checkpoint observation shape {model.observation_space.shape}")
    seeds = list(range(args.seed_offset, args.seed_offset + args.episodes))
    if args.live or args.show_camera:
        print(f"live env: {ENV_ID}")
        print(f"checkpoint: {CHECKPOINT}")
        print("model XML: scene_cube3cm.xml (includes ur5e_4c2.xml)")
        print(f"mode: {args.mode}; seeds: {seeds}")
        print(f"goal source: {args.goal_source}")
    gt = rgbd = []
    if args.mode in ("both", "ground_truth"):
        gt, _ = evaluate_mode("ground_truth", seeds, model,
                              video_path=args.video if args.mode == "ground_truth" else None,
                              live=args.live, show_camera=args.show_camera,
                              viewer_size=(args.viewer_width, args.viewer_height),
                              hide_pinch_site=args.hide_pinch_site,
                              observe_move_extra_seconds=args.observe_move_extra_seconds,
                              observe_hold_seconds=args.observe_hold_seconds,
                              goal_source=args.goal_source,
                              aruco_config=aruco_config)
    if args.mode in ("both", "rgbd"):
        rgbd, _ = evaluate_mode("rgbd", seeds, model,
                                video_path=args.video if args.mode == "rgbd" else None,
                                live=args.live, show_camera=args.show_camera,
                                viewer_size=(args.viewer_width, args.viewer_height),
                                hide_pinch_site=args.hide_pinch_site,
                                observe_move_extra_seconds=args.observe_move_extra_seconds,
                                observe_hold_seconds=args.observe_hold_seconds,
                                goal_source=args.goal_source,
                                aruco_config=aruco_config)
    if args.mode == "both":
        summary = {"configuration": {"checkpoint": str(CHECKPOINT), "environment": ENV_ID,
            "seeds": seeds, "deterministic": True, "max_tcp_lead": .03,
            "goal_source": args.goal_source,
            "ik_posture_mode": "off", "mocap_weld_enabled": False,
            "arm_qpos_written_during_policy": False},
            "ground_truth": summarize(gt), "rgbd": summarize(rgbd)}
        make_outputs(gt, rgbd, summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
