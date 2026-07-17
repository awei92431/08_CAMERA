"""ArUco goal localization and the gated, multi-frame capture state machine."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import cv2
import numpy as np

from fourc2.camera_geometry import transform_point
from fourc2.goal_estimate import GoalEstimate


class GoalCaptureState(str, Enum):
    MOVE_TO_OBSERVE = "MOVE_TO_OBSERVE"
    WAIT_OBSERVE_STABLE = "WAIT_OBSERVE_STABLE"
    CAPTURE_GOAL = "CAPTURE_GOAL"
    GOAL_FROZEN = "GOAL_FROZEN"
    RETURN_HOME = "RETURN_HOME"
    POLICY_EXECUTION = "POLICY_EXECUTION"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ArucoGoalConfig:
    dictionary: str = "DICT_4X4_50"
    marker_id: int = 0
    marker_size_m: float = 0.04
    cube_size_m: float = 0.03
    required_valid_frames: int = 8
    maximum_capture_frames: int = 30
    position_std_threshold_m: float = 0.003
    max_reprojection_error_px: float = 3.0
    min_marker_area_px: float = 100.0
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    # This model's named UR5e base body is yawed 180 degrees relative to
    # MuJoCo world, so the table is at negative base X. Bounds are base-frame.
    workspace_base_low: tuple = (-0.85, -0.38, 0.29)
    workspace_base_high: tuple = (-0.15, 0.38, 0.34)
    table_normal_base: tuple = (0.0, 0.0, 1.0)
    observe_q_tolerance_rad: float = 0.06
    observe_qvel_tolerance_rad_s: float = 0.03
    observe_stable_steps: int = 10
    observe_settle_seconds: float = 0.5


def load_aruco_goal_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return ArucoGoalConfig(**values)


def _invalid(timestamp, marker_id, reason, source="aruco_color"):
    return GoalEstimate(
        position=np.full(3, np.nan), rotation=None, timestamp=float(timestamp),
        valid=False, confidence=0.0, source=source, estimate_id="unfrozen",
        marker_id=int(marker_id), frame="base", failure_reason=str(reason),
    )


def _dictionary(dictionary_name):
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def detect_aruco_goal(color_rgb, camera_intrinsics, marker_size_m,
                      expected_marker_id, T_base_color_optical, cube_size_m,
                      *, dictionary_name="DICT_4X4_50", timestamp=0.0,
                      max_reprojection_error_px=3.0, min_marker_area_px=100.0,
                      min_depth_m=0.10, max_depth_m=1.50,
                      workspace_base_low=(-0.85, -0.38, 0.29),
                      workspace_base_high=(-0.15, 0.38, 0.34),
                      table_normal_base=(0.0, 0.0, 1.0)):
    """Estimate the 3 cm goal center in the UR5e base frame.

    OpenCV returns `T_color_optical_marker`.  The existing geometry module
    supplies `T_base_color_optical`; their product maps the marker center into
    base.  The goal center is the marker center plus half the cube edge along
    the configured table normal.  Invalid results contain NaNs, never zeros.
    """
    rgb = np.asarray(color_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _invalid(timestamp, expected_marker_id, "invalid_rgb_shape")
    if not np.isfinite(rgb).all():
        return _invalid(timestamp, expected_marker_id, "non_finite_rgb")
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(_dictionary(dictionary_name), parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return _invalid(timestamp, expected_marker_id, "marker_not_detected")
    ids_flat = ids.reshape(-1).astype(int)
    matches = np.flatnonzero(ids_flat == int(expected_marker_id))
    if matches.size == 0:
        return _invalid(
            timestamp, expected_marker_id,
            "wrong_marker_id:" + ",".join(str(value) for value in ids_flat),
        )
    image_points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
    pixel_area = float(abs(cv2.contourArea(image_points.astype(np.float32))))
    if not np.isfinite(pixel_area) or pixel_area < float(min_marker_area_px):
        return _invalid(timestamp, expected_marker_id,
                        f"marker_area_too_small:{pixel_area:.3f}")

    half = 0.5 * float(marker_size_m)
    # Required ordering for SOLVEPNP_IPPE_SQUARE: top-left, top-right,
    # bottom-right, bottom-left as returned by ArUcoDetector.
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float64)
    camera_matrix = np.asarray(
        getattr(camera_intrinsics, "matrix", camera_intrinsics),
        dtype=np.float64,
    ).reshape(3, 3)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return _invalid(timestamp, expected_marker_id, "solvepnp_failed")
    transform = np.asarray(T_base_color_optical, dtype=np.float64).reshape(4, 4)
    normal = np.asarray(table_normal_base, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        return _invalid(timestamp, expected_marker_id, "invalid_table_normal")
    normal /= normal_norm
    low = np.asarray(workspace_base_low, dtype=np.float64).reshape(3)
    high = np.asarray(workspace_base_high, dtype=np.float64).reshape(3)
    candidates = []
    rejection_reasons = []
    for candidate_rvec, candidate_tvec in zip(rvecs, tvecs):
        rvec = np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
        marker_color = np.asarray(candidate_tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(rvec).all() or not np.isfinite(marker_color).all():
            rejection_reasons.append("non_finite_pose")
            continue
        if not float(min_depth_m) <= marker_color[2] <= float(max_depth_m):
            rejection_reasons.append(f"depth_out_of_range:{marker_color[2]:.6f}")
            continue
        rotation_color_marker, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            object_points, rvec, marker_color.reshape(3, 1),
            camera_matrix, np.zeros(5),
        )
        reprojection_error = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        if (not np.isfinite(reprojection_error)
                or reprojection_error > float(max_reprojection_error_px)):
            rejection_reasons.append(
                f"reprojection_error:{reprojection_error:.6f}")
            continue
        marker_base = transform_point(transform, marker_color)
        goal_base = marker_base + normal * (0.5 * float(cube_size_m))
        if not np.isfinite(goal_base).all():
            rejection_reasons.append("non_finite_goal")
            continue
        if np.any(goal_base < low) or np.any(goal_base > high):
            rejection_reasons.append(
                "goal_outside_workspace:" + np.array2string(goal_base))
            continue
        candidates.append((
            reprojection_error, rvec, marker_color,
            rotation_color_marker, marker_base, goal_base,
        ))
    if not candidates:
        reason = ("no_geometrically_valid_pnp_solution:"
                  + "|".join(rejection_reasons))
        return _invalid(timestamp, expected_marker_id, reason)
    (reprojection_error, rvec, marker_color, rotation_color_marker,
     marker_base, goal_base) = min(candidates, key=lambda item: item[0])
    rotation_base_marker = transform[:3, :3] @ rotation_color_marker
    confidence = float(np.clip(
        np.exp(-reprojection_error / max(float(max_reprojection_error_px), 1e-9))
        * min(1.0, pixel_area / max(4.0 * float(min_marker_area_px), 1.0)),
        0.0, 1.0,
    ))
    diagnostics = {
        "corners": image_points, "pixel_area": pixel_area,
        "reprojection_error_px": reprojection_error,
        "marker_position_color_optical": marker_color,
        "goal_position_base": goal_base.copy(),
        "pnp_candidate_count": len(rvecs),
        "geometrically_valid_candidate_count": len(candidates),
    }
    return GoalEstimate(
        position=goal_base, rotation=rotation_base_marker,
        timestamp=float(timestamp), valid=True, confidence=confidence,
        source="aruco_color_multiframe_candidate", estimate_id="unfrozen",
        marker_id=int(expected_marker_id), frame="base",
        diagnostics=diagnostics,
    )


@dataclass
class GoalCaptureSession:
    config: ArucoGoalConfig
    state: GoalCaptureState = GoalCaptureState.MOVE_TO_OBSERVE
    stable_steps: int = 0
    stable_since: float = None
    capture_frames: int = 0
    valid_estimates: list = field(default_factory=list)
    failure_counts: Counter = field(default_factory=Counter)
    frozen_estimate: GoalEstimate = None
    failure_reason: str = None

    @property
    def goal_is_frozen(self):
        return self.frozen_estimate is not None

    def reached_observe_command(self):
        if self.state != GoalCaptureState.MOVE_TO_OBSERVE:
            raise RuntimeError(f"cannot enter wait from {self.state}")
        self.state = GoalCaptureState.WAIT_OBSERVE_STABLE

    def update_stability(self, qpos, qvel, observe_q, now):
        if self.state != GoalCaptureState.WAIT_OBSERVE_STABLE:
            return False
        q_error = float(np.max(np.abs(np.asarray(qpos) - np.asarray(observe_q))))
        qvel_max = float(np.max(np.abs(np.asarray(qvel))))
        stable = (q_error <= self.config.observe_q_tolerance_rad
                  and qvel_max <= self.config.observe_qvel_tolerance_rad_s)
        if stable:
            if self.stable_steps == 0:
                self.stable_since = float(now)
            self.stable_steps += 1
        else:
            self.stable_steps = 0
            self.stable_since = None
        settled = (self.stable_since is not None
                   and float(now) - self.stable_since
                   >= self.config.observe_settle_seconds)
        if stable and settled and self.stable_steps >= self.config.observe_stable_steps:
            self.state = GoalCaptureState.CAPTURE_GOAL
            return True
        return False

    def submit_detection(self, detection, timestamp):
        """Accept a formal detection only inside the dedicated capture window."""
        if self.state != GoalCaptureState.CAPTURE_GOAL:
            return None
        self.capture_frames += 1
        estimate = detection
        if (estimate.valid and estimate.marker_id == self.config.marker_id
                and np.isfinite(estimate.position).all()):
            self.valid_estimates.append(estimate)
        else:
            self.failure_counts[estimate.failure_reason or "invalid"] += 1
        if len(self.valid_estimates) >= self.config.required_valid_frames:
            positions = np.stack([item.position for item in self.valid_estimates])
            position_std = np.std(positions, axis=0)
            if float(np.max(position_std)) <= self.config.position_std_threshold_m:
                position = np.median(positions, axis=0)
                rotations = [item.rotation for item in self.valid_estimates
                             if item.rotation is not None]
                rotation = None if not rotations else rotations[-1]
                confidence = float(np.mean([
                    item.confidence for item in self.valid_estimates]))
                self.frozen_estimate = GoalEstimate(
                    position=position, rotation=rotation,
                    timestamp=float(timestamp), valid=True,
                    confidence=confidence, source="aruco_color_multiframe_frozen",
                    estimate_id="pending-world-id", marker_id=self.config.marker_id,
                    frame="base",
                )
                self.state = GoalCaptureState.GOAL_FROZEN
                return self.frozen_estimate
            if self.capture_frames >= self.config.maximum_capture_frames:
                self.failure_reason = (
                    "goal_position_std_too_large:"
                    + np.array2string(position_std, precision=6))
                self.state = GoalCaptureState.FAILED
                return None
        if self.capture_frames >= self.config.maximum_capture_frames:
            self.failure_reason = (
                f"insufficient_valid_frames:{len(self.valid_estimates)}/"
                f"{self.config.required_valid_frames};failures={dict(self.failure_counts)}")
            self.state = GoalCaptureState.FAILED
        return None

    def begin_return_home(self):
        if self.state != GoalCaptureState.GOAL_FROZEN:
            raise RuntimeError("goal must be frozen before return home")
        self.state = GoalCaptureState.RETURN_HOME

    def begin_policy_execution(self):
        if self.state != GoalCaptureState.RETURN_HOME or not self.goal_is_frozen:
            raise RuntimeError("invalid transition to policy execution")
        self.state = GoalCaptureState.POLICY_EXECUTION
