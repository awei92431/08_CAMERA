"""Build a 39-D PPO-compatible shadow observation from visual localization.

The adapter is deliberately independent of MuJoCo object truth.  It receives
an object centre in the UR5e base frame and converts it to MuJoCo world before
rebuilding every object-position-derived policy field.  The returned shadow
observation is for diagnostics only in stage 6A.
"""

from dataclasses import dataclass

import numpy as np

from fourc2.camera_geometry import invert_transform, transform_point


FIELD_SLICES = {
    "pinch_position": slice(12, 15),
    "object_position": slice(15, 18),
    "pregrasp_position": slice(18, 21),
    "grasp_position": slice(21, 24),
    "pinch_to_pregrasp": slice(24, 27),
    "pinch_to_grasp": slice(27, 30),
    "object_to_goal": slice(30, 33),
    "gripper_state": slice(33, 34),
    "object_lift": slice(34, 35),
    "unchanged_privileged": slice(35, 39),
}


@dataclass
class VisualObservationResult:
    valid: bool
    failure_reason: str | None
    observation: np.ndarray | None
    estimated_object_world: np.ndarray | None
    estimated_initial_object_z: float | None


class VisualObservationAdapter:
    """Stateful per-episode adapter for visual object observations."""

    def __init__(self, pregrasp_height, grasp_height_offset):
        self.pregrasp_height = float(pregrasp_height)
        self.grasp_height_offset = float(grasp_height_offset)
        self.estimated_initial_object_z = None

    def reset(self):
        """Start an episode; initial visual Z is acquired on first valid frame."""
        self.estimated_initial_object_z = None

    def build(self, original_observation, estimated_object_center_base,
              t_base_world, goal_position_world):
        """Return a coherent shadow observation without reading object truth.

        T_A_B maps B-frame points into frame A, hence Base -> World uses
        T_world_base = inverse(T_base_world).
        """
        if estimated_object_center_base is None:
            return VisualObservationResult(
                False, "visual_localization_failed", None, None,
                self.estimated_initial_object_z,
            )
        estimated_base = np.asarray(
            estimated_object_center_base, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(estimated_base).all():
            return VisualObservationResult(
                False, "non_finite_visual_position", None, None,
                self.estimated_initial_object_z,
            )
        original = np.asarray(original_observation)
        if original.shape != (39,):
            return VisualObservationResult(
                False, f"expected_39d_observation_got_{original.shape}",
                None, None, self.estimated_initial_object_z,
            )
        goal_world = np.asarray(goal_position_world, dtype=np.float64).reshape(3)
        if not np.isfinite(goal_world).all():
            return VisualObservationResult(
                False, "non_finite_goal_position", None, None,
                self.estimated_initial_object_z,
            )

        t_world_base = invert_transform(t_base_world)
        estimated_world = transform_point(t_world_base, estimated_base)
        if self.estimated_initial_object_z is None:
            self.estimated_initial_object_z = float(estimated_world[2])

        shadow = original.copy()
        pinch = original[FIELD_SLICES["pinch_position"]].astype(np.float64)
        pregrasp = estimated_world + np.array(
            [0.0, 0.0, self.pregrasp_height]
        )
        grasp = estimated_world + np.array(
            [0.0, 0.0, self.grasp_height_offset]
        )
        # Preserve the environment's original stage gate exactly.  In all
        # non-Place stages the original object_to_goal slice is identically 0.
        place_gate_open = bool(np.any(
            original[FIELD_SLICES["object_to_goal"]] != 0.0
        ))
        object_to_goal = (
            goal_world - estimated_world if place_gate_open
            else np.zeros(3, dtype=np.float64)
        )
        lift = max(
            0.0, float(estimated_world[2]) - self.estimated_initial_object_z
        )

        values = {
            "object_position": estimated_world,
            "pregrasp_position": pregrasp,
            "grasp_position": grasp,
            "pinch_to_pregrasp": pregrasp - pinch,
            "pinch_to_grasp": grasp - pinch,
            "object_to_goal": object_to_goal,
            "object_lift": np.array([lift], dtype=np.float64),
        }
        for name, value in values.items():
            shadow[FIELD_SLICES[name]] = value
        return VisualObservationResult(
            True, None, shadow, estimated_world,
            self.estimated_initial_object_z,
        )


def internal_consistency_residual(observation, pregrasp_height,
                                  grasp_height_offset,
                                  estimated_initial_object_z,
                                  goal_position_world=None):
    """Return the largest residual among the adapter's defining equations."""
    obs = np.asarray(observation, dtype=np.float64)
    obj = obs[FIELD_SLICES["object_position"]]
    pinch = obs[FIELD_SLICES["pinch_position"]]
    pregrasp = obs[FIELD_SLICES["pregrasp_position"]]
    grasp = obs[FIELD_SLICES["grasp_position"]]
    residuals = [
        np.max(np.abs(pregrasp - obj - [0.0, 0.0, pregrasp_height])),
        np.max(np.abs(grasp - obj - [0.0, 0.0, grasp_height_offset])),
        np.max(np.abs(
            obs[FIELD_SLICES["pinch_to_pregrasp"]] - (pregrasp - pinch)
        )),
        np.max(np.abs(
            obs[FIELD_SLICES["pinch_to_grasp"]] - (grasp - pinch)
        )),
        abs(
            float(obs[FIELD_SLICES["object_lift"]][0])
            - max(0.0, float(obj[2]) - float(estimated_initial_object_z))
        ),
    ]
    object_to_goal = obs[FIELD_SLICES["object_to_goal"]]
    if goal_position_world is None or not np.any(object_to_goal != 0.0):
        residuals.append(np.max(np.abs(object_to_goal)))
    else:
        residuals.append(np.max(np.abs(
            object_to_goal - (np.asarray(goal_position_world) - obj)
        )))
    return float(max(residuals))
