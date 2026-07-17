from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from fourc2.object_estimate import (
    ObjectEstimate,
    ObjectEstimateAuthority,
    ObjectEstimateUnavailable,
)
from fourc2.goal_estimate import (
    GoalEstimate,
    GoalEstimateAuthority,
    GoalEstimateUnavailable,
)
from fourc2.task_supervisor import (
    GripperState,
    RobotState,
    SupervisorInputUnavailable,
    TaskSupervisor,
    TaskSupervisorConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "scene.xml"

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_ACTUATOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

PINCH_SITE_NAME = "pinch"
ATTACHMENT_SITE_NAME = "attachment_site"
OBJECT_SITE_NAME = "object_site"
GOAL_SITE_NAME = "goal_site"
OBJECT_JOINT_NAME = "object_freejoint"
GOAL_BODY_NAME = "goal"
MOCAP_TARGET_BODY_NAME = "mocap_target"
MOCAP_TARGET_SITE_NAME = "mocap_target_site"
MOCAP_TCP_WELD_NAME = "mocap_tcp_weld"
HOME_KEY_NAME = "home"
GRIPPER_ACTUATOR_NAME = "fingers_actuator"
TABLE_GEOM_NAME = "table_top"
OBJECT_GEOM_NAME = "object_geom"
LEFT_PAD_GEOM_NAMES = [
    # l_3 is the visible terminal finger link that contacts the workpiece.
    "l_3_link_collision",
    "left_pad1_collision",
    "left_pad2_collision",
]
RIGHT_PAD_GEOM_NAMES = [
    # r_3 is the visible terminal finger link that contacts the workpiece.
    "r_3_link_collision",
    "right_pad1_collision",
    "right_pad2_collision",
]

STAGE_REACH = 0
STAGE_GRASP = 1
STAGE_LIFT = 2
STAGE_FULL = 3
STAGE_REACH_GRASP = 4
STAGE_PLACE = 5

STAGE_NAMES = {
    STAGE_REACH: "reach",
    STAGE_GRASP: "grasp",
    STAGE_LIFT: "lift",
    STAGE_FULL: "full",
    STAGE_REACH_GRASP: "reach_grasp",
    STAGE_PLACE: "place",
}

GRASP_PHASE_ALIGN = 0
GRASP_PHASE_DESCEND = 1
GRASP_PHASE_CLOSE = 2
GRASP_PHASE_CONFIRM = 3

GRASP_PHASE_NAMES = {
    GRASP_PHASE_ALIGN: "align",
    GRASP_PHASE_DESCEND: "descend",
    GRASP_PHASE_CLOSE: "close",
    GRASP_PHASE_CONFIRM: "confirm",
}

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.4,
    "azimuth": 135.0,
    "elevation": -30.0,
    "lookat": np.array([0.35, 0.0, 0.35]),
}


class My4C2AllStageEnv(gym.Env):
    """Self-contained staged UR5e + 4C2 grasp environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        render_mode=None,
        frame_skip=10,
        distance_threshold=0.04,
        action_scale=0.05,
        max_tcp_lead=0.03,
        ik_axis_weight=0.35,
        ik_posture_weight=0.02,
        ik_posture_mode="off",
        place_xy_control_mode="combined",
        place_oracle_xy_gain=0.30,
        pregrasp_height=0.07,
        grasp_height_offset=0.0,
        lift_height=0.08,
        training_stage=STAGE_FULL,
        model_xml_path=None,
        object_half_size=0.03,
        close_reward_distance=None,
        contact_reward_distance=None,
        rough_close_reward_distance=None,
        rough_grasp_xy_close_threshold=None,
        rough_grasp_z_close_threshold=None,
        grasp_descend_xy_threshold=None,
        grasp_xy_close_threshold=None,
        grasp_z_close_threshold=None,
        latch_distance_threshold=None,
        max_pregrasp_object_xy_drift=None,
        stable_grasp_xy_threshold=None,
        latch_grasp_xy_threshold=None,
        latch_grasp_z_threshold=None,
        place_handoff_xy_threshold=None,
        release_open_xy_threshold=None,
        release_success_lift=None,
        release_min_open_steps=None,
        release_descent_action_scale=None,
        place_descent_xy_threshold=None,
        place_xy_servo_gain=None,
        place_xy_servo_max_delta=None,
        place_success_bonus=None,
        reset_stage_probabilities=None,
        include_stage_observation=False,
        curriculum_place_reset_reward_scale=0.05,
        sequential_training=False,
        success_stage=None,
        place_xy_action_scale=0.20,
        place_handoff_hold_steps=5,
        place_reset_variation=False,
        object_observation_mode="ground_truth",
        object_estimate_max_age=60.0,
        goal_observation_mode="ground_truth",
        fsm_mode="privileged_fsm",
        simulated_latch_physics=True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.distance_threshold = distance_threshold
        self.action_scale = action_scale
        self.max_tcp_lead = None if max_tcp_lead is None else float(max_tcp_lead)
        self.ik_axis_weight = float(ik_axis_weight)
        self.ik_posture_weight = float(ik_posture_weight)
        # The arm position-servo 2x gains are already fixed in ur5e_4c2.xml
        # (2000 for the first three axes and 500 for the wrists).  Do not
        # multiply model gains again here.
        self.arm_kp_scale = 2.0
        # The verified default disables the old raw joint-space correction:
        # directly adding it can override the primary Cartesian DLS task.
        # This six-joint, near-full-rank six-dimensional task has almost no
        # useful null space.  The nullspace mode remains available for future
        # redundant robots or lower-dimensional Cartesian tasks.
        self.ik_posture_mode = str(ik_posture_mode)
        if self.ik_posture_mode not in ("raw", "off", "nullspace"):
            raise ValueError(f"Unknown IK posture mode: {ik_posture_mode}")
        self.place_xy_control_mode = str(place_xy_control_mode)
        if self.place_xy_control_mode not in ("combined", "servo_only", "policy_only", "oracle"):
            raise ValueError(f"Unknown place_xy_control_mode: {place_xy_control_mode}")
        self.place_oracle_xy_gain = float(place_oracle_xy_gain)
        self.object_observation_mode = str(object_observation_mode)
        self.object_estimate_authority = ObjectEstimateAuthority(
            self.object_observation_mode, max_age=float(object_estimate_max_age)
        )
        self.goal_observation_mode = str(goal_observation_mode)
        self.goal_estimate_authority = GoalEstimateAuthority(
            self.goal_observation_mode
        )
        self.control_step_index = 0
        self.object_estimate_sequence = 0
        self.goal_estimate_sequence = 0
        self.fsm_mode = str(fsm_mode)
        if self.fsm_mode not in ("privileged_fsm", "deployable_fsm"):
            raise ValueError(f"unknown fsm_mode: {fsm_mode}")
        self.simulated_latch_physics = bool(simulated_latch_physics)
        if self.max_tcp_lead is not None and self.max_tcp_lead <= 0.0:
            raise ValueError("max_tcp_lead must be positive or None")
        self.pregrasp_height = pregrasp_height
        self.grasp_height_offset = grasp_height_offset
        self.lift_height = lift_height
        self.training_stage = int(training_stage)
        if self.training_stage not in STAGE_NAMES:
            raise ValueError(f"Unknown training_stage: {training_stage}")
        self.reset_stage_probabilities = self._normalize_reset_stage_probabilities(
            reset_stage_probabilities
        )
        self.sequential_training = bool(sequential_training)
        default_success_stage = {
            STAGE_REACH_GRASP: STAGE_GRASP,
            STAGE_FULL: STAGE_PLACE,
        }.get(self.training_stage, self.training_stage)
        self.success_stage = (
            default_success_stage if success_stage is None else int(success_stage)
        )
        if self.success_stage not in (STAGE_REACH, STAGE_GRASP, STAGE_LIFT, STAGE_PLACE):
            raise ValueError(f"Unknown success_stage: {success_stage}")
        self.include_stage_observation = bool(include_stage_observation)
        self.curriculum_place_reset_reward_scale = float(
            curriculum_place_reset_reward_scale
        )
        if not 0.0 < self.curriculum_place_reset_reward_scale <= 1.0:
            raise ValueError(
                "curriculum_place_reset_reward_scale must be in (0, 1]"
            )

        if model_xml_path is None:
            self.model_path = XML_PATH
        else:
            model_path = Path(model_xml_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            self.model_path = model_path
        self.table_top_z = 0.30
        self.object_half_size = float(object_half_size)
        self.object_center_z = self.table_top_z + self.object_half_size
        self.gripper_lowest_point_below_pinch = 0.027
        self.gripper_table_clearance_margin = 0.002
        self.pinch_min_z_over_table = (
            self.table_top_z
            + self.gripper_lowest_point_below_pinch
            + self.gripper_table_clearance_margin
        )
        self.attachment_min_z_over_table = self.table_top_z + 0.06
        self.desired_approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.approach_axis_index = 2
        self.approach_threshold = min(self.distance_threshold, 0.035)
        self.reach_xy_threshold = 0.018
        self.reach_z_threshold = 0.025
        self.reach_tcp_error_threshold = 0.045
        self.reach_vertical_threshold = 0.95
        self.grasp_threshold = 0.055
        self.close_gripper_threshold = 0.35
        self.close_reward_distance = (
            0.045 if close_reward_distance is None else float(close_reward_distance)
        )
        self.contact_reward_distance = (
            0.06 if contact_reward_distance is None else float(contact_reward_distance)
        )
        self.rough_close_reward_distance = (
            0.085
            if rough_close_reward_distance is None
            else float(rough_close_reward_distance)
        )
        self.rough_grasp_xy_close_threshold = (
            0.070
            if rough_grasp_xy_close_threshold is None
            else float(rough_grasp_xy_close_threshold)
        )
        self.rough_grasp_z_close_threshold = (
            0.070
            if rough_grasp_z_close_threshold is None
            else float(rough_grasp_z_close_threshold)
        )
        self.grasp_descend_xy_threshold = (
            0.040
            if grasp_descend_xy_threshold is None
            else float(grasp_descend_xy_threshold)
        )
        self.grasp_alignment_height = self.pregrasp_height * 0.85
        self.grasp_xy_close_threshold = (
            0.035 if grasp_xy_close_threshold is None else float(grasp_xy_close_threshold)
        )
        self.grasp_z_close_threshold = (
            0.035 if grasp_z_close_threshold is None else float(grasp_z_close_threshold)
        )
        self.latch_distance_threshold = (
            0.085 if latch_distance_threshold is None else float(latch_distance_threshold)
        )
        self.lift_grasp_distance_threshold = 0.10
        self.max_pregrasp_object_xy_drift = (
            0.02
            if max_pregrasp_object_xy_drift is None
            else float(max_pregrasp_object_xy_drift)
        )
        self.max_grasp_object_xy_drift = 0.12
        self.max_latched_object_xy_drift = 0.015
        self.latch_grasp_xy_threshold = (
            self.grasp_xy_close_threshold
            if latch_grasp_xy_threshold is None
            else float(latch_grasp_xy_threshold)
        )
        self.latch_grasp_z_threshold = (
            self.grasp_z_close_threshold
            if latch_grasp_z_threshold is None
            else float(latch_grasp_z_threshold)
        )
        # Stable success accepts a physically secure grasp anywhere inside the
        # object's top footprint.  The tighter latch threshold remains the
        # strict/centered quality threshold for diagnostics and reward shaping.
        self.stable_grasp_xy_threshold = (
            self.object_half_size
            if stable_grasp_xy_threshold is None
            else float(stable_grasp_xy_threshold)
        )
        self.strict_grasp_xy_threshold = self.latch_grasp_xy_threshold
        self.required_bilateral_contact_steps = 5
        # This counter is updated once per policy/control step, not per MuJoCo
        # substep.  With frame_skip=10 this represents about 100 ms.
        self.grasp_stable_required_steps = 5
        self.object_table_margin = 0.02
        self.table_x_bounds = (0.20, 1.10)
        self.table_y_bounds = (-0.35, 0.35)
        self.task_workspace_low = np.array(
            [-0.20, -0.60, 0.20],
            dtype=np.float64,
        )
        self.task_workspace_high = np.array(
            [0.80, 0.60, 0.85],
            dtype=np.float64,
        )
        self.tcp_table_z_min = self.pinch_min_z_over_table
        self.tcp_target_smoothing = 0.50
        self.reach_target_z_margin = 0.12
        self.grasp_xy_action_scale = 0.55
        self.grasp_z_action_scale = 0.32
        self.grasp_descend_min_action = 0.25
        self.grasp_target_min_z = max(
            self.pinch_min_z_over_table,
            self.table_top_z + max(
                0.012,
                0.80 * self.object_half_size,
            ),
        )
        self.lift_target_extra_height = 0.04
        self.grasp_target_xy_margin = 0.055
        self.grasp_target_z_margin = 0.015
        self.early_close_gripper_limit = 0.25
        self.min_close_steps = 4
        self.max_close_attempt_steps = 24
        self.place_xy_threshold = max(0.03, 2.0 * self.object_half_size)
        self.place_handoff_xy_threshold = (
            0.85 * self.place_xy_threshold
            if place_handoff_xy_threshold is None
            else float(place_handoff_xy_threshold)
        )
        if place_handoff_xy_threshold is not None:
            self.place_xy_threshold = self.place_handoff_xy_threshold
        self.place_descent_xy_threshold = (
            max(self.place_xy_threshold, 0.028)
            if place_descent_xy_threshold is None
            else float(place_descent_xy_threshold)
        )
        self.place_xy_servo_gain = (
            0.55 if place_xy_servo_gain is None else float(place_xy_servo_gain)
        )
        self.place_xy_servo_max_delta = (
            0.012
            if place_xy_servo_max_delta is None
            else float(place_xy_servo_max_delta)
        )
        self.place_target_extra_height = 0.025
        # PPO is a residual around the deterministic Place XY servo.  Keep the
        # learned residual smaller than the 12 mm servo correction; the old
        # 1.80 multiplier allowed a 90 mm opposing action and caused the
        # merged policy to fight a controller that already solved Place.
        self.place_xy_action_scale = float(place_xy_action_scale)
        self.place_handoff_hold_steps = max(int(place_handoff_hold_steps), 0)
        self.place_reset_variation = bool(place_reset_variation)
        self.place_handoff_count = 0
        self.place_goal_radius_range = (0.05, 0.13)
        self.latched_gripper_normalized = 0.70
        self.release_descent_distance = 0.04
        self.release_success_lift = (
            0.012 if release_success_lift is None else float(release_success_lift)
        )
        self.release_tcp_min_lift = min(self.release_success_lift, 0.010)
        self.release_tcp_min_z = (
            self.table_top_z + self.object_half_size + self.release_tcp_min_lift
        )
        self.release_open_after_steps = 30
        self.release_min_open_steps = (
            6 if release_min_open_steps is None else int(release_min_open_steps)
        )
        self.release_open_xy_threshold = (
            self.place_handoff_xy_threshold
            if release_open_xy_threshold is None
            else float(release_open_xy_threshold)
        )
        self.release_descent_action_scale = (
            0.35
            if release_descent_action_scale is None
            else float(release_descent_action_scale)
        )
        self.penetration_tolerance = 0.004
        self.penetration_failure_tolerance = 1.5 * self.penetration_tolerance
        self.grasp_upright_threshold = 0.90
        self.grasp_success_max_xy_drift = self.max_pregrasp_object_xy_drift
        self.grasp_success_penetration_tolerance = self.penetration_tolerance
        self.gripper_command_rate = 0.04
        self.table_side_margin = 0.05
        self.table_side_clearance_z = self.table_top_z + 0.10

        self.reach_success_bonus = 8.0
        self.grasp_success_bonus = 12.0
        self.lift_success_bonus = 12.0
        self.place_success_bonus = (
            24.0 if place_success_bonus is None else float(place_success_bonus)
        )
        self.reach_completed_base_reward = 0.05
        self.grasp_completed_base_reward = 0.08
        self.lift_completed_base_reward = 0.10

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config=DEFAULT_CAMERA_CONFIG,
        )

        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.arm_joint_ids < 0):
            raise ValueError(f"Missing arm joints: {ARM_JOINT_NAMES}")

        self.arm_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.arm_actuator_ids < 0):
            raise ValueError(f"Missing arm actuators: {ARM_ACTUATOR_NAMES}")


        self.object_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            OBJECT_JOINT_NAME,
        )
        if self.object_joint_id < 0:
            raise ValueError(f"Missing object freejoint: {OBJECT_JOINT_NAME}")

        self.pinch_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            PINCH_SITE_NAME,
        )
        self.attachment_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            ATTACHMENT_SITE_NAME,
        )
        self.object_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            OBJECT_SITE_NAME,
        )
        self.goal_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            GOAL_SITE_NAME,
        )
        if min(
            self.pinch_site_id,
            self.attachment_site_id,
            self.object_site_id,
            self.goal_site_id,
        ) < 0:
            raise ValueError("Missing one of pinch/attachment/object/goal sites")

        self.goal_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            GOAL_BODY_NAME,
        )
        if self.goal_body_id < 0:
            raise ValueError(f"Missing goal body: {GOAL_BODY_NAME}")
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body_id])
        if self.goal_mocap_id < 0:
            raise ValueError(f"{GOAL_BODY_NAME} is not a mocap body")

        self.tcp_mocap_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            MOCAP_TARGET_BODY_NAME,
        )
        if self.tcp_mocap_body_id < 0:
            raise ValueError(f"Missing mocap body: {MOCAP_TARGET_BODY_NAME}")
        self.tcp_mocap_id = int(self.model.body_mocapid[self.tcp_mocap_body_id])
        if self.tcp_mocap_id < 0:
            raise ValueError(f"{MOCAP_TARGET_BODY_NAME} is not a mocap body")

        self.tcp_mocap_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            MOCAP_TARGET_SITE_NAME,
        )
        if self.tcp_mocap_site_id < 0:
            raise ValueError(f"Missing mocap target site: {MOCAP_TARGET_SITE_NAME}")

        self.tcp_weld_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            MOCAP_TCP_WELD_NAME,
        )
        # The IK test model intentionally has no TCP mocap weld.  Keeping the
        # mocap marker is harmless and useful for visualising the requested
        # target, but it must not exert forces on the robot.
        self.tcp_weld_enabled = self.tcp_weld_id >= 0

        self.home_key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            HOME_KEY_NAME,
        )
        if self.home_key_id < 0:
            raise ValueError(f"Missing keyframe: {HOME_KEY_NAME}")

        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            GRIPPER_ACTUATOR_NAME,
        )
        if self.gripper_actuator_id < 0:
            raise ValueError(f"Missing gripper actuator: {GRIPPER_ACTUATOR_NAME}")
        self.gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_1_joint"
        )
        if self.gripper_joint_id < 0:
            raise ValueError("Missing gripper feedback joint: r_1_joint")
        self.gripper_qpos_id = int(self.model.jnt_qposadr[self.gripper_joint_id])
        self.gripper_qvel_id = int(self.model.jnt_dofadr[self.gripper_joint_id])

        self.table_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            TABLE_GEOM_NAME,
        )
        self.object_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            OBJECT_GEOM_NAME,
        )
        self.left_pad_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in LEFT_PAD_GEOM_NAMES
        }
        self.right_pad_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in RIGHT_PAD_GEOM_NAMES
        }
        self.left_pad_geom_ids.discard(-1)
        self.right_pad_geom_ids.discard(-1)
        if not self.left_pad_geom_ids or not self.right_pad_geom_ids:
            raise ValueError("Missing gripper pad geoms")

        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_qvel_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.home_arm_qpos = self.model.key_qpos[
            self.home_key_id,
            self.arm_qpos_ids,
        ].copy()

        self.object_qpos_id = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_id = int(self.model.jnt_dofadr[self.object_joint_id])
        self.object_qpos_slice = slice(self.object_qpos_id, self.object_qpos_id + 7)
        self.object_qvel_slice = slice(self.object_qvel_id, self.object_qvel_id + 6)

        ctrl_range = self.model.actuator_ctrlrange.astype(np.float64)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]
        self.arm_ctrl_low = self.ctrl_low[self.arm_actuator_ids]
        self.arm_ctrl_high = self.ctrl_high[self.arm_actuator_ids]
        self.gripper_ctrl_low = self.ctrl_low[self.gripper_actuator_id]
        self.gripper_ctrl_high = self.ctrl_high[self.gripper_actuator_id]

        self.object_initial_z = 0.0
        self.object_initial_position = np.zeros(3, dtype=np.float64)
        self.previous_object_position = np.zeros(3, dtype=np.float64)
        self.previous_distance = 0.0
        self.previous_pregrasp_xy_error = 0.0
        self.previous_pregrasp_abs_z_error = 0.0
        self.previous_grasp_xy_error = 0.0
        self.previous_grasp_abs_z_error = 0.0
        self.previous_lift = 0.0
        self.tcp_target_pos = np.zeros(3, dtype=np.float64)
        self.tcp_target_quat = np.array(
            [0.0, 0.70710678, 0.70710678, 0.0],
            dtype=np.float64,
        )
        self.stage = STAGE_REACH
        self.episode_start_stage = STAGE_REACH
        self.is_grasp_latched = False
        self.grasp_object_offset = np.zeros(3, dtype=np.float64)
        self.latched_object_xy = np.zeros(2, dtype=np.float64)
        self.max_pad_object_penetration = 0.0
        self.gripper_command_normalized = 0.0
        self.bilateral_contact_steps = 0
        self.grasp_stable_count = 0
        self.grasp_phase = GRASP_PHASE_ALIGN
        self.grasp_phase_steps = 0
        self.close_steps = 0
        self.release_has_opened = False
        self.place_descent_active = False
        self.place_handoff_count = 0
        self._episode_done = False
        self.ik_solve_calls = 0
        self.lead_clip_count = 0
        self.max_raw_tcp_lead = 0.0
        self.max_clipped_tcp_lead = 0.0
        self.last_ik_joint_target = self.home_arm_qpos.copy()
        self.task_supervisor = TaskSupervisor(TaskSupervisorConfig(
            pregrasp_height=self.pregrasp_height,
            grasp_height_offset=self.grasp_height_offset,
            reach_xy_threshold=self.reach_xy_threshold,
            reach_z_threshold=self.reach_z_threshold,
            reach_vertical_threshold=self.reach_vertical_threshold,
            grasp_descend_xy_threshold=self.grasp_descend_xy_threshold,
            grasp_xy_close_threshold=self.grasp_xy_close_threshold,
            grasp_z_close_threshold=self.grasp_z_close_threshold,
            stable_grasp_xy_threshold=self.stable_grasp_xy_threshold,
            latch_grasp_z_threshold=self.latch_grasp_z_threshold,
            lift_height=self.lift_height,
            release_success_lift=self.release_success_lift,
            release_open_xy_threshold=self.release_open_xy_threshold,
            release_min_open_steps=self.release_min_open_steps,
            min_close_steps=self.min_close_steps,
            max_close_attempt_steps=self.max_close_attempt_steps,
            stable_required_steps=self.grasp_stable_required_steps,
            latched_gripper_closure=self.latched_gripper_normalized,
        ))

        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43 if self.include_stage_observation else 39,),
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode is None:
            return None
        return self.mujoco_renderer.render(self.render_mode)

    def _normalize_reset_stage_probabilities(self, reset_stage_probabilities):
        if reset_stage_probabilities is None:
            return None

        if isinstance(reset_stage_probabilities, dict):
            items = reset_stage_probabilities.items()
        else:
            items = reset_stage_probabilities

        stages = []
        probabilities = []
        for stage, probability in items:
            stage = int(stage)
            probability = float(probability)
            if stage not in STAGE_NAMES:
                raise ValueError(f"Unknown reset stage: {stage}")
            if probability < 0.0:
                raise ValueError("Reset stage probabilities must be non-negative")
            if probability > 0.0:
                stages.append(stage)
                probabilities.append(probability)

        total_probability = float(sum(probabilities))
        if total_probability <= 0.0:
            raise ValueError("At least one reset stage probability must be positive")

        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities /= total_probability
        return tuple(stages), probabilities

    def _sample_reset_stage(self):
        if self.reset_stage_probabilities is None:
            return self.training_stage

        stages, probabilities = self.reset_stage_probabilities
        stage_index = int(self.np_random.choice(len(stages), p=probabilities))
        return stages[stage_index]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.object_estimate_authority.reset()
        self.goal_estimate_authority.reset()
        self.control_step_index = 0
        self.object_estimate_sequence = 0
        self.goal_estimate_sequence = 0
        self.task_supervisor.reset()
        self._episode_done = False
        self.lead_clip_count = 0
        self.max_raw_tcp_lead = 0.0
        self.max_clipped_tcp_lead = 0.0
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.data.ctrl[self.arm_actuator_ids] = self.home_arm_qpos
        self.stage = STAGE_REACH
        self.is_grasp_latched = False
        self.grasp_object_offset[:] = 0.0
        self.latched_object_xy[:] = 0.0
        self.max_pad_object_penetration = 0.0
        self._set_gripper_command_immediate(0.0)
        self.bilateral_contact_steps = 0
        self.grasp_stable_count = 0
        self.release_steps = 0
        self.release_start_lift = self.lift_height
        self.release_has_opened = False
        self.place_descent_active = False
        self._set_grasp_phase(GRASP_PHASE_ALIGN)

        self._set_object_position(self._sample_object_position())
        goal_position = self._sample_goal_position()
        self.data.mocap_pos[self.goal_mocap_id] = goal_position
        self.data.mocap_quat[self.goal_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)
        if self.goal_observation_mode == "ground_truth":
            self._publish_ground_truth_goal_estimate()
        self.object_initial_position = self.data.site_xpos[self.object_site_id].copy()
        self.previous_object_position = self.object_initial_position.copy()
        self.object_initial_z = self.object_initial_position[2]
        self._sync_mocap_target_to_tcp()
        if self.object_observation_mode == "ground_truth":
            self._publish_ground_truth_object_estimate()

        setup_stage = self._sample_reset_stage()
        self.episode_start_stage = setup_stage

        if setup_stage == STAGE_REACH:
            self.stage = STAGE_REACH
            self._set_gripper_command_immediate(0.0)
        elif setup_stage == STAGE_GRASP:
            self._set_object_position(self._sample_object_position())
            self.stage = STAGE_GRASP
            self._drive_to_target(
                self._pregrasp_position,
                max_steps=320,
                gripper_action=-1.0,
                tolerance=0.035,
            )
            self._set_gripper_command_immediate(0.0)
        elif setup_stage == STAGE_LIFT:
            self._set_object_position(self._sample_stage_object_position())
            self.stage = STAGE_GRASP
            self._drive_to_target(
                self._pregrasp_position,
                max_steps=320,
                gripper_action=-1.0,
                tolerance=0.030,
            )
            self._set_grasp_phase(GRASP_PHASE_DESCEND)
            self._drive_to_target(
                self._grasp_position,
                max_steps=240,
                gripper_action=-1.0,
                tolerance=0.010,
            )
            self._close_gripper_until_latched()
            self.stage = STAGE_LIFT
            self._set_grasp_phase(GRASP_PHASE_CONFIRM)
        elif setup_stage == STAGE_PLACE:
            self._set_object_position(self._sample_stage_object_position())
            goal_position = self._sample_goal_position()
            self.data.mocap_pos[self.goal_mocap_id] = goal_position
            self.data.mocap_quat[self.goal_mocap_id] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            mujoco.mj_forward(self.model, self.data)
            self.object_initial_position = self.data.site_xpos[
                self.object_site_id
            ].copy()
            self.previous_object_position = self.object_initial_position.copy()
            self.object_initial_z = self.object_initial_position[2]
            self.stage = STAGE_GRASP
            self._drive_to_target(
                self._pregrasp_position,
                max_steps=320,
                gripper_action=-1.0,
                tolerance=0.030,
            )
            self._set_grasp_phase(GRASP_PHASE_DESCEND)
            self._drive_to_target(
                self._grasp_position,
                max_steps=240,
                gripper_action=-1.0,
                tolerance=0.010,
            )
            self._close_gripper_until_latched()
            self.stage = STAGE_LIFT
            self._set_grasp_phase(GRASP_PHASE_CONFIRM)
            for _ in range(80):
                self._apply_raw_action(
                    np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                )
                lift_info = self._get_info()
                if lift_info["lift_success"]:
                    break
            self.stage = STAGE_PLACE
            self.place_descent_active = False
            self.place_handoff_count = self.place_handoff_hold_steps
            if self.place_reset_variation and self.is_grasp_latched:
                # Cover the measured states produced by the learned Lift
                # policy, instead of exposing Place only to a perfectly
                # centered scripted grasp.  Bounds cover the observed
                # ~6.2 mm XY and ~4.2 mm Z handoff errors.
                self.grasp_object_offset[:2] += self.np_random.uniform(
                    -0.0065, 0.0065, size=2
                )
                self.grasp_object_offset[2] += self.np_random.uniform(
                    -0.0045, 0.0045
                )
                self._set_gripper_command_immediate(
                    self.np_random.uniform(0.68, 0.78)
                )
                self._update_grasp_latch(update_counter=False)
        else:
            self.stage = STAGE_REACH
            self._set_gripper_command_immediate(0.0)

        mujoco.mj_forward(self.model, self.data)
        if setup_stage in (STAGE_LIFT, STAGE_PLACE) and not self.is_grasp_latched:
            self.object_initial_position = self.data.site_xpos[
                self.object_site_id
            ].copy()
            fallback_substeps = self.frame_skip * (
                self.required_bilateral_contact_steps + 1
            )
            for substep in range(fallback_substeps):
                self._neutralize_arm_actuators()
                mujoco.mj_step(self.model, self.data)
                self._update_grasp_latch(
                    update_counter=(substep + 1) % self.frame_skip == 0
                )
                if self.is_grasp_latched:
                    break
            mujoco.mj_forward(self.model, self.data)

        current_object_position = self.data.site_xpos[self.object_site_id].copy()
        if setup_stage != STAGE_PLACE:
            self.object_initial_position = current_object_position.copy()
            self.object_initial_z = current_object_position[2]
        self.previous_object_position = current_object_position.copy()

        self.object_estimate_authority.reset()
        self.object_estimate_sequence = 0
        if self.object_observation_mode == "ground_truth":
            self._publish_ground_truth_object_estimate()
        self.goal_estimate_authority.reset()
        self.goal_estimate_sequence = 0
        if self.goal_observation_mode == "ground_truth":
            self._publish_ground_truth_goal_estimate()
        obs = self._get_obs(
            allow_unavailable=(self.object_observation_mode == "rgbd"
                               or self.goal_observation_mode == "aruco")
        )
        info = self._get_info()
        self.previous_distance = info["distance"]
        self.previous_pregrasp_xy_error = info["pregrasp_xy_error"]
        self.previous_pregrasp_abs_z_error = abs(info["pregrasp_z_error"])
        self.previous_grasp_xy_error = info["grasp_xy_error"]
        self.previous_grasp_abs_z_error = abs(info["grasp_z_error"])
        self.previous_lift = info["object_lift"]
        self.previous_release_lift = info["object_lift"]

        if self.render_mode == "human":
            self.render()
        return obs, info

    def publish_object_estimate(self, estimate):
        """Publish the sole object state allowed on the runtime control path."""
        return self.object_estimate_authority.publish(estimate)

    def invalidate_object_estimate(self, reason, source="rgbd_invalid"):
        self.object_estimate_sequence += 1
        self.object_estimate_authority.invalidate(
            self.data.time, source,
            f"episode-{self.object_estimate_sequence:06d}", reason,
        )

    def _publish_ground_truth_object_estimate(self):
        if self.object_observation_mode != "ground_truth":
            raise RuntimeError("ground-truth publication forbidden outside ground_truth mode")
        self.object_estimate_sequence += 1
        return self.publish_object_estimate(ObjectEstimate(
            position=self.data.site_xpos[self.object_site_id].copy(),
            timestamp=float(self.data.time), valid=True, confidence=1.0,
            source="ground_truth_simulation",
            estimate_id=f"gt-{self.object_estimate_sequence:06d}",
        ))

    def _control_object_estimate(self, consumer):
        return self.object_estimate_authority.require(
            self.data.time, consumer, self.control_step_index
        )

    def _control_object_position(self, consumer):
        return self._control_object_estimate(consumer).position.copy()

    def publish_goal_estimate(self, estimate):
        """Publish the sole goal state allowed on the runtime control path."""
        return self.goal_estimate_authority.publish(estimate)

    def invalidate_goal_estimate(self, reason, source="aruco_invalid"):
        self.goal_estimate_sequence += 1
        self.goal_estimate_authority.invalidate(
            self.data.time, source,
            f"episode-goal-{self.goal_estimate_sequence:06d}", reason,
        )

    def _publish_ground_truth_goal_estimate(self):
        if self.goal_observation_mode != "ground_truth":
            raise RuntimeError(
                "ground-truth goal publication forbidden outside ground_truth mode")
        self.goal_estimate_sequence += 1
        return self.publish_goal_estimate(GoalEstimate(
            position=self.data.site_xpos[self.goal_site_id].copy(),
            timestamp=float(self.data.time), valid=True, confidence=1.0,
            source="ground_truth_simulation",
            estimate_id=f"gt-goal-{self.goal_estimate_sequence:06d}",
            frame="world",
        ))

    def _control_goal_estimate(self, consumer):
        return self.goal_estimate_authority.require(
            self.data.time, consumer, self.control_step_index
        )

    def _control_goal_position(self, consumer):
        return self._control_goal_estimate(consumer).position.copy()

    def control_observation(self):
        """Return the observation that may be sent to PPO for this control step."""
        return self._get_obs()

    def deployable_robot_state(self):
        tcp = self.data.site_xpos[self.pinch_site_id].copy()
        rotation = self.data.site_xmat[self.pinch_site_id].reshape(3, 3).copy()
        vertical = float(np.clip(np.dot(
            self._pinch_approach_axis(self.data), self.desired_approach_axis
        ), -1.0, 1.0))
        return RobotState(
            tcp_position=tcp, world_from_tcp=rotation,
            tcp_velocity=np.zeros(3, dtype=np.float64),
            vertical_alignment=vertical, timestamp=float(self.data.time),
            valid=True, source="mujoco_joint_fk_proxy",
        )

    def deployable_gripper_state(self):
        qpos = float(self.data.qpos[self.gripper_qpos_id])
        qvel = float(self.data.qvel[self.gripper_qvel_id])
        effort = float(self.data.actuator_force[self.gripper_actuator_id])
        closure = float(np.clip(qpos / 0.9, 0.0, 1.0))
        stopped = abs(qvel) < self.task_supervisor.config.stopped_velocity
        blocked = closure < self.task_supervisor.config.fully_closed_fraction
        effort_score = float(np.clip(
            abs(effort) / max(self.task_supervisor.config.effort_threshold, 1e-9),
            0.0, 1.0,
        ))
        hold_confidence = float(max(
            effort_score, 1.0 if stopped and blocked
            and self.gripper_command_normalized > 0.5 else 0.0
        ))
        return GripperState(
            commanded_opening=1.0 - self.gripper_command_normalized,
            actual_opening=1.0 - closure, velocity=-qvel,
            motion_status="stopped" if stopped else "moving",
            actuator_effort=effort, fault=False,
            grasp_hold_confidence=hold_confidence,
            timestamp=float(self.data.time), valid=True,
            source="mujoco_gripper_sensor_proxy",
        )

    def update_task_supervisor(self):
        """Update deployable supervisor without object/contact/reward truth."""
        estimate = self._control_object_estimate("task_supervisor")
        previous_stage = int(self.task_supervisor.stage)
        diagnostics = self.task_supervisor.update(
            self.data.time, estimate, self.deployable_robot_state(),
            self.deployable_gripper_state(),
            self._control_goal_position("task_supervisor"),
        )
        if self.fsm_mode == "deployable_fsm":
            self.stage = int(self.task_supervisor.stage)
            self.release_has_opened = bool(
                self.task_supervisor.release_commanded
            )
            if previous_stage != STAGE_PLACE and self.stage == STAGE_PLACE:
                self.release_steps = 0
                self.place_descent_active = False
                self.place_handoff_count = self.place_handoff_hold_steps
        return diagnostics

    def step(self, action):
        if self._episode_done:
            raise RuntimeError(
                "step() called after this episode terminated; call reset() first"
            )
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_tcp_action(action)

        self.control_step_index += 1
        if self.object_observation_mode == "ground_truth":
            self._publish_ground_truth_object_estimate()
        if self.goal_observation_mode == "ground_truth":
            self._publish_ground_truth_goal_estimate()

        obs = self._get_obs(consumer="post_step_observation")
        info = self._get_info()
        stage_before_reward = self.stage
        reward = self._compute_reward(info, action)
        if self.include_stage_observation and self.stage != stage_before_reward:
            # Full/reach-grasp reward handlers perform the stage transition.
            # Return an observation for the controller mode that will consume
            # the next action, rather than a one-step-stale previous stage.
            obs = self._get_obs()
        self._refresh_task_fields(info)
        # ReachGrasp ends on the exact control step where the existing
        # multi-step stable-grasp predicate becomes true.  Do not collect a
        # post-success hold/no-lift action: Lift owns the next state, in its
        # own episode or in an explicit sequential-transition environment.
        reach_grasp_success = bool(
            self.training_stage == STAGE_REACH_GRASP
            and info["stable_grasp_success"]
        )
        if reach_grasp_success:
            info["is_success"] = True
            info["stage_success"] = True
        if self.fsm_mode == "deployable_fsm":
            terminated = bool(self.task_supervisor.failed)
        else:
            terminated = bool(
                reach_grasp_success
                or info["is_success"]
                or info.get("stage_failure", False)
            )
        self._episode_done = terminated
        truncated = False

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _set_object_position(self, object_position):
        self.data.qpos[self.object_qpos_slice] = np.array(
            [
                object_position[0],
                object_position[1],
                object_position[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.data.qvel[self.object_qvel_slice] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_stage_object_position(self):
        return np.array(
            [
                self.np_random.uniform(0.25, 0.35),
                self.np_random.uniform(0.12, 0.22),
                self.object_center_z,
            ],
            dtype=np.float64,
        )

    def _drive_to_target(self, target_fn, max_steps, gripper_action, tolerance=0.06):
        for _ in range(max_steps):
            target_position = target_fn()
            pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
            delta = target_position - pinch_position
            if np.linalg.norm(delta) < tolerance:
                break
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(delta / self.action_scale, -1.0, 1.0)
            action[3] = gripper_action
            self._apply_raw_action(action)

    def _apply_raw_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_tcp_action(action)

    def _apply_tcp_action(self, action):
        self.diag_policy_action = np.asarray(action, dtype=np.float64).copy()
        self.diag_tcp_actual_before = self.data.site_xpos[self.pinch_site_id].copy()
        self.diag_object_before = self.data.site_xpos[self.object_site_id].copy()
        self.diag_qpos_before = self.data.qpos[self.arm_qpos_ids].copy()
        self.diag_tcp_target_before = self.tcp_target_pos.copy()
        self.diag_stage_before = int(self.stage)
        if self.stage == STAGE_GRASP and not self.is_grasp_latched:
            self._update_grasp_phase()

        delta_direction = action[:3].astype(np.float64)
        delta_norm = np.linalg.norm(delta_direction)
        if delta_norm > 1.0:
            delta_direction = delta_direction / delta_norm

        target_delta = self.action_scale * delta_direction
        if self.stage == STAGE_GRASP and not self.is_grasp_latched:
            target_delta[:2] *= self.grasp_xy_action_scale
            target_delta[2] = (
                self.action_scale
                * self.grasp_z_action_scale
                * float(np.clip(action[2], -1.0, 0.35))
            )
            if (
                self.grasp_phase == GRASP_PHASE_ALIGN
                or not self._grasp_xy_aligned()
            ):
                target_delta[2] = max(target_delta[2], 0.0)
            elif self.grasp_phase == GRASP_PHASE_DESCEND:
                target_delta[2] = min(
                    target_delta[2],
                    -self.action_scale
                    * self.grasp_z_action_scale
                    * self.grasp_descend_min_action,
                )
        if self.stage == STAGE_PLACE:
            object_position = self._control_object_position("place_servo")
            goal_position = self._control_goal_position("place_servo")
            goal_delta_xy = (goal_position - object_position)[:2]
            goal_xy = float(np.linalg.norm(goal_delta_xy))
            target_delta[:2] *= self.place_xy_action_scale
            self.diag_place_policy_xy_initial = target_delta[:2].copy()
            handoff_hold_active = self.place_handoff_count > 0
            if handoff_hold_active:
                # Do not combine the final upward Lift command with an
                # immediate horizontal Place target jump.  This short hold is
                # part of the physical transition, not a separate controller.
                target_delta[:] = 0.0
                self.place_handoff_count -= 1
            if goal_xy < self.place_descent_xy_threshold:
                self.place_descent_active = True
            if self.release_has_opened:
                target_delta[:2] = 0.0
                # Keep the TCP at the release pose while the actuator opens.
                # The gripper command is rate-limited, so retracting as soon
                # as release_has_opened becomes true carries the object upward
                # for several control steps before dropping it.  The episode
                # terminates after the jaws are open and the latch is gone;
                # no post-release retract is needed inside this task.
                target_delta[2] = 0.0
            elif self.place_descent_active:
                target_delta[:2] *= 0.15
                if self._release_low_enough(
                    object_position, consumer="place_descent"
                ) or self._release_should_open():
                    target_delta[2] = 0.0
                else:
                    target_delta[2] = (
                        -self.action_scale * self.release_descent_action_scale
                    )
            else:
                target_delta[2] *= 0.10
            policy_xy = target_delta[:2].copy()
            servo_xy = np.zeros(2, dtype=np.float64)
            if (
                not handoff_hold_active
                and not self.release_has_opened
                and goal_xy > 1e-6
            ):
                servo_xy = self.place_xy_servo_gain * goal_delta_xy
                servo_norm = float(np.linalg.norm(servo_xy))
                if servo_norm > self.place_xy_servo_max_delta:
                    servo_xy *= self.place_xy_servo_max_delta / servo_norm
            if self.place_xy_control_mode == "combined":
                target_delta[:2] = policy_xy + servo_xy
            elif self.place_xy_control_mode == "servo_only":
                target_delta[:2] = servo_xy
            elif self.place_xy_control_mode == "policy_only":
                target_delta[:2] = policy_xy
            else:
                oracle_xy = self.place_oracle_xy_gain * goal_delta_xy
                oracle_norm = float(np.linalg.norm(oracle_xy))
                if oracle_norm > self.place_xy_servo_max_delta:
                    oracle_xy *= self.place_xy_servo_max_delta / oracle_norm
                target_delta[:2] = (
                    np.zeros(2, dtype=np.float64)
                    if handoff_hold_active or self.release_has_opened
                    else oracle_xy
                )
            self.diag_place_policy_xy = policy_xy.copy()
            self.diag_place_servo_xy = servo_xy.copy()
            self.diag_place_final_delta_xy = target_delta[:2].copy()
        elif self.stage >= STAGE_LIFT or self.is_grasp_latched:
            target_delta[:2] = 0.0
            # Lift is an upward-only task.  Map the full policy range onto an
            # upward command so the inherited negative ReachGrasp Z mean does
            # not sit in a zero-motion dead zone.  -1 means no lift and +1
            # means full lift; the action space and all other stages remain
            # unchanged.
            lift_up_action = 0.5 * (float(np.clip(action[2], -1.0, 1.0)) + 1.0)
            target_delta[2] = self.action_scale * lift_up_action

        target_base_pos = self.tcp_target_pos
        if self.stage == STAGE_PLACE:
            if self.release_has_opened:
                target_base_pos = self.tcp_target_pos.copy()
            else:
                target_base_pos = self.data.site_xpos[self.pinch_site_id].copy()

        raw_target_pos = target_base_pos + target_delta
        safe_target_pos = self._safe_tcp_target_pos(raw_target_pos)
        self.diag_scaled_tcp_action = target_delta.copy()
        self.diag_target_base = np.asarray(target_base_pos).copy()
        self.diag_raw_target = raw_target_pos.copy()
        self.diag_safe_target = safe_target_pos.copy()
        self.diag_workspace_clip = bool(not np.allclose(raw_target_pos, safe_target_pos, atol=1e-12))
        alpha = 0.75 if self.stage == STAGE_PLACE else self.tcp_target_smoothing
        self.diag_target_before_smoothing = self.tcp_target_pos.copy()
        self.tcp_target_pos = (1.0 - alpha) * self.tcp_target_pos + alpha * safe_target_pos
        self.tcp_target_pos = self._safe_tcp_target_pos(self.tcp_target_pos)
        self.diag_target_after_smoothing = self.tcp_target_pos.copy()

        actual_tcp = self.data.site_xpos[self.pinch_site_id].copy()
        lead = self.tcp_target_pos - actual_tcp
        lead_norm = float(np.linalg.norm(lead))
        self.diag_raw_tcp_lead = lead_norm
        self.diag_target_before_lead_clip = self.tcp_target_pos.copy()
        self.max_raw_tcp_lead = max(self.max_raw_tcp_lead, lead_norm)
        self.diag_lead_clip = False
        if self.max_tcp_lead is not None and lead_norm > self.max_tcp_lead:
            self.tcp_target_pos = (
                actual_tcp
                + lead / (lead_norm + 1e-8) * self.max_tcp_lead
            )
            self.lead_clip_count += 1
            self.diag_lead_clip = True
        clipped_lead = float(np.linalg.norm(self.tcp_target_pos - actual_tcp))
        self.diag_clipped_tcp_lead = clipped_lead
        self.diag_target_after_lead_clip = self.tcp_target_pos.copy()
        self.max_clipped_tcp_lead = max(self.max_clipped_tcp_lead, clipped_lead)

        self.data.mocap_pos[self.tcp_mocap_id] = self.tcp_target_pos
        self.data.mocap_quat[self.tcp_mocap_id] = self.tcp_target_quat
        self._set_gripper_from_action(action)

        # Execution-layer replacement: the unchanged Cartesian target is
        # converted by the existing Jacobian DLS helper into six position
        # actuator commands.  No arm qpos is written here.
        arm_joint_target = self._solve_ik(self.tcp_target_pos)
        self.ik_solve_calls += 1
        self.last_ik_joint_target = arm_joint_target.copy()
        self.diag_q_target = arm_joint_target.copy()
        self.diag_actuator_ctrl_before_step = self.data.ctrl[self.arm_actuator_ids].copy()

        for substep in range(self.frame_skip):
            self.data.ctrl[self.arm_actuator_ids] = np.clip(
                arm_joint_target,
                self.arm_ctrl_low,
                self.arm_ctrl_high,
            )
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_latch(
                update_counter=substep == self.frame_skip - 1
            )
        self.diag_tcp_actual_after = self.data.site_xpos[self.pinch_site_id].copy()
        self.diag_object_after = self.data.site_xpos[self.object_site_id].copy()
        self.diag_qpos_after = self.data.qpos[self.arm_qpos_ids].copy()
        self.diag_actuator_ctrl_after = self.data.ctrl[self.arm_actuator_ids].copy()
        if self.stage == STAGE_PLACE:
            self.release_steps += 1

    def _set_gripper_from_action(self, action):
        if self.fsm_mode == "deployable_fsm":
            gripper_normalized = self.task_supervisor.commanded_closure
        else:
            gripper_normalized = self._scripted_gripper_normalized()
        self._set_gripper_command(gripper_normalized)

    def _set_gripper_command_immediate(self, normalized):
        self.gripper_command_normalized = float(np.clip(normalized, 0.0, 1.0))
        self.data.ctrl[self.gripper_actuator_id] = (
            self.gripper_ctrl_low
            + self.gripper_command_normalized
            * (self.gripper_ctrl_high - self.gripper_ctrl_low)
        )

    def _set_gripper_command(self, normalized):
        target = float(np.clip(normalized, 0.0, 1.0))
        delta = np.clip(
            target - self.gripper_command_normalized,
            -self.gripper_command_rate,
            self.gripper_command_rate,
        )
        self._set_gripper_command_immediate(self.gripper_command_normalized + delta)

    def _scripted_gripper_normalized(self):
        if self.stage == STAGE_REACH:
            self._set_grasp_phase(GRASP_PHASE_ALIGN)
            return 0.0

        if self.stage == STAGE_PLACE:
            self._set_grasp_phase(GRASP_PHASE_CONFIRM)
            if (
                not self.release_has_opened
                and not self._place_release_phase()
            ):
                return self.latched_gripper_normalized
            if self.release_has_opened or self._release_should_open():
                self.release_has_opened = True
                return 0.0
            return self.latched_gripper_normalized

        if self.stage >= STAGE_LIFT or self.is_grasp_latched:
            self._set_grasp_phase(GRASP_PHASE_CONFIRM)
            return self.latched_gripper_normalized

        if self.stage != STAGE_GRASP:
            return 0.0

        if self.grasp_phase in (GRASP_PHASE_CLOSE, GRASP_PHASE_CONFIRM):
            return 1.0
        return 0.0

    def _release_should_open(self):
        object_position = self._control_object_position("place_release")
        goal_position = self._control_goal_position("place_release")
        object_to_goal_xy_distance = float(
            np.linalg.norm((goal_position - object_position)[:2])
        )
        return (
            self.release_steps >= self.release_min_open_steps
            and self._release_low_enough(
                object_position, consumer="place_release"
            )
            and object_to_goal_xy_distance < self.release_open_xy_threshold
        )

    def _release_low_enough(self, object_position=None, consumer="place_release"):
        if object_position is None:
            object_position = self._control_object_position(consumer)
        initial_z = self.object_estimate_authority.initial_z
        if initial_z is None:
            raise ObjectEstimateUnavailable(
                "missing_initial_z", consumer, self.data.time,
                self.object_estimate_authority.current,
            )
        object_lift = float(max(0.0, object_position[2] - initial_z))
        return object_lift <= self.release_success_lift

    def _place_release_phase(self):
        object_position = self._control_object_position("place_descent")
        goal_position = self._control_goal_position("place_descent")
        object_to_goal_xy_distance = float(
            np.linalg.norm((goal_position - object_position)[:2])
        )
        return object_to_goal_xy_distance < self.place_descent_xy_threshold

    def _set_grasp_phase(self, phase):
        phase = int(phase)
        if phase != self.grasp_phase:
            self.grasp_phase = phase
            self.grasp_phase_steps = 0
            if phase != GRASP_PHASE_CLOSE:
                self.close_steps = 0
        else:
            self.grasp_phase_steps += 1

    def _update_grasp_phase(self):
        left_contact, right_contact = self._raw_pad_object_contact_flags()
        has_any_contact = bool(left_contact or right_contact)
        has_bilateral_contact = bool(left_contact and right_contact)

        if self.grasp_phase == GRASP_PHASE_ALIGN:
            if self._pregrasp_handoff_ready():
                self._set_grasp_phase(GRASP_PHASE_DESCEND)
            else:
                self._set_grasp_phase(GRASP_PHASE_ALIGN)
            return

        if self.grasp_phase == GRASP_PHASE_DESCEND:
            if not self._grasp_xy_aligned():
                self._set_grasp_phase(GRASP_PHASE_ALIGN)
            elif self._fine_grasp_close_allowed():
                self._set_grasp_phase(GRASP_PHASE_CLOSE)
            else:
                self._set_grasp_phase(GRASP_PHASE_DESCEND)
            return

        if self.grasp_phase == GRASP_PHASE_CLOSE:
            self.close_steps += 1
            if self.close_steps >= self.min_close_steps and has_bilateral_contact:
                self._set_grasp_phase(GRASP_PHASE_CONFIRM)
            elif self.close_steps > self.max_close_attempt_steps and not has_bilateral_contact:
                self._set_grasp_phase(GRASP_PHASE_ALIGN)
            else:
                self._set_grasp_phase(GRASP_PHASE_CLOSE)
            return

        if self.grasp_phase == GRASP_PHASE_CONFIRM:
            if not has_any_contact and not self.is_grasp_latched:
                self._set_grasp_phase(GRASP_PHASE_CLOSE)
            else:
                self._set_grasp_phase(GRASP_PHASE_CONFIRM)

    def _neutralize_arm_actuators(self):
        self.data.ctrl[self.arm_actuator_ids] = np.clip(
            self.data.qpos[self.arm_qpos_ids],
            self.arm_ctrl_low,
            self.arm_ctrl_high,
        )

    def _sync_mocap_target_to_tcp(self):
        mujoco.mj_forward(self.model, self.data)
        self.tcp_target_pos = self.data.site_xpos[self.pinch_site_id].copy()
        self.tcp_target_quat = self._site_quat(self.pinch_site_id)
        self.data.mocap_pos[self.tcp_mocap_id] = self.tcp_target_pos
        self.data.mocap_quat[self.tcp_mocap_id] = self.tcp_target_quat
        self._neutralize_arm_actuators()
        mujoco.mj_forward(self.model, self.data)

    def _site_quat(self, site_id):
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id].copy())
        if quat[0] < 0:
            quat *= -1.0
        return quat

    def _object_upright_alignment(self):
        quat = self.data.qpos[self.object_qpos_id + 3 : self.object_qpos_id + 7].copy()
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return 1.0
        quat /= norm

        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        return float(abs(rotation[2, 2]))

    def _safe_tcp_target_pos(self, target_pos):
        safe_pos = np.clip(
            target_pos,
            self.task_workspace_low,
            self.task_workspace_high,
        )
        if self.stage == STAGE_REACH:
            object_position = self._control_object_position("reach_safety")
            reach_target_z = object_position[2] + self.pregrasp_height
            if self.include_stage_observation:
                safe_pos[2] = np.clip(
                    safe_pos[2],
                    reach_target_z,
                    reach_target_z + self.reach_target_z_margin,
                )
            else:
                safe_pos[2] = max(safe_pos[2], reach_target_z)
            return safe_pos
        if self.stage == STAGE_GRASP and not self.is_grasp_latched:
            object_position = self._control_object_position("grasp_safety")
            pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
            grasp_xy_error = np.linalg.norm((object_position - pinch_position)[:2])
            safe_pos[:2] = np.clip(
                safe_pos[:2],
                object_position[:2] - self.grasp_target_xy_margin,
                object_position[:2] + self.grasp_target_xy_margin,
            )
            if (
                self.grasp_phase == GRASP_PHASE_ALIGN
                or grasp_xy_error > self.grasp_descend_xy_threshold
            ):
                safe_pos[2] = max(
                    safe_pos[2],
                    object_position[2] + self.grasp_alignment_height,
                )
            safe_pos[2] = np.clip(
                safe_pos[2],
                self.grasp_target_min_z,
                object_position[2] + self.pregrasp_height + self.grasp_target_z_margin,
            )
            return safe_pos
        if self.stage == STAGE_PLACE:
            if self.release_has_opened or self.place_descent_active or self._place_release_phase():
                release_min_z = self.release_tcp_min_z
                release_max_z = (
                    self.object_initial_z
                    + self.lift_height
                    + self.lift_target_extra_height
                )
                safe_pos[2] = np.clip(safe_pos[2], release_min_z, release_max_z)
                return safe_pos
            place_target_z_min = (
                self.object_initial_z
                + self.lift_height
                + self.place_target_extra_height
            )
            lift_target_z_max = (
                self.object_initial_z
                + self.lift_height
                + self.lift_target_extra_height
            )
            safe_pos[2] = np.clip(safe_pos[2], place_target_z_min, lift_target_z_max)
        elif self.stage >= STAGE_LIFT or self.is_grasp_latched:
            lift_target_z_max = (
                self.object_initial_z
                + self.lift_height
                + self.lift_target_extra_height
            )
            safe_pos[2] = min(safe_pos[2], lift_target_z_max)
        if self._is_over_table_xy(safe_pos):
            safe_pos[2] = max(safe_pos[2], self.tcp_table_z_min)
        return safe_pos

    def _grasp_close_allowed(self):
        return self._fine_grasp_close_allowed()

    def _grasp_xy_aligned(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        object_position = self.data.site_xpos[self.object_site_id].copy()
        return (
            np.linalg.norm((object_position - pinch_position)[:2])
            < self.grasp_descend_xy_threshold
        )

    def _pregrasp_handoff_ready(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        pregrasp_position = self._pregrasp_position()
        pregrasp_delta = pregrasp_position - pinch_position
        vertical_alignment = float(
            np.dot(self._pinch_approach_axis(self.data), self.desired_approach_axis)
        )
        return (
            np.linalg.norm(pregrasp_delta[:2]) < self.reach_xy_threshold
            and abs(pregrasp_delta[2]) < self.reach_z_threshold
            and vertical_alignment > self.reach_vertical_threshold
        )

    def _rough_grasp_close_allowed(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        grasp_position = self._grasp_position()
        delta = grasp_position - pinch_position
        return (
            np.linalg.norm(delta) < self.rough_close_reward_distance
            and np.linalg.norm(delta[:2]) < self.rough_grasp_xy_close_threshold
            and abs(delta[2]) < self.rough_grasp_z_close_threshold
        )

    def _fine_grasp_close_allowed(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        grasp_position = self._grasp_position()
        delta = grasp_position - pinch_position
        return (
            np.linalg.norm(delta) < self.close_reward_distance
            and np.linalg.norm(delta[:2]) < self.grasp_xy_close_threshold
            and abs(delta[2]) < self.grasp_z_close_threshold
        )

    def _close_gripper_until_latched(self, max_steps=80):
        self._set_grasp_phase(GRASP_PHASE_CONFIRM)
        for substep in range(max_steps):
            self._set_gripper_command(1.0)
            self._neutralize_arm_actuators()
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_latch(
                update_counter=(substep + 1) % self.frame_skip == 0
            )
            if self.is_grasp_latched:
                break

    def _get_obs(self, allow_unavailable=False, consumer="ppo_observation"):
        arm_qpos = self.data.qpos[self.arm_qpos_ids].copy()
        arm_qvel = self.data.qvel[self.arm_qvel_ids].copy()
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        try:
            object_estimate = self._control_object_estimate(consumer)
            object_position = object_estimate.position.copy()
            initial_object_z = self.object_estimate_authority.initial_z
        except ObjectEstimateUnavailable:
            if not allow_unavailable:
                raise
            object_position = np.full(3, np.nan, dtype=np.float64)
            initial_object_z = None
        pregrasp_position = object_position + np.array(
            [0.0, 0.0, self.pregrasp_height], dtype=np.float64
        )
        grasp_position = object_position + np.array(
            [0.0, 0.0, self.grasp_height_offset], dtype=np.float64
        )
        try:
            goal_position = self._control_goal_position(consumer)
        except GoalEstimateUnavailable:
            if not allow_unavailable and self.stage == STAGE_PLACE:
                raise
            goal_position = np.full(3, np.nan, dtype=np.float64)
        pinch_to_pregrasp = pregrasp_position - pinch_position
        pinch_to_grasp = grasp_position - pinch_position
        object_to_goal = goal_position - object_position
        if self.stage != STAGE_PLACE:
            object_to_goal = np.zeros(3, dtype=np.float64)
        left_contact, right_contact = self._pad_object_contact_flags()
        object_lift = np.array(
            [np.nan if initial_object_z is None else
             max(0.0, object_position[2] - initial_object_z)],
            dtype=np.float64,
        )
        contact_flags = np.array(
            [float(left_contact), float(right_contact)],
            dtype=np.float64,
        )
        vertical_alignment = np.array(
            [
                np.clip(
                    np.dot(
                        self._pinch_approach_axis(self.data),
                        self.desired_approach_axis,
                    ),
                    -1.0,
                    1.0,
                )
            ],
            dtype=np.float64,
        )
        grasp_phase = np.array(
            [float(self.grasp_phase) / float(GRASP_PHASE_CONFIRM)],
            dtype=np.float64,
        )

        gripper_ctrl = self.data.ctrl[self.gripper_actuator_id]
        gripper_state = np.array(
            [
                (gripper_ctrl - self.gripper_ctrl_low)
                / (self.gripper_ctrl_high - self.gripper_ctrl_low)
            ],
            dtype=np.float64,
        )

        obs = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                pinch_position,
                object_position,
                pregrasp_position,
                grasp_position,
                pinch_to_pregrasp,
                pinch_to_grasp,
                object_to_goal,
                gripper_state,
                object_lift,
                contact_flags,
                vertical_alignment,
                grasp_phase,
            ]
        )
        if self.include_stage_observation:
            # Full/curriculum policies must not infer the active controller
            # mode from geometry alone.  Reach, grasp, lift and place can have
            # very similar physical observations while requiring opposite Z
            # actions, which previously caused lift actions to leak into the
            # earlier stages.
            stage_one_hot = np.zeros(4, dtype=np.float64)
            stage_to_index = {
                STAGE_REACH: 0,
                STAGE_GRASP: 1,
                STAGE_LIFT: 2,
                STAGE_PLACE: 3,
            }
            stage_index = stage_to_index.get(self.stage)
            if stage_index is not None:
                stage_one_hot[stage_index] = 1.0
            obs = np.concatenate([obs, stage_one_hot])
        obs = obs.astype(np.float32)
        return obs

    def _get_info(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        attachment_position = self.data.site_xpos[self.attachment_site_id].copy()
        object_position = self.data.site_xpos[self.object_site_id].copy()
        goal_truth_evaluator_only = self.data.site_xpos[self.goal_site_id].copy()
        try:
            # These distances feed Place reward/success and therefore use the
            # same frozen goal as the action path in ArUco mode. MuJoCo goal
            # truth is exposed only by explicitly named evaluator fields.
            goal_estimate = self._control_goal_estimate("task_metrics")
            goal_position = goal_estimate.position.copy()
            goal_estimate_id = goal_estimate.estimate_id
            goal_estimate_timestamp = float(goal_estimate.timestamp)
        except GoalEstimateUnavailable:
            goal_position = np.full(3, np.nan, dtype=np.float64)
            goal_estimate_id = None
            goal_estimate_timestamp = None
        pregrasp_position = self._pregrasp_position()
        grasp_position = self._grasp_position()

        pinch_to_pregrasp_distance = float(
            np.linalg.norm(pregrasp_position - pinch_position)
        )
        pregrasp_delta = pregrasp_position - pinch_position
        pregrasp_xy_error = float(np.linalg.norm(pregrasp_delta[:2]))
        pregrasp_z_error = float(pregrasp_delta[2])
        pinch_to_grasp_distance = float(np.linalg.norm(grasp_position - pinch_position))
        pinch_to_object_distance = float(np.linalg.norm(object_position - pinch_position))
        grasp_xy_error = float(np.linalg.norm((grasp_position - pinch_position)[:2]))
        grasp_z_error = float(grasp_position[2] - pinch_position[2])
        grasp_height_above_object = float(pinch_position[2] - object_position[2])
        grasp_xy_aligned = grasp_xy_error < self.grasp_descend_xy_threshold
        grasp_descent_allowed = grasp_xy_aligned
        object_to_goal_distance = float(np.linalg.norm(goal_position - object_position))
        object_to_goal_xy_distance = float(
            np.linalg.norm((goal_position - object_position)[:2])
        )
        object_goal_z_error = float(goal_position[2] - object_position[2])
        object_lift = float(max(0.0, object_position[2] - self.object_initial_z))
        lift_distance = float(max(0.0, self.lift_height - object_lift))
        place_height_ok = object_lift >= self.lift_height
        object_horizontal_drift = float(
            np.linalg.norm((object_position - self.object_initial_position)[:2])
        )
        object_xy_step = float(
            np.linalg.norm((object_position - self.previous_object_position)[:2])
        )
        object_linear_velocity = self.data.qvel[
            self.object_qvel_id : self.object_qvel_id + 3
        ].copy()
        object_speed = float(np.linalg.norm(object_linear_velocity))
        object_xy_speed = float(np.linalg.norm(object_linear_velocity[:2]))
        object_table_boundary_penalty = self._object_table_boundary_penalty(
            object_position
        )
        gripper_state = self._gripper_state()
        left_contact, right_contact = self._pad_object_contact_flags()
        raw_left_contact, raw_right_contact = self._raw_pad_object_contact_flags()
        pad_object_penetration = self._pad_object_contact_penetration()
        self.max_pad_object_penetration = max(
            self.max_pad_object_penetration,
            pad_object_penetration,
        )
        has_bilateral_contact = bool(left_contact and right_contact)
        has_unilateral_contact = bool(left_contact != right_contact)
        has_any_contact = bool(left_contact or right_contact)
        has_raw_bilateral_contact = bool(raw_left_contact and raw_right_contact)
        has_raw_any_contact = bool(raw_left_contact or raw_right_contact)

        approach_axis = self._pinch_approach_axis(self.data)
        vertical_alignment = float(
            np.clip(np.dot(approach_axis, self.desired_approach_axis), -1.0, 1.0)
        )
        orientation_penalty = 1.0 - vertical_alignment
        tcp_target_error = float(np.linalg.norm(self.tcp_target_pos - pinch_position))
        table_clearance_penalty = self._table_clearance_penalty(
            pinch_position,
            attachment_position,
        )
        table_side_penalty = self._table_side_penalty(
            pinch_position
        ) + self._table_side_penalty(attachment_position)
        table_contact_count = self._robot_table_contact_count()
        object_upright = self._object_upright_alignment()
        object_upright_ok = object_upright >= self.grasp_upright_threshold
        object_still_near_start = (
            object_horizontal_drift <= self.grasp_success_max_xy_drift
        )
        contact_penetration_ok = (
            pad_object_penetration <= self.grasp_success_penetration_tolerance
            and self.max_pad_object_penetration
            <= self.grasp_success_penetration_tolerance
        )
        robot_table_contact_ok = table_contact_count == 0
        object_inside_table = object_table_boundary_penalty == 0.0
        low_away_from_object_penalty = self._low_away_from_object_penalty(
            pinch_position,
            object_position,
        )
        rough_grasp_close_allowed = (
            pinch_to_grasp_distance < self.rough_close_reward_distance
            and grasp_xy_error < self.rough_grasp_xy_close_threshold
            and abs(grasp_z_error) < self.rough_grasp_z_close_threshold
        )
        fine_grasp_close_allowed = (
            pinch_to_grasp_distance < self.close_reward_distance
            and grasp_xy_error < self.grasp_xy_close_threshold
            and abs(grasp_z_error) < self.grasp_z_close_threshold
        )
        object_drift_failure = (
            self.stage == STAGE_GRASP
            and not self.is_grasp_latched
            and object_horizontal_drift > self.max_grasp_object_xy_drift
        )
        object_tip_failure = (
            self.stage == STAGE_GRASP
            and has_raw_any_contact
            and gripper_state > self.close_gripper_threshold
            and not object_upright_ok
        )
        contact_penetration_failure = (
            self.stage == STAGE_GRASP
            and has_raw_any_contact
            and self.max_pad_object_penetration > self.penetration_failure_tolerance
        )
        robot_table_contact_failure = (
            self.stage == STAGE_GRASP
            and gripper_state > self.close_gripper_threshold
            and table_contact_count > 0
        )

        reach_distance_ok = pinch_to_pregrasp_distance < self.approach_threshold
        reach_xy_centered = pregrasp_xy_error < self.reach_xy_threshold
        reach_z_centered = abs(pregrasp_z_error) < self.reach_z_threshold
        reach_tcp_tracked = tcp_target_error < self.reach_tcp_error_threshold
        reach_handoff_success = (
            reach_distance_ok
            and reach_xy_centered
            and reach_z_centered
            and vertical_alignment > self.reach_vertical_threshold
            and reach_tcp_tracked
        )
        reach_success = reach_handoff_success
        grasp_result = self._check_grasp_success(
            {
                "gripper_state": gripper_state,
                "has_left_contact": bool(left_contact),
                "has_right_contact": bool(right_contact),
                "pinch_to_object_distance": pinch_to_object_distance,
                "grasp_xy_error": grasp_xy_error,
                "grasp_z_error": grasp_z_error,
                "pad_object_penetration": pad_object_penetration,
                "max_pad_object_penetration": float(self.max_pad_object_penetration),
            },
            update_counter=False,
        )
        coarse_grasp_success = grasp_result["coarse_grasp_success"]
        stable_grasp_success = grasp_result["stable_grasp_success"]
        strict_grasp_success = grasp_result["strict_grasp_success"]
        grasp_success = stable_grasp_success
        lift_contact_ok = bool(
            has_raw_bilateral_contact
            or has_bilateral_contact
            or self.is_grasp_latched
        )
        lift_success = (
            object_lift >= self.lift_height
            and self.is_grasp_latched
            and lift_contact_ok
            and table_clearance_penalty == 0.0
            and table_contact_count == 0
            and object_table_boundary_penalty == 0.0
        )
        place_height_ok = object_lift <= self.release_success_lift
        place_xy_ready = object_to_goal_xy_distance < self.place_handoff_xy_threshold
        place_low_ready = place_height_ok
        place_open_ready = bool(
            place_xy_ready
            and place_low_ready
            and self.release_steps >= self.release_min_open_steps
        )
        place_opened = gripper_state < 0.20 and not self.is_grasp_latched
        place_success = (
            place_open_ready
            and place_opened
            and object_xy_speed < 0.04
            and table_contact_count == 0
            and object_table_boundary_penalty == 0.0
        )
        place_distance = float(
            object_to_goal_xy_distance
            + max(0.0, object_lift - self.release_success_lift)
        )

        reach_success = reach_success or self.stage > STAGE_REACH
        coarse_grasp_success = coarse_grasp_success or self.stage > STAGE_GRASP
        grasp_success = grasp_success or self.stage > STAGE_GRASP
        lift_success = lift_success or self.stage > STAGE_LIFT

        if self.success_stage == STAGE_REACH:
            stage_success = reach_success
        elif self.success_stage == STAGE_GRASP:
            stage_success = grasp_success
        elif self.success_stage == STAGE_LIFT:
            stage_success = lift_success
        elif self.success_stage == STAGE_PLACE:
            stage_success = place_success
        else:
            stage_success = place_success

        if self.stage == STAGE_REACH:
            task_distance = pinch_to_pregrasp_distance
        elif self.stage == STAGE_GRASP:
            task_distance = pinch_to_grasp_distance
        elif self.stage == STAGE_LIFT:
            task_distance = lift_distance
        elif self.stage == STAGE_PLACE:
            task_distance = place_distance
        else:
            task_distance = object_to_goal_xy_distance

        return {
            "distance": task_distance,
            "pinch_to_pregrasp_distance": pinch_to_pregrasp_distance,
            "pregrasp_xy_error": pregrasp_xy_error,
            "pregrasp_z_error": pregrasp_z_error,
            "reach_distance_ok": bool(reach_distance_ok),
            "reach_xy_centered": bool(reach_xy_centered),
            "reach_z_centered": bool(reach_z_centered),
            "reach_tcp_tracked": bool(reach_tcp_tracked),
            "reach_centered": bool(reach_xy_centered and reach_z_centered),
            "reach_handoff_success": bool(reach_handoff_success),
            "pinch_to_grasp_distance": pinch_to_grasp_distance,
            "pinch_to_object_distance": pinch_to_object_distance,
            "grasp_xy_error": grasp_xy_error,
            "grasp_z_error": grasp_z_error,
            "grasp_height_above_object": grasp_height_above_object,
            "grasp_xy_aligned": bool(grasp_xy_aligned),
            "grasp_descent_allowed": bool(grasp_descent_allowed),
            "object_to_goal_distance": object_to_goal_distance,
            "object_to_goal_xy_distance": object_to_goal_xy_distance,
            "object_goal_z_error": object_goal_z_error,
            "is_success": bool(stage_success),
            "stage_success": bool(stage_success),
            "reach_success": bool(reach_success),
            "grasp_success": bool(grasp_success),
            "coarse_grasp_success": bool(coarse_grasp_success),
            "stable_grasp_success": bool(stable_grasp_success),
            "strict_grasp_success": bool(strict_grasp_success),
            "lift_success": bool(lift_success),
            "lift_contact_ok": bool(lift_contact_ok),
            "place_success": bool(place_success),
            "place_height_ok": bool(place_height_ok),
            "place_xy_ready": bool(place_xy_ready),
            "place_low_ready": bool(place_low_ready),
            "place_open_ready": bool(place_open_ready),
            "place_opened": bool(place_opened),
            "place_distance": place_distance,
            "release_opened": bool(place_opened),
            "release_steps": int(self.release_steps),
            "place_has_opened": bool(self.release_has_opened),
            "training_stage": self.training_stage,
            "success_stage": self.success_stage,
            "active_stage": self.stage,
            "pinch_position": pinch_position,
            "attachment_position": attachment_position,
            "object_position": object_position,
            "pregrasp_position": pregrasp_position,
            "grasp_position": grasp_position,
            "goal_position": goal_position,
            "goal_source": self.goal_observation_mode,
            "goal_estimate_id": goal_estimate_id,
            "goal_estimate_timestamp": goal_estimate_timestamp,
            "goal_truth_position_evaluator_only": goal_truth_evaluator_only,
            "goal_estimation_error_m_evaluator_only": (
                None if not np.isfinite(goal_position).all() else float(
                    np.linalg.norm(goal_position - goal_truth_evaluator_only))
            ),
            "tcp_target_position": self.tcp_target_pos.copy(),
            "tcp_target_error": tcp_target_error,
            "grasp_phase": int(self.grasp_phase),
            "grasp_phase_steps": int(self.grasp_phase_steps),
            "close_steps": int(self.close_steps),
            "grasp_close_allowed": bool(rough_grasp_close_allowed),
            "rough_grasp_close_allowed": bool(rough_grasp_close_allowed),
            "fine_grasp_close_allowed": bool(fine_grasp_close_allowed),
            "object_drift_failure": bool(object_drift_failure),
            "object_tip_failure": bool(object_tip_failure),
            "contact_penetration_failure": bool(contact_penetration_failure),
            "robot_table_contact_failure": bool(robot_table_contact_failure),
            "stage_failure": bool(
                object_drift_failure
                or object_tip_failure
                or contact_penetration_failure
                or robot_table_contact_failure
            ),
            "object_lift": object_lift,
            "lift_distance": lift_distance,
            "object_horizontal_drift": object_horizontal_drift,
            "object_xy_step": object_xy_step,
            "object_speed": object_speed,
            "object_xy_speed": object_xy_speed,
            "object_table_boundary_penalty": object_table_boundary_penalty,
            "gripper_state": gripper_state,
            "left_pad_contact": bool(left_contact),
            "right_pad_contact": bool(right_contact),
            "has_left_contact": bool(left_contact),
            "has_right_contact": bool(right_contact),
            "raw_left_pad_contact": bool(raw_left_contact),
            "raw_right_pad_contact": bool(raw_right_contact),
            "has_bilateral_contact": has_bilateral_contact,
            "has_unilateral_contact": has_unilateral_contact,
            "has_any_contact": has_any_contact,
            "has_raw_bilateral_contact": has_raw_bilateral_contact,
            "has_raw_any_contact": has_raw_any_contact,
            "pad_object_penetration": pad_object_penetration,
            "max_pad_object_penetration": float(self.max_pad_object_penetration),
            "contact_penetration_ok": bool(contact_penetration_ok),
            "object_upright": object_upright,
            "object_upright_ok": bool(object_upright_ok),
            "object_still_near_start": bool(object_still_near_start),
            "stable_latched_grasp": bool(stable_grasp_success),
            "grasp_stable_count": int(grasp_result["grasp_stable_count"]),
            "grasp_success_gripper_closed": bool(grasp_result["gripper_closed"]),
            "grasp_success_bilateral_contact": bool(
                grasp_result["has_bilateral_contact"]
            ),
            "grasp_success_distance_ok": bool(grasp_result["distance_ok"]),
            "grasp_success_pose_ok": bool(grasp_result["grasp_pose_ok"]),
            "grasp_success_strict_pose_ok": bool(
                grasp_result["strict_grasp_pose_ok"]
            ),
            "grasp_success_penetration_ok": bool(grasp_result["penetration_ok"]),
            "is_grasp_latched": bool(self.is_grasp_latched),
            "has_grasp": bool(grasp_success),
            "vertical_alignment": vertical_alignment,
            "orientation_penalty": orientation_penalty,
            "table_clearance_penalty": table_clearance_penalty,
            "table_side_penalty": table_side_penalty,
            "low_away_from_object_penalty": low_away_from_object_penalty,
            "table_contact_count": table_contact_count,
            "stage": self.stage,
        }

    def _refresh_task_fields(self, info):
        if self.stage == STAGE_REACH:
            info["distance"] = info["pinch_to_pregrasp_distance"]
        elif self.stage == STAGE_GRASP:
            info["distance"] = info["pinch_to_grasp_distance"]
        elif self.stage == STAGE_LIFT:
            info["distance"] = info["lift_distance"]
        elif self.stage == STAGE_PLACE:
            info["distance"] = info["place_distance"]
        else:
            info["distance"] = info["object_to_goal_xy_distance"]
        info["stage"] = self.stage
        info["active_stage"] = self.stage

    def _compute_reward(self, info, action):
        if self.fsm_mode == "deployable_fsm":
            return self._full_reward_without_transitions(info, action)
        if self.sequential_training:
            return self._full_reward(info, action)
        if self.training_stage == STAGE_REACH:
            return self._reach_reward(info)
        if self.training_stage == STAGE_GRASP:
            return self._grasp_reward(info)
        if self.training_stage == STAGE_LIFT:
            return self._lift_reward(info, action)
        if self.training_stage == STAGE_PLACE:
            return self._place_reward(info, action)
        if self.training_stage == STAGE_REACH_GRASP:
            return self._reach_grasp_reward(info)
        return self._full_reward(info, action)

    def _full_reward_without_transitions(self, info, action):
        """Preserve shaping while the independent supervisor owns stages."""
        if self.stage == STAGE_REACH:
            return self._reach_reward(info)
        if self.stage == STAGE_GRASP:
            return self._completed_stage_base_reward(
                STAGE_GRASP, info
            ) + self._grasp_reward(info)
        if self.stage == STAGE_LIFT:
            return self._completed_stage_base_reward(
                STAGE_LIFT, info
            ) + self._lift_reward(info, action)
        return self._completed_stage_base_reward(
            STAGE_PLACE, info
        ) + self._place_reward(info, action)

    def _completed_stage_base_reward(self, stage, info):
        reward = 0.0
        if stage >= STAGE_GRASP:
            reward += self.reach_completed_base_reward
        if stage >= STAGE_LIFT and info["is_grasp_latched"]:
            reward += self.grasp_completed_base_reward
        if (
            stage >= STAGE_PLACE
            and info["is_grasp_latched"]
            and info["object_lift"] >= 0.8 * self.lift_height
        ):
            reward += self.lift_completed_base_reward
        return reward

    def _reach_grasp_reward(self, info):
        if self.stage == STAGE_REACH and info["reach_success"]:
            reward = self._reach_reward(info)
            if self.success_stage == STAGE_REACH:
                return reward
            self.stage = STAGE_GRASP
            self._set_grasp_phase(GRASP_PHASE_DESCEND)
            self.previous_distance = info["pinch_to_grasp_distance"]
            self.previous_grasp_xy_error = info["grasp_xy_error"]
            self.previous_grasp_abs_z_error = abs(info["grasp_z_error"])
            return reward

        if self.stage == STAGE_REACH:
            return self._reach_reward(info)
        return self._completed_stage_base_reward(STAGE_GRASP, info) + self._grasp_reward(info)

    def _full_reward(self, info, action):
        if self.stage == STAGE_REACH and info["reach_success"]:
            reward = self._reach_reward(info)
            self.stage = STAGE_GRASP
            self._set_grasp_phase(GRASP_PHASE_DESCEND)
            self.previous_distance = info["pinch_to_grasp_distance"]
            self.previous_grasp_xy_error = info["grasp_xy_error"]
            self.previous_grasp_abs_z_error = abs(info["grasp_z_error"])
            return reward

        if self.stage == STAGE_GRASP and info["grasp_success"]:
            reward = (
                self._completed_stage_base_reward(STAGE_GRASP, info)
                + self._grasp_reward(info)
            )
            if self.success_stage == STAGE_GRASP:
                return reward
            self.stage = STAGE_LIFT
            self._set_grasp_phase(GRASP_PHASE_CONFIRM)
            self.previous_distance = info["lift_distance"]
            self.previous_lift = info["object_lift"]
            return reward

        if self.stage == STAGE_LIFT and info["lift_success"]:
            reward = (
                self._completed_stage_base_reward(STAGE_LIFT, info)
                + self._lift_reward(info, action)
            )
            if self.success_stage == STAGE_LIFT:
                return reward
            self.stage = STAGE_PLACE
            self.previous_distance = info["object_to_goal_xy_distance"]
            self.release_steps = 0
            self.release_start_lift = info["object_lift"]
            self.release_has_opened = False
            self.place_descent_active = False
            self.place_handoff_count = self.place_handoff_hold_steps
            return reward

        if self.stage == STAGE_REACH:
            return self._reach_reward(info)
        if self.stage == STAGE_GRASP:
            return self._completed_stage_base_reward(STAGE_GRASP, info) + self._grasp_reward(info)
        if self.stage == STAGE_LIFT:
            return self._completed_stage_base_reward(STAGE_LIFT, info) + self._lift_reward(info, action)
        if self.stage == STAGE_PLACE:
            return self._completed_stage_base_reward(STAGE_PLACE, info) + self._place_reward(info, action)
        return self._completed_stage_base_reward(STAGE_PLACE, info) + self._place_reward(info, action)

    def _reach_reward(self, info):
        current_distance = info["pinch_to_pregrasp_distance"]
        xy_error = info["pregrasp_xy_error"]
        abs_z_error = abs(info["pregrasp_z_error"])
        xy_progress = self.previous_pregrasp_xy_error - xy_error
        z_progress = self.previous_pregrasp_abs_z_error - abs_z_error

        xy_cost = np.tanh(18.0 * xy_error)
        z_cost = np.tanh(18.0 * abs_z_error)

        reward = -(0.80 * xy_cost + 0.40 * z_cost)
        reward += 2.0 * xy_progress + 1.0 * z_progress
        if info["gripper_state"] > 0.05:
            reward -= 0.20 * info["gripper_state"]
        if info["object_xy_speed"] > 0.05 and current_distance < 0.08:
            reward -= 0.30
        if info["has_any_contact"]:
            reward -= 0.50
        reward -= 2.0 * info["object_horizontal_drift"]
        reward -= 0.30 * info["table_contact_count"]
        reward -= 0.02

        if info["reach_success"]:
            reward += self.reach_success_bonus
        self.previous_distance = current_distance
        self.previous_pregrasp_xy_error = xy_error
        self.previous_pregrasp_abs_z_error = abs_z_error
        self.previous_object_position = info["object_position"].copy()
        return float(reward)

    def _grasp_reward(self, info):
        current_distance = info["pinch_to_grasp_distance"]
        xy_error = info["grasp_xy_error"]
        abs_z_error = abs(info["grasp_z_error"])
        xy_progress = self.previous_grasp_xy_error - xy_error
        z_progress = self.previous_grasp_abs_z_error - abs_z_error
        phase = int(info["grasp_phase"])
        xy_cost = np.tanh(18.0 * xy_error)
        z_cost = np.tanh(18.0 * abs_z_error)

        if phase == GRASP_PHASE_ALIGN:
            reward = -(0.80 * xy_cost)
            reward += 2.0 * xy_progress
            height_deficit = max(
                0.0,
                self.grasp_alignment_height - info["grasp_height_above_object"],
            )
            reward -= 0.50 * np.tanh(12.0 * height_deficit)

        elif phase == GRASP_PHASE_DESCEND:
            reward = 0.10 - (0.45 * xy_cost + 0.75 * z_cost)
            reward += 1.0 * xy_progress + 2.0 * z_progress
            if not info["grasp_descent_allowed"]:
                reward -= 0.30
            if info["fine_grasp_close_allowed"]:
                reward += 1.0

        elif phase == GRASP_PHASE_CLOSE:
            reward = 0.05 - (0.55 * xy_cost + 0.30 * z_cost)
            reward += 1.50 * xy_progress + 0.50 * z_progress
            if info["has_any_contact"]:
                reward += 0.15
            if info["has_bilateral_contact"]:
                reward += 0.25
            if info["grasp_success_pose_ok"]:
                reward += 0.20
            if info["grasp_success_strict_pose_ok"]:
                reward += 0.15
            if info["object_xy_speed"] > 0.05 and not info["has_bilateral_contact"]:
                reward -= 0.40

        else:
            # Confirmation must not become a positive-reward waiting room.
            # Off-center coarse grasps receive approximately zero/negative
            # reward, while centered stable contact remains clearly positive.
            reward = -0.10 - (0.75 * xy_cost + 0.25 * z_cost)
            reward += 2.0 * xy_progress + 0.50 * z_progress
            if info["has_any_contact"]:
                reward += 0.10
            if info["has_bilateral_contact"]:
                reward += 0.15
            if info["grasp_success_pose_ok"]:
                reward += 0.25
            if info["grasp_success_strict_pose_ok"]:
                reward += 0.20
            if not info["has_any_contact"]:
                reward -= 0.20

        if info["stable_grasp_success"]:
            reward += self.grasp_success_bonus
            if info["strict_grasp_success"]:
                reward += 0.25 * self.grasp_success_bonus
        elif info["coarse_grasp_success"]:
            reward += 0.05

        reward -= 4.0 * info["object_horizontal_drift"]
        reward -= 0.50 * info["object_xy_speed"]
        reward -= 0.30 * info["table_contact_count"]
        reward -= 1.0 * info["object_table_boundary_penalty"]
        reward -= 8.0 * max(
            0.0,
            info["pad_object_penetration"] - self.penetration_tolerance,
        )
        reward -= 0.02
        if info["object_drift_failure"]:
            reward -= 4.0

        self.previous_distance = current_distance
        self.previous_grasp_xy_error = xy_error
        self.previous_grasp_abs_z_error = abs_z_error
        self.previous_object_position = info["object_position"].copy()
        return float(reward)

    def _lift_reward(self, info, action):
        current_distance = info["lift_distance"]
        progress = self.previous_distance - current_distance
        horizontal_action = float(np.linalg.norm(action[:2]))
        upward_action = 0.5 * (
            float(np.clip(action[2], -1.0, 1.0)) + 1.0
        )
        downward_action = 0.0
        lift_norm = np.clip(info["object_lift"] / self.lift_height, 0.0, 1.0)
        lift_progress = info["object_lift"] - self.previous_lift
        has_stable_grasp = bool(
            info["is_grasp_latched"] and info.get("lift_contact_ok", False)
        )

        reward = -2.0 * (1.0 - lift_norm)
        reward += 10.0 * progress
        reward += 8.0 * max(lift_progress, 0.0)
        reward -= 4.0 * max(-lift_progress, 0.0)
        # The accepted ReachGrasp policy enters Lift with a strongly negative
        # Z mean.  Lift clips negative Z motion to zero for safety, so weak
        # shaping leaves PPO on a long flat plateau with no physical lift
        # progress.  Strengthen only the pre-success action signal; the task,
        # controller and success threshold are unchanged.
        reward += 1.50 * upward_action
        reward -= 1.50 * downward_action
        reward -= 1.00 * horizontal_action
        reward -= 2.0 * info["object_horizontal_drift"]
        reward -= 0.30 * info["table_contact_count"]
        reward -= 1.0 * info["object_table_boundary_penalty"]
        reward -= 8.0 * max(0.0, info["pad_object_penetration"] - self.penetration_tolerance)
        reward -= 12.0 * max(
            0.0,
            info["max_pad_object_penetration"] - self.penetration_tolerance,
        )
        reward -= 0.02

        if not info["is_grasp_latched"]:
            reward -= 1.5
        if not info.get("lift_contact_ok", info["has_raw_bilateral_contact"]):
            reward -= 1.0
        if has_stable_grasp and not info["lift_success"]:
            if lift_progress < 0.0002:
                reward -= 0.25
            if upward_action < 0.20:
                reward -= 0.50
        if info["lift_success"]:
            reward += self.lift_success_bonus

        self.previous_distance = current_distance
        self.previous_lift = info["object_lift"]
        self.previous_object_position = info["object_position"].copy()
        return float(reward)

    def _place_reward(self, info, action):
        goal_xy = info["object_to_goal_xy_distance"]
        progress = self.previous_distance - goal_xy

        object_lift = info["object_lift"]

        near_goal = goal_xy < self.place_xy_threshold
        xy_score = 1.0 - np.tanh(5.0 * goal_xy)

        reward = 0.0

        reward += 6.0 * xy_score
        reward += 60.0 * progress

        if not near_goal:
            lift_deficit = max(0.0, self.lift_height * 0.8 - object_lift)
            reward -= 1.0 * lift_deficit

            if not info["is_grasp_latched"]:
                reward -= 1.0

            if info.get("release_opened", False):
                reward -= 2.0

        else:
            place_height_error = abs(object_lift - self.release_success_lift)
            height_score = 1.0 - np.tanh(10.0 * place_height_error)

            reward += 2.0 * height_score
            reward += 1.0

            release_ready = place_height_error < 0.02

            if release_ready:
                reward += 1.0

            if info.get("release_opened", False):
                if release_ready:
                    reward += 2.0
                else:
                    reward -= 1.0

        if info.get("place_success", False):
            reward += self.place_success_bonus

        reward -= 0.02
        reward -= 0.2 * info["table_contact_count"]

        self.previous_distance = goal_xy
        self.previous_object_position = info["object_position"].copy()

        if (
            self.reset_stage_probabilities is not None
            and self.episode_start_stage == STAGE_PLACE
        ):
            # A scripted Place reset already starts with a valid grasp and
            # completed lift.  Without importance correction its easy 3000
            # point terminal bonus dominates Reach/Grasp learning.
            reward *= self.curriculum_place_reset_reward_scale

        return float(reward)

    def _solve_ik(self, target_pos):
        qpos = self.data.qpos[self.arm_qpos_ids].copy()
        initial_qpos = qpos.copy()
        self._copy_data_for_ik()

        damping = 1e-4
        position_weight = 1.0
        axis_weight = self.ik_axis_weight
        posture_weight = self.ik_posture_weight

        any_dq_clip = False
        dq_clip_iterations = 0
        any_joint_clip = False
        iterations = 0
        converged = False
        last_raw_dq = np.zeros(6, dtype=np.float64)
        last_dq = np.zeros(6, dtype=np.float64)
        last_dq_task = np.zeros(6, dtype=np.float64)
        last_posture_raw = np.zeros(6, dtype=np.float64)
        last_posture_projected = np.zeros(6, dtype=np.float64)
        last_j_posture = np.zeros(6, dtype=np.float64)
        max_j_posture_norm = 0.0
        sum_j_posture_norm = 0.0
        sum_sq_j_posture_norm = 0.0
        projection_iterations = 0
        min_joint_limit_margin_by_joint = np.minimum(
            qpos - self.arm_ctrl_low,
            self.arm_ctrl_high - qpos,
        )
        for _ in range(15):
            iterations += 1
            self.ik_data.qpos[self.arm_qpos_ids] = qpos
            mujoco.mj_forward(self.model, self.ik_data)

            pinch_position = self.ik_data.site_xpos[self.pinch_site_id].copy()
            position_error = target_pos - pinch_position

            current_axis = self._pinch_approach_axis(self.ik_data)
            axis_error = np.cross(current_axis, self.desired_approach_axis)

            if (
                np.linalg.norm(position_error) < 1e-3
                and np.linalg.norm(axis_error) < 3e-2
            ):
                converged = True
                break

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self.model,
                self.ik_data,
                jacp,
                jacr,
                self.pinch_site_id,
            )

            j_pos = jacp[:, self.arm_qvel_ids]
            j_rot = jacr[:, self.arm_qvel_ids]
            jacobian = np.vstack(
                [
                    position_weight * j_pos,
                    axis_weight * j_rot,
                ]
            )
            error = np.concatenate(
                [
                    position_weight * position_error,
                    axis_weight * axis_error,
                ]
            )

            damped_task_matrix = (
                jacobian @ jacobian.T + damping * np.eye(6)
            )
            # Use exactly the same damped inverse for the primary task and
            # the null-space projector.
            j_pinv = jacobian.T @ np.linalg.solve(
                damped_task_matrix,
                np.eye(6),
            )
            dq_task = j_pinv @ error
            dq_posture_raw = posture_weight * (self.home_arm_qpos - qpos)
            if self.ik_posture_mode == "raw":
                dq_posture = dq_posture_raw
            elif self.ik_posture_mode == "nullspace":
                nullspace = np.eye(6) - j_pinv @ jacobian
                dq_posture = nullspace @ dq_posture_raw
            else:
                dq_posture = np.zeros(6, dtype=np.float64)
            j_posture = jacobian @ dq_posture
            dq = dq_task + dq_posture
            last_dq_task = dq_task.copy()
            last_posture_raw = dq_posture_raw.copy()
            last_posture_projected = dq_posture.copy()
            last_j_posture = j_posture.copy()
            max_j_posture_norm = max(
                max_j_posture_norm,
                float(np.linalg.norm(j_posture)),
            )
            posture_leak_norm = float(np.linalg.norm(j_posture))
            sum_j_posture_norm += posture_leak_norm
            sum_sq_j_posture_norm += posture_leak_norm**2
            projection_iterations += 1
            last_raw_dq = dq.copy()
            clipped_this_iteration = bool(np.any(np.abs(dq) > 0.04))
            any_dq_clip |= clipped_this_iteration
            dq_clip_iterations += int(clipped_this_iteration)
            dq = np.clip(dq, -0.04, 0.04)
            last_dq = dq.copy()

            proposed_qpos = qpos + dq
            any_joint_clip |= bool(np.any((proposed_qpos < self.arm_ctrl_low) | (proposed_qpos > self.arm_ctrl_high)))
            qpos = np.clip(proposed_qpos, self.arm_ctrl_low, self.arm_ctrl_high)
            min_joint_limit_margin_by_joint = np.minimum(
                min_joint_limit_margin_by_joint,
                np.minimum(qpos - self.arm_ctrl_low, self.arm_ctrl_high - qpos),
            )

        self.ik_data.qpos[self.arm_qpos_ids] = qpos
        mujoco.mj_forward(self.model, self.ik_data)
        fk_target_tcp = self.ik_data.site_xpos[self.pinch_site_id].copy()
        final_axis = self._pinch_approach_axis(self.ik_data)
        final_axis_error = np.cross(final_axis, self.desired_approach_axis)
        self.diag_ik = {
            "damping": damping,
            "iterations": iterations,
            "converged": converged,
            "dq_clip": any_dq_clip,
            "dq_clip_iterations": dq_clip_iterations,
            "joint_limit_clip": any_joint_clip,
            "last_raw_dq": last_raw_dq.copy(),
            "last_dq": last_dq.copy(),
            "last_dq_task": last_dq_task.copy(),
            "posture_raw_increment": last_posture_raw.copy(),
            "posture_projected_increment": last_posture_projected.copy(),
            "posture_raw_increment_norm": float(np.linalg.norm(last_posture_raw)),
            "posture_projected_increment_norm": float(np.linalg.norm(last_posture_projected)),
            "j_posture": last_j_posture.copy(),
            "j_posture_norm": float(np.linalg.norm(last_j_posture)),
            "max_j_posture_norm": max_j_posture_norm,
            "mean_j_posture_norm": float(
                sum_j_posture_norm / max(projection_iterations, 1)
            ),
            "rms_j_posture_norm": float(
                np.sqrt(sum_sq_j_posture_norm / max(projection_iterations, 1))
            ),
            "projection_iterations": projection_iterations,
            "total_dq": (qpos - initial_qpos).copy(),
            "initial_qpos": initial_qpos.copy(),
            "q_target": qpos.copy(),
            "fk_target_tcp": fk_target_tcp,
            "position_error": (target_pos - fk_target_tcp).copy(),
            "approach_axis_error": final_axis_error.copy(),
            "approach_axis_error_norm": float(np.linalg.norm(final_axis_error)),
            "posture_error": (self.home_arm_qpos - qpos).copy(),
            "posture_error_norm": float(np.linalg.norm(self.home_arm_qpos - qpos)),
            "weighted_posture_error_norm": float(posture_weight * np.linalg.norm(self.home_arm_qpos - qpos)),
            "axis_weight": axis_weight,
            "posture_weight": posture_weight,
            "posture_mode": self.ik_posture_mode,
            "min_joint_limit_margin": float(np.min(min_joint_limit_margin_by_joint)),
            "min_joint_limit_margin_by_joint": min_joint_limit_margin_by_joint.copy(),
            "final_joint_limit_margin_by_joint": np.minimum(
                qpos - self.arm_ctrl_low,
                self.arm_ctrl_high - qpos,
            ),
        }

        return qpos

    def _copy_data_for_ik(self):
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        self.ik_data.ctrl[:] = self.data.ctrl
        if self.model.nmocap > 0:
            self.ik_data.mocap_pos[:] = self.data.mocap_pos
            self.ik_data.mocap_quat[:] = self.data.mocap_quat

    def _clip_target_position(self, target_pos):
        return np.clip(
            target_pos,
            self.task_workspace_low,
            self.task_workspace_high,
        )

    def _pinch_approach_axis(self, data):
        rotation = data.site_xmat[self.pinch_site_id].reshape(3, 3)
        axis = rotation[:, self.approach_axis_index].copy()
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return self.desired_approach_axis.copy()
        return axis / norm

    def _pregrasp_position(self):
        object_position = self.data.site_xpos[self.object_site_id].copy()
        return object_position + np.array(
            [0.0, 0.0, self.pregrasp_height],
            dtype=np.float64,
        )

    def _grasp_position(self):
        object_position = self.data.site_xpos[self.object_site_id].copy()
        return object_position + np.array(
            [0.0, 0.0, self.grasp_height_offset],
            dtype=np.float64,
        )

    def _place_goal_hover_position(self):
        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        object_position = self.data.site_xpos[self.object_site_id].copy()
        goal_position = self._control_goal_position("place_goal_hover")
        target_position = pinch_position.copy()
        target_position[:2] += goal_position[:2] - object_position[:2]
        return target_position

    def _gripper_state(self):
        gripper_ctrl = float(self.data.ctrl[self.gripper_actuator_id])
        denom = self.gripper_ctrl_high - self.gripper_ctrl_low
        if abs(denom) < 1e-8:
            return 0.0
        return float(
            np.clip(
                (gripper_ctrl - self.gripper_ctrl_low) / denom,
                0.0,
                1.0,
            )
        )

    def _pad_object_contact_flags(self):
        if self.is_grasp_latched:
            return True, True
        return self._raw_pad_object_contact_flags()

    def _check_grasp_success(self, info, update_counter=True):
        gripper_closed = info["gripper_state"] > self.close_gripper_threshold
        has_left_contact = bool(info.get("has_left_contact", info.get("left_pad_contact", False)))
        has_right_contact = bool(info.get("has_right_contact", info.get("right_pad_contact", False)))
        has_bilateral_contact = has_left_contact and has_right_contact
        distance_ok = info["pinch_to_object_distance"] < self.latch_distance_threshold

        coarse_grasp_success = (
            self.stage >= STAGE_GRASP
            and gripper_closed
            and has_bilateral_contact
            and distance_ok
        )

        stable_grasp_pose_ok = (
            info["grasp_xy_error"] < self.stable_grasp_xy_threshold
            and abs(info["grasp_z_error"]) < self.latch_grasp_z_threshold
        )
        strict_grasp_pose_ok = (
            info["grasp_xy_error"] < self.strict_grasp_xy_threshold
            and abs(info["grasp_z_error"]) < self.latch_grasp_z_threshold
        )
        penetration_ok = (
            info["pad_object_penetration"] <= self.grasp_success_penetration_tolerance
            and info["max_pad_object_penetration"]
            <= self.grasp_success_penetration_tolerance
        )

        stable_grasp_candidate = (
            coarse_grasp_success
            and stable_grasp_pose_ok
            and penetration_ok
        )

        if update_counter:
            if stable_grasp_candidate:
                self.grasp_stable_count += 1
            else:
                self.grasp_stable_count = 0
            self.bilateral_contact_steps = self.grasp_stable_count

        effective_count = self.grasp_stable_count if stable_grasp_candidate else 0
        stable_grasp_success = (
            effective_count >= self.grasp_stable_required_steps
        )
        strict_grasp_success = (
            stable_grasp_success and strict_grasp_pose_ok
        )

        return {
            "coarse_grasp_success": bool(coarse_grasp_success),
            "stable_grasp_success": bool(stable_grasp_success),
            "strict_grasp_success": bool(strict_grasp_success),
            "stable_grasp_candidate": bool(stable_grasp_candidate),
            "grasp_stable_count": int(effective_count),
            "gripper_closed": bool(gripper_closed),
            "has_bilateral_contact": bool(has_bilateral_contact),
            "distance_ok": bool(distance_ok),
            "grasp_pose_ok": bool(stable_grasp_pose_ok),
            "strict_grasp_pose_ok": bool(strict_grasp_pose_ok),
            "penetration_ok": bool(penetration_ok),
        }

    def _update_grasp_latch(self, update_counter=True):
        gripper_state = self._gripper_state()
        if self.is_grasp_latched and gripper_state < 0.15:
            self.is_grasp_latched = False
            self.grasp_object_offset[:] = 0.0
            self.latched_object_xy[:] = 0.0
            self.bilateral_contact_steps = 0
            self.grasp_stable_count = 0
            return

        pinch_position = self.data.site_xpos[self.pinch_site_id].copy()
        object_position = self.data.site_xpos[self.object_site_id].copy()
        pinch_to_object_distance = np.linalg.norm(object_position - pinch_position)
        grasp_position = self._grasp_position()
        grasp_delta = grasp_position - pinch_position
        grasp_xy_error = np.linalg.norm(grasp_delta[:2])
        grasp_z_error = abs(grasp_delta[2])
        left_contact, right_contact = self._raw_pad_object_contact_flags()
        pad_object_penetration = self._pad_object_contact_penetration()
        self.max_pad_object_penetration = max(
            self.max_pad_object_penetration,
            pad_object_penetration,
        )

        if not self.is_grasp_latched:
            grasp_result = self._check_grasp_success(
                {
                    "gripper_state": gripper_state,
                    "has_left_contact": bool(left_contact),
                    "has_right_contact": bool(right_contact),
                    "pinch_to_object_distance": float(pinch_to_object_distance),
                    "grasp_xy_error": float(grasp_xy_error),
                    "grasp_z_error": float(grasp_z_error),
                    "pad_object_penetration": float(pad_object_penetration),
                    "max_pad_object_penetration": float(self.max_pad_object_penetration),
                },
                update_counter=update_counter,
            )
        else:
            grasp_result = {"stable_grasp_success": False}

        latch_enable_allowed = (
            self.fsm_mode == "privileged_fsm"
            or self.task_supervisor.grasp_confirmed
        )
        if (not self.is_grasp_latched
                and latch_enable_allowed
                and grasp_result["stable_grasp_success"]):
            self.is_grasp_latched = True
            self.grasp_object_offset = object_position - pinch_position
            self.latched_object_xy = object_position[:2].copy()

        if self.is_grasp_latched and self.simulated_latch_physics:
            new_object_position = pinch_position + self.grasp_object_offset
            if self.stage == STAGE_PLACE:
                x_min, x_max = self.table_x_bounds
                y_min, y_max = self.table_y_bounds
                margin = self.object_table_margin
                new_object_position[0] = np.clip(
                    new_object_position[0],
                    x_min + margin,
                    x_max - margin,
                )
                new_object_position[1] = np.clip(
                    new_object_position[1],
                    y_min + margin,
                    y_max - margin,
                )
            else:
                new_object_position[:2] = np.clip(
                    new_object_position[:2],
                    self.latched_object_xy - self.max_latched_object_xy_drift,
                    self.latched_object_xy + self.max_latched_object_xy_drift,
                )
            new_object_position[2] = max(
                new_object_position[2],
                self.object_center_z,
            )
            self.data.qpos[self.object_qpos_id : self.object_qpos_id + 3] = (
                new_object_position
            )
            self.data.qpos[self.object_qpos_id + 3 : self.object_qpos_id + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            self.data.qvel[self.object_qvel_slice] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _raw_pad_object_contact_flags(self):
        if self.object_geom_id < 0:
            return False, False

        left_contact = False
        right_contact = False
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.object_geom_id not in (geom1, geom2):
                continue

            other_geom = geom2 if geom1 == self.object_geom_id else geom1
            if other_geom in self.left_pad_geom_ids:
                left_contact = True
            if other_geom in self.right_pad_geom_ids:
                right_contact = True

        return left_contact, right_contact

    def _pad_object_contact_penetration(self):
        if self.object_geom_id < 0:
            return 0.0

        max_penetration = 0.0
        pad_geom_ids = self.left_pad_geom_ids | self.right_pad_geom_ids
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.object_geom_id not in (geom1, geom2):
                continue

            other_geom = geom2 if geom1 == self.object_geom_id else geom1
            if other_geom not in pad_geom_ids:
                continue
            max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))

        return float(max_penetration)

    def _table_clearance_penalty(self, pinch_position, attachment_position):
        if not self._is_over_table_xy(pinch_position):
            return 0.0

        pinch_violation = max(0.0, self.pinch_min_z_over_table - pinch_position[2])
        attachment_violation = max(
            0.0,
            self.attachment_min_z_over_table - attachment_position[2],
        )
        return float(pinch_violation + attachment_violation)

    def _low_away_from_object_penalty(self, pinch_position, object_position):
        horizontal_distance = np.linalg.norm((pinch_position - object_position)[:2])
        low_violation = max(0.0, self.table_top_z + 0.06 - pinch_position[2])
        if horizontal_distance < 0.08:
            return 0.0
        return float(low_violation)

    def _is_over_table_xy(self, position):
        x_min, x_max = self.table_x_bounds
        y_min, y_max = self.table_y_bounds
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def _table_side_penalty(self, position):
        x_min, x_max = self.table_x_bounds
        y_min, y_max = self.table_y_bounds
        margin = self.table_side_margin

        near_table = (
            x_min - margin <= position[0] <= x_max + margin
            and y_min - margin <= position[1] <= y_max + margin
        )
        if not near_table:
            return 0.0

        edge_violation = (
            max(0.0, x_min + margin - position[0])
            + max(0.0, position[0] - (x_max - margin))
            + max(0.0, y_min + margin - position[1])
            + max(0.0, position[1] - (y_max - margin))
        )
        if edge_violation <= 0.0:
            return 0.0

        low_violation = max(0.0, self.table_side_clearance_z - position[2])
        return float(edge_violation * (1.0 + 4.0 * low_violation))

    def _object_table_boundary_penalty(self, object_position):
        x_min, x_max = self.table_x_bounds
        y_min, y_max = self.table_y_bounds
        margin = self.object_table_margin

        x_low_violation = max(0.0, x_min + margin - object_position[0])
        x_high_violation = max(0.0, object_position[0] - (x_max - margin))
        y_low_violation = max(0.0, y_min + margin - object_position[1])
        y_high_violation = max(0.0, object_position[1] - (y_max - margin))
        return float(
            x_low_violation
            + x_high_violation
            + y_low_violation
            + y_high_violation
        )

    def _robot_table_contact_count(self):
        if self.table_geom_id < 0:
            return 0

        count = 0
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.table_geom_id not in (geom1, geom2):
                continue

            other_geom = geom2 if geom1 == self.table_geom_id else geom1
            if other_geom == self.object_geom_id:
                continue
            count += 1

        return count

    def _sample_object_position(self):
        return np.array(
            [
                self.np_random.uniform(0.35, 0.65),
                self.np_random.uniform(-0.20, 0.20),
                self.object_center_z,
            ],
            dtype=np.float64,
        )

    def _sample_goal_position(self):
        if self.training_stage in (STAGE_PLACE, STAGE_FULL):
            object_position = self.data.site_xpos[self.object_site_id].copy()
            radius = self.np_random.uniform(*self.place_goal_radius_range)
            angle = self.np_random.uniform(-np.pi, np.pi)
            goal_position = object_position + np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0,
                ],
                dtype=np.float64,
            )
            x_min, x_max = self.table_x_bounds
            y_min, y_max = self.table_y_bounds
            margin = self.object_table_margin + self.object_half_size
            goal_position[0] = np.clip(goal_position[0], x_min + margin, x_max - margin)
            goal_position[1] = np.clip(goal_position[1], y_min + margin, y_max - margin)
            goal_position[2] = self.object_center_z
            return goal_position

        return np.array(
            [
                self.np_random.uniform(0.35, 0.75),
                self.np_random.uniform(-0.25, 0.25),
                self.object_center_z,
            ],
            dtype=np.float64,
        )

    def close(self):
        self.mujoco_renderer.close()


My4C2StagedEnv = My4C2AllStageEnv
