"""Deployable task supervision interfaces with no simulator dependencies."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from fourc2.object_estimate import ObjectEstimate


class SupervisorInputUnavailable(RuntimeError):
    def __init__(self, reason, source):
        super().__init__(f"supervisor input unavailable: {source}: {reason}")
        self.reason = str(reason); self.source = str(source)


class TaskStage(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT = 2
    PLACE = 5


class GraspPhase(IntEnum):
    ALIGN = 0
    DESCEND = 1
    CLOSE = 2
    CONFIRM = 3


@dataclass(frozen=True)
class RobotState:
    tcp_position: np.ndarray
    world_from_tcp: np.ndarray
    tcp_velocity: np.ndarray
    vertical_alignment: float
    timestamp: float
    valid: bool
    source: str

    def __post_init__(self):
        object.__setattr__(self, "tcp_position", np.asarray(
            self.tcp_position, dtype=np.float64).reshape(3).copy())
        object.__setattr__(self, "world_from_tcp", np.asarray(
            self.world_from_tcp, dtype=np.float64).reshape(3, 3).copy())
        object.__setattr__(self, "tcp_velocity", np.asarray(
            self.tcp_velocity, dtype=np.float64).reshape(3).copy())


@dataclass(frozen=True)
class GripperState:
    commanded_opening: float
    actual_opening: float
    velocity: float
    motion_status: str
    actuator_effort: float
    fault: bool
    grasp_hold_confidence: float
    timestamp: float
    valid: bool
    source: str

    @property
    def commanded_closure(self):
        return float(np.clip(1.0 - self.commanded_opening, 0.0, 1.0))

    @property
    def actual_closure(self):
        return float(np.clip(1.0 - self.actual_opening, 0.0, 1.0))


@dataclass(frozen=True)
class TaskSupervisorConfig:
    pregrasp_height: float
    grasp_height_offset: float
    reach_xy_threshold: float
    reach_z_threshold: float
    reach_vertical_threshold: float
    grasp_descend_xy_threshold: float
    grasp_xy_close_threshold: float
    grasp_z_close_threshold: float
    stable_grasp_xy_threshold: float
    latch_grasp_z_threshold: float
    lift_height: float
    release_success_lift: float
    release_open_xy_threshold: float
    release_min_open_steps: int
    min_close_steps: int
    max_close_attempt_steps: int
    stable_required_steps: int
    latched_gripper_closure: float
    max_input_age: float = 1.0
    stopped_velocity: float = 0.02
    effort_threshold: float = 0.10
    fully_closed_fraction: float = 0.97


class TaskSupervisor:
    """Task FSM consuming only estimate, robot, gripper and timers."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.stage = TaskStage.REACH
        self.grasp_phase = GraspPhase.ALIGN
        self.commanded_closure = 0.0
        self.close_steps = 0
        self.stable_steps = 0
        self.place_steps = 0
        self.release_commanded = False
        self.grasp_confirmed = False
        self.task_complete = False
        self.failed = False
        self.failure_reason = None
        self.initial_object_z = None
        self.events = []
        self.last_diagnostics = {}

    def _validate(self, now, object_estimate, robot, gripper):
        values = (("object_estimate", object_estimate),
                  ("robot_state", robot), ("gripper_state", gripper))
        for name, value in values:
            if value is None:
                raise SupervisorInputUnavailable("missing", name)
            if not value.valid:
                reason = "fault" if name == "gripper_state" and value.fault else "invalid"
                raise SupervisorInputUnavailable(reason, name)
            if now - float(value.timestamp) > self.config.max_input_age:
                raise SupervisorInputUnavailable("stale", name)
            if value.timestamp > now + 1e-9:
                raise SupervisorInputUnavailable("future_timestamp", name)
        if gripper.fault:
            raise SupervisorInputUnavailable("fault", "gripper_state")

    def _event(self, now, name):
        self.events.append({"timestamp": float(now), "event": str(name),
                            "stage": int(self.stage),
                            "grasp_phase": int(self.grasp_phase)})

    def update(self, now, object_estimate, robot, gripper, goal_position):
        self._validate(now, object_estimate, robot, gripper)
        cfg = self.config
        obj = object_estimate.position
        tcp = robot.tcp_position
        goal = np.asarray(goal_position, dtype=np.float64).reshape(3)
        pregrasp = obj + np.array([0., 0., cfg.pregrasp_height])
        grasp = obj + np.array([0., 0., cfg.grasp_height_offset])
        pre_delta = pregrasp - tcp; grasp_delta = grasp - tcp
        pre_ready = (np.linalg.norm(pre_delta[:2]) < cfg.reach_xy_threshold
                     and abs(pre_delta[2]) < cfg.reach_z_threshold
                     and robot.vertical_alignment > cfg.reach_vertical_threshold)
        grasp_xy = float(np.linalg.norm(grasp_delta[:2]))
        grasp_z = float(abs(grasp_delta[2]))
        fine_close = (grasp_xy < cfg.grasp_xy_close_threshold
                      and grasp_z < cfg.grasp_z_close_threshold)
        stable_geometry = (grasp_xy < cfg.stable_grasp_xy_threshold
                           and grasp_z < cfg.latch_grasp_z_threshold)
        stopped = (gripper.motion_status == "stopped"
                   or abs(gripper.velocity) < cfg.stopped_velocity)
        blocked = gripper.actual_closure < cfg.fully_closed_fraction
        effort_contact = abs(gripper.actuator_effort) >= cfg.effort_threshold
        sensor_hold = (stopped and (blocked or effort_contact)
                       and gripper.grasp_hold_confidence >= 0.5)
        if self.initial_object_z is None:
            self.initial_object_z = float(obj[2])
        object_lift = max(0.0, float(obj[2]) - self.initial_object_z)
        goal_xy = float(np.linalg.norm((goal - obj)[:2]))

        old_stage = self.stage
        if self.stage == TaskStage.REACH:
            self.commanded_closure = 0.0
            if pre_ready:
                self.stage = TaskStage.GRASP
                self.grasp_phase = GraspPhase.DESCEND
                self._event(now, "reach_to_grasp")
        elif self.stage == TaskStage.GRASP:
            if self.grasp_phase == GraspPhase.ALIGN:
                self.commanded_closure = 0.0
                if pre_ready:
                    self.grasp_phase = GraspPhase.DESCEND
            elif self.grasp_phase == GraspPhase.DESCEND:
                self.commanded_closure = 0.0
                if grasp_xy >= cfg.grasp_descend_xy_threshold:
                    self.grasp_phase = GraspPhase.ALIGN
                elif fine_close:
                    self.grasp_phase = GraspPhase.CLOSE
                    self.close_steps = 0
            elif self.grasp_phase == GraspPhase.CLOSE:
                self.commanded_closure = 1.0
                closing_command_active = gripper.commanded_closure > 0.5
                if closing_command_active:
                    self.close_steps += 1
                candidate = (closing_command_active
                             and self.close_steps >= cfg.min_close_steps
                             and stable_geometry and sensor_hold)
                self.stable_steps = self.stable_steps + 1 if candidate else 0
                if self.stable_steps >= cfg.stable_required_steps:
                    self.grasp_phase = GraspPhase.CONFIRM
                    self.grasp_confirmed = True
                    self.stage = TaskStage.LIFT
                    self.commanded_closure = cfg.latched_gripper_closure
                    self._event(now, "grasp_confirmed")
                    self._event(now, "grasp_to_lift")
                elif (closing_command_active
                      and self.close_steps > cfg.max_close_attempt_steps
                      and not sensor_hold):
                    self.grasp_phase = GraspPhase.ALIGN
                    self.close_steps = 0; self.stable_steps = 0
            else:
                self.commanded_closure = cfg.latched_gripper_closure
        elif self.stage == TaskStage.LIFT:
            self.commanded_closure = cfg.latched_gripper_closure
            if object_lift >= cfg.lift_height and sensor_hold:
                self.stage = TaskStage.PLACE
                self.place_steps = 0
                self._event(now, "lift_to_place")
        elif self.stage == TaskStage.PLACE:
            self.place_steps += 1
            low_enough = object_lift <= cfg.release_success_lift
            if (self.place_steps >= cfg.release_min_open_steps
                    and goal_xy < cfg.release_open_xy_threshold and low_enough):
                if not self.release_commanded:
                    self._event(now, "release_commanded")
                self.release_commanded = True
            self.commanded_closure = 0.0 if self.release_commanded else cfg.latched_gripper_closure
            if self.release_commanded and gripper.actual_closure < 0.20 and stopped:
                self.task_complete = True
                self._event(now, "release_complete")

        self.last_diagnostics = {
            "pregrasp_ready": bool(pre_ready), "grasp_xy_error": grasp_xy,
            "grasp_z_error": grasp_z, "stable_geometry": bool(stable_geometry),
            "gripper_stopped": bool(stopped), "blocked_before_full_close": bool(blocked),
            "effort_contact": bool(effort_contact), "sensor_hold": bool(sensor_hold),
            "closing_command_active": bool(gripper.commanded_closure > 0.5),
            "object_lift": object_lift, "goal_xy": goal_xy,
            "stage_changed": bool(self.stage != old_stage),
        }
        return self.last_diagnostics.copy()
