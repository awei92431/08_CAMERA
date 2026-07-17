"""Authoritative runtime object estimate shared by policy and control."""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


class ObjectEstimateUnavailable(RuntimeError):
    """Raised when runtime control has no valid, fresh object estimate."""

    def __init__(self, reason, consumer, now, estimate=None):
        super().__init__(f"object estimate unavailable: {reason}; consumer={consumer}")
        self.reason = str(reason)
        self.consumer = str(consumer)
        self.now = float(now)
        self.estimate = estimate


@dataclass(frozen=True)
class ObjectEstimate:
    position: np.ndarray
    timestamp: float
    valid: bool
    confidence: float
    source: str
    estimate_id: str
    orientation_wxyz: Optional[np.ndarray] = None

    def __post_init__(self):
        position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        position.setflags(write=False)
        object.__setattr__(self, "position", position)
        if self.orientation_wxyz is not None:
            quat = np.asarray(self.orientation_wxyz, dtype=np.float64).reshape(4).copy()
            quat.setflags(write=False)
            object.__setattr__(self, "orientation_wxyz", quat)

    def with_position(self, position, timestamp, source, estimate_id):
        return replace(self, position=position, timestamp=float(timestamp),
                       source=str(source), estimate_id=str(estimate_id))


class ObjectEstimateAuthority:
    """Fixed-mode, fail-closed owner of the runtime object estimate."""

    MODES = ("ground_truth", "rgbd")

    def __init__(self, mode="ground_truth", max_age=60.0):
        if mode not in self.MODES:
            raise ValueError(f"unknown object observation mode: {mode}")
        if max_age <= 0:
            raise ValueError("object estimate max_age must be positive")
        self.mode = str(mode)
        self.max_age = float(max_age)
        self.current = None
        self.initial_z = None
        self.uses = []
        self.failure_reason = None

    def reset(self):
        self.current = None
        self.initial_z = None
        self.uses.clear()
        self.failure_reason = None

    def publish(self, estimate):
        if not isinstance(estimate, ObjectEstimate):
            raise TypeError("publish requires ObjectEstimate")
        if self.mode == "rgbd" and estimate.source.startswith("ground_truth"):
            raise ValueError("ground-truth estimate cannot enter rgbd authority")
        if self.mode == "ground_truth" and not estimate.source.startswith("ground_truth"):
            raise ValueError("non-ground-truth estimate cannot enter ground_truth authority")
        if estimate.valid and not np.isfinite(estimate.position).all():
            raise ValueError("valid object estimate must have finite position")
        if not 0.0 <= estimate.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.current = estimate
        if estimate.valid and self.initial_z is None:
            self.initial_z = float(estimate.position[2])
        return estimate

    def invalidate(self, timestamp, source, estimate_id, reason):
        previous = np.full(3, np.nan) if self.current is None else self.current.position
        self.current = ObjectEstimate(previous, float(timestamp), False, 0.0,
                                      str(source), str(estimate_id))
        self.failure_reason = str(reason)

    def require(self, now, consumer, control_step):
        estimate = self.current
        if estimate is None:
            reason = "missing"
        elif not estimate.valid:
            reason = self.failure_reason or "invalid"
        elif not np.isfinite(estimate.position).all():
            reason = "non_finite"
        elif float(now) - estimate.timestamp > self.max_age:
            reason = "stale"
        elif estimate.timestamp > float(now) + 1e-9:
            reason = "future_timestamp"
        else:
            reason = None
        if reason is not None:
            self.failure_reason = reason
            raise ObjectEstimateUnavailable(reason, consumer, now, estimate)
        self.uses.append({
            "control_step": int(control_step), "consumer": str(consumer),
            "estimate_id": estimate.estimate_id,
            "timestamp": float(estimate.timestamp), "source": estimate.source,
        })
        return estimate

    def uses_for_step(self, control_step):
        return [entry.copy() for entry in self.uses
                if entry["control_step"] == int(control_step)]


class TcpObjectTracker:
    """Propagate a visually initialized object using a rigid TCP transform."""

    def __init__(self, initial_world_position):
        self.world_position = np.asarray(
            initial_world_position, dtype=np.float64
        ).reshape(3).copy()
        self.tcp_local_offset = None

    def update(self, tcp_world_position, world_from_tcp, grasp_confirmed):
        tcp = np.asarray(tcp_world_position, dtype=np.float64).reshape(3)
        rotation = np.asarray(world_from_tcp, dtype=np.float64).reshape(3, 3)
        if grasp_confirmed and self.tcp_local_offset is None:
            self.tcp_local_offset = rotation.T @ (self.world_position - tcp)
        if grasp_confirmed and self.tcp_local_offset is not None:
            self.world_position = tcp + rotation @ self.tcp_local_offset
        return self.world_position.copy()
