"""Authoritative, episode-frozen goal state for policy and runtime control.

`T_A_B` conventions elsewhere in this project mean points are stored in the
fixed MuJoCo world frame before publication.  ArUco mode accepts exactly one
valid estimate per episode; later publications are rejected so returning home
or seeing the marker again cannot move the task goal.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


class GoalEstimateUnavailable(RuntimeError):
    """Raised when a runtime consumer has no valid, frozen episode goal."""

    def __init__(self, reason, consumer, now, estimate=None):
        super().__init__(f"goal estimate unavailable: {reason}; consumer={consumer}")
        self.reason = str(reason)
        self.consumer = str(consumer)
        self.now = float(now)
        self.estimate = estimate


@dataclass(frozen=True)
class GoalEstimate:
    position: np.ndarray
    timestamp: float
    valid: bool
    confidence: float
    source: str
    estimate_id: str
    marker_id: int = -1
    rotation: Optional[np.ndarray] = None
    frame: str = "world"
    failure_reason: Optional[str] = None
    diagnostics: Optional[dict] = None

    def __post_init__(self):
        position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        position.setflags(write=False)
        object.__setattr__(self, "position", position)
        if self.rotation is not None:
            rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3).copy()
            rotation.setflags(write=False)
            object.__setattr__(self, "rotation", rotation)


class GoalEstimateAuthority:
    """Fixed-mode, fail-closed owner of the goal used by every consumer."""

    MODES = ("ground_truth", "aruco")

    def __init__(self, mode="ground_truth"):
        if mode not in self.MODES:
            raise ValueError(f"unknown goal observation mode: {mode}")
        self.mode = str(mode)
        self.current = None
        self.frozen = False
        self.uses = []
        self.failure_reason = None

    def reset(self):
        self.current = None
        self.frozen = False
        self.uses.clear()
        self.failure_reason = None

    def publish(self, estimate):
        if not isinstance(estimate, GoalEstimate):
            raise TypeError("publish requires GoalEstimate")
        if estimate.frame != "world":
            raise ValueError("runtime goal must be published in the world frame")
        if self.mode == "aruco" and estimate.source.startswith("ground_truth"):
            raise ValueError("ground-truth goal cannot enter aruco authority")
        if self.mode == "ground_truth" and not estimate.source.startswith("ground_truth"):
            raise ValueError("non-ground-truth goal cannot enter ground_truth authority")
        if estimate.valid and not np.isfinite(estimate.position).all():
            raise ValueError("valid goal estimate must have finite position")
        if not 0.0 <= float(estimate.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.mode == "aruco" and self.frozen:
            raise RuntimeError("episode ArUco goal is already frozen")
        self.current = estimate
        self.failure_reason = estimate.failure_reason
        if self.mode == "aruco" and estimate.valid:
            self.frozen = True
        return estimate

    def invalidate(self, timestamp, source, estimate_id, reason):
        if self.mode == "aruco" and self.frozen:
            raise RuntimeError("cannot invalidate a frozen episode goal")
        self.current = GoalEstimate(
            position=np.full(3, np.nan), timestamp=float(timestamp), valid=False,
            confidence=0.0, source=str(source), estimate_id=str(estimate_id),
            failure_reason=str(reason),
        )
        self.failure_reason = str(reason)

    def require(self, now, consumer, control_step):
        estimate = self.current
        if estimate is None:
            reason = "missing"
        elif not estimate.valid:
            reason = estimate.failure_reason or self.failure_reason or "invalid"
        elif not np.isfinite(estimate.position).all():
            reason = "non_finite"
        elif self.mode == "aruco" and not self.frozen:
            reason = "not_frozen"
        else:
            reason = None
        if reason is not None:
            self.failure_reason = reason
            raise GoalEstimateUnavailable(reason, consumer, now, estimate)
        self.uses.append({
            "control_step": int(control_step),
            "consumer": str(consumer),
            "estimate_id": estimate.estimate_id,
            "timestamp": float(estimate.timestamp),
            "source": estimate.source,
        })
        return estimate

    def uses_for_step(self, control_step):
        return [entry.copy() for entry in self.uses
                if entry["control_step"] == int(control_step)]
