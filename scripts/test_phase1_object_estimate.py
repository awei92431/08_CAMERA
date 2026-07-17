"""Fail-closed and rigid-propagation tests for Phase 1 ObjectEstimate."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourc2  # noqa: E402,F401
from fourc2.object_estimate import (  # noqa: E402
    ObjectEstimate, ObjectEstimateUnavailable, TcpObjectTracker,
)

OUTPUT = ROOT / "outputs" / "phase1_unified_object_estimate"


def expect_unavailable(env, expected_reason):
    qpos = env.data.qpos.copy(); time_before = float(env.data.time)
    try:
        env.control_observation()
    except ObjectEstimateUnavailable as exc:
        assert exc.reason == expected_reason, (exc.reason, expected_reason)
    else:
        raise AssertionError(f"expected {expected_reason}")
    assert np.array_equal(qpos, env.data.qpos)
    assert float(env.data.time) == time_before
    return {"reason": expected_reason, "physics_unchanged": True,
            "behavior": "diagnostic_stop_before_model_predict_or_env.step"}


def main():
    wrapped = gym.make(
        "My4C2AllStageSinglePPOV22Cube3cm-v0",
        object_observation_mode="rgbd", object_estimate_max_age=1.0,
        disable_env_checker=True,
    )
    env = wrapped.unwrapped
    wrapped.reset(seed=0)
    results = {"missing": expect_unavailable(env, "missing")}

    env.invalidate_object_estimate("rgbd_dropout")
    results["dropout"] = expect_unavailable(env, "rgbd_dropout")

    env.publish_object_estimate(ObjectEstimate(
        [0.5, 0.0, 0.315], env.data.time - 1.01, True, 0.8,
        "rgbd_visual", "stale-001"))
    results["stale"] = expect_unavailable(env, "stale")

    mixing_rejected = False
    try:
        env.publish_object_estimate(ObjectEstimate(
            [0.5, 0.0, 0.315], env.data.time, True, 1.0,
            "ground_truth_simulation", "forbidden-gt"))
    except ValueError:
        mixing_rejected = True
    assert mixing_rejected
    results["mode_mixing_rejected"] = True

    estimate = ObjectEstimate(
        [0.5, -0.1, 0.315], env.data.time, True, 0.9,
        "rgbd_visual", "visual-001")
    env.publish_object_estimate(estimate)
    obs = env.control_observation()
    assert np.allclose(obs[15:18], estimate.position)
    assert np.allclose(obs[18:21], estimate.position + [0, 0, env.pregrasp_height])
    assert np.allclose(obs[21:24], estimate.position + [0, 0, env.grasp_height_offset])
    results["valid_observation"] = {
        "estimate_id": estimate.estimate_id,
        "finite": bool(np.isfinite(obs).all()),
        "object_position_matches": True,
    }
    wrapped.close()

    tracker = TcpObjectTracker([1.0, 0.0, 0.0])
    identity = np.eye(3)
    tracker.update([0, 0, 0], identity, True)
    rz90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    propagated = tracker.update([0, 2, 0], rz90, True)
    expected = np.array([0., 3., 0.])
    error = float(np.linalg.norm(propagated - expected))
    assert error < 1e-12
    results["tcp_fk_propagation"] = {
        "expected": expected.tolist(), "actual": propagated.tolist(),
        "error_m": error, "uses_object_truth": False,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "fail_closed_and_propagation_tests.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"PASS: {path}")


if __name__ == "__main__":
    main()
