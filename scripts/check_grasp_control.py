import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "glfw")

import fourc2


def main():
    parser = argparse.ArgumentParser(
        description="Check GraspStage with a simple hand-written TCP policy."
    )
    parser.add_argument("--env-id", default="My4C2GraspStage-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--close-distance", type=float, default=0.045)
    args = parser.parse_args()

    wrapped_env = gym.make(args.env_id, render_mode="human" if args.render else None)
    max_episode_steps = wrapped_env.spec.max_episode_steps or 250
    env = wrapped_env.unwrapped

    successes = []
    latched = []
    bilateral = []
    drifts = []
    final_distances = []
    first_success_steps = []

    print("env_id:", args.env_id)
    print("control: descend to grasp point, then close gripper")
    print("action_scale:", env.action_scale)
    print("close_distance:", args.close_distance)
    print("max_grasp_object_xy_drift:", env.max_grasp_object_xy_drift)

    for episode in range(args.episodes):
        obs, info = wrapped_env.reset(seed=args.seed + episode)
        first_success_step = None
        final_info = info

        for step in range(max_episode_steps):
            target = final_info["grasp_position"].copy()
            tcp_target = final_info["tcp_target_position"].copy()
            delta = target - tcp_target

            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            if (
                final_info["pinch_to_grasp_distance"] < args.close_distance
                and final_info["grasp_xy_error"] < env.grasp_xy_close_threshold
                and abs(final_info["grasp_z_error"]) < env.grasp_z_close_threshold
            ):
                action[3] = 1.0
            else:
                action[3] = -1.0

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            final_info = info

            if first_success_step is None and info["grasp_success"]:
                first_success_step = step + 1
            if terminated or truncated:
                break
            if args.render and args.sleep > 0:
                time.sleep(args.sleep)

        success = bool(final_info["grasp_success"])
        successes.append(float(success))
        latched.append(float(final_info["is_grasp_latched"]))
        bilateral.append(float(final_info["has_bilateral_contact"]))
        drifts.append(float(final_info["object_horizontal_drift"]))
        final_distances.append(float(final_info["pinch_to_grasp_distance"]))
        if first_success_step is not None:
            first_success_steps.append(first_success_step)

        print(
            f"episode={episode} "
            f"success={int(success)} "
            f"first_success_step={first_success_step} "
            f"steps={step + 1} "
            f"grasp_d={final_info['pinch_to_grasp_distance']:.4f} "
            f"xy={final_info['grasp_xy_error']:.4f} "
            f"z={final_info['grasp_z_error']:.4f} "
            f"grip={final_info['gripper_state']:.2f} "
            f"latched={int(final_info['is_grasp_latched'])} "
            f"bilateral={int(final_info['has_bilateral_contact'])} "
            f"unilateral={int(final_info['has_unilateral_contact'])} "
            f"drift={final_info['object_horizontal_drift']:.4f} "
            f"fail={int(final_info['stage_failure'])} "
            f"tcp_err={final_info['tcp_target_error']:.4f}"
        )

    print("\nsummary")
    print("success_rate:", f"{np.mean(successes):.3f}")
    print("latched_rate:", f"{np.mean(latched):.3f}")
    print("bilateral_rate:", f"{np.mean(bilateral):.3f}")
    print("mean_final_distance:", f"{np.mean(final_distances):.4f}")
    print("mean_xy_drift:", f"{np.mean(drifts):.4f}")
    if first_success_steps:
        print("mean_first_success_step:", f"{np.mean(first_success_steps):.1f}")
    else:
        print("mean_first_success_step: none")

    wrapped_env.close()


if __name__ == "__main__":
    main()
