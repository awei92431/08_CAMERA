import argparse
import os
import sys
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
        description="Check whether the Cartesian action + IK layer can reach pregrasp."
    )
    parser.add_argument("--env-id", default="My4C2ReachStage-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    import time

    wrapped_env = gym.make(args.env_id, render_mode="human" if args.render else None)
    max_episode_steps = wrapped_env.spec.max_episode_steps or 250
    env = wrapped_env.unwrapped
    successes = []
    final_distances = []
    first_success_steps = []

    print("env_id:", args.env_id)
    print("action_scale:", env.action_scale)
    print("approach_threshold:", env.approach_threshold)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        target = env._pregrasp_position().copy()
        first_success_step = None
        final_info = info

        for step in range(max_episode_steps):
            pinch = env.data.site_xpos[env.pinch_site_id].copy()
            delta = target - pinch
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = -1.0

            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info

            if first_success_step is None and info["reach_success"]:
                first_success_step = step + 1
            if terminated or truncated:
                break
            if args.render and args.sleep > 0:
                time.sleep(args.sleep)

        final_distance = float(final_info["pinch_to_pregrasp_distance"])
        success = bool(final_info["reach_success"])
        successes.append(float(success))
        final_distances.append(final_distance)
        if first_success_step is not None:
            first_success_steps.append(first_success_step)

        print(
            f"episode={episode} "
            f"success={int(success)} "
            f"first_success_step={first_success_step} "
            f"final_pre_dist={final_distance:.4f} "
            f"pinch={np.round(final_info['pinch_position'], 4)} "
            f"pregrasp={np.round(final_info['pregrasp_position'], 4)} "
            f"tc={final_info['table_contact_count']} "
            f"clear={final_info['table_clearance_penalty']:.4f} "
            f"low={final_info['low_away_from_object_penalty']:.4f}"
        )

    print("\nsummary")
    print("success_rate:", f"{np.mean(successes):.3f}")
    print("mean_final_pre_dist:", f"{np.mean(final_distances):.4f}")
    if first_success_steps:
        print("mean_first_success_step:", f"{np.mean(first_success_steps):.1f}")
    else:
        print("mean_first_success_step: none")

    wrapped_env.close()


if __name__ == "__main__":
    main()
