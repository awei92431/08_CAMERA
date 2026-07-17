from pathlib import Path
import argparse
import sys

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fourc2


def fmt(array):
    return np.array2string(np.asarray(array), precision=4, suppress_small=True)


def site_positions(env):
    return {
        "pinch": env.data.site_xpos[env.pinch_site_id].copy(),
        "object": env.data.site_xpos[env.object_site_id].copy(),
        "goal": env.data.site_xpos[env.goal_site_id].copy(),
    }


def print_reset_summary(env, obs, info):
    positions = site_positions(env)
    print("\n=== reset ===")
    print(f"obs.shape={obs.shape} obs.dtype={obs.dtype} finite={np.isfinite(obs).all()}")
    print(f"action_space={env.action_space}")
    print(f"observation_space={env.observation_space}")
    print(f"pinch={fmt(positions['pinch'])}")
    print(f"object={fmt(positions['object'])}")
    print(f"goal={fmt(positions['goal'])}")
    print(f"info={info}")


def print_step_summary(step_id, action, obs, reward, terminated, truncated, info, env, prev_pinch):
    positions = site_positions(env)
    pinch_delta = positions["pinch"] - prev_pinch
    print(f"\n=== step {step_id} ===")
    print(f"action={fmt(action)}")
    print(f"obs.shape={obs.shape} obs.dtype={obs.dtype} finite={np.isfinite(obs).all()}")
    print(
        "reward="
        f"{reward:.6f} terminated={terminated} truncated={truncated} "
        f"distance={info['distance']:.6f}"
    )
    print(f"pinch={fmt(positions['pinch'])} delta={fmt(pinch_delta)}")
    print(f"object={fmt(positions['object'])}")
    print(f"goal={fmt(positions['goal'])}")
    print(f"arm_ctrl={fmt(env.data.ctrl[env.arm_actuator_ids])}")
    print(f"gripper_ctrl={env.data.ctrl[env.gripper_actuator_id]:.4f}")
    for key in [
        "stage",
        "object_lift",
        "lift_distance",
        "object_horizontal_drift",
        "object_xy_step",
        "is_grasp_latched",
        "has_bilateral_contact",
        "has_unilateral_contact",
        "vertical_alignment",
        "table_contact_count",
        "object_table_boundary_penalty",
    ]:
        if key in info:
            print(f"{key}={info[key]}")
    return positions["pinch"]


def scripted_actions():
    return [
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="My4C2AllStage-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-steps", type=int, default=5)
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--skip-scripted", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env_id, render_mode=args.render_mode).unwrapped

    try:
        obs, info = env.reset(seed=args.seed)
        print_reset_summary(env, obs, info)

        prev_pinch = site_positions(env)["pinch"]
        step_id = 0

        if not args.skip_scripted:
            for action in scripted_actions():
                step_id += 1
                obs, reward, terminated, truncated, info = env.step(action)
                prev_pinch = print_step_summary(
                    step_id,
                    action,
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                    env,
                    prev_pinch,
                )
                if terminated or truncated:
                    obs, info = env.reset(seed=args.seed + step_id)
                    print_reset_summary(env, obs, info)
                    prev_pinch = site_positions(env)["pinch"]

        rng = np.random.default_rng(args.seed)
        for _ in range(args.random_steps):
            step_id += 1
            action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            prev_pinch = print_step_summary(
                step_id,
                action,
                obs,
                reward,
                terminated,
                truncated,
                info,
                env,
                prev_pinch,
            )
            if terminated or truncated:
                obs, info = env.reset(seed=args.seed + step_id)
                print_reset_summary(env, obs, info)
                prev_pinch = site_positions(env)["pinch"]

    finally:
        env.close()


if __name__ == "__main__":
    main()
