import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fourc2
from fourc2.envs.allstage import STAGE_GRASP, STAGE_LIFT, STAGE_PLACE, STAGE_REACH


STAGE_NAMES = {
    STAGE_REACH: "reach",
    STAGE_GRASP: "grasp",
    STAGE_LIFT: "lift",
    STAGE_PLACE: "place",
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_verify_bundle(manifest_path):
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)

    expected_pipeline = manifest["environment_pipeline_version"]
    if fourc2.PIPELINE_VERSION != expected_pipeline:
        raise RuntimeError(
            "Environment pipeline mismatch: "
            f"current={fourc2.PIPELINE_VERSION!r}, expected={expected_pipeline!r}"
        )

    for relative_path, expected_hash in manifest["environment_sha256"].items():
        path = PROJECT_ROOT / relative_path
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Environment hash mismatch for {relative_path}: "
                f"actual={actual_hash}, expected={expected_hash}"
            )

    experts = {}
    for name, spec in manifest["experts"].items():
        path = PROJECT_ROOT / spec["path"]
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(
                f"Expert hash mismatch for {name}: "
                f"actual={actual_hash}, expected={spec['sha256']}"
            )
        run_config_path = path.parent.parent / "run_config.json"
        with run_config_path.open("r", encoding="utf-8") as file_obj:
            run_config = json.load(file_obj)
        if run_config.get("pipeline_version") != spec["pipeline_version"]:
            raise RuntimeError(f"Expert pipeline mismatch for {name}")
        if run_config.get("output_model_sha256", {}).get(path.name) != actual_hash:
            raise RuntimeError(f"Expert is not recorded by its source run: {name}")
        model = PPO.load(path)
        if model.observation_space.shape != (39,):
            raise RuntimeError(
                f"Expert {name} must consume 39-D observations, got "
                f"{model.observation_space.shape}"
            )
        experts[name] = model
    return manifest, experts


def mean(infos, key):
    return float(np.mean([float(info.get(key, 0.0)) for info in infos]))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the reproducible stage-routed 4C2 policy bundle"
    )
    parser.add_argument(
        "--manifest",
        default="configs/stage_router_v2_v21.json",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose-episodes", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest, experts = load_and_verify_bundle(manifest_path)

    print("bundle_version:", manifest["bundle_version"])
    print("environment_pipeline_version:", manifest["environment_pipeline_version"])
    print("manifest:", manifest_path)
    for name, spec in manifest["experts"].items():
        print(f"expert_{name}: {spec['path']} sha256={spec['sha256']}")
    print("place_controller: zero_action_place_servo")

    final_infos = []
    episode_rewards = []
    episode_lengths = []
    raw_bilateral_ever = []
    latched_ever = []
    opened_ever = []
    max_lifts = []

    for episode in range(args.episodes):
        env = gym.make(
            manifest["env_id"],
            render_mode="human" if args.render else None,
        )
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        saw_raw_bilateral = False
        saw_latched = False
        saw_opened = False
        max_lift = float(info.get("object_lift", 0.0))

        for step in range(env.spec.max_episode_steps):
            stage = int(info["active_stage"])
            if stage == STAGE_PLACE:
                action = np.zeros(4, dtype=np.float32)
            else:
                expert_name = manifest["router"][STAGE_NAMES[stage]]
                action, _ = experts[expert_name].predict(
                    obs[:39],
                    deterministic=True,
                )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            saw_raw_bilateral |= bool(info.get("has_raw_bilateral_contact", False))
            saw_latched |= bool(info.get("is_grasp_latched", False))
            saw_opened |= bool(info.get("place_has_opened", False))
            max_lift = max(max_lift, float(info.get("object_lift", 0.0)))
            if terminated or truncated:
                break

        final_infos.append(info.copy())
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        raw_bilateral_ever.append(float(saw_raw_bilateral))
        latched_ever.append(float(saw_latched))
        opened_ever.append(float(saw_opened))
        max_lifts.append(max_lift)
        env.close()

        if args.verbose_episodes or args.render:
            print(
                f"episode={episode} steps={step + 1} reward={total_reward:.3f} "
                f"success={int(bool(info.get('place_success', False)))} "
                f"stage={STAGE_NAMES.get(int(info['active_stage']), info['active_stage'])} "
                f"goal_xy={float(info['object_to_goal_xy_distance']):.4f} "
                f"lift={float(info['object_lift']):.4f} "
                f"pad_pen_max={float(info['max_pad_object_penetration']):.4f}"
            )

    stage_counts = Counter(
        STAGE_NAMES.get(int(info["active_stage"]), str(info["active_stage"]))
        for info in final_infos
    )
    print("\nStage-router 评估汇总")
    print(
        f"score: reward={np.mean(episode_rewards):.3f} "
        f"success={mean(final_infos, 'place_success'):.3f} "
        f"steps_mean={np.mean(episode_lengths):.1f} "
        f"steps_max={max(episode_lengths)}"
    )
    print(
        "stage: "
        f"reach={mean(final_infos, 'reach_success'):.3f} "
        f"grasp={mean(final_infos, 'grasp_success'):.3f} "
        f"lift={mean(final_infos, 'lift_success'):.3f} "
        f"place={mean(final_infos, 'place_success'):.3f}"
    )
    print(
        "contact: "
        f"raw_bi_ever={np.mean(raw_bilateral_ever):.3f} "
        f"latched_ever={np.mean(latched_ever):.3f} "
        f"opened_ever={np.mean(opened_ever):.3f}"
    )
    print(
        "place: "
        f"goal_xy={mean(final_infos, 'object_to_goal_xy_distance'):.4f} "
        f"goal_abs_z={mean(final_infos, 'object_lift'):.4f} "
        f"max_lift={np.mean(max_lifts):.4f}"
    )
    print(
        "safety: "
        f"fail={mean(final_infos, 'stage_failure'):.3f} "
        f"table_contacts={mean(final_infos, 'table_contact_count'):.3f} "
        f"pad_pen={mean(final_infos, 'pad_object_penetration'):.4f} "
        f"pad_pen_max={max(float(i['max_pad_object_penetration']) for i in final_infos):.4f}"
    )
    print("final_stage_counts:", dict(stage_counts))


if __name__ == "__main__":
    main()
