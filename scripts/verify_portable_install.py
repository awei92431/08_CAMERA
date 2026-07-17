#!/usr/bin/env python3
"""Verify that a cloned repository can load and execute the frozen policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fourc2  # noqa: E402,F401  Registers project environments.


EVIDENCE_DIR = PROJECT_ROOT / "eval/final_v22_4c2_20260713"
MANIFEST_PATH = EVIDENCE_DIR / "manifest.json"
MODEL_PATH = (
    PROJECT_ROOT
    / "runs/v22_4c2_lift_place_transition_200k/models/best_handoff_model.zip"
)
ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_model_hash = manifest["model_sha256"]
    actual_model_hash = sha256_file(MODEL_PATH)
    if actual_model_hash != expected_model_hash:
        raise RuntimeError(
            f"Model SHA256 mismatch: expected {expected_model_hash}, got {actual_model_hash}"
        )

    for relative_path, expected_hash in manifest["source_sha256"].items():
        path = PROJECT_ROOT / relative_path
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Source SHA256 mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    print(f"hashes: OK ({len(manifest['source_sha256'])} source files + model)")


def run_episode(env, model: PPO, seed: int) -> tuple[bool, int, float, str]:
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0
    final_info = {}
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        steps += 1
    reason = "success" if final_info.get("is_success", False) else (
        "truncated" if truncated else "failure"
    )
    return bool(final_info.get("is_success", False)), steps, total_reward, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-failed-episode",
        action="store_true",
        help="Only verify execution; do not fail the command if an episode misses the task.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    verify_hashes()
    model = PPO.load(str(MODEL_PATH), device="cpu")
    env = gym.make(ENV_ID)
    observation, _ = env.reset(seed=args.seed)
    if observation.shape != (39,):
        raise RuntimeError(f"Expected observation shape (39,), got {observation.shape}")
    if env.action_space.shape != (4,):
        raise RuntimeError(f"Expected action shape (4,), got {env.action_space.shape}")
    if model.observation_space.shape != env.observation_space.shape:
        raise RuntimeError("Policy/environment observation spaces do not match")
    if model.action_space.shape != env.action_space.shape:
        raise RuntimeError("Policy/environment action spaces do not match")
    print("spaces: OK (observation=(39,), action=(4,))")
    print("checkpoint: OK (PPO loaded on CPU)")

    successes = 0
    for episode in range(args.episodes):
        success, steps, reward, reason = run_episode(env, model, args.seed + episode)
        successes += int(success)
        print(
            f"episode={episode} seed={args.seed + episode} steps={steps} "
            f"reward={reward:.3f} result={reason}"
        )
    env.close()

    if successes != args.episodes and not args.allow_failed_episode:
        raise RuntimeError(
            f"Full-task regression failed: {successes}/{args.episodes} episodes succeeded"
        )
    print(f"portable verification: PASS ({successes}/{args.episodes} task successes)")


if __name__ == "__main__":
    main()

