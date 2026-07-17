"""Evaluate PPO robustness to explicit object-position observation errors only."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import fourc2  # noqa: F401,E402


ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best_full_flow_v22.zip"
OBJECT_POSITION_SLICE = slice(15, 18)
DEFAULT_LEVELS_MM = (0, 1, 2, 3, 5, 8, 10, 15)
STAGE_KEYS = ("reach", "grasp", "lift", "place")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate object-position error robustness without changing the environment."
    )
    parser.add_argument(
        "--error-mode", nargs="+", required=True,
        choices=("fixed", "jitter"),
        help="Explicitly enable one or both separate error experiments.",
    )
    parser.add_argument(
        "--levels-mm", nargs="+", type=float, default=DEFAULT_LEVELS_MM
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "position_error_robustness",
    )
    return parser.parse_args()


def noise_for_step(mode, sigma_m, rng, fixed_bias):
    if sigma_m == 0.0:
        return np.zeros(3, dtype=np.float64)
    if mode == "fixed":
        return fixed_bias
    if mode == "jitter":
        return sigma_m * rng.standard_normal(3)
    raise ValueError(mode)


def failure_stage(ever, success):
    if success:
        return "none"
    for stage in STAGE_KEYS:
        if not ever[stage]:
            return stage
    return "place"


def evaluate_condition(model, mode, level_mm, seeds):
    sigma_m = float(level_mm) / 1000.0
    episodes = []
    injected_vectors = []
    for seed in seeds:
        env = gym.make(ENV_ID)
        try:
            observation, _ = env.reset(seed=seed)
            if observation.shape != (39,):
                raise RuntimeError(f"expected 39D observation, got {observation.shape}")
            # Reinitialize identically for every error level. Thus condition N
            # sees the same standard samples as condition M, merely rescaled.
            rng = np.random.default_rng(810_000 + int(seed))
            fixed_bias = sigma_m * rng.standard_normal(3)
            total_reward = 0.0
            steps = 0
            ever = {stage: False for stage in STAGE_KEYS}
            final_info = {}
            while True:
                injected = noise_for_step(mode, sigma_m, rng, fixed_bias)
                policy_observation = observation.copy()
                policy_observation[OBJECT_POSITION_SLICE] += injected.astype(
                    policy_observation.dtype
                )
                # Only the policy sees policy_observation. The environment step
                # receives the action and keeps all true state/reward metrics.
                action, _ = model.predict(policy_observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                injected_vectors.append(injected.copy())
                ever["reach"] |= bool(info.get("reach_success", False))
                ever["grasp"] |= bool(
                    info.get("grasp_success", False)
                    or info.get("stable_grasp_success", False)
                )
                ever["lift"] |= bool(info.get("lift_success", False))
                ever["place"] |= bool(info.get("place_success", False))
                final_info = info
                if terminated or truncated:
                    break
            success = bool(
                final_info.get("is_success", False) or ever["place"]
            )
            episodes.append({
                "seed": int(seed),
                "return": total_reward,
                "steps": steps,
                "success": success,
                "reach_success": ever["reach"],
                "grasp_success": ever["grasp"],
                "lift_success": ever["lift"],
                "place_success": ever["place"],
                "failure_stage": failure_stage(ever, success),
                "fixed_bias_m": (
                    fixed_bias.tolist() if mode == "fixed" else None
                ),
            })
        finally:
            env.close()

    vectors = np.asarray(injected_vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    failure_counts = Counter(episode["failure_stage"] for episode in episodes)
    result = {
        "mode": mode,
        "level_mm_per_axis_sigma": float(level_mm),
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "reach_success_rate": float(np.mean([e["reach_success"] for e in episodes])),
        "grasp_success_rate": float(np.mean([e["grasp_success"] for e in episodes])),
        "lift_success_rate": float(np.mean([e["lift_success"] for e in episodes])),
        "place_success_rate": float(np.mean([e["place_success"] for e in episodes])),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "failure_stage_counts": dict(failure_counts),
        "injected_error": {
            "policy_steps": int(vectors.shape[0]),
            "axis_mean_m": np.mean(vectors, axis=0).tolist(),
            "axis_std_m": np.std(vectors, axis=0).tolist(),
            "norm_mean_m": float(np.mean(norms)),
            "norm_median_m": float(np.median(norms)),
            "norm_p95_m": float(np.percentile(norms, 95)),
            "norm_max_m": float(np.max(norms)),
        },
        "episode_records": episodes,
    }
    return result


def write_csv(results, path):
    fields = [
        "mode", "level_mm_per_axis_sigma", "episodes", "success_rate",
        "reach_success_rate", "grasp_success_rate", "lift_success_rate",
        "place_success_rate", "mean_return", "failure_stage_counts",
        "injected_axis_mean_m", "injected_axis_std_m", "injected_norm_mean_m",
        "injected_norm_median_m", "injected_norm_p95_m", "injected_norm_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            injected = result["injected_error"]
            writer.writerow({
                **{field: result.get(field) for field in fields},
                "failure_stage_counts": json.dumps(result["failure_stage_counts"]),
                "injected_axis_mean_m": json.dumps(injected["axis_mean_m"]),
                "injected_axis_std_m": json.dumps(injected["axis_std_m"]),
                "injected_norm_mean_m": injected["norm_mean_m"],
                "injected_norm_median_m": injected["norm_median_m"],
                "injected_norm_p95_m": injected["norm_p95_m"],
                "injected_norm_max_m": injected["norm_max_m"],
            })


def plot_success(results, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mode in sorted({result["mode"] for result in results}):
        rows = sorted(
            (result for result in results if result["mode"] == mode),
            key=lambda row: row["level_mm_per_axis_sigma"],
        )
        ax.plot(
            [row["level_mm_per_axis_sigma"] for row in rows],
            [row["success_rate"] for row in rows],
            marker="o", linewidth=2, label=mode,
        )
    ax.set(xlabel="Object position error sigma per axis (mm)",
           ylabel="Full-task success rate", ylim=(-0.03, 1.03))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stages(results, path):
    modes = sorted({result["mode"] for result in results})
    fig, axes = plt.subplots(1, len(modes), figsize=(7.2 * len(modes), 4.8), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        rows = sorted(
            (result for result in results if result["mode"] == mode),
            key=lambda row: row["level_mm_per_axis_sigma"],
        )
        levels = [row["level_mm_per_axis_sigma"] for row in rows]
        for stage in STAGE_KEYS:
            ax.plot(
                levels, [row[f"{stage}_success_rate"] for row in rows],
                marker="o", label=stage,
            )
        ax.set(title=mode, xlabel="Error sigma per axis (mm)",
               ylabel="Stage success rate", ylim=(-0.03, 1.03))
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = PPO.load(checkpoint)
    if model.observation_space.shape != (39,):
        raise RuntimeError(
            f"checkpoint observation shape is {model.observation_space.shape}, expected (39,)"
        )
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    results = []
    for mode in args.error_mode:
        for level_mm in args.levels_mm:
            print(f"evaluating mode={mode} level={level_mm:g}mm episodes={len(seeds)}", flush=True)
            result = evaluate_condition(model, mode, level_mm, seeds)
            results.append(result)
            print(
                f"  success={result['success_rate']:.3f} "
                f"reach={result['reach_success_rate']:.3f} "
                f"grasp={result['grasp_success_rate']:.3f} "
                f"lift={result['lift_success_rate']:.3f} "
                f"place={result['place_success_rate']:.3f} "
                f"return={result['mean_return']:.2f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "environment": ENV_ID,
        "policy_object_position_coordinate_frame": "MuJoCo world",
        "policy_object_position_observation_slice": [15, 18],
        "error_definition": (
            "level is Gaussian sigma per XYZ axis; fixed samples once per episode, "
            "jitter samples once per policy step; modes are never mixed"
        ),
        "seeds": seeds,
        "results": results,
    }
    with (args.output_dir / "position_error_robustness.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    write_csv(results, args.output_dir / "position_error_robustness.csv")
    plot_success(results, args.output_dir / "success_rate_vs_error.png")
    plot_stages(results, args.output_dir / "stage_success_rates_vs_error.png")
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
