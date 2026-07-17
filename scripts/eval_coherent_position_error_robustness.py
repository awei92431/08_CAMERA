"""Evaluate PPO with internally coherent object-position observation errors."""

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
from fourc2.envs.allstage import STAGE_PLACE  # noqa: E402


ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
CHECKPOINT = ROOT / "checkpoints" / "best_full_flow_v22.zip"
DEFAULT_LEVELS_MM = (0, 2, 5, 10, 15, 20, 30)
OUTPUT_DIR = ROOT / "outputs" / "coherent_position_error_robustness"
STAGES = ("reach", "grasp", "lift", "place")
FIELDS = {
    "pinch_position": slice(12, 15),
    "object_position": slice(15, 18),
    "pregrasp_position": slice(18, 21),
    "grasp_position": slice(21, 24),
    "pinch_to_pregrasp": slice(24, 27),
    "pinch_to_grasp": slice(27, 30),
    "object_to_goal": slice(30, 33),
    "object_lift": slice(34, 35),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--error-mode", nargs="*", choices=("fixed", "jitter"), default=[]
    )
    parser.add_argument(
        "--levels-mm", nargs="+", type=float, default=DEFAULT_LEVELS_MM
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def reconstruct_observation(observation, raw_env, position_error,
                            estimated_initial_object_z):
    """Rebuild every object-geometry field from one estimated world position."""
    corrupted = observation.copy()
    estimated_object = (
        observation[FIELDS["object_position"]].astype(np.float64)
        + np.asarray(position_error, dtype=np.float64)
    )
    pinch = observation[FIELDS["pinch_position"]].astype(np.float64)
    pregrasp = estimated_object + np.array(
        [0.0, 0.0, float(raw_env.pregrasp_height)]
    )
    grasp = estimated_object + np.array(
        [0.0, 0.0, float(raw_env.grasp_height_offset)]
    )
    if raw_env.stage == STAGE_PLACE:
        goal = raw_env.data.site_xpos[raw_env.goal_site_id].copy()
        object_to_goal = goal - estimated_object
    else:
        object_to_goal = np.zeros(3, dtype=np.float64)
    object_lift = max(
        0.0, float(estimated_object[2] - estimated_initial_object_z)
    )
    replacements = {
        "object_position": estimated_object,
        "pregrasp_position": pregrasp,
        "grasp_position": grasp,
        "pinch_to_pregrasp": pregrasp - pinch,
        "pinch_to_grasp": grasp - pinch,
        "object_to_goal": object_to_goal,
        "object_lift": np.array([object_lift]),
    }
    for name, value in replacements.items():
        corrupted[FIELDS[name]] = np.asarray(value, dtype=corrupted.dtype)
    return corrupted


def assert_internal_consistency(observation, raw_env,
                                estimated_initial_object_z):
    obj = observation[FIELDS["object_position"]]
    pinch = observation[FIELDS["pinch_position"]]
    pregrasp = observation[FIELDS["pregrasp_position"]]
    grasp = observation[FIELDS["grasp_position"]]
    checks = {
        "pregrasp_from_object": pregrasp - (
            obj + np.array([0.0, 0.0, raw_env.pregrasp_height])
        ),
        "grasp_from_object": grasp - (
            obj + np.array([0.0, 0.0, raw_env.grasp_height_offset])
        ),
        "pinch_to_pregrasp": observation[FIELDS["pinch_to_pregrasp"]]
        - (pregrasp - pinch),
        "pinch_to_grasp": observation[FIELDS["pinch_to_grasp"]]
        - (grasp - pinch),
        "object_lift": observation[FIELDS["object_lift"]]
        - max(0.0, float(obj[2] - estimated_initial_object_z)),
    }
    if raw_env.stage == STAGE_PLACE:
        expected_goal = (
            raw_env.data.site_xpos[raw_env.goal_site_id] - obj
        )
    else:
        expected_goal = np.zeros(3)
    checks["object_to_goal"] = (
        observation[FIELDS["object_to_goal"]] - expected_goal
    )
    maximum = max(float(np.max(np.abs(value))) for value in checks.values())
    if maximum > 2e-7:
        raise AssertionError(f"coherent observation check failed: {checks}")
    return maximum


def single_state_audit(model):
    env = gym.make(ENV_ID)
    try:
        clean, _ = env.reset(seed=0)
        raw = env.unwrapped
        error = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        estimated_initial_z = float(raw.object_initial_z + error[2])
        corrupted = reconstruct_observation(
            clean, raw, error, estimated_initial_z
        )
        consistency_error = assert_internal_consistency(
            corrupted, raw, estimated_initial_z
        )
        clean_action, _ = model.predict(clean, deterministic=True)
        # This exact corrupted array is passed directly to model.predict: no
        # VecNormalize exists in this manual Gymnasium evaluation path.
        corrupted_action, _ = model.predict(corrupted, deterministic=True)
        audit = {
            "position_error_m": error.tolist(),
            "pregrasp_height_from_env_m": float(raw.pregrasp_height),
            "grasp_height_offset_from_env_m": float(raw.grasp_height_offset),
            "estimated_initial_object_z_m": estimated_initial_z,
            "stage": int(raw.stage),
            "maximum_consistency_residual": consistency_error,
            "model_predict_receives_corrupted_directly": True,
            "clean_action": clean_action.tolist(),
            "corrupted_action": corrupted_action.tolist(),
            "action_delta": (corrupted_action - clean_action).tolist(),
            "fields": {},
        }
        for name, field_slice in FIELDS.items():
            audit["fields"][name] = {
                "indices": [field_slice.start, field_slice.stop],
                "clean": clean[field_slice].tolist(),
                "corrupted": corrupted[field_slice].tolist(),
                "delta": (
                    corrupted[field_slice] - clean[field_slice]
                ).tolist(),
            }
        print(json.dumps(audit, indent=2), flush=True)
        return audit
    finally:
        env.close()


def failure_stage(ever, success):
    if success:
        return "none"
    for stage in STAGES:
        if not ever[stage]:
            return stage
    return "place"


def evaluate_condition(model, mode, level_mm, seeds):
    sigma_m = float(level_mm) / 1000.0
    episode_rows = []
    injected_errors = []
    for seed in seeds:
        env = gym.make(ENV_ID)
        try:
            observation, _ = env.reset(seed=seed)
            raw = env.unwrapped
            rng = np.random.default_rng(920_000 + int(seed))
            initial_standard_error = rng.standard_normal(3)
            fixed_error = sigma_m * initial_standard_error
            initial_error = fixed_error.copy()
            estimated_initial_z = float(
                raw.object_initial_z + initial_error[2]
            )
            total_reward = 0.0
            ever = {stage: False for stage in STAGES}
            final_info = {}
            steps = 0
            while True:
                if mode == "fixed":
                    position_error = fixed_error
                elif mode == "jitter":
                    position_error = sigma_m * rng.standard_normal(3)
                else:
                    raise ValueError(mode)
                corrupted = reconstruct_observation(
                    observation, raw, position_error, estimated_initial_z
                )
                assert_internal_consistency(
                    corrupted, raw, estimated_initial_z
                )
                action, _ = model.predict(corrupted, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                injected_errors.append(position_error.copy())
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
            success = bool(final_info.get("is_success", False) or ever["place"])
            episode_rows.append({
                "seed": int(seed),
                "return": total_reward,
                "steps": steps,
                "success": success,
                **{f"{stage}_success": ever[stage] for stage in STAGES},
                "failure_stage": failure_stage(ever, success),
                "initial_position_error_m": initial_error.tolist(),
                "estimated_initial_object_z_m": estimated_initial_z,
            })
        finally:
            env.close()

    errors = np.asarray(injected_errors, dtype=np.float64)
    norms = np.linalg.norm(errors, axis=1)
    failures = Counter(row["failure_stage"] for row in episode_rows)
    return {
        "mode": mode,
        "level_mm_per_axis_sigma": float(level_mm),
        "episodes": len(episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        **{
            f"{stage}_success_rate": float(np.mean([
                row[f"{stage}_success"] for row in episode_rows
            ])) for stage in STAGES
        },
        "mean_return": float(np.mean([row["return"] for row in episode_rows])),
        "failure_stage_counts": dict(failures),
        "actual_3d_error": {
            "mean_m": float(np.mean(norms)),
            "p95_m": float(np.percentile(norms, 95)),
            "max_m": float(np.max(norms)),
            "axis_std_m": np.std(errors, axis=0).tolist(),
        },
        "episode_records": episode_rows,
    }


def save_csv(results, path):
    fields = [
        "mode", "level_mm_per_axis_sigma", "success_rate",
        "reach_success_rate", "grasp_success_rate", "lift_success_rate",
        "place_success_rate", "mean_return", "failure_stage_counts",
        "actual_3d_error_mean_m", "actual_3d_error_p95_m",
        "actual_3d_error_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            error = row["actual_3d_error"]
            writer.writerow({
                **{name: row.get(name) for name in fields},
                "failure_stage_counts": json.dumps(row["failure_stage_counts"]),
                "actual_3d_error_mean_m": error["mean_m"],
                "actual_3d_error_p95_m": error["p95_m"],
                "actual_3d_error_max_m": error["max_m"],
            })


def save_plots(results, output_dir):
    for metric, filename, ylabel in (
        ("success_rate", "success_rate_vs_error.png", "Full success rate"),
        ("mean_return", "mean_return_vs_error.png", "Mean return"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for mode in sorted({row["mode"] for row in results}):
            rows = sorted(
                [row for row in results if row["mode"] == mode],
                key=lambda row: row["level_mm_per_axis_sigma"],
            )
            ax.plot(
                [row["level_mm_per_axis_sigma"] for row in rows],
                [row[metric] for row in rows], marker="o", label=mode
            )
        ax.set(xlabel="Position error sigma per axis (mm)", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    model = PPO.load(CHECKPOINT)
    if model.observation_space.shape != (39,):
        raise RuntimeError(model.observation_space.shape)
    audit = single_state_audit(model)
    if args.audit_only:
        return
    if not args.error_mode:
        raise SystemExit("use --error-mode fixed and/or jitter, or --audit-only")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    results = []
    for mode in args.error_mode:
        for level in args.levels_mm:
            print(f"evaluating coherent mode={mode} level={level:g}mm", flush=True)
            result = evaluate_condition(model, mode, level, seeds)
            results.append(result)
            print(
                f"  success={result['success_rate']:.3f} "
                f"reach={result['reach_success_rate']:.3f} "
                f"grasp={result['grasp_success_rate']:.3f} "
                f"lift={result['lift_success_rate']:.3f} "
                f"place={result['place_success_rate']:.3f} "
                f"return={result['mean_return']:.2f}", flush=True
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(CHECKPOINT),
        "environment": ENV_ID,
        "coordinate_frame": "MuJoCo world",
        "error_definition": "Gaussian sigma per XYZ axis",
        "reconstructed_fields": {
            name: [value.start, value.stop] for name, value in FIELDS.items()
            if name != "pinch_position"
        },
        "remaining_privileged_observation": {
            "35:37": "true left/right object contact flags",
            "37": "true gripper vertical alignment",
            "38": "true grasp FSM phase",
        },
        "seeds": seeds,
        "single_state_audit": audit,
        "results": results,
    }
    with (args.output_dir / "coherent_position_error_robustness.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    save_csv(results, args.output_dir / "coherent_position_error_robustness.csv")
    save_plots(results, args.output_dir)
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
