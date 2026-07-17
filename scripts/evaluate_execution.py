#!/usr/bin/env python3
"""Evaluate the frozen full-flow checkpoint without training."""
import csv, json, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("FOURC2_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
import fourc2  # noqa: F401

MODE = os.environ.get("EXECUTION_MODE", "ik")
OUT = Path(os.environ.get("EVAL_OUTPUT", ROOT / "results" / MODE))
MODEL = Path(os.environ.get("CHECKPOINT", ROOT / "checkpoints/best_full_flow_v22.zip"))
ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
SEEDS = list(range(10))
STAGES = {0:"reach", 1:"grasp", 2:"lift", 3:"full", 4:"reach_grasp", 5:"place"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = PPO.load(MODEL, device="cpu")
    rows = []
    for ep, seed in enumerate(SEEDS):
        video = ep < 2
        env = gym.make(ENV_ID, render_mode="rgb_array" if video else None)
        if video:
            env.unwrapped.mujoco_renderer.width = 960
            env.unwrapped.mujoco_renderer.height = 720
        obs, info = env.reset(seed=seed)
        frames = [env.render()] if video else []
        errors, stages = [], []
        ever = {k:False for k in ("reach","grasp","lift")}
        entered_place = False
        place_success = False
        reward_sum = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1; reward_sum += float(reward)
            errors.append(float(info.get("tcp_target_error", 0.0)))
            stage = STAGES.get(int(info.get("stage", -1)), "unknown")
            stages.append(stage)
            entered_place |= stage == "place"
            place_success |= bool(info.get("place_success", False))
            for key in ever:
                ever[key] |= bool(info.get(f"{key}_success", False))
            if video: frames.append(env.render())
        if video:
            imageio.mimsave(OUT / f"{MODE}_seed_{seed}.mp4", frames, fps=20, macro_block_size=1)
        raw = env.unwrapped
        row = dict(episode=ep, seed=seed, success=bool(info.get("is_success", False)),
                   steps=steps, reward=reward_sum, mean_tcp_error=float(np.mean(errors)),
                   max_tcp_error=float(np.max(errors)), final_stage=stages[-1],
                   dominant_stage=Counter(stages).most_common(1)[0][0],
                   ik_solve_calls=int(getattr(raw, "ik_solve_calls", 0)),
                   weld_present=bool(getattr(raw, "tcp_weld_enabled", True)), **{f"ever_{k}":v for k,v in ever.items()})
        row["entered_place"] = entered_place
        row["place_success"] = place_success
        rows.append(row); env.close()
        print(f"{MODE} seed={seed} success={int(row['success'])} steps={steps} mean_tcp_error={1000*row['mean_tcp_error']:.2f}mm stage={row['final_stage']}")
    failures = [r["final_stage"] for r in rows if not r["success"]]
    summary = dict(mode=MODE, checkpoint=str(MODEL), seeds=SEEDS, episodes=len(rows),
                   success_rate=float(np.mean([r["success"] for r in rows])),
                   mean_tcp_tracking_error=float(np.mean([r["mean_tcp_error"] for r in rows])),
                   max_tcp_tracking_error=float(np.max([r["max_tcp_error"] for r in rows])),
                   main_failure_stage=Counter(failures).most_common(1)[0][0] if failures else None,
                   ik_solve_calls=sum(r["ik_solve_calls"] for r in rows),
                   weld_present=any(r["weld_present"] for r in rows))
    (OUT / "evaluation.json").write_text(json.dumps({"summary":summary,"episodes":rows}, indent=2), encoding="utf-8")
    with (OUT / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
