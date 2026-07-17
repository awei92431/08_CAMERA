#!/usr/bin/env python3
"""Dedicated ArUco-goal + RGB-D-object full closed-loop evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_full_visual_closed_loop import (  # noqa: E402
    ARUCO_CONFIG, CHECKPOINT, ENV_ID, evaluate_mode, summarize, write_csv,
)
from fourc2.aruco_goal_localizer import load_aruco_goal_config  # noqa: E402

OUTPUT = ROOT / "outputs" / "aruco_goal_closed_loop"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--show-camera", action="store_true")
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    if args.video is not None and args.episodes != 1:
        parser.error("video recording requires --episodes 1")
    model = PPO.load(CHECKPOINT, device="cpu")
    seeds = list(range(args.seed_offset, args.seed_offset + args.episodes))
    rows, step_rows = evaluate_mode(
        "rgbd", seeds, model, save_steps=True,
        video_path=args.video, live=args.live, show_camera=args.show_camera,
        goal_source="aruco",
        aruco_config=load_aruco_goal_config(ARUCO_CONFIG),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "aruco_rgbd_episodes.csv", rows)
    if step_rows:
        write_csv(OUTPUT / "aruco_rgbd_steps.csv", step_rows)
    summary = {
        "configuration": {
            "checkpoint": str(CHECKPOINT), "environment": ENV_ID,
            "object_source": "rgbd", "goal_source": "aruco",
            "seeds": seeds, "deterministic": True,
            "goal_failure_behavior": "fail_closed_before_PPO",
        },
        "results": summarize(rows),
        "goal_localization_error_m_evaluator_only": summarize_goal(rows),
        "goal_source_consistency_failures": int(sum(
            row["goal_source_consistency_failures"] for row in rows)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def summarize_goal(rows):
    import numpy as np
    values = np.asarray([
        row["goal_visual_error_m_evaluator_only"] for row in rows
        if row["goal_visual_error_m_evaluator_only"] is not None], dtype=float)
    if not len(values):
        return None
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


if __name__ == "__main__":
    main()
