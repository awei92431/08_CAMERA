#!/usr/bin/env python3
"""Measure 4 cm ArUco goal, RGB-D object and final placement XY errors."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_full_visual_closed_loop import (  # noqa: E402
    ARUCO_CONFIG, CHECKPOINT, evaluate_mode,
)
from fourc2.aruco_goal_localizer import load_aruco_goal_config  # noqa: E402

OUTPUT = ROOT / "outputs" / "aruco4cm_xy_error_100seeds"


def stats_mm(values):
    values = np.asarray([
        value for value in values
        if value is not None and np.isfinite(value)], dtype=float)
    if not len(values):
        return None
    millimeters = values * 1000.0
    return {
        "count": int(len(values)),
        "mean": float(millimeters.mean()),
        "median": float(np.median(millimeters)),
        "p95": float(np.percentile(millimeters, 95)),
        "max": float(millimeters.max()),
        "std": float(millimeters.std()),
    }


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows):
    valid = [row for row in rows if row["control_executed"]]
    series = [
        ("goal_xy_error_m", "ArUco goal XY"),
        ("object_xy_error_m", "RGB-D object XY"),
        ("final_true_goal_xy_error_m", "Final placement XY"),
    ]
    data = [[row[key] * 1000.0 for row in valid
             if row.get(key) is not None] for key, _ in series]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot(data, tick_labels=[label for _, label in series], showfliers=True)
    axis.set_ylabel("XY error (mm)")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "three_xy_errors_boxplot.png", dpi=180)
    plt.close(fig)

    if valid:
        goal = np.asarray([row["goal_xy_error_m"] * 1000 for row in valid])
        obj = np.asarray([row["object_xy_error_m"] * 1000 for row in valid])
        final = np.asarray([row["final_true_goal_xy_error_m"] * 1000
                            for row in valid])
        fig, axis = plt.subplots(figsize=(7, 5))
        scatter = axis.scatter(goal, final, c=obj, cmap="viridis", alpha=.8)
        axis.set_xlabel("ArUco goal XY error (mm)")
        axis.set_ylabel("Final true-goal placement XY error (mm)")
        axis.grid(alpha=.25)
        fig.colorbar(scatter, ax=axis, label="RGB-D object XY error (mm)")
        fig.tight_layout()
        fig.savefig(OUTPUT / "vision_vs_placement_xy_error.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    model = PPO.load(CHECKPOINT, device="cpu")
    config = load_aruco_goal_config(ARUCO_CONFIG)
    if not np.isclose(config.marker_size_m, 0.04):
        raise RuntimeError(
            f"this evaluation requires a physical/configured 4 cm marker; "
            f"got {config.marker_size_m}")

    rows = []
    for index, seed in enumerate(range(
            args.seed_offset, args.seed_offset + args.episodes), start=1):
        try:
            episode_rows, _ = evaluate_mode(
                "rgbd", [seed], model, goal_source="aruco",
                aruco_config=config)
            source = episode_rows[0]
            final_delta_xy = (
                np.asarray(source["final_object_xyz_evaluator_only"], dtype=float)[:2]
                - np.asarray(source["initial_goal_xyz"], dtype=float)[:2]
            )
            footprint_margin_m = 0.5 * (0.040 - 0.030)
            row = {
                "seed": seed,
                "goal_detected": True,
                "goal_failure_reason": None,
                "object_detected": bool(source["rgbd_valid"]),
                "object_failure_reason": source["rgbd_failure_reason"],
                "control_executed": True,
                "full_success": bool(source["full_success"]),
                "place_success": bool(source["place_success"]),
                "goal_xy_error_m": source[
                    "goal_visual_xy_error_m_evaluator_only"],
                "object_xy_error_m": source["visual_xy_error_m"],
                "final_true_goal_xy_error_m": source[
                    "final_true_goal_xy_error_m_evaluator_only"],
                "final_estimated_goal_xy_error_m": source[
                    "final_goal_xy_error_m"],
                "final_true_goal_abs_x_error_m": float(abs(final_delta_xy[0])),
                "final_true_goal_abs_y_error_m": float(abs(final_delta_xy[1])),
                "cube_fully_inside_40mm_goal_xy": bool(
                    np.max(np.abs(final_delta_xy)) <= footprint_margin_m),
                "episode_steps": source["episode_steps"],
                "initial_object_xyz": source["initial_object_xyz"],
                "true_goal_xyz_evaluator_only": source["initial_goal_xyz"],
                "final_object_xyz_evaluator_only": source[
                    "final_object_xyz_evaluator_only"],
            }
        except RuntimeError as exc:
            message = str(exc)
            if "ArUco goal capture failed closed before PPO" not in message:
                raise
            row = {
                "seed": seed,
                "goal_detected": False,
                "goal_failure_reason": message,
                "object_detected": False,
                "object_failure_reason": "not_attempted_after_goal_fail_closed",
                "control_executed": False,
                "full_success": False,
                "place_success": False,
                "goal_xy_error_m": None,
                "object_xy_error_m": None,
                "final_true_goal_xy_error_m": None,
                "final_estimated_goal_xy_error_m": None,
                "final_true_goal_abs_x_error_m": None,
                "final_true_goal_abs_y_error_m": None,
                "cube_fully_inside_40mm_goal_xy": False,
                "episode_steps": 0,
            }
        rows.append(row)
        print(
            f"XY sweep {index:03d}/{args.episodes}: seed={seed:03d} "
            f"goal={row['goal_detected']} object={row['object_detected']} "
            f"full={row['full_success']}", flush=True)

    executed = [row for row in rows if row["control_executed"]]
    successful = [row for row in rows if row["full_success"]]
    summary = {
        "configuration": {
            "episodes": args.episodes,
            "seeds": [args.seed_offset,
                      args.seed_offset + args.episodes - 1],
            "aruco_marker_size_mm": 40.0,
            "cube_size_mm": 30.0,
            "errors": "XY only; Z excluded",
            "checkpoint": str(CHECKPOINT),
        },
        "rates": {
            "goal_detection": {
                "count": int(sum(row["goal_detected"] for row in rows)),
                "rate": float(np.mean([row["goal_detected"] for row in rows])),
            },
            "object_detection_over_all": {
                "count": int(sum(row["object_detected"] for row in rows)),
                "rate": float(np.mean([row["object_detected"] for row in rows])),
            },
            "full_success_over_all": {
                "count": len(successful),
                "rate": float(len(successful) / len(rows)),
            },
            "full_success_given_control": {
                "count": len(successful),
                "denominator": len(executed),
                "rate": None if not executed else float(
                    len(successful) / len(executed)),
            },
            "cube_fully_inside_40mm_goal_given_control": {
                "count": int(sum(
                    row["cube_fully_inside_40mm_goal_xy"] for row in executed)),
                "denominator": len(executed),
                "rate": None if not executed else float(np.mean([
                    row["cube_fully_inside_40mm_goal_xy"] for row in executed])),
                "criterion": "abs(dx) <= 5 mm and abs(dy) <= 5 mm",
            },
        },
        "xy_error_mm": {
            "aruco_goal_valid_detections": stats_mm(
                [row["goal_xy_error_m"] for row in rows]),
            "rgbd_object_valid_detections": stats_mm(
                [row["object_xy_error_m"] for row in rows]),
            "final_object_to_true_goal_all_executed": stats_mm(
                [row["final_true_goal_xy_error_m"] for row in executed]),
            "final_object_to_true_goal_successful_only": stats_mm(
                [row["final_true_goal_xy_error_m"] for row in successful]),
            "final_object_to_estimated_goal_all_executed": stats_mm(
                [row["final_estimated_goal_xy_error_m"] for row in executed]),
            "final_true_goal_abs_x_all_executed": stats_mm(
                [row["final_true_goal_abs_x_error_m"] for row in executed]),
            "final_true_goal_abs_y_all_executed": stats_mm(
                [row["final_true_goal_abs_y_error_m"] for row in executed]),
        },
        "failure_counts": {
            "goal_fail_closed": int(sum(
                not row["goal_detected"] for row in rows)),
            "object_detection": int(sum(
                row["goal_detected"] and not row["object_detected"]
                for row in rows)),
            "task_after_control_started": int(sum(
                row["control_executed"] and not row["full_success"]
                for row in rows)),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "episodes.csv", rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
