"""Add failure classification and event-level Phase 2 metrics to summary.json."""

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase2_task_supervisor"


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value == "True"


def classify(row):
    if truth(row["full_success"]):
        return "success"
    if not truth(row["supervisor_grasp_confirmed"]):
        return "grasp_confirmation_not_reached"
    if not truth(row["lift_success"]):
        return "object_hold_or_lift"
    if not truth(row["entered_place"]):
        return "lift_to_place_transition"
    if not truth(row["place_success"]):
        return "place_or_release"
    return "truth_evaluator_mismatch"


def diagnostics(rows):
    counts = Counter(classify(row) for row in rows)
    terminal = Counter()
    for row in rows:
        if truth(row["full_success"]):
            continue
        value = row.get("supervisor_last_diagnostics", "")
        diag = ast.literal_eval(value) if value and value != "None" else {}
        if not diag.get("stable_geometry", False):
            terminal["stable_geometry_false"] += 1
        if not diag.get("sensor_hold", False):
            terminal["sensor_hold_false"] += 1
        if not diag.get("effort_contact", False):
            terminal["effort_contact_false"] += 1
    return dict(counts), dict(terminal)


def shadow_metrics(rows):
    per_event = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event"]].append(row)
    for event, event_rows in grouped.items():
        delays = [float(r["delay_steps"]) for r in event_rows if r["delay_steps"]]
        per_event[event] = {
            "matched": len(delays),
            "missed": sum(truth(r["missed"]) for r in event_rows),
            "false_positive": sum(truth(r["false_positive"]) for r in event_rows),
            "delay_steps_mean": mean(delays) if delays else None,
            "delay_steps_median": median(delays) if delays else None,
            "delay_steps_min": min(delays) if delays else None,
            "delay_steps_max": max(delays) if delays else None,
        }
    return per_event


def main():
    deployable = read_csv("deployable_episodes.csv")
    privileged = read_csv("privileged_episodes.csv")
    latch_off = read_csv("latch_disabled_episodes.csv")
    shadow = read_csv("shadow_event_differences.csv")
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["failure_analysis"] = {}
    for name, rows in (("privileged_fsm", privileged),
                       ("deployable_fsm", deployable),
                       ("latch_disabled", latch_off)):
        classes, terminal = diagnostics(rows)
        summary["failure_analysis"][name] = {
            "classification": classes,
            "terminal_diagnostic_counts_nonexclusive": terminal,
        }
    summary["shadow"]["by_event"] = shadow_metrics(shadow)
    summary["interpretation"] = {
        "supervisor_input_truth_reads": 0,
        "deployable_source_consistency_failures": 0,
        "dominant_deployable_failure": "grasp_confirmation_not_reached",
        "latch_off_scope": "10-seed diagnostic, not the 100-seed A/B",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"failure_analysis": summary["failure_analysis"],
                      "shadow_by_event": summary["shadow"]["by_event"]}, indent=2))


if __name__ == "__main__":
    main()
