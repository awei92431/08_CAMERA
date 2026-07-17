#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lenovo/mujoco_learning/01_rl_baselines3_zoo/.conda_zoo/bin/python}"
SEED="${SEED:-0}"
N_ENVS="${N_ENVS:-2}"
N_STEPS="${N_STEPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GAMMA="${GAMMA:-0.98}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-2}"
PREFIX="${PREFIX:-cube3cm_closed_loop_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_DIR="${PIPELINE_DIR:-runs/${PREFIX}_pipeline}"
MANIFEST="${MANIFEST:-${PIPELINE_DIR}/manifest.txt}"

# These are additional timesteps per stage. The early-stage defaults follow the
# historical cube3cm runs more closely: short, high-lr skill stages, then lower
# lr for merged place/full fine-tuning.
REACH_STEPS="${REACH_STEPS:-200000}"
GRASP_STEPS="${GRASP_STEPS:-100000}"
REACH_GRASP_STEPS="${REACH_GRASP_STEPS:-100000}"
LIFT_STEPS="${LIFT_STEPS:-150000}"
PLACE_STEPS="${PLACE_STEPS:-150000}"
FULL_TRANSITION_STEPS="${FULL_TRANSITION_STEPS:-200000}"
FULL_STEPS="${FULL_STEPS:-150000}"

REACH_LR="${REACH_LR:-0.0003}"
GRASP_LR="${GRASP_LR:-0.0003}"
REACH_GRASP_LR="${REACH_GRASP_LR:-0.0003}"
LIFT_LR="${LIFT_LR:-0.0003}"
PLACE_LR="${PLACE_LR:-0.00008}"
FULL_TRANSITION_LR="${FULL_TRANSITION_LR:-0.00003}"
FULL_LR="${FULL_LR:-0.00002}"

CHECK_STAGE_SUCCESS="${CHECK_STAGE_SUCCESS:-1}"
MIN_REACH_SUCCESS="${MIN_REACH_SUCCESS:-0.98}"
MIN_GRASP_SUCCESS="${MIN_GRASP_SUCCESS:-0.95}"
MIN_REACH_GRASP_SUCCESS="${MIN_REACH_GRASP_SUCCESS:-0.95}"
MIN_LIFT_SUCCESS="${MIN_LIFT_SUCCESS:-0.95}"
MIN_PLACE_SUCCESS="${MIN_PLACE_SUCCESS:-0.90}"
MIN_FULL_TRANSITION_SUCCESS="${MIN_FULL_TRANSITION_SUCCESS:-0.50}"
MIN_FULL_SUCCESS="${MIN_FULL_SUCCESS:-0.90}"
HANDOFF_MODEL_KIND="${HANDOFF_MODEL_KIND:-final}"
REACH_MODEL_OVERRIDE="${REACH_MODEL_OVERRIDE:-}"
GRASP_MODEL_OVERRIDE="${GRASP_MODEL_OVERRIDE:-}"
REACH_GRASP_MODEL_OVERRIDE="${REACH_GRASP_MODEL_OVERRIDE:-}"
LIFT_MODEL_OVERRIDE="${LIFT_MODEL_OVERRIDE:-}"
PLACE_MODEL_OVERRIDE="${PLACE_MODEL_OVERRIDE:-}"
FULL_TRANSITION_MODEL_OVERRIDE="${FULL_TRANSITION_MODEL_OVERRIDE:-}"

mkdir -p "$PIPELINE_DIR"

{
  echo "# Cube3cm current-code closed-loop training"
  echo "created_at: $(date -Is)"
  echo "python: ${PYTHON_BIN}"
  echo "seed: ${SEED}"
  echo "prefix: ${PREFIX}"
  echo "n_envs: ${N_ENVS}"
  echo "n_steps: ${N_STEPS}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "gamma: ${GAMMA}"
  echo "eval_freq: ${EVAL_FREQ}"
  echo "eval_episodes: ${EVAL_EPISODES}"
  echo "steps: reach=${REACH_STEPS} grasp=${GRASP_STEPS} reach_grasp=${REACH_GRASP_STEPS} lift=${LIFT_STEPS} place=${PLACE_STEPS} full_transition=${FULL_TRANSITION_STEPS} full=${FULL_STEPS}"
  echo "learning_rates: reach=${REACH_LR} grasp=${GRASP_LR} reach_grasp=${REACH_GRASP_LR} lift=${LIFT_LR} place=${PLACE_LR} full_transition=${FULL_TRANSITION_LR} full=${FULL_LR}"
  echo "success_gates_enabled: ${CHECK_STAGE_SUCCESS}"
  echo "success_gates: reach=${MIN_REACH_SUCCESS} grasp=${MIN_GRASP_SUCCESS} reach_grasp=${MIN_REACH_GRASP_SUCCESS} lift=${MIN_LIFT_SUCCESS} place=${MIN_PLACE_SUCCESS} full_transition=${MIN_FULL_TRANSITION_SUCCESS} full=${MIN_FULL_SUCCESS}"
  echo "handoff_model_kind: ${HANDOFF_MODEL_KIND}"
  echo "reach_model_override: ${REACH_MODEL_OVERRIDE}"
  echo "grasp_model_override: ${GRASP_MODEL_OVERRIDE}"
  echo "reach_grasp_model_override: ${REACH_GRASP_MODEL_OVERRIDE}"
  echo "lift_model_override: ${LIFT_MODEL_OVERRIDE}"
  echo "place_model_override: ${PLACE_MODEL_OVERRIDE}"
  echo "full_transition_model_override: ${FULL_TRANSITION_MODEL_OVERRIDE}"
  echo
} > "$MANIFEST"

run_train() {
  local env_id="$1"
  local eval_env_id="$2"
  local run_name="$3"
  local steps="$4"
  local lr="$5"
  local load_model="${6:-}"

  local cmd=(
    "$PYTHON_BIN" -B scripts/trainenv.py
    --env-id "$env_id"
    --eval-env-id "$eval_env_id"
    --total-timesteps "$steps"
    --n-envs "$N_ENVS"
    --n-steps "$N_STEPS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$lr"
    --gamma "$GAMMA"
    --eval-freq "$EVAL_FREQ"
    --eval-episodes "$EVAL_EPISODES"
    --seed "$SEED"
    --run-name "$run_name"
    --terminal-log-interval "$LOG_INTERVAL"
  )

  if [[ -n "$load_model" ]]; then
    cmd+=(--load-model "$load_model")
  fi

  echo
  echo "==> ${run_name}"
  printf ' %q' "${cmd[@]}"
  echo

  {
    echo "## ${run_name}"
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    echo
    echo
  } >> "$MANIFEST"

  "${cmd[@]}"
}

pick_model_by_kind() {
  local run_name="$1"
  local kind="$2"
  local model_dir="runs/${run_name}/models"

  case "$kind" in
    final)
      if [[ -f "${model_dir}/final_model.zip" ]]; then
        echo "${model_dir}/final_model.zip"
        return
      fi
      ;;
    best_success)
      if [[ -f "${model_dir}/best_success_model.zip" ]]; then
        echo "${model_dir}/best_success_model.zip"
        return
      fi
      ;;
    best_reward)
      if [[ -f "${model_dir}/best_model.zip" ]]; then
        echo "${model_dir}/best_model.zip"
        return
      fi
      ;;
    *)
      echo "Unknown model kind: ${kind}" >&2
      exit 1
      ;;
  esac

  if [[ -f "${model_dir}/best_success_model.zip" ]]; then
    echo "${model_dir}/best_success_model.zip"
    return
  fi
  if [[ -f "${model_dir}/best_model.zip" ]]; then
    echo "${model_dir}/best_model.zip"
    return
  fi
  if [[ -f "${model_dir}/final_model.zip" ]]; then
    echo "${model_dir}/final_model.zip"
    return
  fi

  echo "No model found in ${model_dir}" >&2
  exit 1
}

best_success_for_run() {
  local run_name="$1"
  "$PYTHON_BIN" - "$run_name" <<'PY'
import sys
from pathlib import Path

import numpy as np

run_name = sys.argv[1]
eval_path = Path("runs") / run_name / "logs" / "evaluations.npz"
if not eval_path.exists():
    raise SystemExit(f"missing evaluations.npz: {eval_path}")

data = np.load(eval_path, allow_pickle=True)
if "successes" not in data.files:
    raise SystemExit(f"missing successes in {eval_path}")

success = data["successes"].mean(axis=1)
print(f"{float(success.max()):.6f}")
PY
}

check_stage_success() {
  local label="$1"
  local run_name="$2"
  local min_success="$3"

  if [[ "$CHECK_STAGE_SUCCESS" == "0" ]]; then
    return
  fi

  local best_success
  best_success="$(best_success_for_run "$run_name")"
  echo "${label} best_success=${best_success} min_required=${min_success}"
  {
    echo "${label}_best_success: ${best_success}"
    echo "${label}_min_required: ${min_success}"
    echo
  } >> "$MANIFEST"

  "$PYTHON_BIN" - "$label" "$best_success" "$min_success" <<'PY'
import sys

label = sys.argv[1]
best_success = float(sys.argv[2])
min_success = float(sys.argv[3])
if best_success + 1e-12 < min_success:
    raise SystemExit(
        f"{label} stage failed success gate: "
        f"best_success={best_success:.3f} < required={min_success:.3f}. "
        "Stop here; do not continue a broken curriculum."
    )
PY
}

REACH_RUN="${PREFIX}_reach_${REACH_STEPS}"
GRASP_RUN="${PREFIX}_grasp_${GRASP_STEPS}"
REACH_GRASP_RUN="${PREFIX}_reach_grasp_${REACH_GRASP_STEPS}"
LIFT_RUN="${PREFIX}_lift_${LIFT_STEPS}"
PLACE_RUN="${PREFIX}_place_${PLACE_STEPS}"
FULL_TRANSITION_RUN="${PREFIX}_full_transition_${FULL_TRANSITION_STEPS}"
FULL_RUN="${PREFIX}_full_${FULL_STEPS}"

if [[ -n "$REACH_MODEL_OVERRIDE" ]]; then
  REACH_MODEL="$REACH_MODEL_OVERRIDE"
  echo "Using reach override model: ${REACH_MODEL}"
  echo "reach_override_model: ${REACH_MODEL}" >> "$MANIFEST"
else
  run_train My4C2ReachStageCube3cm-v0 My4C2ReachStageCube3cm-v0 "$REACH_RUN" "$REACH_STEPS" "$REACH_LR"
  check_stage_success reach "$REACH_RUN" "$MIN_REACH_SUCCESS"
  REACH_MODEL="$(pick_model_by_kind "$REACH_RUN" "$HANDOFF_MODEL_KIND")"
fi

if [[ -n "$GRASP_MODEL_OVERRIDE" ]]; then
  GRASP_MODEL="$GRASP_MODEL_OVERRIDE"
  echo "Using grasp override model: ${GRASP_MODEL}"
  echo "grasp_override_model: ${GRASP_MODEL}" >> "$MANIFEST"
else
  run_train My4C2GraspStageCube3cm-v0 My4C2GraspStageCube3cm-v0 "$GRASP_RUN" "$GRASP_STEPS" "$GRASP_LR" "$REACH_MODEL"
  check_stage_success grasp "$GRASP_RUN" "$MIN_GRASP_SUCCESS"
  GRASP_MODEL="$(pick_model_by_kind "$GRASP_RUN" "$HANDOFF_MODEL_KIND")"
fi

if [[ -n "$REACH_GRASP_MODEL_OVERRIDE" ]]; then
  REACH_GRASP_MODEL="$REACH_GRASP_MODEL_OVERRIDE"
  echo "Using reach_grasp override model: ${REACH_GRASP_MODEL}"
  echo "reach_grasp_override_model: ${REACH_GRASP_MODEL}" >> "$MANIFEST"
else
  run_train My4C2ReachGraspStageCube3cm-v0 My4C2ReachGraspStageCube3cm-v0 "$REACH_GRASP_RUN" "$REACH_GRASP_STEPS" "$REACH_GRASP_LR" "$GRASP_MODEL"
  check_stage_success reach_grasp "$REACH_GRASP_RUN" "$MIN_REACH_GRASP_SUCCESS"
  REACH_GRASP_MODEL="$(pick_model_by_kind "$REACH_GRASP_RUN" "$HANDOFF_MODEL_KIND")"
fi

if [[ -n "$LIFT_MODEL_OVERRIDE" ]]; then
  LIFT_MODEL="$LIFT_MODEL_OVERRIDE"
  echo "Using lift override model: ${LIFT_MODEL}"
  echo "lift_override_model: ${LIFT_MODEL}" >> "$MANIFEST"
else
  run_train My4C2GraspLiftStageCube3cm-v0 My4C2GraspLiftStageCube3cm-v0 "$LIFT_RUN" "$LIFT_STEPS" "$LIFT_LR" "$REACH_GRASP_MODEL"
  check_stage_success lift "$LIFT_RUN" "$MIN_LIFT_SUCCESS"
  LIFT_MODEL="$(pick_model_by_kind "$LIFT_RUN" "$HANDOFF_MODEL_KIND")"
fi

if [[ -n "$PLACE_MODEL_OVERRIDE" ]]; then
  PLACE_MODEL="$PLACE_MODEL_OVERRIDE"
  echo "Using place override model: ${PLACE_MODEL}"
  echo "place_override_model: ${PLACE_MODEL}" >> "$MANIFEST"
else
  run_train My4C2TransportPlaceStageCube3cm-v0 My4C2TransportPlaceStageCube3cm-v0 "$PLACE_RUN" "$PLACE_STEPS" "$PLACE_LR" "$LIFT_MODEL"
  check_stage_success place "$PLACE_RUN" "$MIN_PLACE_SUCCESS"
  PLACE_MODEL="$(pick_model_by_kind "$PLACE_RUN" "$HANDOFF_MODEL_KIND")"
fi

if [[ -n "$FULL_TRANSITION_MODEL_OVERRIDE" ]]; then
  FULL_TRANSITION_MODEL="$FULL_TRANSITION_MODEL_OVERRIDE"
  echo "Using full_transition override model: ${FULL_TRANSITION_MODEL}"
  echo "full_transition_override_model: ${FULL_TRANSITION_MODEL}" >> "$MANIFEST"
else
  run_train My4C2CurriculumCube3cm-v0 My4C2AllStageCube3cm-v0 "$FULL_TRANSITION_RUN" "$FULL_TRANSITION_STEPS" "$FULL_TRANSITION_LR" "$PLACE_MODEL"
  check_stage_success full_transition "$FULL_TRANSITION_RUN" "$MIN_FULL_TRANSITION_SUCCESS"
  FULL_TRANSITION_MODEL="$(pick_model_by_kind "$FULL_TRANSITION_RUN" "$HANDOFF_MODEL_KIND")"
fi

run_train My4C2AllStageCube3cm-v0 My4C2AllStageCube3cm-v0 "$FULL_RUN" "$FULL_STEPS" "$FULL_LR" "$FULL_TRANSITION_MODEL"
check_stage_success full "$FULL_RUN" "$MIN_FULL_SUCCESS"
FULL_MODEL="$(pick_model_by_kind "$FULL_RUN" best_success)"

echo
echo "Closed-loop training finished."
echo "Reach model: ${REACH_MODEL}"
echo "Grasp model: ${GRASP_MODEL}"
echo "Reach-grasp model: ${REACH_GRASP_MODEL}"
echo "Lift model: ${LIFT_MODEL}"
echo "Place model: ${PLACE_MODEL}"
echo "Full-transition model: ${FULL_TRANSITION_MODEL}"
echo "Full model: ${FULL_MODEL}"
echo
echo "Suggested final eval:"
printf ' %q' "$PYTHON_BIN" scripts/eval.py --env-id My4C2AllStageCube3cm-v0 --model-path "$FULL_MODEL" --episodes 100 --seed "$SEED" --show-safety
echo

{
  echo "## Selected models"
  echo "reach: ${REACH_MODEL}"
  echo "grasp: ${GRASP_MODEL}"
  echo "reach_grasp: ${REACH_GRASP_MODEL}"
  echo "lift: ${LIFT_MODEL}"
  echo "place: ${PLACE_MODEL}"
  echo "full_transition: ${FULL_TRANSITION_MODEL}"
  echo "full: ${FULL_MODEL}"
  echo
  echo "## Suggested final eval"
  printf 'command:'
  printf ' %q' "$PYTHON_BIN" scripts/eval.py --env-id My4C2AllStageCube3cm-v0 --model-path "$FULL_MODEL" --episodes 100 --seed "$SEED" --show-safety
  echo
} >> "$MANIFEST"

echo "Manifest: ${MANIFEST}"
