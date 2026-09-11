#!/usr/bin/env bash
# Train and evaluate every controller on the large paired collections.
#
# Answers two questions the first study left open:
#   is the transformer's failure data starvation? Same model, same settings,
#   four times the data, so it compares directly with the 20-minute arms.
#
# Transformer only. The CNN's comparison runs on the original collections
# (run_cnn_same_data.sh) so the model is the only variable there.
#
# Stages, each appending markers to logs/big_study.log:
#   train   transformer on the clean and ghost big sets
#   gates   acceptance probe + ghost counterfactual for each
#   arms    the same four scenarios and ten seeds as the 20-minute arms
set -u
export CARLA_ROOT="${CARLA_ROOT:-/storage/CARLA_0.9.16}"
export CARLA_TM_PORT="${CARLA_TM_PORT:-8050}"
OV="${OV:-artifacts/rgd_calibration_v7s_drive/calibrated_overrides.json}"
HORIZON="${LABEL_HORIZON:-40}"
WORKERS="${WORKERS:-8}"
SCENARIOS="${SCENARIOS:-1 2 4 5}"
LOG=logs/big_study.log
mkdir -p logs
mark() { echo "$1 $(date +%H:%M:%S)" >> "$LOG"; }

# An epoch on the large sets costs about 9.4 minutes, so the 60-epoch ceiling
# the small study used would run for days. On 20 minutes of data the best
# checkpoint landed at epoch 19 and everything after it overfitted, so a
# ceiling of EPOCHS with PATIENCE of no improvement bounds the schedule while
# leaving room for the optimum to arrive later on four times the data.
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"

# CNN before transformer: it is the new question, so if anything runs out of
# time it should be the model we already have a result for.
stage_train() {
  for d in clean ghost; do
    local data=dataset_${d}_big
    [ -d "$data" ] || { mark "SKIP_${d}_no_data"; continue; }
    if [ -f model_tf_${d}_big/model_config.json ]; then
      mark "HAVE_tf_${d}"
      continue
    fi
    python3 -u train_target_speed_transformer.py --data "$data" --label-horizon "$HORIZON" \
        --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" \
        --num-workers "$WORKERS" --output model_tf_${d}_big > logs/big_train_tf_$d.log 2>&1
    mark "TRAIN_tf_${d}=$?"
  done
}

stage_gates() {
  for m in model_tf_clean_big model_tf_ghost_big; do
    [ -d "$m" ] || continue
    python3 acceptance_test.py --model-dir "$m" --background-from dataset_ghost_big \
        > logs/big_accept_$m.log 2>&1
    mark "ACCEPT_${m}=$?"
    python3 -u counterfactual_ghost_test.py --model-dir "$m" --data dataset_ghost_big \
        --limit 20000 --output artifacts/big_counterfactual_$m.json \
        > logs/big_cf_$m.log 2>&1
    mark "CF_${m}=$?"
  done
}

run_arm() {  # label driver model extra...
  local label=$1 driver=$2 model=$3; shift 3
  (cd scenarios && python3 -u run_all.py --driver "$driver" --model-dir "../$model" \
     --scenarios $SCENARIOS \
     --radar-backend realistic --radar-profile rgd_regime_v1 --radar-config "../$OV" \
     --output-root "results_big_$label" "$@") > "logs/big_arm_$label.log" 2>&1
  mark "ARM_${label}=$?"
}

stage_arms() {
  run_arm tf_clean transformer model_tf_clean_big --radar-multipath-mode geometry
  run_arm tf_ghost transformer model_tf_ghost_big --radar-multipath-mode geometry
}

case "${1:-all}" in
  train) stage_train ;;
  gates) stage_gates ;;
  arms)  stage_arms ;;
  all)   stage_train; stage_gates; stage_arms ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
mark "STAGE_${1:-all}_DONE"
