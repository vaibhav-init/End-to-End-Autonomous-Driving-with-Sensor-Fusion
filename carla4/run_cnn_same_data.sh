#!/usr/bin/env bash
# CNN on the ORIGINAL 20-minute collections, so it is directly comparable
# with the transformer and MLP arms already measured on the same data.
#
# Everything matches those runs: same datasets, same 40-frame label horizon,
# same 60-epoch ceiling and patience, same four scenarios and ten seeds.
# The only variable is the model.
set -u
export CARLA_ROOT="${CARLA_ROOT:-/storage/CARLA_0.9.16}"
export CARLA_TM_PORT="${CARLA_TM_PORT:-8050}"
OV="${OV:-artifacts/rgd_calibration_v7s_drive/calibrated_overrides.json}"
HORIZON="${LABEL_HORIZON:-40}"
WORKERS="${WORKERS:-8}"
SCENARIOS="${SCENARIOS:-1 2 4 5}"
LOG=logs/cnn_same_data.log
mkdir -p logs
mark() { echo "$1 $(date +%H:%M:%S)" >> "$LOG"; }

for d in clean ghost; do
  if [ ! -f model_cnn_${d}/model_config.json ]; then
    python3 -u train_target_speed_cnn.py --data dataset_${d} --label-horizon "$HORIZON" \
        --num-workers "$WORKERS" --output model_cnn_${d} > logs/cnn_same_train_$d.log 2>&1
    mark "TRAIN_cnn_${d}=$?"
  else
    mark "HAVE_cnn_${d}"
  fi
done

for m in model_cnn_clean model_cnn_ghost; do
  [ -d "$m" ] || continue
  python3 acceptance_test.py --model-dir "$m" --background-from dataset_ghost > logs/cnn_same_accept_$m.log 2>&1
  mark "ACCEPT_${m}=$?"
  python3 -u counterfactual_ghost_test.py --model-dir "$m" --data dataset_ghost \
      --output artifacts/counterfactual_$m.json > logs/cnn_same_cf_$m.log 2>&1
  mark "CF_${m}=$?"
done

run_arm() {  # label model
  local label=$1 model=$2
  (cd scenarios && python3 -u run_all.py --driver cnn --model-dir "../$model" \
     --scenarios $SCENARIOS --radar-backend realistic --radar-profile rgd_regime_v1 \
     --radar-config "../$OV" --radar-multipath-mode geometry \
     --output-root "results_$label") > "logs/cnn_same_arm_$label.log" 2>&1
  mark "ARM_${label}=$?"
}
run_arm A_cnn_clean_ghost model_cnn_clean
run_arm D_cnn_ghost       model_cnn_ghost
mark "ALL_DONE"
