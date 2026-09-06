#!/usr/bin/env bash
# Closed-loop ghost study, staged. Run from carla4/ inside tmux with CARLA up:
#   bash run_closed_loop_pipeline.sh smoke      # 60 s collector smoke + gate
#   bash run_closed_loop_pipeline.sh collect    # paired 1800 s collections (ghosts off / on)
#   bash run_closed_loop_pipeline.sh train      # MLP + transformer, acceptance, counterfactual
#   bash run_closed_loop_pipeline.sh s5smoke    # one S5 seed per arm
#   bash run_closed_loop_pipeline.sh arms       # full arm matrix + comparison
#   bash run_closed_loop_pipeline.sh all        # everything in order, stops at the first failed gate
# Every stage appends STAGE_<name>_EXIT=<code> to logs/closed_loop_pipeline.log.
set -u
export CARLA_ROOT="${CARLA_ROOT:-/storage/CARLA_0.9.16}"
OV="${OV:-artifacts/rgd_calibration_v7s/calibrated_overrides.json}"
RADAR="--radar-backend realistic --radar-profile rgd_regime_v1 --radar-config $OV"
SEED="${SEED:-42}"
COLLECT_S="${COLLECT_S:-900}"
LOG=logs/closed_loop_pipeline.log
mkdir -p logs

mark() { echo "STAGE_$1_EXIT=$2 $(date +%H:%M:%S)" >> "$LOG"; }

gate_collection() {
  # A collection is only worth training on if the radar saw a target in a
  # reasonable share of frames and the sidecar exists.
  python3 - "$1" <<'PY'
import sys, os, pandas as pd
d = sys.argv[1]
df = pd.read_csv(os.path.join(d, "data.csv"))
seen = float((df["distance_t-0"] < 90.0).mean())
side = os.path.exists(os.path.join(d, "data.detections.npz"))
print(f"  gate {d}: rows={len(df)} target-seen share={seen:.2f} sidecar={side}")
sys.exit(0 if (seen >= 0.25 and side and len(df) > 200) else 1)
PY
}

stage_smoke() {
  python3 -u collect_throttle_brake_data.py --teacher gapkeep --duration 60 --seed "$SEED" \
    --output dataset_ctl_smoke $RADAR --radar-multipath-mode geometry > logs/ctl_smoke.log 2>&1
  code=$?; [ $code -eq 0 ] && python3 inspect_dataset.py dataset_ctl_smoke >> logs/ctl_smoke.log 2>&1
  [ $code -eq 0 ] && gate_collection dataset_ctl_smoke >> logs/ctl_smoke.log 2>&1; code=$?
  mark smoke $code; return $code
}

stage_collect() {
  python3 -u collect_throttle_brake_data.py --teacher gapkeep --duration "$COLLECT_S" --seed "$SEED" \
    --output dataset_clean $RADAR --radar-multipath-mode off > logs/collect_clean.log 2>&1
  code=$?; [ $code -eq 0 ] && gate_collection dataset_clean >> logs/collect_clean.log 2>&1; code=$?
  mark collect_clean $code; [ $code -ne 0 ] && return $code
  python3 -u collect_throttle_brake_data.py --teacher gapkeep --duration "$COLLECT_S" --seed "$SEED" \
    --output dataset_ghost $RADAR --radar-multipath-mode geometry > logs/collect_ghost.log 2>&1
  code=$?; [ $code -eq 0 ] && gate_collection dataset_ghost >> logs/collect_ghost.log 2>&1; code=$?
  python3 inspect_dataset.py dataset_clean dataset_ghost >> logs/collect_ghost.log 2>&1
  mark collect_ghost $code; return $code
}

stage_train() {
  local code=0
  for d in clean ghost; do
    python3 -u train_throttle_brake.py --data dataset_$d --config dataset_$d/dataset_config.json \
      --output model_mlp_$d > logs/train_mlp_$d.log 2>&1 || code=1
    python3 -u train_target_speed_transformer.py --data dataset_$d --output model_tf_$d \
      > logs/train_tf_$d.log 2>&1 || code=1
  done
  for m in model_mlp_clean model_mlp_ghost model_tf_clean model_tf_ghost; do
    python3 acceptance_test.py --model-dir $m > logs/accept_$m.log 2>&1; echo "ACCEPT_${m}_EXIT=$?" >> "$LOG"
  done
  for m in model_tf_clean model_tf_ghost; do
    python3 -u counterfactual_ghost_test.py --model-dir $m --data dataset_ghost \
      --output artifacts/counterfactual_$m.json > logs/counterfactual_$m.log 2>&1 || code=1
  done
  mark train $code; return $code
}

run_arm() {  # name driver model extra...
  local name=$1 driver=$2 model=$3; shift 3
  (cd scenarios && python3 -u run_all.py --driver "$driver" --model-dir "../$model" \
     --radar-backend realistic --radar-profile rgd_regime_v1 --radar-config "../$OV" \
     --output-root "results_$name" "$@") > "logs/arm_$name.log" 2>&1
  echo "ARM_${name}_EXIT=$? $(date +%H:%M:%S)" >> "$LOG"
}

stage_s5smoke() {
  run_arm s5smoke_A_mlp mlp model_mlp_clean --radar-multipath-mode geometry --scenarios 5 --seeds "$SEED"
  run_arm s5smoke_D_tf transformer model_tf_ghost --radar-multipath-mode geometry --scenarios 5 --seeds "$SEED"
  mark s5smoke 0
}

stage_arms() {
  run_arm 0_mlp_clean_noghost mlp model_mlp_clean --radar-multipath-mode off
  run_arm A_mlp_clean_ghost mlp model_mlp_clean --radar-multipath-mode geometry
  run_arm B_mlp_clean_oracle mlp model_mlp_clean --radar-multipath-mode geometry --radar-ghost-oracle
  run_arm D_mlp_ghost mlp model_mlp_ghost --radar-multipath-mode geometry
  run_arm A_tf_clean_ghost transformer model_tf_clean --radar-multipath-mode geometry
  run_arm D_tf_ghost transformer model_tf_ghost --radar-multipath-mode geometry
  (cd scenarios && python3 compare_drivers.py --runs \
     noghost=results_0_mlp_clean_noghost A_mlp=results_A_mlp_clean_ghost B_mlp=results_B_mlp_clean_oracle \
     D_mlp=results_D_mlp_ghost A_tf=results_A_tf_clean_ghost D_tf=results_D_tf_ghost) > logs/compare_arms.log 2>&1
  mark arms $?
}

case "${1:-all}" in
  smoke) stage_smoke ;;
  collect) stage_collect ;;
  train) stage_train ;;
  s5smoke) stage_s5smoke ;;
  arms) stage_arms ;;
  all) stage_smoke && stage_collect && stage_train && stage_s5smoke && stage_arms ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
echo "PIPELINE_${1:-all}_DONE $(date +%H:%M:%S)" >> "$LOG"
