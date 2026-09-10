#!/usr/bin/env bash
# Large paired collection for the data-starvation test.
#
# The 20-minute sets the first study used made the transformer overfit by
# epoch 19. This collects three times as much, in 1800 s chunks so a crash
# costs one chunk rather than the whole run, and writes every chunk into the
# same dataset directory (the trainers glob all CSVs in a directory).
#
#   bash collect_large_dataset.sh            # ghost branch, then clean
#   BRANCHES=ghost bash collect_large_dataset.sh
#
# Progress markers land in logs/large_collect.log.
set -u
export CARLA_ROOT="${CARLA_ROOT:-/storage/CARLA_0.9.16}"
export CARLA_TM_PORT="${CARLA_TM_PORT:-8050}"
OV="${OV:-artifacts/rgd_calibration_v7s_drive/calibrated_overrides.json}"
RADAR="--radar-backend realistic --radar-profile rgd_regime_v1 --radar-config $OV"
CHUNK_S="${CHUNK_S:-1800}"      # seconds of simulated driving per chunk
CHUNKS="${CHUNKS:-2}"           # free-driving chunks per branch
SUPPLEMENT_S="${SUPPLEMENT_S:-1200}"
SEED="${SEED:-42}"
BRANCHES="${BRANCHES:-ghost clean}"
LOG=logs/large_collect.log
mkdir -p logs

mark() { echo "$1 $(date +%H:%M:%S)" >> "$LOG"; }

collect() {  # branch chunk_index duration scenarios seed suffix
  local branch=$1 idx=$2 dur=$3 scen=$4 seed=$5 suffix=$6
  local mode=off; [ "$branch" = ghost ] && mode=geometry
  local out=dataset_${branch}_big
  local tmp=.chunk_${branch}_${suffix}
  rm -rf "$tmp"
  local extra=""; [ -n "$scen" ] && extra="--scenarios $scen"
  python3 -u collect_throttle_brake_data.py --teacher gapkeep --duration "$dur" \
      --seed "$seed" --output "$tmp" $extra $RADAR --radar-multipath-mode $mode \
      > "logs/large_${branch}_${suffix}.log" 2>&1
  local code=$?
  if [ $code -eq 0 ] && [ -f "$tmp/data.csv" ]; then
    mkdir -p "$out"
    cp "$tmp/data.csv" "$out/data_${suffix}.csv"
    cp "$tmp/data.detections.npz" "$out/data_${suffix}.detections.npz"
    cp "$tmp/dataset_config.json" "$out/dataset_config.json"
    rm -rf "$tmp"
    mark "OK ${branch}_${suffix} rows=$(($(wc -l < "$out/data_${suffix}.csv") - 1))"
  else
    mark "FAIL ${branch}_${suffix} exit=$code"
  fi
}

for branch in $BRANCHES; do
  mark "START $branch"
  for i in $(seq 1 "$CHUNKS"); do
    collect "$branch" "$i" "$CHUNK_S" "" "$((SEED + i * 13))" "drive$i"
  done
  # Emergency-only chunk: a stopped car appears every few seconds at speed.
  collect "$branch" 0 "$SUPPLEMENT_S" emergency "$((SEED + 7))" emerg
  mark "DONE $branch"
  python3 inspect_dataset.py "dataset_${branch}_big" >> "logs/inspect_${branch}_big.log" 2>&1
done
mark "ALL_DONE"
