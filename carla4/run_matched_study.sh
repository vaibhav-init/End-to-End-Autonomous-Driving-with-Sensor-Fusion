#!/usr/bin/env bash
# Matched synthetic -> real ghost-detector study.
#
#   1. prepare the matched CARLA collection (pedestrian/cyclist, tangential
#      motion, calibrated RGD profile)
#   2. pretrain from the zero-shot checkpoint on it
#   3. score it zero-shot on the real RGD test split
#   4. two-stage fine-tune on real data for three seeds
#   5. label-scarce fine-tunes (5 and 10 real sequences)
#
# Run inside tmux with the PCLA environment active:
#   tmux new -d -s matched './run_matched_study.sh'

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_REAL="${DATA_REAL:-artifacts/ghost_real_official_v2}"
MATCHED_ROOT="${MATCHED_ROOT:-artifacts/carla_ghost_matched_v1}"
MATCHED_PREPARED="${MATCHED_PREPARED:-artifacts/ghost_carla_matched_v1_prepared}"
ZEROSHOT="${ZEROSHOT:-artifacts/zeroshot_v3_carla_pretrain/best_detector.pt}"
PRETRAIN_OUT="${PRETRAIN_OUT:-artifacts/ft_matched_v1}"
OUT="${OUT:-artifacts/s2r_matched}"
SEEDS="${SEEDS:-42 123 256}"
WORKERS="${WORKERS:-4}"

fail() {
    echo "[fail] $*" >&2
    exit 1
}

echo "=== 1. prepare matched collection ==="
python3 -u prepare_radar_ghost_dataset.py \
    --input "$MATCHED_ROOT" \
    --output "$MATCHED_PREPARED" \
    --split-mode official \
    || fail "prepare"

python3 - "$MATCHED_PREPARED" <<'PY' || exit 1
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
for split in ("train", "val", "test"):
    records = [r for r in manifest["sequences"] if r.get("split") == split]
    real = sum(r.get("real_points", 0) for r in records)
    ghost = sum(r.get("ghost_points", 0) for r in records)
    print(f"{split:5s} sequences={len(records):2d} real={real} ghost={ghost}")
    if split == "train" and (real == 0 or ghost == 0):
        raise SystemExit("matched training split is missing a class")
PY

echo "=== 2. synthetic-only pretrain ==="
python3 -u train_radar_ghost_detector.py \
    --data "$MATCHED_PREPARED" \
    --output "$PRETRAIN_OUT" \
    --pretrained "$ZEROSHOT" \
    --epochs 10 \
    --batch-size 16 \
    --num-workers "$WORKERS" \
    --cache-sequences 64 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --label-smoothing 0.02 \
    --max-real-fpr 0.01 \
    --select-metric auprc \
    --early-stop-patience 0 \
    > "$PRETRAIN_OUT.log" 2>&1 \
    || fail "matched pretrain (see $PRETRAIN_OUT.log)"

echo "=== 3. zero-shot synthetic -> real test ==="
python3 -u evaluate_radar_ghost_detector.py \
    --data "$DATA_REAL" \
    --checkpoint "$PRETRAIN_OUT/best_detector.pt" \
    --split test \
    --batch-size 32 \
    --num-workers "$WORKERS" \
    --output "$PRETRAIN_OUT/real_test.json" \
    > "$PRETRAIN_OUT/real_test.log" 2>&1 \
    || fail "matched zero-shot evaluation"

echo "=== 4. matched fine-tune, seeds: $SEEDS ==="
for seed in $SEEDS; do
    PRETRAINED="$PRETRAIN_OUT/best_detector.pt" \
    OUT="$OUT" \
    EPOCHS=30 \
    PATIENCE=8 \
    ./run_synthetic_to_real.sh train ft_twostage "$seed" \
        || fail "matched fine-tune seed $seed"
done

echo "=== 5. label-scarce matched fine-tunes ==="
for count in 5 10; do
    subset="artifacts/ghost_real_subset_${count}"
    if [[ ! -f "$subset/manifest.json" ]]; then
        python3 make_real_subset.py \
            --input "$DATA_REAL" \
            --output "$subset" \
            --train-sequences "$count" \
            --seed 42 \
            --force \
            || fail "subset $count"
    fi
    PRETRAINED="$PRETRAIN_OUT/best_detector.pt" \
    DATA_REAL="$subset" \
    OUT="artifacts/s2r_matched_few${count}" \
    EPOCHS=30 \
    PATIENCE=8 \
    ./run_synthetic_to_real.sh train ft_twostage 42 \
        || fail "matched few-shot $count"
done

echo "=== summary ==="
OUT="$OUT" ./run_synthetic_to_real.sh summary
echo "=== done ==="
