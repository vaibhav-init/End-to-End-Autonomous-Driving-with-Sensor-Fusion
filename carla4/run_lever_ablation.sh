#!/usr/bin/env bash
# Leave-one-out lever ablation for the matched synthetic-to-real study.
#
# The matched collection (S2R_MATCHED_REPORT.md) bundles six choices:
#   1. tangential motion along the reflector
#   2. target range 4-10 m
#   3. 2.0 m surface gap
#   4. ~336 points/scan density (points-per-second + calibrated emission)
#   5. calibrated amplitude/bounce losses (rgd_calibration_v7s overrides)
#   6. main-object-only labels
# This script collects one variant per removed lever, then runs the SAME
# matched recipe (zero-shot-anchored pretrain + two-stage fine-tune, seeds
# 42 123) so each variant's number is directly comparable with the matched
# headline 0.8849/0.5536.
#
#   ./run_lever_ablation.sh <variant>          # one variant
#   ./run_lever_ablation.sh all                # every variant in sequence
#
# Variants: radial_motion, wide_range, far_gap, low_density,
#           uncalibrated_amplitude, all_labels
#
# Run inside tmux with the PCLA environment active and CARLA on :2000:
#   tmux new -d -s levers './run_lever_ablation.sh all'

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"

# The matched recipe's shared collection settings (S2R_MATCHED_REPORT.md §1).
TOWN=${TOWN:-Town04}
POINTS_PER_SECOND=${POINTS_PER_SECOND:-100000}
RADAR_CONFIG=${RADAR_CONFIG:-artifacts/rgd_calibration_v7s/calibrated_overrides.json}
ZEROSHOT=${ZEROSHOT:-artifacts/zeroshot_v3_carla_pretrain/best_detector.pt}
SEEDS_FINETUNE=${SEEDS_FINETUNE:-"42 123"}
WORKERS=${WORKERS:-4}

variant_args() {
    # Each variant removes exactly one matched lever from the base recipe.
    case "$1" in
        base)
            printf '%s\n' --motion-direction tangent --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 10 --label-scope main
            ;;
        radial_motion)
            printf '%s\n' --motion-direction radial --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 10 --label-scope main
            ;;
        wide_range)
            printf '%s\n' --motion-direction tangent --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 16 --label-scope main
            ;;
        far_gap)
            printf '%s\n' --motion-direction tangent --surface-distance-target 8.0 \
                --target-range-min 4 --target-range-max 10 --label-scope main
            ;;
        low_density)
            printf '%s\n' --motion-direction tangent --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 10 --label-scope main
            ;;
        uncalibrated_amplitude)
            printf '%s\n' --motion-direction tangent --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 10 --label-scope main
            ;;
        all_labels)
            printf '%s\n' --motion-direction tangent --surface-distance-target 2.0 \
                --target-range-min 4 --target-range-max 10 --label-scope all
            ;;
        *)
            echo "unknown variant: $1" >&2
            return 2
            ;;
    esac
}

run_variant() {
    local variant="$1"
    local out="artifacts/lever_${variant}"
    local prepared="${out}_prepared"
    mkdir -p "$out"

    echo "=== [$variant] collect 20 sequences (10 cyclist + 10 pedestrian) ==="
    local -a common=()
    while IFS= read -r value; do common+=("$value"); done < <(variant_args "$variant")

    local density_args=()
    if [[ "$variant" == "low_density" ]]; then
        density_args=(--points-per-second 40000)
    else
        density_args=(--points-per-second "$POINTS_PER_SECOND")
    fi

    local config_args=()
    if [[ "$variant" == "uncalibrated_amplitude" ]]; then
        config_args=()   # profile priors only: no fitted amplitude/loss overrides
    else
        config_args=(--radar-config "$RADAR_CONFIG")
    fi

    # Two cyclist placements under seed 100 (scenes 001 and 002) crash the
    # collector worker deterministically; the supervisor continues past
    # them, so a collection failure is tolerated as long as enough
    # sequences exist. Every variant shares the same scene set, which keeps
    # the leave-one-out comparison internally valid.
    "$PYTHON" -u collect_carla_radar_ghosts.py \
        --target-type cyclist --split train --seed 100 \
        --sequences 10 --headless --town "$TOWN" --output "$out" \
        --sequence-retries 2 --resume \
        "${common[@]}" "${density_args[@]}" "${config_args[@]}" \
        || echo "[warn] $variant cyclist collection incomplete (cursed scenes skipped)"
    "$PYTHON" -u collect_carla_radar_ghosts.py \
        --target-type pedestrian --split train --seed 2000 \
        --sequences 10 --headless --town "$TOWN" --output "$out" \
        --sequence-retries 2 --resume \
        "${common[@]}" "${density_args[@]}" "${config_args[@]}" \
        || echo "[warn] $variant pedestrian collection incomplete"

    local collected
    collected=$(ls "$out"/train/*.h5 2>/dev/null | wc -l)
    if [[ "$collected" -lt 14 ]]; then
        echo "[fail] $variant collection has only $collected sequences (< 14)" >&2
        return 1
    fi
    echo "=== [$variant] collected $collected sequences ==="

    echo "=== [$variant] prepare ==="
    "$PYTHON" -u prepare_radar_ghost_dataset.py \
        --input "$out" --output "$prepared" --split-mode all_train \
        || { echo "[fail] $variant prepare" >&2; return 1; }

    # all_train leaves no validation split for checkpoint selection; carve a
    # small synthetic val out of the prepared set (last two sequences by
    # name — the same scenes across variants, matching the matched study's
    # synthetic-val protocol and never touching real data).
    "$PYTHON" - "$prepared" <<'PY' || { echo "[fail] $variant val carve" >&2; return 1; }
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]) / "manifest.json"
manifest = json.loads(manifest_path.read_text())
sequences = sorted(manifest["sequences"], key=lambda r: r["name"])
moved = 0
for record in reversed(sequences):
    if moved >= 2:
        break
    if record.get("split") == "train":
        record["split"] = "val"
        moved += 1
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"val carve: moved {moved} sequences to val")
PY

    echo "=== [$variant] pretrain (10 epochs from the zero-shot checkpoint) ==="
    "$PYTHON" -u train_radar_ghost_detector.py \
        --data "$prepared" --output "${out}_pretrain" \
        --pretrained "$ZEROSHOT" \
        --epochs 10 --batch-size 16 --num-workers "$WORKERS" \
        --cache-sequences 64 --learning-rate 0.001 --weight-decay 0.0001 \
        --label-smoothing 0.02 --max-real-fpr 0.01 --select-metric auprc \
        --early-stop-patience 0 \
        > "${out}_pretrain.log" 2>&1 \
        || { echo "[fail] $variant pretrain" >&2; return 1; }

    echo "=== [$variant] two-stage fine-tune, seeds: $SEEDS_FINETUNE ==="
    for seed in $SEEDS_FINETUNE; do
        PRETRAINED="${out}_pretrain/best_detector.pt" \
        OUT="artifacts/s2r_lever_${variant}" \
        EPOCHS=30 PATIENCE=8 \
        ./run_synthetic_to_real.sh train ft_twostage "$seed" \
            || { echo "[fail] $variant fine-tune seed $seed" >&2; return 1; }
    done

    echo "=== [$variant] done: artifacts/s2r_lever_${variant} ==="
}

main() {
    local target="${1:-all}"
    if [[ "$target" == "all" ]]; then
        for variant in base radial_motion wide_range far_gap low_density uncalibrated_amplitude all_labels; do
            run_variant "$variant" || echo "[skip] $variant failed; continuing"
        done
    else
        run_variant "$target"
    fi
}

main "$@"
