#!/usr/bin/env bash
# Morning-after analysis for the overnight lever-ablation and 10-seed runs.
#
# Run AFTER both tmux chains finish (check /tmp/lever_ablation.log for
# ABLATION_ALL_DONE and /tmp/extra_seeds.log for SEEDS_ALL_DONE):
#   tmux new -d -s morning './analyze_overnight.sh'
#
# Every step skips itself if its output already exists, so the script can
# be re-run safely as runs trickle in.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/PCLA/bin/python}"
export PYTHON
S=${S:-/storage/vaibhav/s2r_results}
DATA_REAL=${DATA_REAL:-artifacts/ghost_real_official_v2}
SEEDS_ALL="7 42 123 256 2024 5 11 77 99 300"
LOG=${LOG:-/tmp/morning_analysis.log}

log() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. stats
log "paired seed statistics (n=${#SEEDS_ALL})"
"$PYTHON" -u paired_seed_stats.py \
    --matched "$S/s2r_matched" --real "$S/s2r" \
    --seeds $SEEDS_ALL \
    --output "$S/paired_seed_stats.json" >> "$LOG" 2>&1 \
    || log "WARN paired stats failed (some seeds may still be running)"

# ------------------------------------------------- 2. 10-seed score dumps
for split in val test; do
    out="$S/combined10_${split}_scores.npz"
    if [[ -s "$out" ]]; then log "dump $split already present"; continue; fi
    log "dumping 10 checkpoints on $split"
    specs=()
    for seed in $SEEDS_ALL; do
        specs+=(--checkpoint "matched${seed}=$S/s2r_matched/ft_twostage_s${seed}/best_detector.pt")
        specs+=(--checkpoint "ro${seed}=$S/s2r/real_only_s${seed}/best_detector.pt")
    done
    "$PYTHON" -u evaluate_ghost_transfer.py dump "${specs[@]}" \
        --data "$DATA_REAL" --split "$split" --output "$out" \
        --device cuda --batch-size 16 --num-workers 0 \
        >> "$LOG" 2>&1 || log "WARN dump $split failed"
done

# ------------------------------------------------- 3. 10-seed ensembles
if [[ -s "$S/combined10_val_scores.npz" && -s "$S/combined10_test_scores.npz" ]]; then
    log "10-seed ensembles + paired bootstrap"
    "$PYTHON" -u ensemble_ghost_scores.py \
        --val "$S/combined10_val_scores.npz" \
        --test "$S/combined10_test_scores.npz" \
        --group "matched10=matched7,matched42,matched123,matched256,matched2024,matched5,matched11,matched77,matched99,matched300" \
        --group "realonly10=ro7,ro42,ro123,ro256,ro2024,ro5,ro11,ro77,ro99,ro300" \
        --output "$S/ensemble_10seeds.json" >> "$LOG" 2>&1 \
        || log "WARN 10-seed ensemble failed"
fi

# ------------------------------------------------- 4. 10-seed soups
soup_names=$(for seed in $SEEDS_ALL; do printf "matched%s=%s/s2r_matched/ft_twostage_s%s/best_detector.pt " "$seed" "$S" "$seed"; done)
for mode in uniform greedy; do
    out="$S/soup_matched10_${mode}"
    if [[ -s "$out/soup_report.json" ]]; then log "soup $mode already present"; continue; fi
    log "10-seed $mode soup"
    # shellcheck disable=SC2086
    "$PYTHON" -u soup_ghost_checkpoints.py $soup_names \
        --data "$DATA_REAL" --mode "$mode" --output "$out" \
        >> "$LOG" 2>&1 || log "WARN soup $mode failed"
done

# ------------------------------------------------- 5. summaries
log "run summaries"
for dir in "$S/s2r_matched" "$S/s2r"; do
    OUT="$dir" ./run_synthetic_to_real.sh summary >> "$LOG" 2>&1 || true
done

# ------------------------------------------------- 6. ablation summaries
log "lever ablation summaries"
for variant in base radial_motion low_density uncalibrated_amplitude; do
    out="artifacts/s2r_lever_${variant}"
    if [[ -d "$out" ]]; then
        echo "=== lever $variant ===" >> "$LOG"
        OUT="$out" ./run_synthetic_to_real.sh summary >> "$LOG" 2>&1 || true
    fi
done

log "morning analysis complete; see $S/paired_seed_stats.json, $S/ensemble_10seeds.json, soup_matched10_*/soup_report.json and the summaries above"
