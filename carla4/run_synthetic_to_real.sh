#!/usr/bin/env bash
# Synthetic-to-real ghost-detector study.
#
#   ./run_synthetic_to_real.sh train <arm> <seed>
#   ./run_synthetic_to_real.sh screen [seed]
#   ./run_synthetic_to_real.sh summary
#
# Every arm starts from the same synthetic checkpoint and adapts on the real
# RGD train split. The official val split selects the checkpoint and its
# fixed-FPR threshold; the official test split is evaluated once per arm.
#
# Run inside tmux with the PCLA environment active:
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate PCLA
#   tmux new -d -s s2r './run_synthetic_to_real.sh screen 42'

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_REAL="${DATA_REAL:-artifacts/ghost_real_official_v2}"
SYNTH="${SYNTH:-artifacts/ghost_carla_calibrated_v7s_prepared}"
PRETRAINED="${PRETRAINED:-artifacts/zeroshot_v3_carla_pretrain/best_detector.pt}"
OUT="${OUT:-artifacts/s2r}"
PYTHON="${PYTHON:-python3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
CACHE="${CACHE:-128}"
MAX_FPR="${MAX_FPR:-0.01}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"

common_args() {
    printf '%s\n' \
        --data "$DATA_REAL" \
        --cache-sequences "$CACHE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$WORKERS" \
        --epochs "$EPOCHS" \
        --early-stop-patience "$PATIENCE" \
        --max-real-fpr "$MAX_FPR" \
        --label-smoothing 0.02 \
        --weight-decay 0.0001
}

# Each arm is a different answer to "how much of the synthetic encoder should
# survive contact with real data?".
arm_args() {
    local arm="$1"
    case "$arm" in
        real_only)
            common_args
            ;;
        ft_low)
            common_args
            printf '%s\n' --pretrained "$PRETRAINED" \
                --learning-rate 0.0001 --backbone-lr-scale 0.3 \
                --select-metric auroc
            ;;
        ft_auprc)
            common_args
            printf '%s\n' --pretrained "$PRETRAINED" \
                --learning-rate 0.0001 --backbone-lr-scale 0.3 \
                --select-metric auprc
            ;;
        ft_twostage)
            common_args
            printf '%s\n' --pretrained "$PRETRAINED" \
                --learning-rate 0.0003 --backbone-lr-scale 0.1 \
                --freeze-backbone-epochs 3 --select-metric auroc
            ;;
        ft_head)
            common_args
            printf '%s\n' --pretrained "$PRETRAINED" \
                --learning-rate 0.001 --freeze-backbone-epochs "$EPOCHS" \
                --reset-head --select-metric auroc
            ;;
        ft_reset)
            common_args
            printf '%s\n' --pretrained "$PRETRAINED" \
                --learning-rate 0.0003 --backbone-lr-scale 0.1 \
                --freeze-backbone-epochs 2 --reset-head --select-metric auroc
            ;;
        *)
            echo "unknown arm: $arm" >&2
            return 2
            ;;
    esac
}

train_arm() {
    local arm="$1"
    local seed="$2"
    local run_dir="$OUT/${arm}_s${seed}"
    mkdir -p "$run_dir"
    if [[ -s "$run_dir/best_detector.pt" && -s "$run_dir/test_metrics.json" ]]; then
        echo "[skip] $run_dir already complete"
        return 0
    fi
    echo "[train] $arm seed=$seed -> $run_dir"
    local -a extra=()
    while IFS= read -r value; do
        extra+=("$value")
    done < <(arm_args "$arm")
    "$PYTHON" -u train_radar_ghost_detector.py \
        --output "$run_dir" --seed "$seed" "${extra[@]}" \
        > "$run_dir/train.log" 2>&1
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "[fail] $arm seed=$seed training exit=$status" >&2
        return $status
    fi
    "$PYTHON" -u evaluate_radar_ghost_detector.py \
        --data "$DATA_REAL" --checkpoint "$run_dir/best_detector.pt" \
        --split test --batch-size 32 --num-workers "$WORKERS" \
        --output "$run_dir/test_metrics.json" \
        > "$run_dir/eval.log" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "[fail] $arm seed=$seed eval exit=$status" >&2
        return $status
    fi
    echo "[done] $arm seed=$seed"
}

summary() {
    "$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for run in sorted(root.glob("*_s*")):
    metrics = run / "test_metrics.json"
    if not metrics.exists():
        continue
    data = json.loads(metrics.read_text())
    fixed = data["confusion_matrix"]["fixed"]
    summary = run / "training_summary.json"
    selected = json.loads(summary.read_text()) if summary.exists() else {}
    rows.append(
        (
            run.name,
            data["auroc"],
            data["auprc"],
            fixed["threshold"],
            fixed["recall"],
            fixed["false_positive_rate"],
            fixed["f1"],
            selected.get("best_validation_score"),
            selected.get("select_metric", ""),
        )
    )
rows.sort(key=lambda row: row[2], reverse=True)
print(
    f"{'run':22s} {'AUROC':>7s} {'AUPRC':>7s} {'thr':>7s} "
    f"{'recall':>7s} {'FPR':>7s} {'F1':>7s} {'val':>8s} metric"
)
for row in rows:
    best = f"{row[7]:.4f}" if row[7] is not None else "-"
    print(
        f"{row[0]:22s} {row[1]:7.4f} {row[2]:7.4f} {row[3]:7.4f} "
        f"{row[4]:7.4f} {row[5]:7.4f} {row[6]:7.4f} {best:>8s} {row[8]}"
    )
PY
}

main() {
    local command="${1:-}"
    case "$command" in
        train)
            train_arm "$2" "$3"
            ;;
        screen)
            local seed="${2:-42}"
            for arm in real_only ft_low ft_auprc ft_twostage ft_head ft_reset; do
                train_arm "$arm" "$seed"
            done
            summary
            ;;
        summary)
            summary
            ;;
        *)
            echo "usage: $0 {train <arm> <seed>|screen [seed]|summary}" >&2
            return 2
            ;;
    esac
}

main "$@"
