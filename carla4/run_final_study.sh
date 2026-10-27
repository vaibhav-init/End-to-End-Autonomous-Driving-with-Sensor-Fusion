#!/usr/bin/env bash
# Final closed-loop study: S5 ghost exposure with a following vehicle.
#
# Run from carla4:  ./run_final_study.sh smoke|pilot|pcrash|pcrash_close|analyze|all
# Relative paths are used deliberately: the VM's absolute path lookup has a
# stale-dentry problem after heavy I/O, while the relative path is reliable.
set -u

cd "$(dirname "$0")"
SCENARIOS="scenarios"
PY=${PYTHON:-$HOME/anaconda3/envs/PCLA/bin/python}
CONFIG="artifacts/final/model_matched_015b.json"
OUT="artifacts/final"
mkdir -p "$OUT"
export CARLA_ROOT=${CARLA_ROOT:-/storage/CARLA_0.9.16}
export CARLA_TM_PORT=${CARLA_TM_PORT:-8050}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}

check_carla() {
    "$PY" - <<'PY'
import socket, sys
try:
    socket.create_connection(("127.0.0.1", 2000), timeout=5).close()
except OSError as exc:
    sys.exit(f"CARLA not reachable on 127.0.0.1:2000 ({exc})")
PY
}

run_s5() {
    local name=$1 mode=$2 model=$3 duration=$4 seeds=$5
    shift 5
    echo "=== S5 $name | $mode | $model | ${duration}s ==="
    (
        cd "$SCENARIOS"
        "$PY" run_all.py \
            --scenarios 5 --fog 3 --seeds $seeds \
            --towns Town04 \
            --driver mlp --model-dir "$model" \
            --radar-backend realistic --radar-profile rgd_regime_v1 \
            --radar-config "../$CONFIG" \
            --radar-multipath-mode "$mode" \
            --tm-port 8050 \
            --s5-duration-s "$duration" --s5-follower \
            --timeout 420 \
            --output-root "results_final_$name" "$@"
    )
}

smoke() {
    check_carla
    run_s5 smoke off ../model_mlp_clean 15 42
}

pilot() {
    check_carla
    run_s5 pilot_off off ../model_mlp_clean 30 "42 43 44"
    run_s5 pilot_on geometry ../model_mlp_clean 30 "42 43 44"
}

pcrash() {
    check_carla
    run_s5 pcrash_off off ../model_mlp_clean 60 "$SEEDS"
    run_s5 pcrash_on geometry ../model_mlp_clean 60 "$SEEDS"
    run_s5 pcrash_d geometry ../model_mlp_ghost 60 "$SEEDS"
}

pcrash_close() {
    check_carla
    run_s5 pcrash_close_off off ../model_mlp_clean 60 "$SEEDS" \
        --s5-follower-gap-m 10 --s5-follower-reaction-s 1.0
    run_s5 pcrash_close_on geometry ../model_mlp_clean 60 "$SEEDS" \
        --s5-follower-gap-m 10 --s5-follower-reaction-s 1.0
}

analyze() {
    echo "=== final analysis ==="
    (
        cd "$SCENARIOS"
        "$PY" analyze_results.py \
            --runs \
                off=results_final_pcrash_off \
                on=results_final_pcrash_on \
                d=results_final_pcrash_d \
                close_off=results_final_pcrash_close_off \
                close_on=results_final_pcrash_close_on \
            --skip-validation \
            --out ../artifacts/final/comparison_pcrash
        "$PY" power_analysis.py \
            ../artifacts/final/comparison_pcrash/per_run_metrics.csv \
            --out-dir ../artifacts/final/power_pcrash
        "$PY" requirements.py \
            --out ../artifacts/final/requirements.csv \
            --p-crash 1e-2 1e-3 1e-4 1e-5
        "$PY" make_final_report.py
    )
    echo "wrote $OUT/comparison_pcrash and $OUT/requirements.csv"
}

case "${1:-all}" in
    smoke) smoke ;;
    pilot) pilot ;;
    pcrash) pcrash ;;
    pcrash_close) pcrash_close ;;
    analyze) analyze ;;
    all) smoke && pcrash && analyze ;;
    *) echo "usage: $0 smoke|pilot|pcrash|pcrash_close|analyze|all" >&2; exit 2 ;;
esac
