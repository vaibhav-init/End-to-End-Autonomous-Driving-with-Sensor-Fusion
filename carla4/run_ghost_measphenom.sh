#!/usr/bin/env bash
# v7s ghost-event phenomenology on the measured physics profile (rgd_regime_v1).
#
# Recreates the collection recipe behind the one successful sim-to-real
# fine-tune (v7s: stationary ego, controlled target walking tangent to a
# validated reflector, Town04, 21 usable sequences, ~24% ghost share) with
# today's SBR-measured multipath losses and noise floors, which the v7s era
# predated. Requires an already-running CARLA 0.9.16 server on 127.0.0.1:2000;
# it is never started here.
#
# Usage: run_ghost_measphenom.sh [smoke|collect|prepare|all]
set -u
cd "$(dirname "$0")"

OUT=${OUT:-artifacts/carla_ghost_measphenom_v1}
PY=${PYTHON:-python3}
STAGE=${1:-all}

check_carla() {
    "$PY" - <<'EOF'
import socket
import sys
try:
    socket.create_connection(("127.0.0.1", 2000), timeout=5).close()
except OSError as exc:
    sys.exit(f"CARLA not reachable on 127.0.0.1:2000 ({exc}); start it manually first.")
EOF
}

smoke() {
    check_carla
    "$PY" collect_carla_radar_ghosts.py \
        --town Town04 --target-type pedestrian --split train --sequences 1 \
        --duration 10 --output "$OUT/smoke" --resume "$@"
}

collect_all() {
    check_carla
    FAILURES=0
    for spec in "pedestrian:train:8" "pedestrian:val:2" "pedestrian:test:2" \
                "cyclist:train:8" "cyclist:val:1"; do
        IFS=: read -r type split count <<EOF
$spec
EOF
        echo "=== $type / $split / $count sequences ==="
        if ! "$PY" collect_carla_radar_ghosts.py \
            --town Town04 --target-type "$type" --split "$split" --sequences "$count" \
            --output "$OUT/$type" --resume "$@"; then
            # A stalled synchronous-RPC worker (or its failed retry) aborts the
            # supervisor; the failed sequences are retried by rerunning with
            # --resume, so keep going through the remaining invocations.
            echo "INVOCATION FAILED: $type/$split (will be retried by rerunning with --resume)"
            FAILURES=$((FAILURES + 1))
        fi
    done
    exit "$FAILURES"
}

prepare() {
    "$PY" prepare_radar_ghost_dataset.py \
        --input "$OUT" --output "${OUT}_prepared" --split-mode official
}

case "$STAGE" in
    smoke) smoke ;;
    collect) collect_all ;;
    prepare) prepare ;;
    all) collect_all && prepare ;;
    *) echo "unknown stage '$STAGE' (smoke|collect|prepare|all)"; exit 2 ;;
esac
