#!/usr/bin/env bash
#
# run_all.sh -- run the whole Python-vs-MATLAB radar comparison end to end.
#
#   ./run_all.sh              # full run
#   ./run_all.sh --seed 7     # different python-side seed
#   SKIP_MATLAB=1 ./run_all.sh  # truth + python + report only (no MATLAB needed)
#
# Stages:
#   1. generate_truth_scenario.py  analytic ground truth (deterministic)
#   2. run_python_radar.py         this repo's RealisticRadarModel
#   3. run_matlab_radar.m          Radar Toolbox reference, via -batch
#   4. compare_models.py           report + plots
#
# MATLAB must be invoked through the FLEXlm shim on rolling-release distros, so
# this prefers the ~/bin/matlab wrapper and falls back to preloading the shim
# directly.  See README.md Step 1.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

SHIM="$HOME/.local/lib/matlab_flexlm_cpuid_fix.so"

# Resolve a MATLAB launcher that survives the FLEXlm CPUID segfault.
matlab_cmd() {
    if [[ -n "${MATLAB_BIN:-}" ]]; then
        printf '%s' "$MATLAB_BIN"
    elif [[ -x "$HOME/bin/matlab" ]]; then
        printf '%s' "$HOME/bin/matlab"
    elif command -v matlab >/dev/null 2>&1 && [[ -f "$SHIM" ]]; then
        printf 'env LD_PRELOAD=%s matlab' "$SHIM"
    elif command -v matlab >/dev/null 2>&1; then
        printf 'matlab'
    else
        return 1
    fi
}

step() { printf '\n=== %s ===\n' "$1"; }

cd "$HERE"

step "1/4 analytic truth"
python3 generate_truth_scenario.py

step "2/4 python radar model (seed ${SEED})"
python3 run_python_radar.py --seed "$SEED"

step "3/4 MATLAB radar reference"
if [[ -n "${SKIP_MATLAB:-}" ]]; then
    echo "SKIP_MATLAB set; reusing existing matlab_detections.csv"
    if [[ ! -f matlab_detections.csv ]]; then
        echo "error: SKIP_MATLAB set but matlab_detections.csv does not exist" >&2
        exit 1
    fi
else
    if ! MATLAB="$(matlab_cmd)"; then
        echo "error: no MATLAB found. Install it (README.md Step 1) or set" >&2
        echo "       MATLAB_BIN=/path/to/matlab, or SKIP_MATLAB=1 to skip." >&2
        exit 1
    fi
    echo "using: ${MATLAB}"
    # shellcheck disable=SC2086
    $MATLAB -batch "run('matlab/run_matlab_radar.m')"
fi

step "4/4 comparison report"
python3 compare_models.py

printf '\n=== headline ===\n'
sed -n '/^## Headline/,/^Plots:/p' comparison_report.md