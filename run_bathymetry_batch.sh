#!/usr/bin/env bash
#
# Run bathymetry_migrate.py for a batch of melt scenarios.
#
# Each case is "<gmsl> <antarctic_fraction>". Figures are not produced by default.
# The ETOPO resolution (arc-minutes: 2, 5 or 15) is taken from RESOLUTION.
#
# Usage: ./run_bathymetry_batch.sh
#        RESOLUTION=2 ./run_bathymetry_batch.sh
set -euo pipefail

# Line-buffer Python's stdout so progress appears in the log immediately when
# output is piped or redirected (e.g. nohup, tee) rather than sitting in an 8 KB
# buffer until the process exits.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the project virtualenv, fall back to whatever python3 is on PATH.
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

RESOLUTION="${RESOLUTION:-5}"

# Each entry: "<gmsl> <antarctic_fraction>"
cases=(
    "0 0"
    "6 0"
    "6 1"
    "13 0.442"
    "13 1"
    "25 0.710"
)

for case in "${cases[@]}"; do
    read -r gmsl frac <<< "${case}"
    echo "=================================================================="
    echo "Running: gmsl=${gmsl} m, antarctic_fraction=${frac}, resolution=${RESOLUTION} arcmin"
    echo "=================================================================="
    "${PYTHON}" "${SCRIPT_DIR}/bathymetry_migrate.py" \
        --gmsl "${gmsl}" \
        --antarctic-fraction "${frac}" \
        --resolution "${RESOLUTION}"
done

echo "All runs complete."
