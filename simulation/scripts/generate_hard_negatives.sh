#!/usr/bin/env bash
set -euo pipefail

# Set ISAAC_SIM_PATH when Isaac Sim is installed elsewhere.
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/c/Issac-Sim}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/palletjack_sdg/standalone_palletjack_sdg.py"
OUTPUT_DIR="${REPO_ROOT}/data/raw/experiment-03/distractors_only"
ISAAC_PYTHON="${ISAAC_SIM_PATH}/python.sh"

if [[ ! -f "${ISAAC_PYTHON}" ]]; then
    echo "ERROR: Cannot find Isaac Sim Python launcher: ${ISAAC_PYTHON}" >&2
    echo "Set ISAAC_SIM_PATH to the Isaac Sim installation directory." >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: Cannot find the SDG Python script: ${SCRIPT_PATH}" >&2
    exit 1
fi

echo "Generating Experiment 3 target-free hard-negative data..."
"${ISAAC_PYTHON}" "${SCRIPT_PATH}" --height 544 --width 960 --num_frames 1000 --distractors-only --data_dir "${OUTPUT_DIR}"
echo "Hard-negative dataset generated at: ${OUTPUT_DIR}"
