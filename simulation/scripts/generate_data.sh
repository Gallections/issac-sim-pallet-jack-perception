#!/usr/bin/env bash
set -euo pipefail

# Set ISAAC_SIM_PATH when Isaac Sim is installed somewhere other than this default.
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/c/Issac-Sim}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/palletjack_sdg/standalone_palletjack_sdg.py"
OUTPUT_ROOT="${REPO_ROOT}/data/raw/experiment-02"

if [[ ! -f "${ISAAC_SIM_PATH}/python.sh" ]]; then
    echo "ERROR: Cannot find Isaac Sim Linux Python launcher: ${ISAAC_SIM_PATH}/python.sh" >&2
    echo "Set ISAAC_SIM_PATH to the Isaac Sim installation directory, or use generate_data.bat on Windows." >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: Cannot find the SDG Python script: ${SCRIPT_PATH}" >&2
    exit 1
fi

ISAAC_PYTHON="${ISAAC_SIM_PATH}/python.sh"
OUTPUT_WAREHOUSE="${OUTPUT_ROOT}/distractors_warehouse"
OUTPUT_ADDITIONAL="${OUTPUT_ROOT}/distractors_additional"
OUTPUT_NO_DISTRACTORS="${OUTPUT_ROOT}/no_distractors"

echo "Starting synthetic data generation for Experiment 2..."
"${ISAAC_PYTHON}" "${SCRIPT_PATH}" --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir "${OUTPUT_WAREHOUSE}"
"${ISAAC_PYTHON}" "${SCRIPT_PATH}" --height 544 --width 960 --num_frames 2000 --distractors additional --data_dir "${OUTPUT_ADDITIONAL}"
"${ISAAC_PYTHON}" "${SCRIPT_PATH}" --height 544 --width 960 --num_frames 1000 --distractors None --data_dir "${OUTPUT_NO_DISTRACTORS}"

echo "All three Experiment 2 datasets generated successfully."
