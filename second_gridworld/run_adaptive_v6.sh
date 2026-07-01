#!/usr/bin/env bash
# run_adaptive_v6.sh — trains adaptive_v6 with caffeinate and crash-recovery loop.
# Restarts automatically on any crash (OOM, MPS fault, etc.) using --resume auto.
# Exits only when training completes successfully (exit 0).

RUN_NAME="adaptive_v6"
EPISODES=40000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/results/cddp/${RUN_NAME}/train.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo ""
echo "=========================================="
echo "  adaptive_v6  target=${EPISODES} episodes"
echo "  log -> ${LOG_FILE}"
echo "=========================================="

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "=== Attempt ${attempt} — $(date) ===" | tee -a "$LOG_FILE"

    # Disable -e so a non-zero python exit code doesn't kill this script.
    set +e
    caffeinate -i python3 -u "${SCRIPT_DIR}/src/train_adaptive_v6.py" \
        --run-name   "$RUN_NAME"  \
        --episodes   "$EPISODES"  \
        --resume     auto         \
        2>&1 | tee -a "$LOG_FILE"
    py_exit=${PIPESTATUS[0]}
    set -e

    if [ "$py_exit" -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "=== Training finished successfully (exit 0) ===" | tee -a "$LOG_FILE"
        break
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "=== Process exited with code ${py_exit} — restarting in 10s ===" | tee -a "$LOG_FILE"
    sleep 10
done
