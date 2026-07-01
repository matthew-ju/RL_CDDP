#!/bin/bash
# Crash-resilient launcher for adaptive_v1 (compression curriculum, no obstacle reward penalties).
# Usage: bash run_adaptive_v1.sh

set -e
cd "$(dirname "$0")"

RUN="adaptive_v1"
SCRIPT="src/train_sandbox_cddp.py"
LOG="results/cddp/${RUN}/train.log"

mkdir -p "results/cddp/${RUN}"
echo "[runner] Starting: ${RUN} at $(date)"
echo "[runner] Log: ${LOG}"

FIRST_RUN=true
ATTEMPT=0

while [ $ATTEMPT -lt 999 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[runner] Attempt ${ATTEMPT} at $(date)"

    if [ "$FIRST_RUN" = true ]; then
        FIRST_RUN=false
        caffeinate -i python3 "${SCRIPT}" \
            --run-name "${RUN}" \
            --seed-episodes 3000 \
            --pretrain-episodes 10000 \
            --episodes 75000 \
            --timer-curriculum compression \
            2>&1 | tee -a "${LOG}"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        echo "[runner] Crash (exit ${EXIT_CODE}). Resuming in 15s..."
        sleep 15
        caffeinate -i python3 "${SCRIPT}" \
            --run-name "${RUN}" \
            --seed-episodes 3000 \
            --pretrain-episodes 10000 \
            --episodes 75000 \
            --timer-curriculum compression \
            --resume auto \
            2>&1 | tee -a "${LOG}"
        EXIT_CODE=${PIPESTATUS[0]}
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[runner] Completed successfully at $(date)."
        exit 0
    fi
done
