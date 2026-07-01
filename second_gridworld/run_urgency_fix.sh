#!/bin/bash
# Crash-resilient CDDP training runner for urgency_fix.
# Automatically resumes from latest checkpoint on any crash (including OOM).

set -e
cd "$(dirname "$0")"

RUN_NAME="urgency_fix"
SCRIPT="src/train_sandbox_cddp.py"
LOG_FILE="results/cddp/${RUN_NAME}/train.log"

mkdir -p "results/cddp/${RUN_NAME}"

echo "[runner] Starting training run: ${RUN_NAME}"
echo "[runner] Log: ${LOG_FILE}"

FIRST_RUN=true
ATTEMPT=0
MAX_ATTEMPTS=999

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[runner] Attempt ${ATTEMPT} at $(date)"

    if [ "$FIRST_RUN" = true ]; then
        FIRST_RUN=false
        caffeinate -i python3 "${SCRIPT}" \
            --seed-episodes 3000 --pretrain-episodes 10000 --episodes 50000 \
            --run-name "${RUN_NAME}" \
            2>&1 | tee -a "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        echo "[runner] Crash detected. Resuming from checkpoint..."
        caffeinate -i python3 "${SCRIPT}" \
            --seed-episodes 3000 --pretrain-episodes 10000 --episodes 50000 \
            --run-name "${RUN_NAME}" \
            --resume auto \
            2>&1 | tee -a "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[runner] Training completed successfully at $(date)."
        exit 0
    else
        echo "[runner] Process exited with code ${EXIT_CODE} at $(date)."
        echo "[runner] Waiting 10 seconds before restart..."
        sleep 10
    fi
done

echo "[runner] Reached max attempts (${MAX_ATTEMPTS}). Stopping."
exit 1
