#!/bin/bash
# Crash-resilient launcher for adaptive_v5.
#
# v5 changes over v4:
#   - Actor takes stakes as 6th input (gradient flows directly through stakes)
#   - Risk-aligned reward bonus: +alpha*stakes*obstacle_proximity
#   - Pretrain shortened to 2000 episodes (was 10000) to avoid Z_deon
#     collapsing to P_cat=1 before CDDP starts
#   - Bootstrap aggressive policy for first 2000 CDDP episodes to seed
#     tight_util_buffer with successful direct-path trajectories
#
# Usage: bash run_adaptive_v5.sh

set -e
cd "$(dirname "$0")"

RUN="adaptive_v5"
SCRIPT="src/train_adaptive_v5.py"
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
            --pretrain-episodes 2000 \
            --episodes 75000 \
            --timer-curriculum compression \
            --risk-align-alpha 0.05 \
            --bootstrap-cddp-episodes 2000 \
            2>&1 | tee -a "${LOG}"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        echo "[runner] Crash (exit ${EXIT_CODE}). Resuming in 15s..."
        sleep 15
        caffeinate -i python3 "${SCRIPT}" \
            --run-name "${RUN}" \
            --seed-episodes 3000 \
            --pretrain-episodes 2000 \
            --episodes 75000 \
            --timer-curriculum compression \
            --risk-align-alpha 0.05 \
            --bootstrap-cddp-episodes 2000 \
            --resume auto \
            2>&1 | tee -a "${LOG}"
        EXIT_CODE=${PIPESTATUS[0]}
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[runner] Completed successfully at $(date)."
        exit 0
    fi
done
