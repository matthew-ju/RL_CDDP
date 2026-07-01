#!/bin/bash
# Crash-resilient CDDP training runner.
# Usage: bash run_v2.sh <run_name> [--timer-curriculum <name>] [any other train args]
# Example: bash run_v2.sh adaptive_v1 --timer-curriculum compression
# Example: bash run_v2.sh long_clock_only --timer-curriculum long_only
#
# All args after run_name are forwarded directly to train_sandbox_cddp.py.
# On any crash (including OOM), waits 15s and resumes from latest checkpoint.
# Loops up to 999 times — effectively runs forever until training completes.

cd "$(dirname "$0")"

RUN_NAME="${1:-long_clock_only}"
shift  # consume run_name; remaining args forwarded to training script
EXTRA_ARGS="$@"

SCRIPT="src/train_sandbox_cddp.py"
LOG_FILE="results/cddp/${RUN_NAME}/train.log"

mkdir -p "results/cddp/${RUN_NAME}"
echo "[runner] Run: ${RUN_NAME}"
echo "[runner] Extra args: ${EXTRA_ARGS:-none}"
echo "[runner] Log: ${LOG_FILE}"
echo "[runner] Started at $(date)"

FIRST_RUN=true
ATTEMPT=0
EXIT_CODE=0

# Auto-detect existing checkpoints — if they exist, treat this as a resume
# even on the very first launch (handles kill-and-relaunch without losing progress)
CHECKPOINT_DIR="results/cddp/${RUN_NAME}/checkpoints"
if [ -d "${CHECKPOINT_DIR}" ] && ls "${CHECKPOINT_DIR}"/ep* >/dev/null 2>&1; then
    echo "[runner] Found existing checkpoints — will resume automatically on first attempt"
    FIRST_RUN=false
fi

while [ $ATTEMPT -lt 999 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[runner] Attempt ${ATTEMPT} at $(date)"

    if [ "$FIRST_RUN" = true ]; then
        FIRST_RUN=false
        caffeinate -i python3 "${SCRIPT}" \
            --run-name "${RUN_NAME}" \
            ${EXTRA_ARGS} \
            2>&1 | tee -a "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        echo "[runner] Crash detected (exit ${EXIT_CODE}). Resuming in 15s..."
        sleep 15
        caffeinate -i python3 "${SCRIPT}" \
            --run-name "${RUN_NAME}" \
            --resume auto \
            ${EXTRA_ARGS} \
            2>&1 | tee -a "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[runner] Training completed successfully at $(date)."
        echo "[runner] Launching post-training orchestrator..."
        bash post_training_orchestrator.sh "${RUN_NAME}" 1
        exit 0
    fi

    echo "[runner] Non-zero exit (${EXIT_CODE}) — crash assumed, will restart."
done

echo "[runner] Reached 999 attempts. Stopping."
exit 1
