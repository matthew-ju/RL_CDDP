#!/bin/bash
# ============================================================
#  CDDP Autonomous Training Loop
#  - Crash-resilient: auto-restarts on OOM or any crash
#  - Post-training: auto-analyzes results
#  - Good results  → apply efficiency fixes, done
#  - Bad results   → diagnose, fix hyperparams, retrain
#  - Loops until results are good OR max attempts reached
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

MAX_LOOP_ATTEMPTS=5       # max full train→analyze→fix cycles
MAX_CRASH_RESTARTS=999    # restarts per training run (OOM etc.)
RUN_NAME="${1:-long_clock_only}"
ATTEMPT=0

log() { echo "[loop $(date '+%H:%M:%S')] $*"; }

# ── Function: run one training session with crash resilience ─────────────────
run_training() {
    local name="$1"
    local resume_flag="$2"   # "" or "--resume"
    local crash_count=0
    local first=true

    mkdir -p "results/cddp/${name}"
    local LOG="results/cddp/${name}/train.log"

    log "Starting training: ${name} (resume=${resume_flag:-no})"

    while [ $crash_count -lt $MAX_CRASH_RESTARTS ]; do
        if [ "$first" = true ]; then
            first=false
            caffeinate -i python3 src/train_sandbox_cddp.py \
                --run-name "${name}" ${resume_flag} \
                2>&1 | tee -a "${LOG}"
            EXIT="${PIPESTATUS[0]}"
        else
            crash_count=$((crash_count + 1))
            log "Crash #${crash_count} (exit ${EXIT}). Waiting 15s then resuming..."
            sleep 15
            caffeinate -i python3 src/train_sandbox_cddp.py \
                --run-name "${name}" --resume \
                2>&1 | tee -a "${LOG}"
            EXIT="${PIPESTATUS[0]}"
        fi

        if [ "$EXIT" -eq 0 ]; then
            log "Training complete: ${name}"
            return 0
        fi
    done

    log "ERROR: exceeded ${MAX_CRASH_RESTARTS} crash restarts for ${name}"
    return 1
}

# ── Function: analyze results, return verdict ─────────────────────────────────
analyze() {
    local name="$1"
    log "Analyzing results for: ${name}"
    python3 src/analyze_results.py --run-name "${name}"
    return $?
}

# ── Function: apply efficiency fixes ─────────────────────────────────────────
apply_efficiency() {
    log "Applying efficiency fixes..."
    python3 src/apply_efficiency_fixes.py
    log "Efficiency fixes applied."
}

# ── Function: diagnose + fix, return next run name ───────────────────────────
diagnose_and_fix() {
    local name="$1"
    local attempt="$2"
    log "Diagnosing and fixing: ${name} (attempt ${attempt})"
    NEXT_NAME=$(python3 src/diagnose_and_fix.py \
                    --run-name "${name}" \
                    --attempt "${attempt}" \
                | tail -1)
    echo "${NEXT_NAME}"
}

# ── Check if eval log already exists (training may have finished before crash) ─
if [ -f "results/cddp/${RUN_NAME}/cddp_eval_log.json" ]; then
    log "Eval log already exists for ${RUN_NAME} — skipping training, going to analysis"
else
    run_training "${RUN_NAME}" ""
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
CURRENT_RUN="${RUN_NAME}"

while [ $ATTEMPT -lt $MAX_LOOP_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "=== LOOP ATTEMPT ${ATTEMPT}/${MAX_LOOP_ATTEMPTS} | run=${CURRENT_RUN} ==="

    # Analyze results (returns non-zero for BAD/MARGINAL/NO_DATA — capture without set -e killing us)
    set +e
    analyze "${CURRENT_RUN}"
    VERDICT=$?
    set -e

    case $VERDICT in
        0)  # GOOD
            log "Results are GOOD. Applying efficiency fixes. Done."
            apply_efficiency
            log "=== ALL DONE — ${CURRENT_RUN} is good and efficient ==="
            exit 0
            ;;
        2)  # MARGINAL
            log "Results are MARGINAL. Applying efficiency fixes. Done."
            apply_efficiency
            log "=== ALL DONE — ${CURRENT_RUN} is marginal but accepted ==="
            exit 0
            ;;
        1)  # BAD — diagnose, fix, retrain
            log "Results are BAD. Diagnosing..."
            NEXT_RUN=$(diagnose_and_fix "${CURRENT_RUN}" "${ATTEMPT}")
            log "Next run name: ${NEXT_RUN}"
            run_training "${NEXT_RUN}" ""
            CURRENT_RUN="${NEXT_RUN}"
            ;;
        3)  # NO_DATA — training may not have written eval log
            log "WARNING: no eval data found for ${CURRENT_RUN}. Rerunning training..."
            run_training "${CURRENT_RUN}" "--resume"
            ;;
        *)
            log "Unexpected verdict code ${VERDICT}. Stopping."
            exit 1
            ;;
    esac
done

log "Reached max loop attempts (${MAX_LOOP_ATTEMPTS}) without GOOD result."
log "Last run: ${CURRENT_RUN}. Applying efficiency fixes anyway."
apply_efficiency
exit 1
