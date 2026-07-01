#!/bin/bash
# Post-training orchestrator. Runs automatically after training completes.
# Chains: baseline comparison → adaptive behavior analysis → Opus diagnosis if BAD.
# Loops up to MAX_ATTEMPTS times autonomously.
#
# Usage: bash post_training_orchestrator.sh <run_name> <attempt_number>
# Example: bash post_training_orchestrator.sh adaptive_v1 1

set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME="${1:-adaptive_v1}"
ATTEMPT="${2:-1}"
MAX_ATTEMPTS=4

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
printf "║  POST-TRAINING ORCHESTRATOR                              ║\n"
printf "║  Run: %-20s  Attempt: %d/%d              ║\n" "${RUN_NAME}" "${ATTEMPT}" "${MAX_ATTEMPTS}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Baseline comparison (always runs, informational) ─────────────────
echo "[orchestrator] Step 1/2 — Baseline comparison..."
python3 src/compare_with_baselines.py --run-name "${RUN_NAME}" || true
echo ""

# ── Step 2: Adaptive behavior analysis ───────────────────────────────────────
echo "[orchestrator] Step 2/2 — Analyzing adaptive behavior..."
set +e
python3 src/analyze_adaptive_results.py --run-name "${RUN_NAME}"
VERDICT=$?
set -e

# ── GOOD: report and stop ─────────────────────────────────────────────────────
if [ $VERDICT -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  RESULT: ADAPTIVE BEHAVIOR CONFIRMED ✓                  ║"
    echo "║  CDDP is exhibiting intended adaptive safety behavior.   ║"
    echo "║  Waiting for your input on next steps.                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "[orchestrator] Reports available:"
    echo "  Behavior diagnosis: results/cddp/${RUN_NAME}/adaptive_diagnosis.json"
    echo "  Baseline comparison: results/cddp/${RUN_NAME}/baseline_comparison.json"
    echo ""
    echo "[orchestrator] Done. No further autonomous action taken."
    exit 0
fi

# ── BAD: check attempt limit ───────────────────────────────────────────────────
if [ "${ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  STOPPED: Reached max attempts (${MAX_ATTEMPTS})                    ║"
    echo "║  Human review needed.                                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "[orchestrator] Last run: ${RUN_NAME}"
    echo "[orchestrator] Diagnosis: results/cddp/${RUN_NAME}/adaptive_diagnosis.json"
    exit 1
fi

# ── BAD: call Opus for autonomous fix ─────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  VERDICT: BAD — Engaging Opus autonomous diagnosis      ║"
printf "║  Attempt %d/%d → will spawn adaptive_v%d              ║\n" \
    "${ATTEMPT}" "${MAX_ATTEMPTS}" "$((ATTEMPT + 1))"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "[orchestrator] Calling Opus diagnosis pipeline..."
bash src/opus_diagnose.sh "${RUN_NAME}" "${ATTEMPT}"
# Note: opus_diagnose.sh launches the next training run internally via run_v2.sh.
# That run_v2.sh invocation has the success hook wired to call this orchestrator.
# So when the next training run completes, the orchestrator loops automatically.

echo ""
echo "[orchestrator] Opus pipeline complete. Next run (adaptive_v$((ATTEMPT + 1))) launched."
echo "[orchestrator] This orchestrator's work for attempt ${ATTEMPT} is done."
echo "[orchestrator] The next run's completion will trigger attempt $((ATTEMPT + 1))."
exit 0
