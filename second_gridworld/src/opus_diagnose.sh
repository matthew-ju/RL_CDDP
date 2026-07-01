#!/bin/bash
# Calls Claude Opus to diagnose a failed adaptive CDDP run and generate a fix.
# Opus produces a structured diagnosis + Claude Code fix prompt.
# Sonnet implements the fix prompt, then a new training run is launched.
#
# Usage: bash src/opus_diagnose.sh <run_name> <attempt_number>
# Example: bash src/opus_diagnose.sh adaptive_v1 1

set -euo pipefail
cd "$(dirname "$0")/.."   # back to second_gridworld/

RUN_NAME="${1:-adaptive_v1}"
ATTEMPT="${2:-1}"
NEXT_RUN="adaptive_v$((ATTEMPT + 1))"
DIAG_FILE="results/cddp/${RUN_NAME}/adaptive_diagnosis.json"
TRAIN_LOG="results/cddp/${RUN_NAME}/train.log"
COMP_FILE="results/cddp/${RUN_NAME}/baseline_comparison.json"

echo "[opus] =============================================="
echo "[opus] Diagnosing: ${RUN_NAME}  →  next: ${NEXT_RUN}"
echo "[opus] =============================================="
echo "[opus] Gathering evidence..."

# Build the evidence bundle
DIAG=$(cat "${DIAG_FILE}" 2>/dev/null || echo "{}")
COMP=$(cat "${COMP_FILE}" 2>/dev/null || echo "[]")

# Extract last 100 lines of training log (Z_deon diagnostics + final metrics)
TRAIN_TAIL=$(tail -100 "${TRAIN_LOG}" 2>/dev/null || echo "unavailable")

# Extract key hyperparameters from training script
HPARAMS=$(python3 -c "
import re
src = open('src/train_sandbox_cddp.py').read()
keys = ['GAMMA_CBF', 'P_REF', 'TAU_CDDP', 'ALPHA_CDDP', 'TIMER_CURRICULUM',
        'N_EPISODES', 'PRETRAIN_EPISODES', 'OU_SIGMA', 'OU_SIGMA_MIN',
        'BATCH_SIZE', 'REPLAY_CAP', 'TRAIN_EVERY', 'GRADIENT_STEPS',
        'TIGHT_CLOCK_THRESHOLD', 'DEON_HIDDEN_DIM', 'DEON_HIDDEN_LAYERS',
        'MODEL_HIDDEN_DIM']
for k in keys:
    m = re.search(rf'^{k}\s*=\s*(.+)$', src, re.MULTILINE)
    if m: print(f'{k} = {m.group(1).strip()}')
" 2>/dev/null || echo "could not extract hyperparameters")

# Check if obstacle penalties were actually removed from compute_reward
OBS_PENALTY_PRESENT=$(python3 -c "
src = open('src/train_sandbox_cddp.py').read()
import re
in_reward = re.search(r'def compute_reward.*?(?=\ndef )', src, re.DOTALL)
if in_reward and 'obstacle_layer_penalty' in in_reward.group():
    print('YES — obstacle penalties still present in compute_reward()')
else:
    print('NO — obstacle penalties correctly removed from compute_reward()')
" 2>/dev/null || echo "could not check")

# Extract Z_deon diagnostics from training log
ZDEON_DIAG=$(grep -E "Z_deon diag|mean_P_cat" "${TRAIN_LOG}" 2>/dev/null | tail -20 || echo "no diagnostics found")

echo "[opus] Building diagnosis prompt..."

PROMPT="You are diagnosing a failed reinforcement learning training run.
Read all evidence carefully before proposing anything.

== FRAMEWORK OVERVIEW ==
CDDP (Constrained Deontic Duality with Predictive control) is a constrained RL framework
with one actor and three critics:
- Z_free (utilitarian): standard Q-value critic, trains the actor
- Z_free_2: TD3 twin critic
- Z_deon (deontic): binary classifier predicting P(failure under strict CBF compliance)

The CBF (Control Barrier Function) projects actions to avoid obstacles.
stakes = clip(Z_deon(s) / P_REF, 0, 1) controls how much the CBF relaxes:
  stakes=0 → strict CBF (agent must go around obstacles)
  stakes=1 → relaxed CBF (agent can cut near obstacles)

INTENDED BEHAVIOR:
- Tight clock (≤600 steps): Z_deon predicts high failure under strict CBF → stakes=1
  → CBF relaxes → actor cuts near obstacles → succeeds with some violations
- Loose clock (>600 steps): Z_deon predicts low failure under strict CBF → stakes=0
  → strict CBF → actor goes around → succeeds with near-zero violations

The most recent training run (${RUN_NAME}) removed obstacle penalties from the
reward function so the actor only learns to go toward the goal, letting the CBF
be the sole safety enforcer. This was meant to allow the actor to naturally cut
near obstacles when the CBF relaxes. A compression curriculum gradually tightens
the episode clock budget from [2000,2500] down to [420,2500] over training.
Transitions are split into tight_util_buffer (≤800 steps) and loose_util_buffer
(>800 steps), sampled equally so loose experiences don't dominate gradients.

== TRAINING HYPERPARAMETERS ==
${HPARAMS}

== OBSTACLE PENALTY STATUS ==
${OBS_PENALTY_PRESENT}

== ADAPTIVE BEHAVIOR DIAGNOSIS ==
${DIAG}

== BASELINE COMPARISON ==
${COMP}

== Z_DEON DIAGNOSTICS FROM TRAINING LOG ==
${ZDEON_DIAG}

== TRAINING LOG TAIL (last 100 lines) ==
${TRAIN_TAIL}

== YOUR TASK ==
Diagnose why CDDP is not exhibiting the intended adaptive behavior, then produce
a complete, executable Claude Code prompt to fix it.

REQUIREMENTS FOR YOUR DIAGNOSIS:
1. State your PRIMARY hypothesis for why it failed, with a confidence level (high/medium/low)
2. List at least 2 ALTERNATIVE hypotheses you considered but ranked lower, and why
3. For each hypothesis, cite specific evidence from the data above
4. Explicitly flag any part of your diagnosis where you may be HALLUCINATING
   or making assumptions without evidence — use the tag [UNCERTAIN: reason]
5. Do NOT be overconfident — if the evidence is ambiguous, say so

REQUIREMENTS FOR YOUR PROPOSED SOLUTIONS:
Propose 2-3 distinct solutions. For EACH solution:
- Name and describe the mechanism
- State what specific code changes are needed (file, function, change)
- State the expected benefit
- State the NICK: what problem this solution might introduce or worsen
  (e.g., 'fixes tight-clock violations but may destabilize loose-clock safety')
- Rate your confidence it will work (high/medium/low)

Then choose the SINGLE best solution and explain:
- Why it is more robust than the alternatives
- What you will monitor in training to confirm it is working
- What early warning signs would indicate it is causing a new problem (a nick)

REQUIREMENTS FOR YOUR OUTPUT:
Output a complete, self-contained Claude Code prompt that:
1. Reads the relevant source files before touching anything
2. Implements ONLY the chosen solution — no other changes
3. Does not touch GAMMA_CBF, the weighted-BCE Z_deon loss, or terminal rewards
4. Sets run name to '${NEXT_RUN}'
5. Launches training for 60,000 episodes via:
   bash run_v2.sh ${NEXT_RUN} --timer-curriculum compression --seed-episodes 3000 --pretrain-episodes 10000 --episodes 60000
6. After the training command, includes this monitoring note as a comment:
   # Monitor: [specific Z_deon diagnostic or metric that confirms the fix is working]
   # Early warning nick: [specific metric that would indicate a new problem]

Format your output EXACTLY as:
---DIAGNOSIS---
[your diagnosis here]
---SOLUTION CHOSEN---
[chosen solution with nick analysis]
---CLAUDE CODE PROMPT---
[complete prompt starting with '# CDDP Fix ...' that Claude Code can execute]
---END---
"

echo "[opus] Calling Claude Opus (claude-opus-4-6) for diagnosis..."
mkdir -p "results/cddp/${NEXT_RUN}"

# Call Opus and capture full output
claude --model claude-opus-4-6 --print "${PROMPT}" \
    > "results/cddp/${RUN_NAME}/opus_diagnosis.txt" 2>&1

echo "[opus] Opus response received. Extracting fix prompt..."

# Extract the Claude Code prompt from Opus output
python3 - <<'PYEOF'
import sys, os
run = os.environ.get('RUN_NAME', 'adaptive_v1')
text = open(f'results/cddp/{run}/opus_diagnosis.txt').read()
start = text.find('---CLAUDE CODE PROMPT---')
end   = text.find('---END---')
if start == -1 or end == -1:
    print(f'[opus] ERROR: could not find prompt boundaries in Opus output')
    print(f'[opus] Sections found: {[s for s in ["---DIAGNOSIS---","---SOLUTION CHOSEN---","---CLAUDE CODE PROMPT---","---END---"] if s in text]}')
    print(f'[opus] Full output saved to results/cddp/{run}/opus_diagnosis.txt')
    sys.exit(1)
prompt = text[start + len('---CLAUDE CODE PROMPT---'):end].strip()
open(f'results/cddp/{run}/opus_fix_prompt.md', 'w').write(prompt)
print(f'[opus] Fix prompt extracted ({len(prompt)} chars) → results/cddp/{run}/opus_fix_prompt.md')

# Print the diagnosis section for the log
diag_start = text.find('---DIAGNOSIS---')
sol_start  = text.find('---SOLUTION CHOSEN---')
prompt_start = text.find('---CLAUDE CODE PROMPT---')
if diag_start != -1 and sol_start != -1:
    print()
    print('=' * 60)
    print(text[diag_start:prompt_start if prompt_start != -1 else sol_start + 200])
    print('=' * 60)
PYEOF
export RUN_NAME  # needed for the heredoc python above

echo ""
echo "[opus] Running fix prompt via Claude Code (Sonnet implements Opus plan)..."
claude --permission-mode auto \
    < "results/cddp/${RUN_NAME}/opus_fix_prompt.md" \
    2>&1 | tee "results/cddp/${RUN_NAME}/implementation.log"

echo ""
echo "[opus] Implementation complete. Launching ${NEXT_RUN} training (60k episodes)..."
bash run_v2.sh "${NEXT_RUN}" \
    --timer-curriculum compression \
    --seed-episodes 3000 \
    --pretrain-episodes 10000 \
    --episodes 60000 \
    2>&1 | tee -a "results/cddp/${NEXT_RUN}/train.log"
