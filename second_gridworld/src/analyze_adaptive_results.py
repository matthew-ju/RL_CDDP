#!/usr/bin/env python3
"""
Analyzes whether CDDP adaptive_v1 exhibits intended adaptive safety behavior.

Expected signature:
  tight clock → high violations (agent cuts near obstacles to meet budget)
  loose clock → near-zero violations (agent goes around safely)
  Both clocks → high success rates

Exit codes:
  0 = ADAPTIVE_GOOD   — behavior matches intent, competitive with baselines
  1 = ADAPTIVE_BAD    — behavior wrong or underperforming, needs Opus diagnosis
"""
import json, sys, re, argparse
import numpy as np
from pathlib import Path

# ── Thresholds for "intended adaptive behavior" ───────────────────────────────
# Tight clock (≤600 steps)
TIGHT_SUCCESS_MIN      = 0.40   # must beat CPO (40%) to justify the framework
TIGHT_VIOLATIONS_MIN   = 3.0    # must show SOME violations → adaptation is working
                                 # if zero violations on tight → actor still too cautious

# Loose clock (>600 steps)
LOOSE_SUCCESS_MIN      = 0.80   # must stay high
LOOSE_VIOLATIONS_MAX   = 4.0    # must stay low → agent goes around when possible

# Adaptive signature: tight must have meaningfully more violations than loose
# This ratio proves the agent is CHOOSING to violate on tight, not randomly
VIOLATION_RATIO_MIN    = 4.0    # tight_viol / (loose_viol + 0.5) must be ≥ this

# Baseline comparisons (from existing eval logs)
PPO_LAG_TIGHT_SUCCESS  = 0.50   # PPO-Lagrangian tight clock success (our target to beat)
STRICT_LOOSE_VIOLATIONS= 0.0    # Strict model loose violations (our target to match)
# ─────────────────────────────────────────────────────────────────────────────

def load_eval(path):
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("episodes", [])

def split_by_clock(episodes):
    tight, loose = [], []
    for ep in episodes:
        ms = ep.get("max_steps") or ep.get("episode_max_steps")
        if ms is None:
            continue
        bucket = tight if float(ms) <= 600 else loose
        viols = ep.get("total_violations", 0)
        if isinstance(viols, list):
            viols = len([v for v in viols if v])
        bucket.append({
            "success":    ep.get("success", False),
            "violations": float(viols) if viols is not None else 0.0,
            "path_eff":   ep.get("path_efficiency"),
            "steps":      ep.get("total_steps"),
            "max_steps":  float(ms),
        })
    return tight, loose

def summarize(eps):
    if not eps:
        return None
    path_effs = [e["path_eff"] for e in eps if e["path_eff"] is not None]
    return {
        "n":                len(eps),
        "success":          np.mean([e["success"] for e in eps]),
        "avg_viol":         np.mean([e["violations"] for e in eps]),
        "pct_nonzero_viol": np.mean([e["violations"] > 0 for e in eps]),
        "avg_path_eff":     np.mean(path_effs) if path_effs else None,
    }

def parse_training_log(run_name):
    log = Path(f"results/cddp/{run_name}/train.log")
    if not log.exists():
        return {}
    text = log.read_text()
    p_cats = [float(m) for m in re.findall(r'mean_P_cat=([\d.]+)', text)]
    best_scores = [(int(ep), float(sc)) for ep, sc in
                   re.findall(r'Ep (\d+)/\d+.*?BestScore=\+([\d.]+)', text)]
    result = {}
    if p_cats:
        recent = p_cats[-10:]
        result["zdeon_mean_p_cat"] = float(np.mean(recent))
        result["zdeon_healthy"]    = 0.20 < result["zdeon_mean_p_cat"] < 0.85
    if len(best_scores) >= 2:
        last_ep, last_sc = best_scores[-1]
        window = [sc for ep, sc in best_scores if ep >= last_ep - 20000]
        result["plateaued"]   = bool((max(window) - min(window) < 0.005)) if window else False
        result["best_score"]  = last_sc
        result["best_ep"]     = last_ep
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="adaptive_v1")
    args = ap.parse_args()

    eval_path = Path(f"results/cddp/{args.run_name}/cddp_eval_log.json")
    if not eval_path.exists():
        print(f"[analyze] ERROR: {eval_path} not found")
        sys.exit(1)

    episodes         = load_eval(eval_path)
    tight_eps, loose_eps = split_by_clock(episodes)
    tight = summarize(tight_eps)
    loose = summarize(loose_eps)
    train = parse_training_log(args.run_name)

    print("\n" + "="*65)
    print(f"  ADAPTIVE BEHAVIOR ANALYSIS — {args.run_name}")
    print("="*65)

    def row(label, val, target, good_dir="high"):
        if val is None:
            status = "N/A"
        elif good_dir == "high":
            status = "✓" if val >= target else "✗"
        else:
            status = "✓" if val <= target else "✗"
        tgt_str = f"(target {'≥' if good_dir == 'high' else '≤'}{target})"
        val_str = str(round(val, 3)) if val is not None else "N/A"
        print(f"  {status} {label:<38} {val_str:>8}  {tgt_str}")

    print(f"\n  TIGHT CLOCK ({tight['n'] if tight else 0} seeds, ≤600 steps):")
    if tight:
        row("Success rate",          tight["success"],          TIGHT_SUCCESS_MIN)
        row("Avg violations/ep",     tight["avg_viol"],         TIGHT_VIOLATIONS_MIN)
        row("% eps with violations", tight["pct_nonzero_viol"], 0.40)
    else:
        print("  No tight-clock episodes found in eval log")

    print(f"\n  LOOSE CLOCK ({loose['n'] if loose else 0} seeds, >600 steps):")
    if loose:
        row("Success rate",      loose["success"],  LOOSE_SUCCESS_MIN)
        row("Avg violations/ep", loose["avg_viol"], LOOSE_VIOLATIONS_MAX, "low")

    print(f"\n  ADAPTIVE SIGNATURE:")
    if tight and loose:
        ratio = tight["avg_viol"] / (loose["avg_viol"] + 0.5)
        row("Violation ratio (tight/loose)", ratio, VIOLATION_RATIO_MIN)
        print(f"    → Tight avg violations: {tight['avg_viol']:.2f}  "
              f"Loose avg violations: {loose['avg_viol']:.2f}")

    print(f"\n  TRAINING HEALTH:")
    if train.get("zdeon_mean_p_cat") is not None:
        zdeon_status = "✓" if train["zdeon_healthy"] else "✗"
        print(f"  {zdeon_status} Z_deon mean_P_cat: {train['zdeon_mean_p_cat']:.3f}  "
              f"(healthy: 0.20–0.85)")
    if train.get("plateaued"):
        print(f"  ✗ Training PLATEAUED (BestScore stuck near {train.get('best_score','?')} "
              f"after ep {train.get('best_ep','?')})")
    else:
        print(f"  ✓ Training did not plateau (BestScore still improving)")

    print("="*65)

    # ── Verdict ───────────────────────────────────────────────────────────────
    failures = []

    if tight and tight["avg_viol"] < TIGHT_VIOLATIONS_MIN:
        failures.append({
            "code": "NO_TIGHT_VIOLATIONS",
            "detail": (f"Tight-clock avg violations={tight['avg_viol']:.2f} < {TIGHT_VIOLATIONS_MIN}. "
                       f"Actor still avoiding obstacles even when CBF is relaxed. "
                       f"Obstacle aversion is still baked into the policy.")
        })
    if tight and tight["success"] < TIGHT_SUCCESS_MIN:
        failures.append({
            "code": "LOW_TIGHT_SUCCESS",
            "detail": (f"Tight-clock success={tight['success']*100:.0f}% < {TIGHT_SUCCESS_MIN*100:.0f}%. "
                       f"Not competitive with PPO-Lag ({PPO_LAG_TIGHT_SUCCESS*100:.0f}%).")
        })
    if loose and loose["avg_viol"] > LOOSE_VIOLATIONS_MAX:
        failures.append({
            "code": "HIGH_LOOSE_VIOLATIONS",
            "detail": (f"Loose-clock avg violations={loose['avg_viol']:.2f} > {LOOSE_VIOLATIONS_MAX}. "
                       f"Agent is violating even when it has time to go around. "
                       f"Stakes are not calibrating down on loose clocks.")
        })
    if loose and loose["success"] < LOOSE_SUCCESS_MIN:
        failures.append({
            "code": "LOW_LOOSE_SUCCESS",
            "detail": f"Loose-clock success={loose['success']*100:.0f}% < {LOOSE_SUCCESS_MIN*100:.0f}%."
        })
    if tight and loose:
        ratio = tight["avg_viol"] / (loose["avg_viol"] + 0.5)
        if ratio < VIOLATION_RATIO_MIN:
            failures.append({
                "code": "NO_ADAPTIVE_SIGNATURE",
                "detail": (f"Violation ratio={ratio:.2f} < {VIOLATION_RATIO_MIN}. "
                           f"Agent is not adapting between clock types — "
                           f"violations similar across tight and loose.")
            })
    if train.get("plateaued"):
        failures.append({
            "code": "TRAINING_PLATEAUED",
            "detail": "BestScore stopped improving. Policy may have hit architectural ceiling."
        })

    # Save diagnosis
    diag = {
        "run_name": args.run_name,
        "verdict": "BAD" if failures else "GOOD",
        "failures": failures,
        "tight": tight,
        "loose": loose,
        "training": train,
        "thresholds": {
            "tight_success_min":    TIGHT_SUCCESS_MIN,
            "tight_violations_min": TIGHT_VIOLATIONS_MIN,
            "loose_success_min":    LOOSE_SUCCESS_MIN,
            "loose_violations_max": LOOSE_VIOLATIONS_MAX,
            "violation_ratio_min":  VIOLATION_RATIO_MIN,
        }
    }
    out_path = Path(f"results/cddp/{args.run_name}/adaptive_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2))

    if failures:
        print(f"\n  VERDICT: BAD — {len(failures)} issue(s) detected")
        for f in failures:
            print(f"    ✗ {f['code']}: {f['detail']}")
        print(f"\n  Diagnosis saved → {out_path}")
        print("="*65 + "\n")
        sys.exit(1)
    else:
        print(f"\n  VERDICT: GOOD — adaptive behavior confirmed")
        print(f"  Diagnosis saved → {out_path}")
        print("="*65 + "\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
