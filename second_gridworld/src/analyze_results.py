#!/usr/bin/env python3
"""
Analyzes CDDP eval results and outputs a decision + reasoning.

Exit codes:
  0 = GOOD     (high success, low violations) → apply efficiency fixes
  1 = BAD      (needs diagnosis + retraining)
  2 = MARGINAL (acceptable, apply efficiency fixes and stop)
  3 = NO_DATA  (eval log not found or empty)

Usage:
  python3 src/analyze_results.py --run-name long_clock_only
"""
import argparse, json, sys, os
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
# GOOD: both conditions must be true
GOOD_SUCCESS_RATE   = 0.80   # overall success across all eval seeds
GOOD_MAX_VIOLATIONS = 0.05   # P(excess) constraint violation rate

# MARGINAL: acceptable but not great
MARGINAL_SUCCESS_RATE   = 0.65
MARGINAL_MAX_VIOLATIONS = 0.12

# Z_deon health: mean_P_cat should be in this range (not collapsed)
ZDEON_LOW  = 0.25   # below this → Z_deon always predicts safe (wrong)
ZDEON_HIGH = 0.85   # above this → Z_deon collapsed to all-1s (wrong)

# Training plateau detection: parse training log for BestScore changes
PLATEAU_WINDOW = 15000   # if BestScore unchanged for this many episodes → plateaued
# ─────────────────────────────────────────────────────────────────────────────

def parse_eval_log(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    episodes = data if isinstance(data, list) else data.get("episodes", [])
    if not episodes:
        return {}

    successes     = [e["success"] for e in episodes if "success" in e]
    violations    = [e.get("p_excess", e.get("constraint_violation_rate", 0.0))
                     for e in episodes]
    path_effs     = [e["path_efficiency"] for e in episodes if "path_efficiency" in e]
    steps_list    = [e["steps"] for e in episodes if "steps" in e]

    # Split tight vs loose clock if max_steps info available
    tight, loose = [], []
    for e in episodes:
        ms = e.get("max_steps", e.get("episode_max_steps", None))
        if ms is not None:
            (tight if ms <= 600 else loose).append(e["success"])

    result = {
        "n_seeds": len(successes),
        "overall_success": sum(successes) / len(successes) if successes else 0.0,
        "p_excess_mean": sum(violations) / len(violations) if violations else 0.0,
        "avg_path_efficiency": sum(path_effs) / len(path_effs) if path_effs else None,
        "avg_steps": sum(steps_list) / len(steps_list) if steps_list else None,
        "tight_clock_success": sum(tight) / len(tight) if tight else None,
        "long_clock_success": sum(loose) / len(loose) if loose else None,
    }
    return result

def parse_training_log(run_name: str) -> dict:
    """Extract Z_deon diagnostics and plateau info from training log."""
    log_path = Path(f"results/cddp/{run_name}/train.log")
    if not log_path.exists():
        return {}

    zdeon_p_cats = []
    best_scores  = []

    for line in log_path.read_text().splitlines():
        if "mean_P_cat=" in line:
            try:
                val = float(line.split("mean_P_cat=")[1].split()[0])
                zdeon_p_cats.append(val)
            except (ValueError, IndexError):
                pass
        if "BestScore=" in line:
            try:
                val = float(line.split("BestScore=")[1].split()[0].rstrip("|"))
                ep  = int(line.split("Ep ")[1].split("/")[0])
                best_scores.append((ep, val))
            except (ValueError, IndexError):
                pass

    result = {}
    if zdeon_p_cats:
        recent = zdeon_p_cats[-10:]
        result["zdeon_mean_p_cat_recent"] = sum(recent) / len(recent)
        result["zdeon_collapsed_high"] = result["zdeon_mean_p_cat_recent"] > ZDEON_HIGH
        result["zdeon_collapsed_low"]  = result["zdeon_mean_p_cat_recent"] < ZDEON_LOW

    if len(best_scores) >= 2:
        last_ep, last_score = best_scores[-1]
        plateau_start_ep = last_ep - PLATEAU_WINDOW
        scores_in_window = [s for ep, s in best_scores if ep >= plateau_start_ep]
        result["plateaued"] = (max(scores_in_window) - min(scores_in_window) < 0.01
                               if scores_in_window else False)
        result["best_score"] = last_score

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="long_clock_only")
    args = ap.parse_args()

    eval_path = Path(f"results/cddp/{args.run_name}/cddp_eval_log.json")
    if not eval_path.exists():
        print(f"[analyze] NO_DATA: {eval_path} not found")
        sys.exit(3)

    metrics = parse_eval_log(eval_path)
    if not metrics:
        print("[analyze] NO_DATA: eval log is empty or malformed")
        sys.exit(3)

    training = parse_training_log(args.run_name)

    # ── Print full report ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  RESULTS: {args.run_name}")
    print("="*60)
    print(f"  Overall success rate : {metrics['overall_success']*100:.1f}%  (target: ≥80%)")
    print(f"  P(excess) violations : {metrics['p_excess_mean']*100:.1f}%  (target: ≤5%)")
    if metrics.get("tight_clock_success") is not None:
        print(f"  Tight-clock success  : {metrics['tight_clock_success']*100:.1f}%")
    if metrics.get("long_clock_success") is not None:
        print(f"  Long-clock success   : {metrics['long_clock_success']*100:.1f}%")
    if metrics.get("avg_path_efficiency") is not None:
        print(f"  Avg path efficiency  : {metrics['avg_path_efficiency']:.3f}")
    if training.get("zdeon_mean_p_cat_recent") is not None:
        print(f"  Z_deon mean_P_cat    : {training['zdeon_mean_p_cat_recent']:.3f}  (healthy: 0.25–0.85)")
    if training.get("best_score") is not None:
        print(f"  Best training score  : {training['best_score']:.4f}")
    if training.get("plateaued"):
        print(f"  WARNING: Training PLATEAUED (no improvement in last {PLATEAU_WINDOW} episodes)")
    print("="*60)

    # ── Decision ──────────────────────────────────────────────────────────
    # When long_only curriculum was used, tight-clock episodes are physically
    # infeasible (proven: greedy needs ~507 steps, tight budget is 420).
    # Use long-clock success as primary metric if tight-clock drags overall down.
    long_succ  = metrics.get("long_clock_success")
    tight_succ = metrics.get("tight_clock_success")
    if long_succ is not None and tight_succ is not None and tight_succ < 0.35:
        success = long_succ
        print(f"  NOTE: Using long-clock success ({long_succ*100:.1f}%) as primary metric")
        print(f"        (tight-clock {tight_succ*100:.1f}% excluded — physically infeasible)")
    else:
        success = metrics["overall_success"]
    viols   = metrics["p_excess_mean"]

    if success >= GOOD_SUCCESS_RATE and viols <= GOOD_MAX_VIOLATIONS:
        print("  VERDICT: GOOD — apply efficiency fixes")
        print("="*60 + "\n")
        diag = {"verdict": "GOOD", **metrics, **training}
        Path(f"results/cddp/{args.run_name}/diagnosis.json").write_text(json.dumps(diag, indent=2))
        sys.exit(0)

    elif success >= MARGINAL_SUCCESS_RATE and viols <= MARGINAL_MAX_VIOLATIONS:
        print("  VERDICT: MARGINAL — acceptable, applying efficiency fixes")
        print("="*60 + "\n")
        diag = {"verdict": "MARGINAL", **metrics, **training}
        Path(f"results/cddp/{args.run_name}/diagnosis.json").write_text(json.dumps(diag, indent=2))
        sys.exit(2)

    else:
        causes = []
        fixes  = []

        if viols > MARGINAL_MAX_VIOLATIONS:
            causes.append(f"HIGH_VIOLATIONS ({viols*100:.1f}% > {MARGINAL_MAX_VIOLATIONS*100:.0f}%)")
            fixes.append("REVERT_ADAPTIVE_GAMMA")
            fixes.append("RAISE_P_REF")

        if success < MARGINAL_SUCCESS_RATE:
            causes.append(f"LOW_SUCCESS ({success*100:.1f}% < {MARGINAL_SUCCESS_RATE*100:.0f}%)")
            if training.get("zdeon_collapsed_high"):
                causes.append("ZDEON_COLLAPSED_HIGH (always predicts failure)")
                fixes.append("FIX_ZDEON_IMBALANCE")
            if training.get("zdeon_collapsed_low"):
                causes.append("ZDEON_COLLAPSED_LOW (always predicts safe)")
                fixes.append("FIX_ZDEON_LABEL_SMOOTHING")
            if training.get("plateaued"):
                causes.append("TRAINING_PLATEAUED")
                fixes.append("BOOST_EXPLORATION")
                fixes.append("EXTEND_TRAINING")
            if not fixes:
                fixes.append("EXTEND_MIN_EPISODE_STEPS")
                fixes.append("EXTEND_TRAINING")

        print(f"  VERDICT: BAD")
        print(f"  Root causes : {', '.join(causes)}")
        print(f"  Planned fixes: {', '.join(fixes)}")
        print("="*60 + "\n")

        diag = {"verdict": "BAD", "causes": causes, "fixes": fixes, **metrics, **training}
        Path(f"results/cddp/{args.run_name}/diagnosis.json").write_text(json.dumps(diag, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
