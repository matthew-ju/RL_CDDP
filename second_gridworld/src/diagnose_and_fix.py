#!/usr/bin/env python3
"""
Reads diagnosis.json, applies targeted fixes to train_sandbox_cddp.py,
and prints the next --run-name to use.

Usage:
  python3 src/diagnose_and_fix.py --run-name long_clock_only --attempt 2
"""
import argparse, json, re, sys
from pathlib import Path

TRAIN_FILE = Path("src/train_sandbox_cddp.py")

def read_src():
    return TRAIN_FILE.read_text()

def write_src(text):
    TRAIN_FILE.write_text(text)
    print(f"[fix] Updated {TRAIN_FILE}")

def apply_fix_revert_adaptive_gamma(src: str) -> str:
    """Remove adaptive_gamma_cbf function and revert to GAMMA_CBF = 0.7 constant."""
    src = re.sub(
        r'def adaptive_gamma_cbf\(.*?\n(?=\n|\S)',
        '',
        src,
        flags=re.DOTALL
    )
    src = re.sub(
        r'adaptive_gamma_cbf\([^)]+\)',
        'GAMMA_CBF',
        src
    )
    src = re.sub(
        r'GAMMA_CBF\s*=\s*[\d.]+',
        'GAMMA_CBF = 0.7',
        src
    )
    print("[fix] Reverted adaptive GAMMA_CBF → constant 0.7")
    return src

def apply_fix_raise_p_ref(src: str) -> str:
    """Raise P_REF if it is below 0.35 (too aggressive stakes saturation)."""
    match = re.search(r'P_REF\s*=\s*([\d.]+)', src)
    if match:
        current = float(match.group(1))
        if current < 0.35:
            src = re.sub(r'P_REF\s*=\s*[\d.]+', 'P_REF = 0.40', src)
            print(f"[fix] Raised P_REF {current} → 0.40")
        else:
            print(f"[fix] P_REF={current} already OK, no change")
    return src

def apply_fix_zdeon_imbalance(src: str) -> str:
    """Ensure weighted BCE is present in Z_deon loss. If already there, add label smoothing."""
    if 'pos_weight' in src and 'binary_cross_entropy_with_logits' in src:
        old = 'loss_deon = F.binary_cross_entropy_with_logits('
        new = (
            '# Label smoothing prevents Z_deon from collapsing to all-1s\n'
            '        _smooth_labels = deontic_labels * 0.85 + 0.075\n'
            '        loss_deon = F.binary_cross_entropy_with_logits('
        )
        src = src.replace(old, new, 1)
        src = re.sub(
            r'(loss_deon = F\.binary_cross_entropy_with_logits\(\s*\n?\s*)deontic_labels',
            r'\1_smooth_labels',
            src
        )
        print("[fix] Added label smoothing to Z_deon BCE loss")
    else:
        old_pattern = r'loss_deon\s*=\s*F\.binary_cross_entropy_with_logits\(([^)]+)\)'
        replacement = (
            'n_pos = deontic_labels.sum().item()\n'
            '                n_neg = len(deontic_labels) - n_pos\n'
            '                if n_pos > 0 and n_neg > 0:\n'
            '                    _pw = torch.tensor(n_neg / n_pos, device=DEVICE).clamp(max=8.0)\n'
            '                else:\n'
            '                    _pw = torch.ones(1, device=DEVICE)\n'
            '                loss_deon = F.binary_cross_entropy_with_logits(\n'
            '                    deontic_labels, deontic_labels, pos_weight=_pw\n'
            '                )'
        )
        match = re.search(old_pattern, src)
        if match:
            src = src[:match.start()] + replacement + src[match.end():]
            print("[fix] Added weighted BCE to Z_deon loss")
        else:
            print("[fix] WARNING: could not locate Z_deon loss line — skipping")
    return src

def apply_fix_label_smoothing(src: str) -> str:
    """Z_deon always predicts safe — add label noise to deontic labels."""
    old = 'loss_deon = F.binary_cross_entropy_with_logits('
    new = (
        '_smooth_labels = deontic_labels * 0.9 + 0.05  # prevent Z_deon from collapsing to 0\n'
        '                loss_deon = F.binary_cross_entropy_with_logits('
    )
    if '_smooth_labels' not in src:
        src = src.replace(old, new, 1)
        src = re.sub(
            r'(loss_deon = F\.binary_cross_entropy_with_logits\(\s*\n?\s*)deontic_labels',
            r'\1_smooth_labels',
            src
        )
        print("[fix] Added label smoothing (Z_deon was collapsing to all-safe)")
    return src

def apply_fix_boost_exploration(src: str) -> str:
    """Increase OU_SIGMA if training plateaued early."""
    match = re.search(r'OU_SIGMA\s*=\s*([\d.]+)', src)
    if match:
        current = float(match.group(1))
        new_val = min(current * 1.5, 0.6)
        src = re.sub(r'OU_SIGMA\s*=\s*[\d.]+', f'OU_SIGMA = {new_val:.2f}', src, count=1)
        print(f"[fix] Boosted OU_SIGMA {current} → {new_val:.2f} (plateau recovery)")
    return src

def apply_fix_extend_training(src: str) -> str:
    """Increase N_EPISODES if still under 75k."""
    match = re.search(r'N_EPISODES\s*=\s*(\d+)', src)
    if match:
        current = int(match.group(1))
        if current < 75000:
            new_val = 75000
            src = re.sub(r'N_EPISODES\s*=\s*\d+', f'N_EPISODES = {new_val}', src)
            print(f"[fix] Extended N_EPISODES {current} → {new_val}")
        else:
            print(f"[fix] N_EPISODES={current} already at max, no change")
    return src

def apply_fix_extend_min_episode_steps(src: str) -> str:
    """If curriculum min steps is too tight, raise it."""
    match = re.search(r'np\.random\.uniform\((\d+),\s*2500\)', src)
    if match:
        current_min = int(match.group(1))
        if current_min < 800:
            src = src.replace(
                f'np.random.uniform({current_min}, 2500)',
                'np.random.uniform(800, 2500)'
            )
            print(f"[fix] Raised curriculum min steps {current_min} → 800")
    return src

FIX_MAP = {
    "REVERT_ADAPTIVE_GAMMA":      apply_fix_revert_adaptive_gamma,
    "RAISE_P_REF":                apply_fix_raise_p_ref,
    "FIX_ZDEON_IMBALANCE":        apply_fix_zdeon_imbalance,
    "FIX_ZDEON_LABEL_SMOOTHING":  apply_fix_label_smoothing,
    "BOOST_EXPLORATION":          apply_fix_boost_exploration,
    "EXTEND_TRAINING":            apply_fix_extend_training,
    "EXTEND_MIN_EPISODE_STEPS":   apply_fix_extend_min_episode_steps,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--attempt",  type=int, default=1)
    args = ap.parse_args()

    diag_path = Path(f"results/cddp/{args.run_name}/diagnosis.json")
    if not diag_path.exists():
        print(f"[fix] ERROR: {diag_path} not found — cannot diagnose")
        sys.exit(1)

    diag = json.loads(diag_path.read_text())
    fixes = diag.get("fixes", [])
    print(f"\n[fix] Applying {len(fixes)} fix(es): {fixes}")

    src = read_src()
    for fix_name in fixes:
        if fix_name in FIX_MAP:
            src = FIX_MAP[fix_name](src)
        else:
            print(f"[fix] WARNING: unknown fix '{fix_name}' — skipping")
    write_src(src)

    next_name = f"autofix_attempt_{args.attempt}"
    print(f"\n[fix] Next run name: {next_name}")
    print(next_name)  # last line — shell reads this

if __name__ == "__main__":
    main()
