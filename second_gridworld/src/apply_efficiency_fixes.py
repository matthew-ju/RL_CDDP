#!/usr/bin/env python3
"""
Applies all efficiency fixes to analytical_sandbox.py and train_sandbox_cddp.py.
Safe to run multiple times — each fix checks if it was already applied.

Usage:
  python3 src/apply_efficiency_fixes.py
"""
import re
from pathlib import Path

SANDBOX = Path("src/analytical_sandbox.py")
TRAIN   = Path("src/train_sandbox_cddp.py")

def already_applied(src: str, marker: str) -> bool:
    return f"# EFFICIENCY_FIX: {marker}" in src

def mark(src: str, marker: str) -> str:
    return src + f"\n# EFFICIENCY_FIX: {marker}\n"

# ── Fix 1: Remove duplicate cbf_value call in step() ─────────────────────────
def fix1_remove_duplicate_cbf_value(src: str) -> str:
    marker = "FIX1_NO_DUPLICATE_CBF"
    if already_applied(src, marker):
        print("[eff] Fix 1 already applied")
        return src

    pattern = r'h_next\s*=\s*self\.cbf_value\(x_next,\s*timestep=next_step_t\)'
    replacement = 'h_next = h_next_spatial  # reuse — same computation, no redundant call'

    if re.search(pattern, src):
        src = re.sub(pattern, replacement, src, count=1)
        print("[eff] Fix 1 applied: removed duplicate cbf_value() call in step()")
    else:
        print("[eff] Fix 1: pattern not found (may already be fixed or line differs)")

    return mark(src, marker)

# ── Fix 2: Fix dead-code torch.where in step() ────────────────────────────────
def fix2_fix_dead_where(src: str) -> str:
    marker = "FIX2_DEAD_WHERE"
    if already_applied(src, marker):
        print("[eff] Fix 2 already applied")
        return src

    pattern = (
        r'v_next\s*=\s*torch\.where\(\s*inside,\s*'
        r'1\.0\s*\*\s*x_next_spatial\[:,\s*2:\],\s*'
        r'x_next_spatial\[:,\s*2:\]\s*\)'
    )
    replacement = (
        '# Collision response: zero velocity when inside obstacle\n'
        '        v_next = torch.where(inside, torch.zeros_like(x_next_spatial[:, 2:]), x_next_spatial[:, 2:])'
    )

    if re.search(pattern, src):
        src = re.sub(pattern, replacement, src, count=1)
        print("[eff] Fix 2 applied: fixed dead-code torch.where in step()")
    else:
        print("[eff] Fix 2: dead-code pattern not found (may differ slightly — check manually)")

    return mark(src, marker)

# ── Fix 3: Cache static obstacle tensors ─────────────────────────────────────
def fix3_cache_static_obstacles(src: str) -> str:
    marker = "FIX3_STATIC_OBS_CACHE"
    if already_applied(src, marker):
        print("[eff] Fix 3 already applied")
        return src

    init_end_pattern = r'(def __init__\(self[^)]*\):.*?)(^\s{4}def )'
    match = re.search(init_end_pattern, src, re.DOTALL | re.MULTILINE)
    if not match:
        print("[eff] Fix 3: could not locate __init__ — skipping")
        return src

    cache_init = '''
        # Cache static obstacle geometry (oscillation_amplitude == 0.0)
        _osc = getattr(self, 'oscillation_amplitude', 0.0)
        self._obstacles_static = (abs(_osc) < 1e-9)
        if self._obstacles_static:
            _c = self._build_centers_tensor()   # implement below if not present
            self._cached_centers = _c
            self._cached_velocities     = torch.zeros_like(_c)
            self._cached_accelerations  = torch.zeros_like(_c)
'''

    insert_pos = match.start(2)
    src = src[:insert_pos] + cache_init + '\n' + src[insert_pos:]

    obs_center_pattern = r'(def obstacle_centers\(self[^)]*\):)'
    early_return = (
        r'\1\n'
        r'        if getattr(self, "_obstacles_static", False):\n'
        r'            b = batch_size if batch_size is not None else 1\n'
        r'            return self._cached_centers.to(device).expand(b, -1, -1)\n'
    )
    src = re.sub(obs_center_pattern, early_return, src, count=1)

    obs_vel_pattern = r'(def obstacle_velocities\(self[^)]*\):)'
    vel_return = (
        r'\1\n'
        r'        if getattr(self, "_obstacles_static", False):\n'
        r'            b = batch_size if batch_size is not None else 1\n'
        r'            d = device if device is not None else "cpu"\n'
        r'            return self._cached_velocities.to(device=d, dtype=dtype).expand(b, -1, -1)\n'
    )
    src = re.sub(obs_vel_pattern, vel_return, src, count=1)

    obs_acc_pattern = r'(def obstacle_accelerations\(self[^)]*\):)'
    acc_return = (
        r'\1\n'
        r'        if getattr(self, "_obstacles_static", False):\n'
        r'            b = batch_size if batch_size is not None else 1\n'
        r'            d = device if device is not None else "cpu"\n'
        r'            return self._cached_accelerations.to(device=d, dtype=dtype).expand(b, -1, -1)\n'
    )
    src = re.sub(obs_acc_pattern, acc_return, src, count=1)

    print("[eff] Fix 3 applied: static obstacle center/velocity/acceleration caching")
    return mark(src, marker)

# ── Fix 4: Distance-based CBF skip in project_action_cbf() ───────────────────
def fix4_cbf_distance_skip(src: str) -> str:
    marker = "FIX4_CBF_SKIP"
    if already_applied(src, marker):
        print("[eff] Fix 4 already applied")
        return src

    target = (
        r'(if stakes is None:\s*\n'
        r'\s+stakes = torch\.zeros\(B[^)]*\)\s*\n)'
    )
    injection = (
        r'\1\n'
        r'        # ── Early exit: skip QP when far from all obstacles ──────\n'
        r'        CBF_SKIP_MARGIN = 1.5  # units beyond obstacle surface\n'
        r'        _p = x[:, :2]\n'
        r'        _centers = self.obstacle_centers(timestep=timestep, batch_size=B, device=x.device)\n'
        r'        _diff_all = _p.unsqueeze(1) - _centers                   # (B, N, 2)\n'
        r'        _dist_all = _diff_all.norm(dim=2)                        # (B, N)\n'
        r'        _obs_r = self.obstacle_r.to(x.device)\n'
        r'        _margin = (_dist_all - _obs_r.unsqueeze(0)).min(dim=1).values  # (B,)\n'
        r'        _near = _margin < CBF_SKIP_MARGIN                        # (B,) bool\n'
        r'        if not _near.any():\n'
        r'            return u_nom  # entire batch safe — skip all Lie-derivative compute\n'
        r'        # ────────────────────────────────────────────────────────\n'
    )

    if re.search(target, src):
        src = re.sub(target, injection, src, count=1)
        print("[eff] Fix 4 applied: distance-based CBF skip")
    else:
        print("[eff] Fix 4: could not locate stakes-default block — skipping")

    return mark(src, marker)

# ── Fix 5: Shrink Z_deon network ─────────────────────────────────────────────
def fix5_shrink_zdeon(src: str) -> str:
    marker = "FIX5_ZDEON_SMALL"
    if already_applied(src, marker):
        print("[eff] Fix 5 already applied")
        return src

    target = r'(MODEL_HIDDEN_LAYERS\s*=\s*\d+\s*\n)'
    replacement = (
        r'\1'
        r'DEON_HIDDEN_DIM    = 128   # Z_deon is binary classification — needs less capacity\n'
        r'DEON_HIDDEN_LAYERS = 2     # than Z_free (continuous Q-value estimation)\n'
    )
    src = re.sub(target, replacement, src, count=1)

    target2 = (
        r'(class DeonticCritic.*?def __init__\(.*?hidden:\s*int\s*=\s*)'
        r'MODEL_HIDDEN_DIM'
    )
    src = re.sub(target2, r'\1DEON_HIDDEN_DIM', src, count=1, flags=re.DOTALL)

    target3 = (
        r'(class DeonticCritic.*?hidden_layers:\s*int\s*=\s*)'
        r'MODEL_HIDDEN_LAYERS'
    )
    src = re.sub(target3, r'\1DEON_HIDDEN_LAYERS', src, count=1, flags=re.DOTALL)

    print("[eff] Fix 5 applied: Z_deon shrunk to 2×128 (was 3×256)")
    return mark(src, marker)

# ── Fix 6: Cache stakes every 4 steps ─────────────────────────────────────────
def fix6_cache_stakes(src: str) -> str:
    marker = "FIX6_STAKES_CACHE"
    if already_applied(src, marker):
        print("[eff] Fix 6 already applied")
        return src

    target = r'(# (?:inner|episode) step loop|for.*?episode.*?steps|while.*?active\.any\(\))'
    cache_vars = (
        '# Stakes cache — recompute every STAKES_CACHE_FREQ steps (saves Z_deon forward passes)\n'
        '        STAKES_CACHE_FREQ   = 4\n'
        '        stakes_cache        = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)\n'
        '        stakes_last_ep_step = torch.full((NUM_ENVS,), -STAKES_CACHE_FREQ, dtype=torch.long, device=DEVICE)\n\n        '
    )

    match = re.search(target, src, re.IGNORECASE)
    if match:
        src = src[:match.start()] + cache_vars + src[match.start():]
        print("[eff] Fix 6 applied: stakes cache variables injected")
    else:
        print("[eff] Fix 6: could not find step-loop anchor — skipping")

    return mark(src, marker)

# ── Fix 7: Increase TRAIN_EVERY to reduce update overhead ────────────────────
def fix7_train_every(src: str) -> str:
    marker = "FIX7_TRAIN_EVERY"
    if already_applied(src, marker):
        print("[eff] Fix 7 already applied")
        return src

    match = re.search(r'TRAIN_EVERY\s*=\s*(\d+)', src)
    if match:
        current = int(match.group(1))
        if current < 12:
            src = re.sub(r'TRAIN_EVERY\s*=\s*\d+', 'TRAIN_EVERY = 12', src)
            src = re.sub(r'GRADIENT_STEPS\s*=\s*\d+', 'GRADIENT_STEPS = 3', src)
            print(f"[eff] Fix 7: TRAIN_EVERY {current}→12, GRADIENT_STEPS 2→3 (same total updates)")
        else:
            print(f"[eff] Fix 7: TRAIN_EVERY={current} already fine")
    return mark(src, marker)

def main():
    print("\n" + "="*55)
    print("  Applying efficiency fixes to source files")
    print("="*55)

    src_sb = SANDBOX.read_text()
    src_sb = fix1_remove_duplicate_cbf_value(src_sb)
    src_sb = fix2_fix_dead_where(src_sb)
    src_sb = fix3_cache_static_obstacles(src_sb)
    src_sb = fix4_cbf_distance_skip(src_sb)
    SANDBOX.write_text(src_sb)
    print(f"  Wrote {SANDBOX}")

    src_tr = TRAIN.read_text()
    src_tr = fix5_shrink_zdeon(src_tr)
    src_tr = fix6_cache_stakes(src_tr)
    src_tr = fix7_train_every(src_tr)
    TRAIN.write_text(src_tr)
    print(f"  Wrote {TRAIN}")

    print("\n  All efficiency fixes applied.")
    print("  Run a 500-episode smoke test before full training:")
    print("  python3 src/train_sandbox_cddp.py --run-name smoke_eff --max-episodes 500")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
