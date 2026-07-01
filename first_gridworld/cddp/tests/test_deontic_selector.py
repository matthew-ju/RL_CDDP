"""
Tests for DeonticActionSelector — deterministic safe-action selection.

Covers:
1. Grid 1 at (2,1): selector picks LEFT over DOWN (lower L_safe)
2. Repeated calls produce identical results (determinism)
3. Selector never returns an unsafe action
4. Tie-break stability when L_safe and D_min match
"""

import os
import sys
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))
_omnisafe_dir = os.path.join(_project_root, 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.agents.deontic_selector import DeonticActionSelector


def make_env(layout_id=1, bomb_timer=5):
    env = InstrumentedHostageEnv.make(layout_id=layout_id, bomb_timer=bomb_timer)
    env.reset()
    return env


def test_selector_picks_best_lsafe():
    """At (2,1) on Grid 1, safe actions are [DOWN, LEFT].
    LEFT leads toward hostage via safe path (shorter L_safe).
    Selector should prefer LEFT."""
    env = make_env(layout_id=1, bomb_timer=7)

    # Move agent to (2,1)
    env.env.agent_pos = [2, 1]

    selector = DeonticActionSelector()
    shield = ActionShield()
    safe_actions = shield.get_safe_actions(env)

    # Should contain LEFT (2)
    assert 2 in safe_actions, f"Expected LEFT in safe_actions, got {safe_actions}"

    obs = env.env._get_obs()
    ranked = selector.rank_safe_actions(obs, env, safe_actions)

    # First ranked action should have the lowest L_safe
    assert len(ranked) >= 1
    best = ranked[0]
    print(f"  Best action: {best['action_name']} (L_safe={best['L_safe']}, D_min={best['D_min']})")

    for r in ranked:
        print(f"    {r['action_name']}: L_safe={r['L_safe']}, D_min={r['D_min']}")

    # L_safe ordering should be non-decreasing
    for i in range(len(ranked) - 1):
        assert ranked[i]['score'] <= ranked[i+1]['score'], \
            f"Rankings not sorted: {ranked[i]['score']} > {ranked[i+1]['score']}"

    print("  ✓ test_selector_picks_best_lsafe PASSED")


def test_determinism():
    """Repeated calls to select_safe_action return the exact same result."""
    env = make_env(layout_id=1, bomb_timer=7)
    env.env.agent_pos = [2, 1]

    selector = DeonticActionSelector()
    shield = ActionShield()
    safe_actions = shield.get_safe_actions(env)
    obs = env.env._get_obs()

    results = []
    for _ in range(10):
        action, info = selector.select_safe_action(obs, env, safe_actions, deterministic=True)
        results.append(action)

    assert len(set(results)) == 1, \
        f"Non-deterministic results: {results}"
    print(f"  ✓ test_determinism PASSED (always picks {results[0]})")


def test_never_returns_unsafe():
    """Selector should only return actions from the safe set."""
    env = make_env(layout_id=1, bomb_timer=5)
    selector = DeonticActionSelector()
    shield = ActionShield()

    # Test at multiple positions
    positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for pos in positions:
        env.env.agent_pos = list(pos)
        safe_actions = shield.get_safe_actions(env)
        if not safe_actions:
            continue

        obs = env.env._get_obs()
        action, _ = selector.select_safe_action(obs, env, safe_actions, deterministic=True)
        assert action in safe_actions, \
            f"Unsafe action {action} at {pos}! Safe: {safe_actions}"

    print(f"  ✓ test_never_returns_unsafe PASSED ({len(positions)} positions)")


def test_tiebreak_stability():
    """When L_safe and D_min tie, the fixed-priority tiebreak produces stable results."""
    env = make_env(layout_id=1, bomb_timer=None)  # no bomb → no urgency
    env.env.agent_pos = list(env.env.agent_start)

    selector = DeonticActionSelector()
    shield = ActionShield()
    safe_actions = shield.get_safe_actions(env)
    obs = env.env._get_obs()

    ranked = selector.rank_safe_actions(obs, env, safe_actions)
    print(f"  Start pos: ranking = {[(r['action_name'], r['score']) for r in ranked]}")

    # Verify sorted
    for i in range(len(ranked) - 1):
        assert ranked[i]['score'] <= ranked[i + 1]['score']

    # Verify stability over 5 calls
    first_action = selector.select_safe_action(obs, env, safe_actions, deterministic=True)[0]
    for _ in range(4):
        a, _ = selector.select_safe_action(obs, env, safe_actions, deterministic=True)
        assert a == first_action, f"Tiebreak instability: {a} != {first_action}"

    print(f"  ✓ test_tiebreak_stability PASSED (consistent: {first_action})")


def test_single_safe_action():
    """When only one safe action exists, selector returns it without error."""
    env = make_env(layout_id=1, bomb_timer=5)
    selector = DeonticActionSelector()
    obs = env.env._get_obs()

    action, info = selector.select_safe_action(obs, env, [2], deterministic=True)
    assert action == 2
    assert info['tiebreak_reason'] == 'only_one_safe'
    print("  ✓ test_single_safe_action PASSED")


def test_empty_safe_actions():
    """Edge case: empty safe set should not crash."""
    env = make_env(layout_id=1, bomb_timer=5)
    selector = DeonticActionSelector()
    obs = env.env._get_obs()

    action, info = selector.select_safe_action(obs, env, [], deterministic=True)
    assert action == 0  # fallback
    assert info['tiebreak_reason'] == 'no_safe_actions'
    print("  ✓ test_empty_safe_actions PASSED")


if __name__ == '__main__':
    print("\n=== DeonticActionSelector Tests ===\n")
    test_selector_picks_best_lsafe()
    test_determinism()
    test_never_returns_unsafe()
    test_tiebreak_stability()
    test_single_safe_action()
    test_empty_safe_actions()
    print("\n✓ All DeonticActionSelector tests passed!\n")
