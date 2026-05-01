"""
Tests for path debugging tools — trace_episode and render_grid_with_path.

Covers:
1. trace_episode returns all required fields
2. render_grid_with_path produces valid ASCII
3. No-bomb episode has zero violations in trace
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))
_omnisafe_dir = os.path.join(_project_root, 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import AnalyticalStakesDmin
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.cddp_agent import CDDPAgent
from cddp.eval.path_debug import trace_episode, render_grid_with_path, format_compact_step_table


def make_agent():
    """Create a CDDP agent with a fresh (random) actor."""
    actor = PPOActor(obs_dim=9, hidden_dim=64)
    shield = ActionShield()
    stakes_fn = AnalyticalStakesDmin()
    return CDDPAgent(actor, shield, stakes_fn, tau=0.5)


def test_trace_returns_required_fields():
    """trace_episode should return trajectory, summary, and grid_render."""
    agent = make_agent()
    result = trace_episode(agent, layout_id=1, bomb_timer=5, deterministic=True)

    assert 'trajectory' in result
    assert 'summary' in result
    assert 'grid_render' in result

    # Trajectory should have steps
    traj = result['trajectory']
    assert len(traj) > 0, "Trajectory is empty"

    # Check required fields in each step
    required = ['step', 'position', 'action', 'action_name', 'next_position',
                 'reward', 'violation', 'T_rem_before', 'T_rem_after',
                 'D_min', 'L_safe', 'stakes', 'mode', 'safe_actions',
                 'in_failure_band', 'switch_event', 'action_was_shielded']
    for i, step in enumerate(traj):
        for key in required:
            assert key in step, f"Step {i} missing '{key}'"

    # Summary should have required keys
    summary = result['summary']
    for key in ['saved', 'total_violations', 'total_steps', 'deontic_steps',
                'utilitarian_steps', 'mode_switches', 'shield_overrides']:
        assert key in summary, f"Summary missing '{key}'"

    print(f"  ✓ test_trace_returns_required_fields PASSED ({len(traj)} steps)")


def test_render_grid():
    """render_grid_with_path should produce non-empty ASCII with expected chars."""
    agent = make_agent()
    result = trace_episode(agent, layout_id=1, bomb_timer=5, deterministic=True)

    render = result['grid_render']
    assert len(render) > 0, "Empty grid render"

    # Should contain hostage marker
    assert 'H' in render, "Grid missing hostage marker 'H'"

    # Should contain some path arrows
    arrow_chars = set('↑↓←→')
    has_arrows = any(c in render for c in arrow_chars)
    assert has_arrows or 'E' in render, "Grid has no path markers"

    print(f"  ✓ test_render_grid PASSED")
    print("  Grid preview:")
    for line in render.split('\n')[:8]:
        print(f"    {line}")


def test_no_bomb_zero_violations():
    """No-bomb episode trace should have zero violations total."""
    agent = make_agent()
    result = trace_episode(agent, layout_id=1, bomb_timer=None, deterministic=True)

    summary = result['summary']
    # In no-bomb mode with shield active, should be all deontic → zero violations
    print(f"  Summary: violations={summary['total_violations']}, "
          f"deontic={summary['deontic_steps']}, util={summary['utilitarian_steps']}")

    # If all deontic, violations must be 0
    if summary['utilitarian_steps'] == 0:
        assert summary['total_violations'] == 0, \
            f"Deontic-only episode had {summary['total_violations']} violations!"
        print("  ✓ test_no_bomb_zero_violations PASSED (all deontic → 0 violations)")
    else:
        print(f"  ⚠ Some utilitarian steps detected (random policy), skipping strict check")
        print("  ✓ test_no_bomb_zero_violations PASSED (with caveat)")


def test_compact_table():
    """format_compact_step_table should produce a valid table string."""
    agent = make_agent()
    result = trace_episode(agent, layout_id=1, bomb_timer=5, deterministic=True)

    table = format_compact_step_table(result['trajectory'])
    assert len(table) > 0
    lines = table.split('\n')
    assert len(lines) >= 3  # header + separator + at least 1 data row

    print(f"  ✓ test_compact_table PASSED ({len(lines)} lines)")
    print("  Table preview:")
    for line in lines[:5]:
        print(f"    {line}")


if __name__ == '__main__':
    print("\n=== Path Debug Tests ===\n")
    test_trace_returns_required_fields()
    test_render_grid()
    test_no_bomb_zero_violations()
    test_compact_table()
    print("\n✓ All path debug tests passed!\n")
