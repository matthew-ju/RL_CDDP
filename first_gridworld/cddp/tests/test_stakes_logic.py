import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import (
    AnalyticalStakesDmin,
    AnalyticalStakesLsafe,
    AnalyticalFeasibilityTrigger,
)
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.cddp_agent import CDDPAgent


def test_boundary_switching_logic():
    """Verify strictly DEON if S < tau else UTIL (Option A)."""
    print("\n" + "="*60)
    print("TEST: Stakes Switching Boundary (S == tau)")
    print("="*60)
    actor = PPOActor(obs_dim=9, hidden_dim=64)
    shield = ActionShield()
    
    # We will test S = tau behavior.
    # We can mock the stakes_fn to return exactly tau.
    class MockStakes:
        def __init__(self, val):
            self.val = val
        def compute(self, env):
            return self.val
    
    env = InstrumentedHostageEnv.make(layout_id=1, bomb_timer=5)
    obs, info = env.reset()
    
    tau = 0.5
    
    # S < tau -> deontic
    agent_below = CDDPAgent(actor, shield, MockStakes(0.49), tau=tau)
    _, info_below = agent_below.select_action(obs, env)
    assert info_below['mode'] == 'deontic', "S < tau should be deontic"
    
    # S == tau -> utilitarian
    agent_boundary = CDDPAgent(actor, shield, MockStakes(0.50), tau=tau)
    _, info_boundary = agent_boundary.select_action(obs, env)
    assert info_boundary['mode'] == 'utilitarian', "S == tau should be utilitarian"
    
    # S > tau -> utilitarian
    agent_above = CDDPAgent(actor, shield, MockStakes(0.51), tau=tau)
    _, info_above = agent_above.select_action(obs, env)
    assert info_above['mode'] == 'utilitarian', "S > tau should be utilitarian"
    
    print("✓ Boundary rule exactly met (DEON if S < tau, UTIL if S >= tau)")


def test_analytic_mode_grid1():
    """Stakes sanity check using a mocked Grid 1 scenario.
    
    We set up a scenario where D_min = 3, L_safe = 5, T_rem = 6.
    Under AnalyticalStakesDmin:
        S = D_min / T_rem = 3/6 = 0.50
        If tau = 0.5, S >= tau, so it switches to utilitarian!
        But the safe path is 5 <= T_rem=6, so a safe rescue is feasible.
        
    Under AnalyticalFeasibilityTrigger:
        S = 1.0 if L_safe > T_rem else 0.0
        L_safe=5 <= 6, so S = 0.0.
        S < tau (0.0 < 0.5), so it stays deontic.
    """
    print("\n" + "="*60)
    print("TEST: Analytic stakes variants behavior on feasible rescue")
    print("="*60)
    
    class MockEnv:
        def get_T_rem(self): return 6.0
        def get_D_min(self): return 3.0
        def get_L_safe(self): return 5.0
        def get_safe_actions(self): return [0, 1]
    
    mock_env = MockEnv()
    
    dmin_stakes = AnalyticalStakesDmin().compute(mock_env)
    assert abs(dmin_stakes - 0.5) < 1e-6, f"Expected 0.5, got {dmin_stakes}"
    
    lsafe_stakes = AnalyticalStakesLsafe().compute(mock_env)
    assert abs(lsafe_stakes - (5/6)) < 1e-6, f"Expected {5/6}, got {lsafe_stakes}"
    
    feas_stakes = AnalyticalFeasibilityTrigger().compute(mock_env)
    assert feas_stakes == 0.0, f"Feasibility trigger should be 0.0, got {feas_stakes}"
    
    print("✓ Computed stakes match formulas correctly")
    
    # Check mode switching
    actor = PPOActor(obs_dim=9, hidden_dim=64)
    shield = ActionShield()
    obs = (0,) * 9 # mock
    
    tau = 0.5
    
    agent_dmin = CDDPAgent(actor, shield, stakes_fn=AnalyticalStakesDmin(), tau=tau)
    ag_d_mode = agent_dmin.get_mode(dmin_stakes)
    assert ag_d_mode == 'utilitarian', "S=0.5 >= tau=0.5 -> Should switch early"
    
    agent_feas = CDDPAgent(actor, shield, stakes_fn=AnalyticalFeasibilityTrigger(), tau=tau)
    ag_f_mode = agent_feas.get_mode(feas_stakes)
    assert ag_f_mode == 'deontic', "S=0.0 < tau=0.5 -> Should stay deontic"
    
    print("✓ Feasibility vs D_min logic distinguishes 'paranoid' early switches gracefully")


def test_unsorted_quantiles():
    """Verify that unsorted quantiles fed into ProspectiveStakes correctly compute P_cat.
    Because the internal logic strictly sorts before thresholding.
    """
    print("\n" + "="*60)
    print("TEST: Unsorted Quantiles in Prospective Stakes")
    print("="*60)
    
    from cddp.stakes.prospective import ProspectiveStakes
    import torch
    import numpy as np

    class MockOracle:
        def get_action(self, env, pos): return 0
        
    class MockEnv:
        def get_T_rem(self): return 10.0
        def _get_obs(self): return np.zeros(9)
        def get_safe_actions(self, pos): return [0, 1, 2, 3]
        @property
        def agent_pos(self): return (0, 0)
        
    class MockCritics:
        def z_deon(self, obs):
            # Unsorted mock array that crosses c_cat = -0.5
            # 5 values <= -0.5, 5 values > -0.5
            arr = np.array([1.0, -2.0, 0.0, -1.0, 3.0, -0.6, -0.1, -1.5, 2.0, -0.5])
            # shape must be (batch=1, action=4, quantiles=10)
            quantiles = torch.FloatTensor(arr).unsqueeze(0).expand(4, 10).unsqueeze(0)
            return quantiles
            
    stakes = ProspectiveStakes(critics=MockCritics(), oracle=MockOracle(), p_ref=1.0, c_cat=-0.5)
    p_cat = stakes.compute(MockEnv())
    
    # -2.0, -1.0, -0.6, -1.5, -0.5 are <= -0.5. Exactly 5 out of 10.
    assert abs(p_cat - 0.5) < 1e-6, f"Expected P_cat=0.5 but got {p_cat}"
    
    print("✓ Unsorted quantiles safely and correctly computed via internal sorting.")


if __name__ == '__main__':
    test_boundary_switching_logic()
    test_analytic_mode_grid1()
    test_unsorted_quantiles()
    print("\n✅ All Stakes Logic Tests Passed")
