import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.agents.deontic_oracle import DeonticOraclePlanner

def test_deontic_oracle_shortest_path():
    """
    Test that the OraclePlanner successfully decreases L_safe on Grid 1 
    starting from (2,1) and choosing safe actions without ever returning unsafe.
    """
    env = InstrumentedHostageEnv.make(layout_id=1, bomb_timer=10)
    obs, info = env.reset()
    
    oracle = DeonticOraclePlanner()
    pos = tuple(info['agent_pos']) # should be (2, 1)
    
    print(f"Start pos: {pos}")
    L_curr = env.get_L_safe(pos)
    print(f"Initial L_safe: {L_curr}")
    
    # First action should be LEFT in Grid 1 to avoid civilian at (1,1) or (2,2)
    action = oracle.get_action(env)
    assert action in env.get_safe_actions(pos), "Oracle chose an unsafe first action!"
    
    # Let's trace a few steps and ensure L_safe strictly decreases
    for step in range(L_curr):
        pos = tuple(env.agent_pos)
        act = oracle.get_action(env, pos)
        
        # Must be safe
        safe_actions = env.get_safe_actions(pos)
        assert act in safe_actions, f"Unsafe action {act} chosen at pos {pos}"
        
        # Step
        obs, r, term, trunc, info = env.step(act)
        new_pos = tuple(env.agent_pos)
        
        L_new = env.get_L_safe(new_pos)
        print(f"Step {step}: Took action {act}, moved {pos} -> {new_pos}, L_safe {L_curr} -> {L_new}")
        
        # It should strictly decrease unless already 0
        if L_curr > 0:
            assert L_new < L_curr, f"L_safe did not decrease! {L_curr} -> {L_new}"
            
        L_curr = L_new
        if term:
            assert L_curr == 0, "Terminated before L_safe == 0"
            break

    print("Oracle successfully navigated the safe path deterministically.")

if __name__ == "__main__":
    test_deontic_oracle_shortest_path()
