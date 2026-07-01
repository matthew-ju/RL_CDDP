import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield

def test_deontic_shield_invariant():
    """
    Issue D Test: In DEON mode with shield active, violations MUST be impossible (C(s,a) = 0).
    Run 1,000 random shielded steps and assert constraint cost is 0.
    """
    env = InstrumentedHostageEnv.make(layout_id=1, bomb_timer=10)
    
    obs, info = env.reset()
    
    for _ in range(1000):
        pos = tuple(env.agent_pos)
        safe_actions = env.get_safe_actions(pos)
        
        # Pick a random safe action (simulating DEON mode)
        if len(safe_actions) > 0:
            action = np.random.choice(safe_actions)
        else:
            action = env.action_space.sample() # fallback
            
        cost = env.get_constraint_cost(action, pos)
        
        assert cost == 0, f"Deontic invariant violated! Action {action} from pos {pos} yielded cost {cost}"
        
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
            
    print("Deontic shield invariant holds! 1,000 steps with 0 violations.")

if __name__ == "__main__":
    test_deontic_shield_invariant()
