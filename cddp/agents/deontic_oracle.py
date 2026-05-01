import numpy as np
from typing import Optional

class DeonticOraclePlanner:
    """
    Deterministic deontic policy π_d* that follows the shortest SAFE path to the hostage.
    Civilians are treated as walls (cannot be stepped on).
    At each state, choose a safe action that strictly decreases L_safe(s).
    If multiple actions tie, use stable priority order (UP=0, DOWN=1, LEFT=2, RIGHT=3).
    If no safe path exists (L_safe=inf), choose a safe 'stall' action.
    """
    
    def __init__(self):
        # Stable priority order: UP, DOWN, LEFT, RIGHT
        self.priority = [0, 1, 2, 3]

    def get_action_for_state(self, env, pos: tuple) -> int:
        """
        Stateless helper to compute the oracle action for a given state (pos).
        Does not use T_rem.
        """
        safe_actions = env.get_safe_actions(pos)
        
        if not safe_actions:
            # Fallback if somehow no safe actions (shouldn't happen in our grid unless boxed in by civilians)
            return self.priority[0]
            
        best_action = None
        best_L_next = float('inf')
        
        for a in self.priority:
            if a in safe_actions:
                next_pos = env.peek_next_position(a, pos)
                if next_pos == pos:
                    # Wall bounce or boundary, L_safe wouldn't change, but let's check it anyway.
                    L_next = env.get_L_safe(next_pos)
                else:
                    L_next = env.get_L_safe(next_pos)
                    
                if L_next < best_L_next:
                    best_L_next = L_next
                    best_action = a
                    
        if best_action is not None:
            return best_action
            
        # Fallback to stalls
        for a in self.priority:
            if a in safe_actions:
                return a
                
        return self.priority[0]

    def get_action(self, env, pos: Optional[tuple] = None) -> int:
        """
        Get the oracle action for the current environment state.
        """
        if pos is None:
            pos = tuple(env.agent_pos)
        # print(f"[DEBUG] DeonticOracle BFS grid_id={env.grid_id}, ...")
        return self.get_action_for_state(env, pos)
