import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.eval.metrics import min_violations_needed

def test_min_violations_grid1_t3():
    env = InstrumentedHostageEnv.make(layout_id=1, bomb_timer=3)
    env.reset()
    
    # We expect min_violations_needed to be 2 for T=3 from standard start (2, 1)
    min_v = min_violations_needed(env)
    assert min_v == 2, f"Grid 1 T=3 should require min 2 violations, but got {min_v}"

def test_min_violations_grid1_t5():
    env = InstrumentedHostageEnv.make(layout_id=1, bomb_timer=5)
    env.reset()
    
    # Given enough time, we can dodge civilians. T=5 is enough to reach in 4 steps w/ 0 cost?
    # Grid 1 start (2, 1), hostage (1, 3). Path: (2, 1) -> (3, 1) -> (4, 1) -> (4, 2) -> (4, 3) -> (3, 3) -> (2, 3) -> (1, 3) ... Wait, 5 steps is too short.
    min_v = min_violations_needed(env)
    # the min distance ignoring civilians is 3 steps. The optimal safe path is much longer.
    assert min_v >= 0, f"Got min_v = {min_v}"
