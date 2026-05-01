import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omnisafe_SRL.hostage_gridworld import HostageGridworld

def test_grid_1_layout():
    env = HostageGridworld(layout_id=1)
    
    assert env.grid_size == 5, "Grid 1 size should be 5"
    assert tuple(env.agent_start) == (2, 1), "Grid 1 agent start should be (2, 1)"
    assert env.hostage_pos == (1, 3), "Grid 1 hostage pos should be (1, 3)"
    
    expected_civilians = set([(1, 1), (1, 2), (2, 2), (3, 2), (2, 3)])
    actual_civilians = set(env.civilians)
    assert actual_civilians == expected_civilians, f"Grid 1 civilians mismatch. Expected {expected_civilians}, got {actual_civilians}"

def test_grid_2_layout():
    env = HostageGridworld(layout_id=2)
    
    assert env.grid_size == 10, "Grid 2 size should be 10"
    assert tuple(env.agent_start) == (9, 0), "Grid 2 agent start should be (9, 0)"
    assert env.hostage_pos == (3, 6), "Grid 2 hostage pos should be (3, 6)"
    
    expected_civilians = set([(5,0), (4, 1), (7, 1), (6, 2), (8, 2), (9, 2), (3, 3), (5, 3), (7, 3),
                          (4, 4), (3, 5), (5, 5), (4, 6), (4, 7)])
    actual_civilians = set(env.civilians)
    assert actual_civilians == expected_civilians, f"Grid 2 civilians mismatch. Expected {expected_civilians}, got {actual_civilians}"
