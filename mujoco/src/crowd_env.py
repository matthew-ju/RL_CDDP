"""
crowd_env.py
============
Custom MuJoCo environment: "Time-Pressure Crowd"

Topology
--------
The agent (a 2D Point robot) spawns near (0, 0) and must reach a circular
goal zone near (8, 0).

  • DIRECT PATH  (short, ~8 m):  runs straight along y ≈ 0 through a row of
    soft "crowd" obstacles packed between x = 2 and x = 6.
  • DETOUR PATH  (longer, ~12 m):  arcs above (y > 0) and curves around,
    entering the goal from the north-east — completely obstacle-free.

State vector  (15-D)
---------------------
  [qpos_x, qpos_y, qvel_x, qvel_y,   # robot kinematics  (4)
   obs0_dx, obs0_dy, obs1_dx, obs1_dy,  # relative obstacle positions (2×N)
   obs0_vy, obs1_vy, obs2_vy, obs3_vy, obs4_vy,  # (unused filler at 0)
   goal_dx, goal_dy,                 # vector to goal    (2)
   time_remaining_norm]              # ∈ [0, 1]          (1)

Total state dim = 4 + 2*N_OBS + 2 + 1  (N_OBS = 5 → 15-D)

Reward structure
----------------
  • +500  on goal capture
  • -500  if timer expires without goal (catastrophe)
  •  -1   per step spent inside any obstacle core
  • -0.05 per step (dense living cost)
  •  +0.2 dense approach bonus (Δ distance-to-goal)
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from gymnasium.envs.mujoco import MujocoEnv
    _GYM_VER = "gymnasium"
except ImportError:
    from gym.envs.mujoco import MujocoEnv
    _GYM_VER = "gym"


# ---------------------------------------------------------------------------
# XML model for the Point robot in the "Time-Pressure Crowd" arena
# ---------------------------------------------------------------------------

def _build_xml(
    obstacle_centers: list[tuple[float, float]],
    obstacle_radii: list[float],
    goal_center: tuple[float, float] = (8.0, 0.0),
    goal_radius: float = 0.5,
) -> str:
    """Generate a MuJoCo XML with the Point robot, soft crowd obstacles, and
    goal marker.  The robot is a free-sliding disc controlled via x/y forces.
    """

    obs_geoms = ""
    for i, ((cx, cy), r) in enumerate(zip(obstacle_centers, obstacle_radii)):
        obs_geoms += (
            f'    <geom name="obs{i}" type="cylinder" pos="{cx:.3f} {cy:.3f} 0.5" '
            f'size="{r:.3f} 0.5" rgba="0.9 0.3 0.2 0.6" contype="2" conaffinity="2"/>\n'
        )

    gx, gy = goal_center
    goal_geom = (
        f'    <geom name="goal_marker" type="cylinder" '
        f'pos="{gx:.3f} {gy:.3f} 0.05" size="{goal_radius:.3f} 0.05" '
        f'rgba="0.1 0.8 0.2 0.4" contype="0" conaffinity="0"/>\n'
    )

    xml = f"""<mujoco model="crowd_env">
  <compiler angle="radian" coordinate="local"/>
  <option integrator="RK4" timestep="0.02"/>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.7 0.7 0.7 1"/>

{obs_geoms}
{goal_geom}

    <body name="agent" pos="0 0 0.5">
      <joint name="free_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="free_y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom name="agent_geom" type="sphere" size="0.25"
            rgba="0.2 0.4 0.9 1.0" contype="1" conaffinity="1"/>
      <site name="agent_site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_x" joint="free_x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor_y" joint="free_y" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""
    return xml


# ---------------------------------------------------------------------------
# Obstacle layout
# ---------------------------------------------------------------------------

# 5 crowd obstacles arranged in a dense cluster along the direct path
# (y ≈ 0, x from 2 to 6)
CROWD_CENTERS: list[tuple[float, float]] = [
    (2.5, 0.0),
    (3.5, 0.2),
    (4.5, -0.2),
    (5.5, 0.1),
    (4.0, 0.0),
]
CROWD_RADII: list[float] = [0.45, 0.40, 0.45, 0.40, 0.35]

GOAL_CENTER = (8.0, 0.0)
GOAL_RADIUS = 0.6

# State / action dimensions
N_OBS = len(CROWD_CENTERS)
STATE_DIM = 4 + 2 * N_OBS + 2 + 1   # 15


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class CrowdNavEnv:
    """Pure-NumPy wrapper around the MuJoCo Point-crowd scenario.

    Compatible with both ``gymnasium`` and ``gym`` APIs.  Internally wraps
    MujocoEnv via a temp-file XML approach so no filesystem model path is
    needed.

    Parameters
    ----------
    max_steps : int
        Episode horizon.  Determines catastrophe boundary and time-remaining
        normalisation.
    penalty_per_step_inside : float
        Reward deduction per step spent with agent centre inside any obstacle.
    goal_reward : float
        Sparse reward on successful goal capture.
    catastrophe_penalty : float
        Negative reward added when the timer expires without goal.
    frame_skip : int
        MuJoCo physics steps per RL step.
    seed : int | None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        max_steps: int = 500,
        penalty_per_step_inside: float = -1.0,
        goal_reward: float = 500.0,
        catastrophe_penalty: float = -500.0,
        frame_skip: int = 5,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.penalty_per_step_inside = float(penalty_per_step_inside)
        self.goal_reward = float(goal_reward)
        self.catastrophe_penalty = float(catastrophe_penalty)
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        # Write XML to temp file so MujocoEnv can load it
        xml = _build_xml(CROWD_CENTERS, CROWD_RADII, GOAL_CENTER, GOAL_RADIUS)
        self._tmpfile = tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False
        )
        self._tmpfile.write(xml)
        self._tmpfile.flush()
        self._xml_path = self._tmpfile.name

        # Lazy-init: load MuJoCo model on first reset or explicit init
        self._mj: Any = None
        self._step_count: int = 0
        self._prev_dist_to_goal: float = 0.0

        # Gym spaces
        obs_high = np.full(STATE_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._obstacle_centers = np.array(CROWD_CENTERS, dtype=np.float32)
        self._obstacle_radii = np.array(CROWD_RADII, dtype=np.float32)
        self._goal_center = np.array(GOAL_CENTER, dtype=np.float32)
        self._goal_radius = float(GOAL_RADIUS)

    # ------------------------------------------------------------------
    # Internal MuJoCo init
    # ------------------------------------------------------------------

    def _init_mujoco(self) -> None:
        """Lazy-load MuJoCo model.  Called on first reset()."""
        try:
            import mujoco
            import mujoco.viewer
            self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
            self._mj_data = mujoco.MjData(self._mj_model)
            self._mujoco = mujoco
            self._mj = "mujoco"
        except Exception as e:
            raise RuntimeError(
                "Could not load MuJoCo model. "
                "Ensure 'mujoco' Python package is installed: pip install mujoco\n"
                f"Original error: {e}"
            )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        d = self._mj_data
        qpos = d.qpos[:2].copy()   # [x, y]
        qvel = d.qvel[:2].copy()   # [vx, vy]

        # Relative positions to each obstacle
        rel_obs = (self._obstacle_centers - qpos[None]).flatten()  # (2*N,)

        # Relative position to goal
        rel_goal = self._goal_center - qpos   # (2,)

        # Normalized time remaining
        t_norm = max(0.0, (self.max_steps - self._step_count) / self.max_steps)

        obs = np.concatenate(
            [qpos, qvel, rel_obs, rel_goal, [t_norm]], dtype=np.float32
        )
        return obs

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _obstacle_penetrations(self, pos: np.ndarray) -> np.ndarray:
        """Returns penetration depth (>0 means inside obstacle) for each obstacle."""
        dists = np.linalg.norm(self._obstacle_centers - pos[None], axis=1)
        return np.maximum(0.0, self._obstacle_radii - dists)

    def _in_goal(self, pos: np.ndarray) -> bool:
        return bool(np.linalg.norm(pos - self._goal_center) < self._goal_radius)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._mj is None:
            self._init_mujoco()

        self._step_count = 0

        # Spawn agent near origin with small random offset
        d = self._mj_data
        m = self._mj_model
        self._mujoco.mj_resetData(m, d)

        # Randomize start position slightly
        start_x = self._rng.uniform(-0.3, 0.3)
        start_y = self._rng.uniform(-0.3, 0.3)
        d.qpos[0] = start_x
        d.qpos[1] = start_y
        d.qvel[:] = 0.0

        self._mujoco.mj_forward(m, d)
        pos = d.qpos[:2].copy()
        self._prev_dist_to_goal = float(np.linalg.norm(pos - self._goal_center))

        obs = self._get_obs()
        info = {"step": 0, "n_violations": 0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._mj is None:
            raise RuntimeError("Call reset() before step().")

        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        d = self._mj_data
        m = self._mj_model

        # Apply action
        d.ctrl[:2] = action
        for _ in range(self.frame_skip):
            self._mujoco.mj_step(m, d)
        self._step_count += 1

        pos = d.qpos[:2].copy()

        # -- Reward components ------------------------------------------
        reward = 0.0

        # Dense approach bonus
        dist_to_goal = float(np.linalg.norm(pos - self._goal_center))
        reward += 0.2 * (self._prev_dist_to_goal - dist_to_goal)
        self._prev_dist_to_goal = dist_to_goal

        # Living cost
        reward -= 0.05

        # Obstacle penetration penalty
        penetrations = self._obstacle_penetrations(pos)
        n_violations = int((penetrations > 0).sum())
        if n_violations > 0:
            reward += self.penalty_per_step_inside * n_violations

        # -- Termination ------------------------------------------------
        terminated = False
        truncated = False

        if self._in_goal(pos):
            reward += self.goal_reward
            terminated = True
        elif self._step_count >= self.max_steps:
            reward += self.catastrophe_penalty
            truncated = True  # catastrophic timeout

        t_norm = max(0.0, (self.max_steps - self._step_count) / self.max_steps)
        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "n_violations": n_violations,
            "penetration_depths": penetrations.tolist(),
            "dist_to_goal": dist_to_goal,
            "time_remaining_norm": t_norm,
            "goal_reached": terminated,
            "timeout": truncated and not terminated,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass  # Rendering handled externally

    def close(self) -> None:
        if hasattr(self, "_tmpfile") and self._tmpfile:
            try:
                os.unlink(self._xml_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Properties for CDDP / CBF integration
    # ------------------------------------------------------------------

    @property
    def obstacle_centers(self) -> np.ndarray:
        return self._obstacle_centers.copy()

    @property
    def obstacle_radii(self) -> np.ndarray:
        return self._obstacle_radii.copy()

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_center.copy()

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return 2

    def get_robot_pos(self) -> np.ndarray:
        """Return current (x, y) position."""
        return self._mj_data.qpos[:2].copy()

    def get_robot_vel(self) -> np.ndarray:
        """Return current (vx, vy) velocity."""
        return self._mj_data.qvel[:2].copy()

    def get_time_remaining_norm(self) -> float:
        return max(0.0, (self.max_steps - self._step_count) / self.max_steps)

    # ------------------------------------------------------------------
    # Detour path geometry (for visualization)
    # ------------------------------------------------------------------

    @staticmethod
    def detour_waypoints() -> list[tuple[float, float]]:
        """Waypoints for the safe arc route (y > 0 half-circle-ish)."""
        return [
            (0.0, 0.0),
            (1.5, 2.0),
            (4.0, 3.0),
            (6.5, 2.0),
            (8.0, 0.0),
        ]

    @staticmethod
    def direct_waypoints() -> list[tuple[float, float]]:
        """Waypoints for the direct risky route (through the crowd)."""
        return [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]
