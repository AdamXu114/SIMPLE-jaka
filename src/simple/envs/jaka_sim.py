"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

MuJoCo-only Gym environment for the Jaka Khan Mini robot.
No IsaacSim — pure physics simulation for external policy control.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from gymnasium import spaces

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim


class JakaSimEnv(BaseDualSim):
    """MuJoCo-only simulation environment for the Jaka Khan Mini robot.

    This environment runs pure MuJoCo physics with no IsaacSim rendering.
    It is designed to be driven by an external policy through a network interface.

    Observations:
        - joint_qpos: current joint positions (27,)
        - joint_qvel: current joint velocities (27,)

    Actions:
        - A dict with "type" and "parameters" keys (passed through to robot.apply_action).
        - Supported types: "position" (PD target positions), "torque" (raw torques),
          "reset_qpos" (direct joint position set for reset).
    """

    _success: bool

    def __init__(
        self,
        task: str | Task,
        sim_mode: str = "mujoco",
        headless: bool = True,
        *args,
        **kwargs,
    ) -> None:
        # Force MuJoCo-only mode (no IsaacSim)
        if "isaac" in sim_mode:
            import warnings
            warnings.warn(
                f"JakaSimEnv only supports mujoco mode, got '{sim_mode}'. "
                "Forcing sim_mode='mujoco'."
            )
            sim_mode = "mujoco"
        super().__init__(task, sim_mode, headless, *args, **kwargs)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        qpos = np.asarray(
            list(self.mujoco.get_robot_qpos().values()), dtype=np.float32
        )
        qvel = np.asarray(
            list(self.task.robot.get_robot_qvel().values()), dtype=np.float32
        ) if hasattr(self.task.robot, "get_robot_qvel") else np.zeros_like(qpos)

        obs = {
            "joint_qpos": qpos,
            "joint_qvel": qvel,
        }
        return obs

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if hasattr(self.mujoco, "mj_objects"):
            for k, v in self.mujoco.mj_objects.items():
                info[str(k)] = np.concatenate([v.xpos, v.xquat])
        return info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.task.reset(seed, options)
        self.mujoco.update_layout()
        self.mujoco.step(render=False)

        self.step_count = 0
        self._success = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> tuple[
        Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]
    ]:
        """Execute one simulation step.

        Args:
            action: Dict with "type" (str) and "parameters" (dict).
                    Passed to robot.apply_action().

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.mujoco.apply_action(action)
        self.mujoco.step(render=False)

        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()

        reward = self.task.compute_reward(info, mujoco_env=self.mujoco)
        terminated = self.task.check_success(info, mujoco_env=self.mujoco)
        truncated = False

        self._success = terminated
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render (no-op: no rendering in this env)
    # ------------------------------------------------------------------
    def render(self) -> Dict[str, np.ndarray]:
        return {}

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.mujoco.close()
        super().close()
