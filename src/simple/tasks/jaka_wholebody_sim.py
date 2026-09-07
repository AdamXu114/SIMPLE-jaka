"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Minimal simulation-only task for the Jaka Khan Mini robot.
No domain randomization, no IsaacSim — pure MuJoCo physics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from simple.assets.primitive import Box
from simple.core.actor import Actor
from simple.core.layout import Layout
from simple.core.robot import Robot
from simple.core.scene import TabletopScene
from simple.core.task import Task
from simple.dr.manager import DRManager
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("jaka_wholebody_sim")
class JakaWholebodySim(Task):
    """Minimal simulation task for the Jaka Khan Mini robot.

    This task provides a bare-bones MuJoCo simulation environment
    with no domain randomization, no target objects, and no success criteria.
    It is intended to be driven by an external policy via an interface layer.
    """

    uid: str = "jaka_wholebody_sim"
    label: str = "Jaka Wholebody Simulation"
    description: str = (
        "A minimal simulation task for the Jaka Khan Mini humanoid robot. "
        "No domain randomization — raw physics simulation only."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "version": 1.0,
        "max_episode_steps": 10000,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="jaka",
    )

    # No sensors by default (add cameras here if needed)
    sensor_cfgs: dict[str, Any] = {}

    # No domain randomization
    dr_cfgs: dict[str, Any] = {}

    def __init__(
        self,
        robot_uid: str = "jaka",
        split: str = "train",
        render_hz: int | None = None,
        physics_dt: float = 0.002,
        *args,
        **kwargs,
    ) -> None:
        self._instruction = "Stand and wait for commands."
        self._layout: Layout | None = None
        self._robot: Robot | None = None

        self.robot_cfg.update(dict(uid=robot_uid))
        self._robot = RobotRegistry.make(**self.robot_cfg)

        # Empty DR manager (no randomization)
        drmgr = DRManager(level=0)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            physics_dt=physics_dt,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def robot(self) -> Robot:
        assert self._robot is not None
        return self._robot

    @robot.setter
    def robot(self, value: Robot) -> None:
        self._robot = value

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32
            ),
            "joint_qvel": spaces.Box(
                -np.inf, np.inf, shape=(self.robot.dof,), dtype=np.float32
            ),
        })

    # ------------------------------------------------------------------
    # Reset — minimal: just robot + floor
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        """Reset the simulation to initial state.

        Sets up a minimal layout with just the robot positioned at the origin
        and a ground plane. No objects, no randomization.
        """
        split = self.metadata.get("split", "train")

        if options is not None and "state_dict" in options:
            state_dict = options["state_dict"]
            assert state_dict.get("uid") == self.uid, (
                f"Wrong state dict for task {self.uid}"
            )
            self.dr.load_state_dict(state_dict, dr_level=options.get("dr_level"))
        else:
            self.dr.reset(seed=seed)

        # Build minimal layout
        self._layout = Layout()
        self._layout.add_robot(self.robot)

        # Provide a minimal scene with a ground-level table
        # (required by MuJoCo simulator for ground-plane placement)
        ground_table = Box(
            size=[2.0, 2.0, 0.05],
            position=[0.0, 0.0, -0.025],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )
        scene = TabletopScene()
        scene.table = ground_table
        self._layout.scene = scene
        self._layout.add_primitive("table", ground_table)

    # ------------------------------------------------------------------
    # Task logic (no-op for raw simulation)
    # ------------------------------------------------------------------
    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        """Raw simulation has no success criteria."""
        return False

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """Raw simulation has no reward."""
        return 0.0

    def preload_objects(self) -> list[Actor]:
        """No objects to preload."""
        return []

    def decompose(self):
        """No subtask decomposition for raw simulation."""
        raise NotImplementedError("Raw simulation does not support task decomposition")
