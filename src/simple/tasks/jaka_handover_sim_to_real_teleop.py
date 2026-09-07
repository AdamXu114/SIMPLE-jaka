"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Standalone Jaka teleop task: hand over the object (sim-to-real).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor, ObjectActor
from simple.core.layout import Layout
from simple.core.object import Object
from simple.core.scene import Scene
from simple.core.task import Task
from simple.dr import *
from simple.dr.manager import TabletopGraspDRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.tasks.jaka_vla_cameras import jaka_vla_sensor_cfgs
from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("jaka_handover_sim_to_real_teleop")
class JakaHandoverSimToRealTeleop(Task):
    uid: str = "jaka_handover_sim_to_real_teleop"
    label: str = "Jaka Handover SimToReal Teleop"
    description: str = (
        "A task where the Jaka robot must hand over the object in the sim-to-real "
        "table setup."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
        "need_gravity": True,
        "max_episode_steps": 800,
        "keep_ground_at_world_origin": True,
    }

    robot_cfg: dict[str, Any] = dict(uid="jaka")

    sensor_cfgs: dict[str, SensorCfg] = jaka_vla_sensor_cfgs()

    dr_cfgs = dict(
        language=LanguageDRCfg(instructions=["hand over the object"]),
        target=TargetDRCfg(asset_id="graspnet1b:10"),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-1.20, 0.0, 0.0], high=[-1.26, 0.0, 0.0]),
            target_region=Box(low=[-0.28, -0.05], high=[-0.24, 0.03]),
            target_stable_indices=[0], target_rotate_z=Box(low=-0.15, high=0.15),
            placement_attempts=500,
        ),
        camera=CameraDRCfg(cam_id="jaka_camera"),
        scene=TabletopSceneDRCfg(
            scene_mode="random",
            table_size=Box(low=[0.40, 0.80, 0.015], high=[0.40, 0.80, 0.015]),
            table_position=Box(low=[-0.17, 0.0], high=[-0.17, 0.0]),
            table_height=Box(low=0.7575, high=0.7575),
            rotation_z=Box(low=0.0, high=0.0),
            room_choices=["hssd:scene0"], scene_manager="hssd",
            randomize_scene_pose=False,
        ),
        lighting=LightingDRCfg(
            light_mode="random", light_num=(2, 3),
            light_color_temperature=Box(low=2001, high=8001),
            light_intensity=Box(low=1e4 * 0.8, high=1e4 * 1.2),
            light_radius=Box(0.08, 0.12), light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.5, -0.5, 1.3), (0.5, 0.5, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(material_mode="rand_objects"),
    )

    def __init__(
        self,
        robot_uid: str = "jaka",
        scene_uid: str | Scene | None = None,
        target: str | None = None,
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None
        self._contact_started = False

        self.robot_cfg.update(dict(uid=robot_uid))
        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        if target is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target

            target_id = self.dr_cfgs["target"].asset_id.split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if distractor_cfg is not None and isinstance(
                distractor_cfg, DistractorDRCfg
            ):
                if distractor_cfg.exclude is None:
                    distractor_cfg.exclude = []
                if target_id not in distractor_cfg.exclude:
                    distractor_cfg.exclude.append(target_id)

        drmgr = TabletopGraspDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            *args,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction

    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.dof,), dtype=np.float32
            ),
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        lang_dr = self.dr.get_randomizer("language")
        if lang_dr is not None:
            self._instruction = lang_dr(split).format(
                self._target.asset.name if self._target else "object"
            )
        else:
            self._instruction = "perform the task."
        self._init_target_height = None
        self.reward = 0
        self.robot.reset()

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self._target is None:
            return 0.0
        h = info.get("target", [0, 0, 0])[2]
        if self._init_target_height is None:
            self._init_target_height = h
        return float(np.clip((h - self._init_target_height) / 0.04, 0, 1))

    def preload_objects(self) -> list[Actor]:
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self):
        raise NotImplementedError
