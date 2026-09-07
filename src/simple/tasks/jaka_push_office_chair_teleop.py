"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka Khan Mini teleop task — push the office chair.
Mirrors the G1 push office chair teleop task with the robot swapped to "jaka".
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
from simple.dr import *  # noqa: F403
from simple.dr.manager import TabletopGraspDRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.tasks.jaka_vla_cameras import jaka_vla_sensor_cfgs
from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("jaka_push_office_chair_teleop")
class JakaPushOfficeChairTeleop(Task):
    uid: str = "jaka_push_office_chair_teleop"
    label: str = "Jaka PushOfficeChair Teleop"
    description: str = (
        "A task where the Jaka robot must push the office chair."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "dr_level": 0,
        "version": 1.0,
        "need_gravity": True,
        "max_episode_steps": 800,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="jaka",
    )

    sensor_cfgs: dict[str, SensorCfg] = jaka_vla_sensor_cfgs()

    dr_cfgs: dict[str, Any] = dict(
        language=LanguageDRCfg(instructions=["push the office chair"]),
        target=TargetDRCfg(asset_id="graspnet1b:10"),
        distractors=DistractorDRCfg(
            res_id="graspnet1b", number_of_distractors=1,
            allow_duplicates=False, exclude=["10"],
        ),
        articulated=ArticulatedObjectDrCfg(asset_id="articulated:0"),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-2.2, 0, 0.0], high=[-2.3, 0.0, 0.0]),
            target_region=Box(low=[0.6, -0.3], high=[0.7, -0.4]),
            distractors_region=Box(low=[0.5, -0.3], high=[0.7, 0.3]),
            articulated_region=Box(low=[-1.05, -0.3, 0.3], high=[-1, 0.3, 0.3]),
            target_stable_indices=[0], target_rotate_z=Box(low=-0.15, high=0.15),
            articulated_rotate_z=Box(low=-0.12, high=0.12),
        ),
        camera=CameraDRCfg(cam_id="jaka_camera"),
        scene=TabletopSceneDRCfg(
            table_position=Box(low=[1.2, 0], high=[1.2, 0]),
            table_height=Box(low=0.67, high=0.68),
            room_choices=["hssd:scene3"], scene_manager="hssd",
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
        material=MaterialDRCfg(material_mode="rand_all"),
    )

    def __init__(
        self,
        robot_uid: str = "jaka",
        scene_uid: str | Scene | None = None,
        target: str | None = None,
        controller_uid: str = "pd_joint_pos",  # pd_joint_vel, pd_ee_pose, pd_delta_ee_pose
        split: str = "train",  # train, val, test
        render_hz: int | None = None,
        dr_level: int = 0,
        success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        # lazy init instance variables
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None
        self._contact_started = False

        self.robot_cfg.update(
            dict(
                uid=robot_uid,
                # controller_uid=controller_uid,
            )
        )

        self.reward = 0
        self.success_criteria = success_criteria

        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        # domain randomization confs
        if target is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = target  # type:ignore

            # Exclude target object from distractors to avoid duplicates
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
        """Returns the layout of the task."""
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction  # type: ignore

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
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)  # type: ignore
        self._init_target_height = None
        self.reward = 0
        self.robot.reset()

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self._target is None:
            return 0.0
        h = info.get("target", [0, 0, 0])[2]
        if self._init_target_height is None:
            self._init_target_height = h
        return float(np.clip((h - self._init_target_height) / 0.04, 0, 1))

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def decompose(self):
        raise NotImplementedError
