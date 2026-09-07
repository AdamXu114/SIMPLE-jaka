"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka Khan Mini with CuRobo kinematics for motion planning (datagen CLI).
Inherits from Jaka base and adds CuRoboMixin for IK/FK/planning support.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from simple.core.controller import Controller
from simple.robots.jaka import Jaka
from simple.robots.mixin import CuRoboMixin
from simple.robots.protocols import (
    BatchPlannable,
    Controllable,
    Graspable,
    HasDexterousHand,
    HasKinematics,
    HeadCamMountable,
    Humanoid,
)
from simple.robots.registry import RobotRegistry


@RobotRegistry.register("jaka_curobo")
class JakaCuRobo(Jaka, CuRoboMixin, Graspable, HasKinematics):
    """Jaka Khan Mini with CuRobo motion planning support.

    Use this variant for tasks that require motion planning
    (datagen, plan, etc.). For ZMQ bridge / RL policy, use
    the base ``Jaka`` class.
    """

    uid: str = "jaka_curobo"
    label: str = "Jaka Khan Mini (CuRobo)"

    # CuRobo kinematics config
    robot_cfg: Any = "robots/jaka/curobo/khan.yml"

    # EE info for grasp planning
    eef_prim_path: str = "Right_wrist_yaw__Link"
    pregrasp_distance: List[float] = [0.05, 0.08]
    robot_eef_offset: float = 0.0

    # No dexterous hand — wrists are terminal
    hand_dof: int = 0

    def __init__(self) -> None:
        Jaka.__init__(self)
        CuRoboMixin.__init__(self)

    def update_ee_link(self, hand_uid: str) -> None:
        """Switch between left/right arm end-effector for planning."""
        if "left" in hand_uid:
            ee_link = self.LEFT_ARM_EE_LINK
            link_names = [
                "waist_yaw_Link",
                "Left_wrist_yaw_Link",
                "Right_wrist_yaw__Link",
            ]
        else:
            ee_link = self.RIGHT_ARM_EE_LINK
            link_names = [
                "waist_yaw_Link",
                "Right_wrist_yaw__Link",
                "Left_wrist_yaw_Link",
            ]

        self.robot_cfg["kinematics"]["ee_link"] = ee_link
        self.robot_cfg["kinematics"]["link_names"] = link_names

    # ------------------------------------------------------------------
    # Graspable protocol
    # ------------------------------------------------------------------
    def get_grasp_pose_wrt_robot(
        self, grasp_info: dict, pregrasp: bool = False, robot_pose=None
    ):
        """Compute grasp pose relative to robot base."""
        import transforms3d as t3d

        assert robot_pose is not None, "robot_pose required for grasp planning"

        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[:3, 3] = np.array([0, 0, -self.robot_eef_offset], dtype=np.float32)

        T_grasp_ee = np.eye(4, dtype=np.float32)

        R_world_grasp = t3d.quaternions.quat2mat(grasp_info["orientation"])
        T_world_grasp = np.eye(4, dtype=np.float32)
        T_world_grasp[:3, 3] = (
            grasp_info["position"] + grasp_info["depth"] * R_world_grasp[:, 0]
        )
        T_world_grasp[:3, :3] = R_world_grasp

        if robot_pose is None:
            T_world_robot = np.eye(4, dtype=np.float32)
        else:
            T_world_robot = robot_pose

        if pregrasp:
            T_grasp_pregrasp = np.eye(4, dtype=np.float32)
            T_grasp_pregrasp[0, 3] = -np.random.uniform(*self.pregrasp_distance)
            T_robot_hand = (
                np.linalg.inv(robot_pose)
                @ T_world_grasp
                @ T_grasp_pregrasp
                @ T_grasp_ee
                @ T_ee_hand
            )
        else:
            T_robot_hand = (
                np.linalg.inv(robot_pose)
                @ T_world_grasp
                @ T_grasp_ee
                @ T_ee_hand
            )

        grasp_pos = T_robot_hand[:3, 3]
        grasp_ori = t3d.quaternions.mat2quat(T_robot_hand[:3, :3])
        return (grasp_pos, grasp_ori)

    def open_gripper(self):
        pass  # No gripper

    def close_gripper(self):
        pass  # No gripper
