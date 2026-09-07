"""Action manager for the Jaka MF policy (in-process).

Ported from sim2real-jaka ``rl_policy/utils/command_sender.py`` and adapted so
commands are written directly to an in-process bridge (``cmd_q`` / ``cmd_dq`` /
``cmd_tau`` / ``cmd_kp`` / ``cmd_kd``) instead of being published over ZMQ.
Reset requests are applied directly to the bridge's ``mj_data``.
"""

from __future__ import annotations

import numpy as np

from simple.jaka_rl.config import DEFAULT_JOINT_POS, JOINT_KD, JOINT_KP
from simple.jaka_rl.strings import match_param


class ActionManager:
    def __init__(self, joint_names, policy_config, bridge=None):
        self.joint_names = list(joint_names)
        self.policy_config = policy_config
        self.bridge = bridge

        joint_kp_dict = policy_config.get("joint_kp", JOINT_KP)
        self.joint_kp_unitree = np.array(
            [match_param(n, joint_kp_dict) for n in self.joint_names], dtype=np.float32
        )
        joint_kd_dict = policy_config.get("joint_kd", JOINT_KD)
        self.joint_kd_unitree = np.array(
            [match_param(n, joint_kd_dict) for n in self.joint_names], dtype=np.float32
        )
        default_joint_pos_dict = policy_config.get("default_joint_pos", DEFAULT_JOINT_POS)
        self.default_joint_pos_unitree = np.zeros(
            len(self.joint_names), dtype=np.float32
        )
        for jname, jval in default_joint_pos_dict.items():
            if jname in self.joint_names:
                self.default_joint_pos_unitree[self.joint_names.index(jname)] = float(jval)

        self.InitLowCmd()

    def InitLowCmd(self):
        self.cmd_q = np.zeros(len(self.joint_names))
        self.cmd_dq = np.zeros(len(self.joint_names))
        self.cmd_tau = np.zeros(len(self.joint_names))
        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau, reset_qpos=None, reset_qvel=None):
        self.cmd_q[:] = cmd_q
        self.cmd_dq[:] = cmd_dq
        self.cmd_tau[:] = cmd_tau

        if self.bridge is not None:
            if hasattr(self.bridge, "cmd_q"):
                self.bridge.cmd_q[:] = np.asarray(cmd_q, dtype=np.float32)
            if hasattr(self.bridge, "cmd_dq"):
                self.bridge.cmd_dq[:] = np.asarray(cmd_dq, dtype=np.float32)
            if hasattr(self.bridge, "cmd_tau"):
                self.bridge.cmd_tau[:] = np.asarray(cmd_tau, dtype=np.float32)
            if hasattr(self.bridge, "cmd_kp"):
                self.bridge.cmd_kp[:] = self.joint_kp_unitree
            if hasattr(self.bridge, "cmd_kd"):
                self.bridge.cmd_kd[:] = self.joint_kd_unitree

        if reset_qpos is not None and reset_qvel is not None:
            self._apply_reset(reset_qpos, reset_qvel)

    def _apply_reset(self, reset_qpos, reset_qvel):
        import logging

        import mujoco

        bridge = self.bridge
        mj_data = getattr(bridge, "mj_data", None)
        if mj_data is None:
            return
        if getattr(bridge, "reset_requested", None) is not None:
            bridge.reset_requested = True

        # reset_qpos/reset_qvel cover the robot (base 7 + joints 27). The user
        # wants `[` (align) to re-pose only the JOINTS to the reference frame,
        # KEEPING the base pos/quat the scene placed the robot at (no teleport).
        # So we write only the joint slice (reset_qpos[7:]) and zero the robot
        # velocities; qpos[:7] (base) stays. Objects/articulated hinges (after
        # the robot slice) are untouched.
        nq = reset_qpos.size
        nv = reset_qvel.size
        if nq <= mj_data.qpos.size and nv <= mj_data.qvel.size:
            mj_data.qpos[7:nq] = reset_qpos[7:]
            mj_data.qvel[:nv] = reset_qvel
            mujoco.mj_forward(bridge.mj_model, mj_data)
        else:
            logging.getLogger(__name__).warning(
                "Reset qpos size mismatch: got %d, expected %d",
                nq,
                mj_data.qpos.size,
            )


__all__ = ["ActionManager"]
