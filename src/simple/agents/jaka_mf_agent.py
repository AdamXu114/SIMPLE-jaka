"""Teleoperation agent that drives the Jaka whole-body MF RL policy in-process.

The policy is the *same whole-body MF strategy* as sim2real-jaka: a single
ONNX that outputs 27 joint actions from the 620-dim (v1) observation built on
the pico motion buffer. The agent holds:

  - ``RealtimeMotionBuffer`` — subscribes to the pico retarget motion stream.
  - ``BasePolicy`` — control-mode state machine + obs + ONNX; writes PD targets
    into the in-process ``ZMQSimBridge.create_without_zmq`` every 50 Hz.
  - The robot's ``step()`` loop applies PD + elastic band every 500 Hz substep.

Mirrors ``PicoDecoupledAgent`` but is simpler: no decoupled upper/lower split.
"""

from __future__ import annotations

import numpy as np

from simple.agents.base_agent import BaseAgent
from simple.core.action import ActionCmd
from simple.jaka_rl.base_policy import BasePolicy
from simple.jaka_rl.config import ZMQ_DEFAULT_QPOS


class JakaMFAgent(BaseAgent):
    """Runs the whole-body MF policy against a Jaka robot in SIMPLE."""

    def __init__(
        self,
        robot,
        *,
        policy_config: str,
        policy_model: str = "",
        motion_zmq_connect: str = "tcp://127.0.0.1:28701",
        controller: str = "pico",
        inference_backend: str = "onnx-cpu",
        rl_rate: float = 50.0,
        pico_zmq_connect: str = "tcp://127.0.0.1:5592",
        motion_backend: str = "zmq",
        num_substeps: int = 10,
        bridge=None,
    ):
        super().__init__(robot)
        self.num_substeps = int(num_substeps)
        if robot.num_substeps != self.num_substeps:
            robot.num_substeps = self.num_substeps

        # --- in-process PD bridge (no ZMQ) ---
        if bridge is None:
            from simple.interfaces.zmq_bridge import ZMQSimBridge

            bridge = ZMQSimBridge.create_without_zmq(
                robot.mjModel, robot.mjData, list(robot.joint_names)
            )
        bridge.has_received_command = True  # policy owns the command target
        robot.attach_mf_bridge(bridge)
        self._bridge = bridge

        # --- whole-body MF policy ---
        # (its StateProcessor owns the RealtimeMotionBuffer for motion_backend=zmq)
        from simple.jaka_rl.base_policy import BasePolicyArgs

        args = BasePolicyArgs(
            policy_config=policy_config,
            policy_model=policy_model,
            robot="jaka",
            rl_rate=rl_rate,
            inference_backend=inference_backend,
            controller=controller,
            pico_zmq_connect=pico_zmq_connect,
            motion_backend=motion_backend,
            motion_zmq_connect=motion_zmq_connect,
        )
        self.policy = BasePolicy(
            args,
            state_getter=robot.state_getter,
            bridge=bridge,
            mj_model=robot.mjModel,
            mj_data=robot.mjData,
            default_qpos=np.asarray(ZMQ_DEFAULT_QPOS, dtype=np.float32),
        )

        # Band + drop state.
        self._enable_band()

    # ------------------------------------------------------------------
    # Band helpers (mirror run_jaka_sim_server.py semantics)
    # ------------------------------------------------------------------
    def _enable_band(self) -> None:
        if self.robot.elastic_band is None:
            self.robot._init_band_locally(self.robot.mjData.qpos[:2])

    @property
    def reset_requested(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def get_action(self, observation, instruction=None, **kwargs):
        """Run one 50 Hz whole-body MF policy step and return the joint target.

        The policy reads the robot state, builds the MF observation, runs ONNX,
        and writes ``cmd_q`` (27 positions in ``robot.joint_names`` order) into
        the in-process bridge. The robot's ``step()`` applies PD per substep.
        """
        self.policy.step()
        cmd_q = np.asarray(self.policy.action_manager.cmd_q, dtype=np.float32)
        target_qpos = dict(zip(self.robot.joint_names, cmd_q.tolist()))
        return ActionCmd("position", target_qpos=target_qpos)

    def reset_policy(self) -> None:
        """Reset per-episode policy state (history, ref-init, prev action)."""
        self.policy.reset()

    def rebind_bridge(self) -> None:
        """Re-point the PD bridge at the robot's current ``mjModel``/``mjData``.

        Call after ``env.reset()``, which recompiles the MuJoCo model: the bridge is
        mutated in place, so the policy's ``action_manager.bridge`` (same object) and
        ``robot._mf_bridge`` all follow, and ``has_received_command`` stays True.
        """
        self._bridge.rebind(self.robot.mjModel, self.robot.mjData)
        self.robot.attach_mf_bridge(self._bridge)

    def close(self) -> None:
        try:
            self.policy.close()
        except Exception:
            pass


__all__ = ["JakaMFAgent"]
