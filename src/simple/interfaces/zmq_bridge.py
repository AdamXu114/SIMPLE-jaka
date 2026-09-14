"""
ZMQ Bridge: connects SIMPLE simulation to sim2real-jaka RL policy stack.

Uses binary LowStateMessage/LowCmdMessage protocol (compatible with
sim2real-jaka project's SimulationBridge).

Data flow:
  SIMPLE MuJoCo sim ──LowState (PUB:5590)──→ RL policy (sim2real-jaka)
  RL policy (sim2real-jaka) ──LowCmd (PUB:5591)──→ SIMPLE MuJoCo sim
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import mujoco
import numpy as np
import zmq

from simple.interfaces.messages import LowCmdMessage, LowStateMessage
from simple.jaka_rl.config import DEFAULT_JOINT_POS, JOINT_KD, JOINT_KP

logger = logging.getLogger(__name__)

# PD gains matching sim2real-jaka policy config (single source of truth: jaka_rl.config)
DEFAULT_KP: Dict[str, float] = dict(JOINT_KP)
DEFAULT_KD: Dict[str, float] = dict(JOINT_KD)
DEFAULT_JOINT_POS: Dict[str, float] = dict(DEFAULT_JOINT_POS)


def _match_param(joint_name: str, param_dict: Dict[str, float]) -> float:
    """Match joint name against regex patterns in param_dict."""
    import re

    for pattern, value in param_dict.items():
        if pattern == ".*":
            return value
        if re.fullmatch(pattern, joint_name):
            return value
    if ".*" in param_dict:
        return param_dict[".*"]
    raise KeyError(f"No value for joint: {joint_name}")


class ZMQSimBridge:
    """Bridge between SIMPLE MuJoCo simulation and sim2real-jaka ZMQ protocol.

    Usage within the simulation loop::

        bridge = ZMQSimBridge(mj_model, mj_data, joint_names, port_pair=(5590, 5591))
        while running:
            bridge.publish_low_state()
            bridge.poll_and_apply_cmd()
            mujoco.mj_step(mj_model, mj_data)
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        joint_names: list[str],
        *,
        low_state_port: int = 5590,
        low_cmd_port: int = 5591,
        low_state_bind_addr: str = "*",
        low_cmd_host: str = "127.0.0.1",
        kp: Optional[Dict[str, float]] = None,
        kd: Optional[Dict[str, float]] = None,
        default_joint_pos: Optional[Dict[str, float]] = None,
        bind_zmq: bool = True,
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)

        # PD gains
        self._kp_dict = kp or DEFAULT_KP
        self._kd_dict = kd or DEFAULT_KD
        self._default_joint_pos = default_joint_pos or DEFAULT_JOINT_POS

        # Build joint-indexed Kp/Kd/effort arrays
        self._kp = np.array(
            [_match_param(n, self._kp_dict) for n in self.joint_names],
            dtype=np.float32,
        )
        self._kd = np.array(
            [_match_param(n, self._kd_dict) for n in self.joint_names],
            dtype=np.float32,
        )

        # Resolve MuJoCo joint/actuator addresses
        self._init_mujoco_indices()

        # Command buffers
        self.cmd_q = np.zeros(self.num_joints, dtype=np.float32)
        self.cmd_dq = np.zeros(self.num_joints, dtype=np.float32)
        self.cmd_tau = np.zeros(self.num_joints, dtype=np.float32)
        # Default to the config gains; a received LowCmd overwrites them with the
        # policy's own kp/kd (which are the same values in the Jaka pipeline).
        self.cmd_kp = self._kp.copy()
        self.cmd_kd = self._kd.copy()
        self.has_received_command = False
        # Set True when a LowCmd carrying reset_qpos was received; the caller
        # (recorder / server) clears it after handling the episode boundary.
        self.reset_requested = False

        # Computed torques
        self.torques = np.zeros(self.num_joints, dtype=np.float32)

        # ZMQ setup (skipped in the no-ZMQ "PD-only" mode used by replay)
        self._pub = None
        self._sub = None
        if bind_zmq:
            self._ctx = zmq.Context.instance()

            # Publisher: low_state
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.setsockopt(zmq.SNDHWM, 1)
            self._pub.setsockopt(zmq.LINGER, 0)
            self._pub.bind(f"tcp://{low_state_bind_addr}:{low_state_port}")
            logger.info("LowState PUB → tcp://%s:%d", low_state_bind_addr, low_state_port)

            # Subscriber: low_cmd
            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.setsockopt(zmq.SUBSCRIBE, b"")
            self._sub.setsockopt(zmq.CONFLATE, 1)
            self._sub.setsockopt(zmq.RCVTIMEO, 0)
            self._sub.setsockopt(zmq.LINGER, 0)
            self._sub.connect(f"tcp://{low_cmd_host}:{low_cmd_port}")
            logger.info("LowCmd SUB ← tcp://%s:%d", low_cmd_host, low_cmd_port)

    # ------------------------------------------------------------------
    # MuJoCo index resolution
    # ------------------------------------------------------------------
    def _init_mujoco_indices(self) -> None:
        """Map joint names to MuJoCo qpos/qvel/actuator addresses."""
        mj_joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        mj_act_names = [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

        self._qpos_adrs: list[int] = []
        self._qvel_adrs: list[int] = []
        self._act_adrs: list[int] = []
        self._effort_limits: list[float] = []

        for name in self.joint_names:
            if name not in mj_joint_names:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model")
            jid = mj_joint_names.index(name)
            self._qpos_adrs.append(self.mj_model.jnt_qposadr[jid])
            self._qvel_adrs.append(self.mj_model.jnt_dofadr[jid])

        # Find motor actuator for each joint
        # Jaka XML uses "Left_hip_pitch_motor" → joint "Left_hip_pitch_joint"
        motor_to_joint: Dict[str, str] = {}
        for i in range(self.mj_model.nu):
            motor_name = self.mj_model.actuator(i).name
            joint_name = motor_name.replace("_motor", "_joint")
            motor_to_joint[joint_name] = motor_name

        for name in self.joint_names:
            motor_name = motor_to_joint.get(name, name)
            if motor_name in mj_act_names:
                self._act_adrs.append(mj_act_names.index(motor_name))
            else:
                raise KeyError(f"No actuator found for joint '{name}'")

        # Effort limits from XML
        for name in self.joint_names:
            jid = mj_joint_names.index(name)
            frc = self.mj_model.jnt_actfrcrange[jid]
            self._effort_limits.append(float(max(abs(frc[0]), abs(frc[1]))))

        # Root joint (free base) qpos/qvel addresses
        root_joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
                           if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if root_joint_names:
            root_jid = mj_joint_names.index(root_joint_names[0])
            self._root_qpos_adr = self.mj_model.jnt_qposadr[root_jid]
            self._root_qvel_adr = self.mj_model.jnt_dofadr[root_jid]
        else:
            self._root_qpos_adr = 0
            self._root_qvel_adr = 0

        # IMU sensor addresses
        self._imu_gyro_adr: Optional[int] = None
        self._imu_quat_adr: Optional[int] = None
        for i in range(self.mj_model.nsensor):
            s = self.mj_model.sensor(i)
            if s.objtype.item() == mujoco.mjtObj.mjOBJ_SITE:
                site_name = self.mj_model.site(s.objid.item()).name
                if site_name == "waist_imu":
                    if s.type.item() == mujoco.mjtSensor.mjSENS_GYRO:
                        self._imu_gyro_adr = self.mj_model.sensor_adr[i]
                    elif s.type.item() == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                        self._imu_quat_adr = self.mj_model.sensor_adr[i]

        logger.info(
            "MuJoCo indices: %d joints, root_qpos_adr=%d, IMU gyro=%s quat=%s",
            len(self._qpos_adrs),
            self._root_qpos_adr,
            self._imu_gyro_adr,
            self._imu_quat_adr,
        )

    # ------------------------------------------------------------------
    # Publish LowState
    # ------------------------------------------------------------------
    def publish_low_state(self) -> None:
        """Publish current simulation state as LowStateMessage."""
        joint_pos = self.mj_data.qpos[self._qpos_adrs].copy()
        joint_vel = self.mj_data.qvel[self._qvel_adrs].copy()
        joint_tau = self.mj_data.actuator_force[self._act_adrs].copy()

        # IMU quaternion (w,x,y,z)
        if self._imu_quat_adr is not None:
            root_quat = self.mj_data.sensordata[
                self._imu_quat_adr : self._imu_quat_adr + 4
            ].copy()
        else:
            root_quat = self.mj_data.qpos[
                self._root_qpos_adr + 3 : self._root_qpos_adr + 7
            ].copy()

        # IMU gyro (angular velocity in body frame, x,y,z)
        if self._imu_gyro_adr is not None:
            root_ang_vel = self.mj_data.sensordata[
                self._imu_gyro_adr : self._imu_gyro_adr + 3
            ].copy()
        else:
            root_ang_vel = self.mj_data.qvel[
                self._root_qvel_adr + 3 : self._root_qvel_adr + 6
            ].copy()

        msg = LowStateMessage(
            quaternion=root_quat.astype(np.float32),
            gyroscope=root_ang_vel.astype(np.float32),
            joint_positions=joint_pos.astype(np.float32),
            joint_velocities=joint_vel.astype(np.float32),
            joint_torques=joint_tau.astype(np.float32),
            tick=int(self.mj_data.time * 1e3),
        )
        try:
            self._pub.send(msg.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass

    def rebind(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
        """Re-point at a freshly compiled model/data pair.

        ``MujocoSimulator._setup_scene`` (run by ``env.reset()``) recompiles
        ``mjModel``/``mjData``, so the object refs and the qpos/qvel/actuator addresses
        resolved in :meth:`__init__` go stale. Without this, :meth:`apply_pd` would write
        torques into the *dead* ``MjData`` and the new simulation would receive zero
        torque — the robot would simply go limp after a reset.

        Command buffers, gains, and ``has_received_command`` are preserved, and the ZMQ
        sockets are untouched (only the MuJoCo indices are re-resolved).
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self._init_mujoco_indices()

    def get_joint_positions(self) -> np.ndarray:
        """Return the 27 joint positions in ``joint_names`` order.

        Uses the resolved qpos addresses, which is safe in scene mode where the
        model also contains object/articulated joints after the robot joints.
        """
        return self.mj_data.qpos[self._qpos_adrs].copy()

    # ------------------------------------------------------------------
    # Receive and apply LowCmd
    # ------------------------------------------------------------------
    def poll_and_apply_cmd(self) -> bool:
        """Non-blocking poll for LowCmdMessage; compute PD torques if received.

        Returns True if a new command was processed.
        """
        updated = False
        while True:
            try:
                data = self._sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                low_cmd = LowCmdMessage.from_bytes(data)
            except Exception:
                logger.warning("Failed to decode LowCmdMessage", exc_info=True)
                continue

            if low_cmd.q_target.size != self.num_joints:
                logger.warning(
                    "LowCmd size mismatch: got %d, expected %d",
                    low_cmd.q_target.size,
                    self.num_joints,
                )
                continue

            self.cmd_q[:] = low_cmd.q_target
            self.cmd_dq[:] = low_cmd.dq_target
            self.cmd_tau[:] = low_cmd.tau_ff
            self.cmd_kp[:] = low_cmd.kp
            self.cmd_kd[:] = low_cmd.kd
            updated = True

            # Handle reset request
            if low_cmd.reset_qpos is not None and low_cmd.reset_qvel is not None:
                self.reset_requested = True
                if low_cmd.reset_qpos.size == self.mj_data.qpos.size:
                    self.mj_data.qpos[:] = low_cmd.reset_qpos
                    self.mj_data.qvel[:] = low_cmd.reset_qvel
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    logger.info("Sim reset to commanded pose")
                else:
                    logger.warning(
                        "Reset qpos size mismatch: got %d, expected %d",
                        low_cmd.reset_qpos.size,
                        self.mj_data.qpos.size,
                    )

        if updated:
            self.has_received_command = True

        if self.has_received_command:
            self.apply_pd()

        return updated

    def apply_pd(self) -> None:
        """Compute PD torques from ``cmd_q``/``cmd_dq``/``cmd_tau`` and the *current*
        joint state, then write them to the MuJoCo actuators.

        Must be called once per physics substep — the low-level PD on the real
        robot / the recording server runs at the sim rate (500 Hz), so the torque
        is recomputed as the joints move toward their targets. No-op until a
        command has been received (``has_received_command`` is True).
        """
        if not self.has_received_command:
            return

        for i in range(self.num_joints):
            tau = (
                self.cmd_tau[i]
                + self.cmd_kp[i]
                * (self.cmd_q[i] - self.mj_data.qpos[self._qpos_adrs[i]])
                + self.cmd_kd[i]
                * (self.cmd_dq[i] - self.mj_data.qvel[self._qvel_adrs[i]])
            )
            self.torques[i] = float(
                np.clip(tau, -self._effort_limits[i], self._effort_limits[i])
            )

        # Apply torques to MuJoCo actuators
        for i, act_adr in enumerate(self._act_adrs):
            self.mj_data.ctrl[act_adr] = self.torques[i]

    @classmethod
    def create_without_zmq(
        cls,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        joint_names: list[str],
        **kwargs,
    ) -> "ZMQSimBridge":
        """Build a bridge that only applies PD (no ZMQ sockets).

        Used by the closed-loop replay so it runs the exact same per-substep PD
        law / gains / effort limits as the recording server's bridge — the one
        difference is the command target comes from ``cmd_q`` (set by the caller)
        instead of a received LowCmd.
        """
        return cls(mj_model, mj_data, joint_names, bind_zmq=False, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close ZMQ sockets (no-op in no-ZMQ mode)."""
        if self._pub is not None:
            self._pub.close()
        if self._sub is not None:
            self._sub.close()
        # Don't terminate context — it's a singleton
