"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka Khan Mini — 27-DOF humanoid robot.

Joint layout:
  Legs (6×2=12):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  Waist (1):      waist_yaw
  Arms (6×2=12):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw
  Neck (2):       neck_yaw, neck_pitch

When used with the ZMQ bridge (run_jaka_sim_server.py), PD control is handled
by the bridge and torques are written directly to MuJoCo motor actuators.
When used standalone via the Gym env, the robot applies its own PD control.

The Jaka Kham Mini also drives a whole-body MF RL policy in-process (see
``simple.jaka_rl`` + ``simple.agents.jaka_mf_agent``): the agent writes PD
targets into a ``ZMQSimBridge.create_without_zmq`` instance, and the robot's
``step(command)`` loop runs ``apply_pd`` each physics substep — mirroring how
``run_jaka_sim_server.py`` / ``replay_jaka.py`` do closed loop.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from simple.core.controller import Controller, ControllerCfg
from simple.core.robot import Robot
from simple.jaka_rl.config import DEFAULT_QPOS as _DEFAULT_QPOS
from simple.jaka_rl.config import JOINT_NAMES as _CONFIG_JOINT_NAMES
from simple.robots.controllers.qpos import PDJointPosControllerCfg
from simple.robots.protocols import Controllable, HeadCamMountable, Humanoid, WristCamMountable
from simple.robots.registry import RobotRegistry

# Joint groups (matching MuJoCo XML order)
LEFT_LEG_JOINTS = [
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint"]
LEFT_ARM_JOINTS = [
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
]
NECK_JOINTS = ["Neck_yaw_joint", "Neck_pitch_joint"]

ALL_JOINTS = list(_CONFIG_JOINT_NAMES)
assert list(LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + WAIST_JOINTS +
            LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + NECK_JOINTS) == ALL_JOINTS, (
    "jaka group order diverged from simple.jaka_rl.config.JOINT_NAMES"
)


class ElasticBand:
    """Virtual spring-damper from a fixed anchor to the robot's waist.

    Keys: 7→shorter, 8→longer, 9→toggle. Ported from run_jaka_sim_server.py.
    """

    def __init__(
        self,
        stiffness: float = 200.0,
        damping: float = 100.0,
        anchor_point: tuple[float, float, float] = (0.0, 0.0, 2.7),
    ):
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.point = np.array(anchor_point, dtype=np.float64)
        self.length: float = 0.0
        self.enable: bool = True

    def Advance(self, x: np.ndarray, dx: np.ndarray | None = None) -> np.ndarray:
        """Compute elastic force from a full pose or a position+velocity pair.

        Args:
            x: body pose (7) or position (3).
            dx: optional linear velocity (3); if x is a 7-vector, velocity is
                read from x[7:10].
        """
        if x.shape[0] >= 13:
            pos = x[:3]
            vel = x[7:10]
        elif x.shape[0] >= 7:
            pos = x[:3]
            vel = x[7:10] if x.shape[0] >= 10 else np.zeros(3)
        else:
            pos = x
            vel = dx if dx is not None else np.zeros(3)
        delta = self.point - np.asarray(pos, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-10:
            return np.zeros(3, dtype=np.float64)
        direction = delta / distance
        v = float(np.dot(np.asarray(vel, dtype=np.float64), direction))
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f


@RobotRegistry.register("jaka")
class Jaka(Robot, Humanoid, Controllable, HeadCamMountable, WristCamMountable):
    """Jaka Khan Mini — 27-DOF humanoid robot."""

    uid: str = "jaka"
    label: str = "Jaka Khan Mini"
    dof: int = 27

    mjcf_path: str = "robots/jaka/Khan_mini_simplified_new_bigfeet.xml"
    # IsaacSim rendering (USD from sim2real-jaka, defaultPrim = Khan_mini_simplified)
    usd_path: str = "robots/jaka/Khan_mini_simplified/Khan_mini_simplified.usd"
    robot_ns: str = "Khan_mini_simplified"
    eef_prim_path: str = "Left_wrist_yaw_Link"
    hand_prim_path: str = "Left_wrist_yaw_Link"
    robot_eef_offset: float = 0.0
    visulize_spheres: bool = False  # required False — Jaka has no kin_model
    z_offset: float = 0.793  # base_link height in MJCF (same as G1)

    # Control timings (mirror sim2real: sim_dt=0.002, decimation=10 → 50 Hz policy)
    sim_dt: float = 0.002
    viewer_dt: float = 0.02
    reward_dt: float = 0.02
    image_dt: float = 0.02
    num_substeps: int = 10
    use_floating_root_link: bool = True

    def jname_mujoco_to_isaac(self, mujoco_joint_name: str) -> str:
        """Map a MuJoCo joint name to the Isaac/USD articulation joint name.

        The Jaka USD (converted from khan.urdf) preserves the URDF joint names
        ``Left_hip_pitch_joint`` etc., which match MuJoCo — identity mapping.
        """
        return mujoco_joint_name

    # Camera
    wrist_camera_orientation: List[float] = [1.0, 0.0, 0.0, 0.0]
    head_camera_orientation: List[float] = [1.0, 0.0, 0.0, 0.0]

    @property
    def wrist_cam_link(self) -> str:
        return "Left_wrist_yaw_Link"

    @property
    def head_cam_link(self) -> str:
        return "Neck_pitch_Link"

    # Humanoid protocol
    LEFT_ARM_EE_LINK: str = "Left_wrist_yaw_Link"
    RIGHT_ARM_EE_LINK: str = "Right_wrist_yaw__Link"

    # Joint state — default to the MF policy's bent-leg standing pose (matching
    # run_jaka_sim_server.py DEFAULT_QPOS), so align/init modes start cleanly.
    _DEFAULT_QPOS_JOINTS = list(_DEFAULT_QPOS[7:])
    init_joint_states: dict[str, float] = dict(zip(ALL_JOINTS, _DEFAULT_QPOS_JOINTS))
    joint_names: List[str] = ALL_JOINTS
    joint_limits: dict[str, tuple[float, float]] = {}

    # Controller — flat PD (for Gym env mode; MF bridge bypasses this)
    controller_cfg: ControllerCfg = PDJointPosControllerCfg(
        joint_names=ALL_JOINTS,
        init_qpos=_DEFAULT_QPOS_JOINTS,
    )

    _controller: Controller | None = None

    # G1Wholebody compatibility: `command = None` signals to
    # MujocoSimulator that no AMO-policy lower-body control is active.
    command: Any = None
    is_replay: bool = False
    is_eval: bool = False

    def __init__(self, **kwargs) -> None:
        # Accept and drop forwarded kwargs (physics_dt / max_episode_steps /
        # sim_mode / headless / ...) that the env→task→RobotRegistry.make chain
        # passes down, matching G1Sonic. They are consumed by env/sim/task.
        super().__init__(self.uid, self.dof)
        self._joints: dict[str, Any] = {}
        self._actuators: dict[str, Any] = {}
        # Elastic band (optional; set up by the agent / runner).
        self.elastic_band: ElasticBand | None = None
        self.band_attached_link: int = -1
        # In-process MF PD bridge (ZMQSimBridge.create_without_zmq) owned by the
        # agent. The policy writes PD targets into it each 50 Hz step; the robot
        # calls apply_pd() every physics substep in step().
        self._mf_bridge: Any = None

    # ------------------------------------------------------------------
    # MF integration
    # ------------------------------------------------------------------
    def attach_mf_bridge(self, bridge) -> None:
        """Attach the in-process PD bridge used by the whole-body MF policy."""
        self._mf_bridge = bridge
        # Resolve the elastic-band attach body + set default anchor above base.
        import mujoco

        if self.mjData is not None:
            anchor = self.mjData.qpos[:2]
            self._init_band_locally(anchor)

    def _init_band_locally(self, anchor_xy) -> None:
        if self.elastic_band is None:
            self.elastic_band = ElasticBand()
            self.elastic_band.point[:2] = np.asarray(anchor_xy, dtype=np.float64)
        if self.mjModel is not None and self.band_attached_link < 0:
            import mujoco

            bid = mujoco.mj_name2id(
                self.mjModel, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_Link"
            )
            if bid < 0:
                bid = mujoco.mj_name2id(
                    self.mjModel, mujoco.mjtObj.mjOBJ_BODY, "base_link"
                )
            self.band_attached_link = int(bid)

    def prepare_obs(self) -> dict[str, Any]:
        """Build the priv info consumed by the MF policy state processor.

        Returns root IMU quat / angular velocity (waist_imu), joint positions &
        velocities in MuJoCo order, and the floating-base pose/vel.
        """
        import mujoco

        if self.mjData is None:
            raise RuntimeError("prepare_obs() called before mjData is set")
        m = self.mjModel
        d = self.mjData

        # IMU at waist_imu site (framequat + gyro), fall back to root qpos/qvel.
        gyro_adr = quat_adr = None
        if m is not None:
            for i in range(m.nsensor):
                s = m.sensor(i)
                if s.objtype.item() == mujoco.mjtObj.mjOBJ_SITE:
                    if m.site(s.objid.item()).name == "waist_imu":
                        if s.type.item() == mujoco.mjtSensor.mjSENS_GYRO:
                            gyro_adr = int(m.sensor_adr[i])
                        elif s.type.item() == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                            quat_adr = int(m.sensor_adr[i])
        if quat_adr is not None:
            root_quat_w = d.sensordata[quat_adr : quat_adr + 4].copy()
        else:
            root_quat_w = d.qpos[3:7].copy()
        if gyro_adr is not None:
            root_ang_vel_b = d.sensordata[gyro_adr : gyro_adr + 3].copy()
        else:
            root_ang_vel_b = d.qvel[3:6].copy()

        joint_pos = np.asarray(
            list(self.get_robot_qpos().values()), dtype=np.float32
        )
        joint_vel = np.asarray(
            list(self.get_robot_qvel().values()), dtype=np.float32
        )
        return {
            "root_quat_w": root_quat_w.astype(np.float32),
            "root_ang_vel_b": root_ang_vel_b.astype(np.float32),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "floating_base_pose": d.qpos[:7].astype(np.float32),
            "floating_base_vel": d.qvel[:6].astype(np.float32),
        }

    def state_getter(self):
        """Callable matching StateProcessor's state_getter signature."""
        obs = self.prepare_obs()
        return (
            obs["root_quat_w"],
            obs["root_ang_vel_b"],
            obs["joint_pos"],
            obs["joint_vel"],
        )

    # ------------------------------------------------------------------
    # Control setup
    # ------------------------------------------------------------------
    def setup_control(self, mjData, mjModel, **kwargs) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Wire MuJoCo motor actuators to joints.

        The Jaka XML uses motor names like 'Left_hip_pitch_motor' that
        differ from joint names like 'Left_hip_pitch_joint'.
        We build a mapping: joint_name → motor_name.
        """
        actuators: dict[str, Any] = {}
        joints: dict[str, Any] = {}

        # Build joint → motor lookup from naming convention
        mj_motor_names = {mjModel.actuator(i).name for i in range(mjModel.nu)}

        for jname in self.joint_names:
            motor_name = jname.replace("_joint", "_motor")
            if motor_name not in mj_motor_names:
                raise KeyError(
                    f"No motor '{motor_name}' found for joint '{jname}'. "
                    f"Available motors: {sorted(mj_motor_names)}"
                )
            actuators[jname] = mjData.actuator(motor_name)
            joints[jname] = mjData.joint(jname)

        self._joints = joints
        self._actuators = actuators

        # Set initial qpos
        self.controller.set_initial_qpos(actuators, joints)

        # Read joint limits + torque (effort) limits
        if not self.joint_limits:
            for jname, j in self._joints.items():
                limits = mjModel.jnt_range[j.id]
                self.joint_limits[jname] = (float(limits[0]), float(limits[1]))
        self._effort_limits: dict[str, float] = {}
        for jname, j in self._joints.items():
            frc = mjModel.jnt_actfrcrange[j.id]
            self._effort_limits[jname] = float(max(abs(frc[0]), abs(frc[1])))

        # Set default hand-space mid-stance from DEFAULT_QPOS if not already posed.
        self.mjModel = mjModel
        self.mjData = mjData

        return joints, actuators

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def get_robot_qpos(self) -> dict[str, float]:
        return {jname: float(j.qpos[0]) for jname, j in self._joints.items()}

    def get_actuators_action(self) -> dict[str, float]:
        return {jname: float(a.ctrl[0]) for jname, a in self._actuators.items()}

    def get_robot_qvel(self) -> dict[str, float]:
        return {jname: float(j.qvel[0]) for jname, j in self._joints.items()}

    # ------------------------------------------------------------------
    # Action application (for Gym env mode)
    # ------------------------------------------------------------------
    def apply_action(self, action_cmd) -> None:
        """Apply an action command.

        Accepts:
          - ActionCmd / dict: with .type (str) and .parameters / ["key"]
          - np.ndarray: flat joint position targets.

        For the whole-body MF path, ``position`` targets are already written to
        ``_mf_bridge.cmd_q`` by the policy; here we just enable the substep loop.
        """
        import numpy as np

        # Normalize to ActionCmd-like interface
        if isinstance(action_cmd, np.ndarray):
            action_type = "position"
            params = {"target_qpos": dict(zip(self.joint_names, action_cmd))}
        elif hasattr(action_cmd, "type"):
            action_type = action_cmd.type
            params = action_cmd.parameters or {}
        elif isinstance(action_cmd, dict):
            action_type = action_cmd.get("type", "position")
            params = action_cmd.get("parameters", action_cmd)
        else:
            raise TypeError(f"Unsupported action type: {type(action_cmd)}")

        if action_type == "elastic_band":
            import mujoco

            if self.elastic_band is not None and self.band_attached_link >= 0:
                pose = np.concatenate(
                    [self.mjData.xpos[self.band_attached_link],
                     self.mjData.xquat[self.band_attached_link], np.zeros(6)]
                )
                mujoco.mj_objectVelocity(
                    self.mjModel, self.mjData, mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link, pose[7:13], 0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mjData.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)

        elif action_type in ("position", "move_qpos", "move_qpos_with_eef"):
            # Enable the substep loop. PD targets come from _mf_bridge.cmd_q
            # (set by the policy's ActionManager) in MF mode, or are applied
            # directly here for the standalone Gym env path.
            if self._mf_bridge is not None:
                self.command = True
            else:
                target_qpos = params.get("target_qpos", params)
                from simple.interfaces.zmq_bridge import DEFAULT_KP, DEFAULT_KD, _match_param
                for jname, target in target_qpos.items():
                    if jname in self._actuators:
                        a = self._actuators[jname]
                        j = self._joints[jname]
                        kp = _match_param(jname, DEFAULT_KP)
                        kd = _match_param(jname, DEFAULT_KD)
                        limit = self._effort_limits.get(jname, 120.0)
                        torque = float(
                            np.clip(
                                kp * (float(target) - float(j.qpos[0]))
                                - kd * float(j.qvel[0]),
                                -limit,
                                limit,
                            )
                        )
                        a.ctrl = torque

        elif action_type in ("open_eef", "close_eef"):
            # Jaka has no gripper — no-op
            pass

        elif action_type == "reset_qpos":
            self.command = None
            for jname, q in params.get("target_qpos", params).items():
                if jname in self._joints:
                    self._joints[jname].qpos = float(q)
                    self._joints[jname].qvel = 0.0

    # ------------------------------------------------------------------
    # Sub-step loop (invoked by MujocoSimulator.step() when command is set)
    # ------------------------------------------------------------------
    def step(self, command, replay=False, eval=False):
        """Run ``num_substeps`` physics substeps, re-applying band + PD each substep.

        Called once per env.step() from ``MujocoSimulator.step`` when
        ``self.command`` is non-None and ``need_gravity=True``. Mirrors
        sim2real ``base_sim.sim_step``: the elastic band force is applied every
        physics substep (so the robot hangs on the virtual gantry until ``9``),
        and the PD torque is recomputed from the 50 Hz policy command each
        substep (500 Hz PD).
        """
        import mujoco

        for _ in range(self.num_substeps):
            if (self.elastic_band is not None and self.elastic_band.enable
                    and self.band_attached_link >= 0):
                pos = self.mjData.xpos[self.band_attached_link]
                lin_vel = self.mjData.cvel[self.band_attached_link, 3:6]
                self.mjData.xfrc_applied[self.band_attached_link, :3] = (
                    self.elastic_band.Advance(pos, lin_vel)
                )
            if self._mf_bridge is not None and self._mf_bridge.has_received_command:
                self._mf_bridge.apply_pd()
            mujoco.mj_step(self.mjModel, self.mjData)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> None:
        """Reset internal state."""
        self.command = None
        if self.elastic_band is not None:
            self.elastic_band.enable = True
