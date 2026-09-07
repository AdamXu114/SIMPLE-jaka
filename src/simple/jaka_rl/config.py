"""Consolidated Jaka Khan Mini configuration.

Merges constants previously scattered across sim2real-jaka
``config/robots/Jaka.py`` and SIMPLE's ``robots/jaka.py`` /
``interfaces/zmq_bridge.py`` into one place so the policy stack, the ZMQ
bridge, and the robot share a single source of truth.

Joint ordering note
-------------------
There are *two* joint/body orderings in this pipeline:

- ``JOINT_NAMES`` / ``BODY_NAMES`` — canonical **MuJoCo** order (what the
  MJCF uses, what the sim publishes / what `pico_retarget_pub` publishes).
- ``NPZ_JOINT_NAMES`` / ``NPZ_BODY_NAMES`` — **IsaacLab / policy** order (the
  order the ONNX policy was trained on and the motion NPZ rows use).

Observation classes resolve the mapping by match (name) rather than position,
so only the fallback ``ANCHOR_BODY_INDEX`` depends on the ``NPZ_BODY_NAMES``
order (must stay 3 = waist_yaw_Link).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Joints (MuJoCo order, 27)
# ---------------------------------------------------------------------------
JOINT_NAMES = tuple(
    [
        "Left_hip_pitch_joint",
        "Left_hip_roll_joint",
        "Left_hip_yaw_joint",
        "Left_knee_joint",
        "Left_ankle_pitch_joint",
        "Left_ankle_roll_joint",
        "Right_hip_pitch_joint",
        "Right_hip_roll_joint",
        "Right_hip_yaw_joint",
        "Right_knee_joint",
        "Right_ankle_pitch_joint",
        "Right_ankle_roll_joint",
        "waist_yaw_joint",
        "Left_shoulder_pitch_joint",
        "Left_shoulder_roll_joint",
        "Left_shoulder_yaw_joint",
        "Left_elbow_joint",
        "Left_wrist_roll_joint",
        "Left_wrist_yaw_joint",
        "Right_shoulder_pitch_joint",
        "Right_shoulder_roll_joint",
        "Right_shoulder_yaw_joint",
        "Right_elbow_joint",
        "Right_wrist_roll_joint",
        "Right_wrist_yaw_joint",
        "Neck_yaw_joint",
        "Neck_pitch_joint",
    ]
)

# Body names, copied from the MuJoCo XML exactly (case-sensitive).
# NOTE: ``Right_wrist_yaw__Link`` has two underscores; the joint does not.
BODY_NAMES = tuple(
    [
        "base_link",
        "Left_hip_pitch_Link",
        "Left_hip_roll_Link",
        "Left_hip_yaw_Link",
        "Left_knee_Link",
        "Left_ankle_pitch_Link",
        "Left_ankle_roll_Link",
        "Right_hip_pitch_Link",
        "Right_hip_roll_Link",
        "Right_hip_yaw_Link",
        "Right_knee_Link",
        "Right_ankle_pitch_Link",
        "Right_ankle_roll_Link",
        "waist_yaw_Link",
        "Left_shoulder_pitch_Link",
        "Left_shoulder_roll_Link",
        "Left_shoulder_yaw_Link",
        "Left_elbow_Link",
        "Left_wrist_roll_Link",
        "Left_wrist_yaw_Link",
        "Right_shoulder_pitch_Link",
        "Right_shoulder_roll_Link",
        "Right_shoulder_yaw_Link",
        "Right_elbow_Link",
        "Right_wrist_roll_Link",
        "Right_wrist_yaw__Link",
        "Neck_yaw_Link",
        "Neck_pitch_Link",
    ]
)

# ---------------------------------------------------------------------------
# Policy / NPZ (IsaacLab) ordering — from the strategy yaml ``policy_joint_names``
# ---------------------------------------------------------------------------
POLICY_JOINT_NAMES = tuple(
    [
        "Left_hip_pitch_joint",
        "Right_hip_pitch_joint",
        "waist_yaw_joint",
        "Left_hip_roll_joint",
        "Right_hip_roll_joint",
        "Left_shoulder_pitch_joint",
        "Neck_yaw_joint",
        "Right_shoulder_pitch_joint",
        "Left_hip_yaw_joint",
        "Right_hip_yaw_joint",
        "Left_shoulder_roll_joint",
        "Neck_pitch_joint",
        "Right_shoulder_roll_joint",
        "Left_knee_joint",
        "Right_knee_joint",
        "Left_shoulder_yaw_joint",
        "Right_shoulder_yaw_joint",
        "Left_ankle_pitch_joint",
        "Right_ankle_pitch_joint",
        "Left_elbow_joint",
        "Right_elbow_joint",
        "Left_ankle_roll_joint",
        "Right_ankle_roll_joint",
        "Left_wrist_roll_joint",
        "Right_wrist_roll_joint",
        "Left_wrist_yaw_joint",
        "Right_wrist_yaw_joint",
    ]
)

NPZ_JOINT_NAMES = POLICY_JOINT_NAMES

NPZ_BODY_NAMES = tuple(
    [
        "base_link",
        "Left_hip_pitch_Link",
        "Right_hip_pitch_Link",
        "waist_yaw_Link",
        "Left_hip_roll_Link",
        "Right_hip_roll_Link",
        "Left_shoulder_pitch_Link",
        "Right_shoulder_pitch_Link",
        "Neck_yaw_Link",
        "Left_hip_yaw_Link",
        "Right_hip_yaw_Link",
        "Left_shoulder_roll_Link",
        "Right_shoulder_roll_Link",
        "Neck_pitch_Link",
        "Left_knee_Link",
        "Right_knee_Link",
        "Left_shoulder_yaw_Link",
        "Right_shoulder_yaw_Link",
        "Left_ankle_pitch_Link",
        "Right_ankle_pitch_Link",
        "Left_elbow_Link",
        "Right_elbow_Link",
        "Left_ankle_roll_Link",
        "Right_ankle_roll_Link",
        "Left_wrist_roll_Link",
        "Right_wrist_roll_Link",
        "Left_wrist_yaw_Link",
        "Right_wrist_yaw__Link",
    ]
)

# waist_yaw_Link position in ``NPZ_BODY_NAMES`` (0-based).
ANCHOR_BODY_INDEX = 3
ANCHOR_BODY = "waist_yaw_Link"

# Root / IMU
# Candidate names for the floating/base joint (only one is present in the MJCF
# as a free joint; the list mirrors the sim2real reference Jaka cfg).
ROOT_JOINT_NAMES = ("base_joint", "floating_base_joint", "pelvis_root")
IMU_SITE_NAME = "waist_imu"
VIEWER_TRACK_BODY_NAMES = ("base_link",)
ELASTIC_BAND_ATTACH_BODY_NAMES = ("waist_yaw_Link", "base_link")

# ---------------------------------------------------------------------------
# Joint limits (radians) / effort (N·m)
# ---------------------------------------------------------------------------
JOINT_POS_LOWER_LIMIT = {
    "Left_hip_pitch_joint": -1.9198,
    "Left_hip_roll_joint": -0.5236,
    "Left_hip_yaw_joint": -2.0071,
    "Left_knee_joint": -0.3490,
    "Left_ankle_pitch_joint": -0.5236,
    "Left_ankle_roll_joint": -0.3490,
    "Right_hip_pitch_joint": -1.9198,
    "Right_hip_roll_joint": -0.5236,
    "Right_hip_yaw_joint": -2.0071,
    "Right_knee_joint": -0.3490,
    "Right_ankle_pitch_joint": -0.5236,
    "Right_ankle_roll_joint": -0.3490,
    "waist_yaw_joint": -2.0071,
    "Left_shoulder_pitch_joint": -1.5707,
    "Left_shoulder_roll_joint": -1.9189,
    "Left_shoulder_yaw_joint": -1.5707,
    "Left_elbow_joint": -0.8726,
    "Left_wrist_roll_joint": -1.5707,
    "Left_wrist_yaw_joint": 0.0,
    "Right_shoulder_pitch_joint": -1.5707,
    "Right_shoulder_roll_joint": -1.9189,
    "Right_shoulder_yaw_joint": -1.5707,
    "Right_elbow_joint": -0.8726,
    "Right_wrist_roll_joint": -1.5707,
    "Right_wrist_yaw_joint": 0.0,
    "Neck_yaw_joint": -1.5707,
    "Neck_pitch_joint": -0.6108,
}

JOINT_POS_UPPER_LIMIT = {
    "Left_hip_pitch_joint": 1.2217,
    "Left_hip_roll_joint": 2.3562,
    "Left_hip_yaw_joint": 3.5779,
    "Left_knee_joint": 2.2689,
    "Left_ankle_pitch_joint": 0.6108,
    "Left_ankle_roll_joint": 0.3490,
    "Right_hip_pitch_joint": 1.2217,
    "Right_hip_roll_joint": 3.5779,
    "Right_hip_yaw_joint": 3.5779,
    "Right_knee_joint": 2.2689,
    "Right_ankle_pitch_joint": 0.6108,
    "Right_ankle_roll_joint": 0.3490,
    "waist_yaw_joint": 3.5779,
    "Left_shoulder_pitch_joint": 1.5707,
    "Left_shoulder_roll_joint": 1.9189,
    "Left_shoulder_yaw_joint": 1.5707,
    "Left_elbow_joint": 2.2689,
    "Left_wrist_roll_joint": 1.5707,
    "Left_wrist_yaw_joint": 1.5707,
    "Right_shoulder_pitch_joint": 1.5707,
    "Right_shoulder_roll_joint": 1.9189,
    "Right_shoulder_yaw_joint": 1.5707,
    "Right_elbow_joint": 2.2689,
    "Right_wrist_roll_joint": 1.5707,
    "Right_wrist_yaw_joint": 1.5707,
    "Neck_yaw_joint": 1.5707,
    "Neck_pitch_joint": 0.6108,
}

# Note: matches the sim2real reference (uniform 30.0). The runtime velocity
# limit is enforced by the MJCF ``dof`` ``velocity``/``effort`` ranges — this
# table is source-of-truth parity only, not consumed by the in-process path.
JOINT_VELOCITY_LIMIT = {name: 30.0 for name in JOINT_NAMES}

JOINT_EFFORT_LIMIT = {
    "Left_hip_pitch_joint": 120.0,
    "Left_hip_roll_joint": 120.0,
    "Left_hip_yaw_joint": 120.0,
    "Left_knee_joint": 120.0,
    "Left_ankle_pitch_joint": 96.0,
    "Left_ankle_roll_joint": 96.0,
    "Right_hip_pitch_joint": 120.0,
    "Right_hip_roll_joint": 120.0,
    "Right_hip_yaw_joint": 120.0,
    "Right_knee_joint": 120.0,
    "Right_ankle_pitch_joint": 96.0,
    "Right_ankle_roll_joint": 96.0,
    "waist_yaw_joint": 120.0,
    "Left_shoulder_pitch_joint": 96.0,
    "Left_shoulder_roll_joint": 96.0,
    "Left_shoulder_yaw_joint": 36.0,
    "Left_elbow_joint": 36.0,
    "Left_wrist_roll_joint": 8.0,
    "Left_wrist_yaw_joint": 8.0,
    "Right_shoulder_pitch_joint": 96.0,
    "Right_shoulder_roll_joint": 96.0,
    "Right_shoulder_yaw_joint": 36.0,
    "Right_elbow_joint": 36.0,
    "Right_wrist_roll_joint": 8.0,
    "Right_wrist_yaw_joint": 8.0,
    "Neck_yaw_joint": 8.0,
    "Neck_pitch_joint": 8.0,
}

JOINT_ARMATURE = {
    name: 0.03
    for name in JOINT_NAMES
}
for _leg_ankle in (
    "Left_ankle_pitch_joint",
    "Left_ankle_roll_joint",
    "Right_ankle_pitch_joint",
    "Right_ankle_roll_joint",
):
    JOINT_ARMATURE[_leg_ankle] = 0.016
for _arm_light in (
    "Left_shoulder_pitch_joint",
    "Left_shoulder_roll_joint",
    "Right_shoulder_pitch_joint",
    "Right_shoulder_roll_joint",
):
    JOINT_ARMATURE[_arm_light] = 0.016
for _neck_wrist in (
    "Left_shoulder_yaw_joint",
    "Left_elbow_joint",
    "Left_wrist_roll_joint",
    "Left_wrist_yaw_joint",
    "Right_shoulder_yaw_joint",
    "Right_elbow_joint",
    "Right_wrist_roll_joint",
    "Right_wrist_yaw_joint",
    "Neck_yaw_joint",
    "Neck_pitch_joint",
):
    JOINT_ARMATURE[_neck_wrist] = 0.01

JOINT_FRICTIONLOSS = {name: 0.01 for name in JOINT_NAMES}

# ---------------------------------------------------------------------------
# PD gains (regex-keyed), matching the Jaka MF policy config.
# ---------------------------------------------------------------------------
JOINT_KP = {
    ".*_hip_pitch_joint": 187.0,
    ".*_hip_roll_joint": 187.0,
    ".*_hip_yaw_joint": 187.0,
    ".*_knee_joint": 187.0,
    ".*_ankle_pitch_joint": 100.0,
    ".*_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 187.0,
    ".*_shoulder_pitch_joint": 102.0,
    ".*_shoulder_roll_joint": 102.0,
    ".*_shoulder_yaw_joint": 40.8,
    ".*_elbow_joint": 40.8,
    ".*_wrist_roll_joint": 6.7,
    ".*_wrist_yaw_joint": 6.7,
    "Neck_yaw_joint": 6.7,
    "Neck_pitch_joint": 6.7,
}

JOINT_KD = {
    ".*_hip_pitch_joint": 18.7,
    ".*_hip_roll_joint": 18.7,
    ".*_hip_yaw_joint": 18.7,
    ".*_knee_joint": 18.7,
    ".*_ankle_pitch_joint": 2.0,
    ".*_ankle_roll_joint": 0.5,
    "waist_yaw_joint": 18.7,
    ".*_shoulder_pitch_joint": 10.2,
    ".*_shoulder_roll_joint": 10.2,
    ".*_shoulder_yaw_joint": 4.0,
    ".*_elbow_joint": 4.0,
    ".*_wrist_roll_joint": 0.67,
    ".*_wrist_yaw_joint": 0.67,
    "Neck_yaw_joint": 0.67,
    "Neck_pitch_joint": 0.67,
}

# Default joint positions (offset the policy uses; arms in "holding" pose).
DEFAULT_JOINT_POS = {
    "Left_shoulder_roll_joint": -1.57,
    "Left_elbow_joint": 1.57,
    "Left_wrist_yaw_joint": 0.3,
    "Right_shoulder_roll_joint": -1.57,
    "Right_elbow_joint": 1.57,
    "Right_wrist_yaw_joint": 0.3,
}

# qpos = [base xyz(3), base quat(4), 27 joint positions] (34 floats).
#
# DEFAULT_QPOS is the neutral standing pose — matches the sim2real reference
# ``default_qpos`` (base z=0.65, straight legs, arms at the default holding
# pose). This is ALSO the pose the zmq motion stream falls back to when it has
# no data (ZMQ_DEFAULT_QPOS), and the `[`(align) target, so:
#   spawn == align target == no-data reference == sim2real default_qpos.
DEFAULT_QPOS = tuple(
    [
        0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0,        # base pos+quat (z=0.65)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,              # Left leg (straight)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,              # Right leg (straight)
        0.0,                                        # waist_yaw
        0.0, -1.57, 0.0, 1.57, 0.0, 0.3,           # Left arm (holding pose)
        0.0, -1.57, 0.0, 1.57, 0.0, 0.3,           # Right arm (holding pose)
        0.0, 0.0,                                    # Neck
    ]
)

# The static stand the policy holds when the zmq motion stream has NOT received
# any data (no pico / no hub). Identical to DEFAULT_QPOS (the neutral stand), so
# the no-data reference == spawn == align target — one consistent stand pose.
ZMQ_DEFAULT_QPOS = DEFAULT_QPOS

# ---------------------------------------------------------------------------
# ZMQ ports (mirror sim2real utils/common.PORTS)
#
# The in-process MF path uses ONLY the pico motion stream (:28701) and the pico
# handle buttons (:5592). The legacy binary ``low_state`` (5590) / ``low_cmd``
# (5591) ports are kept here (and in interfaces/messages.py) purely for parity
# with the sim2real reference — the single-process path reads state via
# ``StateProcessor._state_getter`` and writes PD directly on the bridge, so
# NO traffic flows on 5590/5591.
# ---------------------------------------------------------------------------
LOW_STATE_PORT = 5590
LOW_CMD_PORT = 5591
PICO_CONTROLLER_PORT = 5592
MOTION_ZMQ_CONNECT = "tcp://127.0.0.1:28701"

PUBLISH_T_NS_KEY = "publish_t_ns"
SMPLX_T_NS_KEY = "smplx_t_ns"
SEQ_KEY = "seq"
JOINT_NAMES_KEY = "joint_names"
BODY_NAMES_KEY = "body_names"
JOINT_POS_KEY = "joint_pos"
JOINT_VEL_KEY = "joint_vel"
BODY_POS_W_KEY = "body_pos_w"
BODY_QUAT_W_KEY = "body_quat_w"


def default_qpos_np() -> "object":
    import numpy as np

    return np.asarray(DEFAULT_QPOS, dtype=np.float32)


__all__ = [
    "JOINT_NAMES",
    "BODY_NAMES",
    "POLICY_JOINT_NAMES",
    "NPZ_JOINT_NAMES",
    "NPZ_BODY_NAMES",
    "ANCHOR_BODY_INDEX",
    "ANCHOR_BODY",
    "ROOT_JOINT_NAMES",
    "IMU_SITE_NAME",
    "VIEWER_TRACK_BODY_NAMES",
    "ELASTIC_BAND_ATTACH_BODY_NAMES",
    "JOINT_POS_LOWER_LIMIT",
    "JOINT_POS_UPPER_LIMIT",
    "JOINT_VELOCITY_LIMIT",
    "JOINT_EFFORT_LIMIT",
    "JOINT_ARMATURE",
    "JOINT_FRICTIONLOSS",
    "JOINT_KP",
    "JOINT_KD",
    "DEFAULT_JOINT_POS",
    "DEFAULT_QPOS",
    "LOW_STATE_PORT",
    "LOW_CMD_PORT",
    "PICO_CONTROLLER_PORT",
    "MOTION_ZMQ_CONNECT",
    "PUBLISH_T_NS_KEY",
    "SMPLX_T_NS_KEY",
    "SEQ_KEY",
    "JOINT_NAMES_KEY",
    "BODY_NAMES_KEY",
    "JOINT_POS_KEY",
    "JOINT_VEL_KEY",
    "BODY_POS_W_KEY",
    "BODY_QUAT_W_KEY",
]
