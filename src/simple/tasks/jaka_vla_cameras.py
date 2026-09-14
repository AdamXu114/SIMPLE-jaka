"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Shared camera configs for Jaka VLA teleop tasks.

Camera parameters mirror G1's task definitions (see
g1_wholebody_tabletop_grasp_mp.py sensor_cfgs). Only the head camera is kept —
Jaka's real head camera mount is the ``Camera_Link`` shell, offset
[0.0665, 0, 0.055] from ``Neck_pitch_Link`` in the URDF. G1 can use an identity
pose because its USD has a dedicated camera link (d435_link); Jaka needs the
forward offset so the camera isn't inside the head mesh (measured empirically).

Views map to LeRobot features:
    head_stereo_left  -> observation.images.ego_view
    front_stereo_left -> observation.images.front_view
"""

from __future__ import annotations

import numpy as np

from simple.sensors import CameraCfg, SensorCfg, StereoCameraCfg


def jaka_vla_sensor_cfgs() -> dict[str, SensorCfg]:
    """Return the VLA camera set for Jaka teleop tasks (head + front)."""
    return {
        # Ego view — mounted at Jaka's real head camera mount (Camera_Link),
        # which is offset [0.0665, 0, 0.055] from Neck_pitch_Link per the URDF.
        "head_stereo": StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=224,
            height=224,
            focal_length=1.93,
            fov=np.deg2rad(110),
            near=0.2,
            far=5,
            baseline=0.05,
            # Identity orientation: both engines already face the camera
            # forward (+X, up +Z) when no rotation is applied — MuJoCo via
            # q_isaac_mujoco, IsaacSim natively. Only the mount offset matters.
            pose=dict(
                position=[0.0665, 0.0, 0.055],
            ),
        ),
        # 3rd-person front view — G1 uses a spherical orbit around the robot.
        "front_stereo": StereoCameraCfg(
            uid="Realsense_D415",
            mount="eye_on_base",
            width=640,
            height=360,
            focal_length=1.88,
            fov=np.deg2rad(90),
            near=0.2,
            far=5,
            baseline=0.055,
            pose=dict(
                distance=2.5,
                polar=np.deg2rad(60),
                azimuth=np.deg2rad(0),
            ),
        ),
    }


__all__ = ["jaka_vla_sensor_cfgs"]
