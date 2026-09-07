"""Motion data container shared by the Jaka MF policy stack.

Ported from sim2real-jaka ``rl_policy/utils/motion.py`` (trimmed): only the
``MotionData`` container and quaternion helpers relevant to the live / npz
motion buffer are kept. The any4hdmi dataset adapter is intentionally dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simple.jaka_rl.math import quat_normalize, quat_slerp_batch

_normalize_quat_batch = quat_normalize
_quat_slerp_batch = quat_slerp_batch


class MotionData:
    """Container for motion data arrays.

    Fields are numpy arrays keyed by name (``joint_pos``, ``body_pos_w``,
    ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``, ``joint_vel``,
    ``motion_id``, ``step``, ``timestamps_ns``). ``__getitem__`` slices the
    leading batch dimension.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key != "batch_size":
                setattr(self, key, value)

    def __getitem__(self, idx):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value[idx]
        return MotionData(**result)


__all__ = [
    "MotionData",
    "quat_slerp_batch",
    "quat_normalize",
]
