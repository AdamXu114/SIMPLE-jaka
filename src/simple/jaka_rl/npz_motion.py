"""Raw NPZ motion dataset adapter for Jaka-style motion data.

NPZ files contain per-frame motion data:
  joint_pos      [T, n_joints]
  joint_vel      [T, n_joints]
  body_pos_w     [T, n_bodies, 3]
  body_quat_w    [T, n_bodies, 4]
  body_lin_vel_w [T, n_bodies, 3]
  body_ang_vel_w [T, n_bodies, 3]
  fps            [1]

Ported from sim2real-jaka ``rl_policy/utils/npz_motion.py``. No joint
reordering is done internally — data is returned in the file order (typically
IsaacLab order for Jaka); reindexing to simulation order happens in the
observation classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from simple.jaka_rl.motion import MotionData


class NpzMotionDataset:
    """Direct NPZ motion loader wrapping raw ``.npz`` into ``MotionData``."""

    def __init__(
        self,
        npz_path: str,
        joint_names: List[str],
        body_names: List[str],
    ):
        npz_path = str(Path(npz_path).expanduser())
        data = np.load(npz_path)

        self.joint_pos_all: np.ndarray = data["joint_pos"].astype(np.float32)        # [T, J]
        self.joint_vel_all: np.ndarray = data["joint_vel"].astype(np.float32)        # [T, J]
        self.body_pos_w_all: np.ndarray = data["body_pos_w"].astype(np.float32)      # [T, B, 3]
        self.body_quat_w_all: np.ndarray = data["body_quat_w"].astype(np.float32)    # [T, B, 4]
        self.body_lin_vel_w_all: np.ndarray = data["body_lin_vel_w"].astype(np.float32)  # [T, B, 3]
        self.body_ang_vel_w_all: np.ndarray = data["body_ang_vel_w"].astype(np.float32)  # [T, B, 3]

        fps = data["fps"]
        self.fps: int = int(fps.item() if fps.ndim > 0 else int(fps))
        self.num_steps: int = self.joint_pos_all.shape[0]

        self.joint_names: List[str] = list(joint_names)
        self.body_names: List[str] = list(body_names)

    def get_slice(
        self,
        motion_ids: np.ndarray,
        starts: np.ndarray,
        steps: np.ndarray,
    ) -> MotionData:
        """Return a ``MotionData`` slice shaped ``[N, S, ...]``.

        Args:
            motion_ids: [N] ignored (single-motion dataset), kept for API compat.
            starts: [N] per-batch starting frame index.
            steps: [S] offsets relative to ``starts`` (e.g. ``[0]`` or a window).
        """
        starts = np.asarray(starts, dtype=np.int64)           # [N]
        steps = np.asarray(steps, dtype=np.int64)              # [S]
        idx = starts[:, None] + steps[None, :]                 # [N, S]
        idx = np.clip(idx, 0, self.num_steps - 1)

        return MotionData(
            joint_pos=self.joint_pos_all[idx],
            joint_vel=self.joint_vel_all[idx],
            body_pos_w=self.body_pos_w_all[idx],
            body_quat_w=self.body_quat_w_all[idx],
            body_lin_vel_w=self.body_lin_vel_w_all[idx],
            body_ang_vel_w=self.body_ang_vel_w_all[idx],
        )


__all__ = ["NpzMotionDataset"]
