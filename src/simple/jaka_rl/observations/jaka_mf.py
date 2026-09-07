"""Jaka robot MF (multi-frame future) observation class.

Ported from sim2real-jaka ``rl_policy/observations/jaka_mf.py``. Produces the
620-dim observation for the MF v1 policy:

  [0:155]   command — root_pos_diff_b (5×3) + root_z_mf (5) + ref_joint_pos (5×27)
  [155:185] anchor_ori — rot6d (5×6)
  [185:200] gravity — 5-frame stack (5×3)
  [200:215] ang_vel — 5-frame stack (5×3)
  [215:350] dof_pos — 5-frame stack (5×27)
  [350:485] dof_vel — 5-frame stack (5×27)
  [485:620] last_action — 5-frame stack (5×27)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np

from simple.jaka_rl.math import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from simple.jaka_rl.observations.base import Observation


# ──────────────────── Single-quaternion helpers (reset only) ────────────── #

def _quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def _quat_inv_single(q: np.ndarray) -> np.ndarray:
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    norm_sq = max(np.sum(q**2), 1e-9)
    return conj / norm_sq


def _yaw_quat_single(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class jaka_frame_stack_mf(Observation):
    """5-frame history stack + 5-future-step command/anchor_ori (620-dim)."""

    _FRAME_DIM = 87
    _STACK_SIZE = 5
    _NUM_FUTURE_STEPS = 5
    _COMMAND_DIM = 155   # 3*5 + 1*5 + 27*5
    _ANCHOR_ORI_DIM = 30  # 6*5

    _SLICES = [
        (0, 3),     # gravity
        (3, 6),     # ang_vel
        (6, 33),    # dof_pos
        (33, 60),   # dof_vel
        (60, 87),   # last_action
    ]

    def __init__(
        self,
        anchor_body_index: int = 3,
        ang_vel_scale: float = 0.25,
        joint_vel_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.anchor_body_index = anchor_body_index
        self.ang_vel_scale = ang_vel_scale
        self.joint_vel_scale = joint_vel_scale

        motion_cfg = self.state_processor.motion_config
        self.isaaclab_joint_names: List[str] = list(motion_cfg.get("npz_joint_names", []))
        sim_joint_names = list(self.state_processor.joint_names)

        self.mujoco_to_isaaclab_reindex = [
            sim_joint_names.index(name) for name in self.isaaclab_joint_names
        ]
        self.n_joints = len(self.isaaclab_joint_names)

        self.default_angles_isaaclab = np.zeros(self.n_joints, dtype=np.float32)
        default_joint_pos_dict = self.env.policy_config.get("default_joint_pos", {})
        for jname, jval in default_joint_pos_dict.items():
            if jname in self.isaaclab_joint_names:
                idx = self.isaaclab_joint_names.index(jname)
                self.default_angles_isaaclab[idx] = float(jval)

        self.ref_to_robot_quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Whether ref_to_robot_quat_init has been aligned. Reset() may run before
        # the live zMQ motion stream has ANY frame, in which case the buffer
        # returns the FK default posture (identity yaw) — aligning against that
        # would lock in a wrong reference orientation. Instead we defer and align
        # on the FIRST real motion frame, using the robot's orientation then.
        self._ref_init_done = False

        self._frame_buffer: deque = deque(maxlen=self._STACK_SIZE)
        self._is_first_frame = True

    def _refresh_motion_indices(self) -> None:
        sp = self.state_processor
        joint_names = tuple(sp.motion_joint_names)
        body_names = tuple(sp.motion_body_names)
        layout = (joint_names, body_names)
        if hasattr(self, "_cached_motion_layout") and self._cached_motion_layout == layout:
            return
        if not joint_names or not body_names:
            raise ValueError("Motion source names are not ready")

        self._motion_joint_indices = [
            joint_names.index(name) for name in self.isaaclab_joint_names
        ]

        motion_cfg = sp.motion_config or {}
        anchor_body_name = motion_cfg.get("anchor_body_name", "waist_yaw_Link")
        if anchor_body_name in body_names:
            self.resolved_anchor_body_index = body_names.index(anchor_body_name)
        else:
            self.resolved_anchor_body_index = self.anchor_body_index

        self._cached_motion_layout = layout

    def _motion_has_real_data(self) -> bool:
        """True if the motion source has actual frames (not the empty-buffer FK
        default posture). For npz/raw_npz the source is always real; for zmq we
        look at whether the live buffer has received a frame yet."""
        sp = self.state_processor
        mb = getattr(sp, "motion_buffer", None)
        if mb is None:
            # npz / raw_npz backends always carry real motion data.
            return sp.motion_data is not None
        return mb.latest_timestamp_ns is not None

    def _align_ref_to_robot(self) -> None:
        """Align the reference motion's yaw to the robot's CURRENT orientation.

        Called when the first real motion frame is available. The reference's
        anchor (waist_yaw_Link) yaw is offset so the robot tracks the motion
        relative to its own heading rather than an absolute world yaw. Only the
        yaw is aligned (pitch/roll of the reference are preserved).
        """
        sp = self.state_processor
        motion_data = sp.motion_data
        if motion_data is None:
            return
        self._refresh_motion_indices()
        ref_anchor_quat = motion_data.body_quat_w[0, 0, self.resolved_anchor_body_index]
        ref_init_yaw = _yaw_quat_single(ref_anchor_quat)
        ref_init_yaw_inv = _quat_inv_single(ref_init_yaw)

        # sp.root_quat_w is the waist_yaw_Link quaternion (IMU framequat) — the
        # robot's orientation at the moment the first real frame becomes usable.
        robot_anchor_quat = sp.root_quat_w.copy()
        robot_init_yaw = _yaw_quat_single(robot_anchor_quat)
        self.ref_to_robot_quat_init = _quat_mul_single(robot_init_yaw, ref_init_yaw_inv)
        self._ref_init_done = True

    def reset(self):
        self._frame_buffer.clear()
        for _ in range(self._STACK_SIZE):
            self._frame_buffer.append(np.zeros(self._FRAME_DIM, dtype=np.float32))
        self._is_first_frame = True
        self._ref_init_done = False
        if self._motion_has_real_data():
            self._align_ref_to_robot()
        else:
            # No real motion frame yet (e.g. empty zMQ buffer). Hold identity and
            # defer the alignment to the first real frame (see update()).
            self.ref_to_robot_quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        if not self._ref_init_done and self._motion_has_real_data():
            self._align_ref_to_robot()
        obs = self._compute_single_frame(data)
        if self._is_first_frame:
            for _ in range(self._STACK_SIZE):
                self._frame_buffer.append(obs.copy())
            self._is_first_frame = False
        else:
            self._frame_buffer.append(obs.copy())

    def _compute_single_frame(self, data: Dict[str, Any]) -> np.ndarray:
        sp = self.state_processor
        motion_data = sp.motion_data

        obs = np.zeros(self._FRAME_DIM, dtype=np.float32)
        if motion_data is None:
            return obs

        # Projected gravity using waist_yaw_Link body quat (IMU framequat).
        anchor_quat = sp.root_quat_w.copy()
        qw, qx, qy, qz = anchor_quat
        obs[0] = 2 * (-qz * qx + qw * qy)
        obs[1] = -2 * (qz * qy + qw * qx)
        obs[2] = 1 - 2 * (qw * qw + qz * qz)

        # Base angular velocity (gyro)
        obs[3:6] = sp.root_ang_vel_b * self.ang_vel_scale

        # Joint positions relative to default (IsaacLab order)
        joint_pos_mujoco = sp.joint_pos
        joint_pos_isaaclab = joint_pos_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[6:33] = joint_pos_isaaclab - self.default_angles_isaaclab

        # Joint velocities (IsaacLab order)
        joint_vel_mujoco = sp.joint_vel
        joint_vel_isaaclab = joint_vel_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[33:60] = joint_vel_isaaclab * self.joint_vel_scale

        # Last action
        last_action = data.get("action", np.zeros(self.n_joints, dtype=np.float32))
        obs[60:87] = last_action[: self.n_joints]
        return obs

    def _compute_command(self) -> np.ndarray:
        """Build 155-dim command from 5 future motion steps (batched)."""
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(self._COMMAND_DIM, dtype=np.float32)

        self._refresh_motion_indices()
        anchor_idx = self.resolved_anchor_body_index

        ref_pos_all = motion_data.body_pos_w[0, :, anchor_idx]

        ref_anchor_quat_cur = motion_data.body_quat_w[0, 0, anchor_idx]  # [4]
        ref_anchor_quat_batch = np.tile(
            ref_anchor_quat_cur[None, :], (self._NUM_FUTURE_STEPS, 1)
        )
        diff_b = quat_rotate_inverse_numpy(
            ref_anchor_quat_batch, ref_pos_all - ref_pos_all[0:1]
        )

        root_pos_diff_b = diff_b.reshape(-1)   # 15
        root_z_mf = ref_pos_all[:, 2:3].reshape(-1)  # 5
        motion_joint_pos = motion_data.joint_pos[0][:, self._motion_joint_indices].copy()  # [5, 27]
        motion_joint_pos_flat = motion_joint_pos.reshape(-1)  # 135

        return np.concatenate(
            [root_pos_diff_b, root_z_mf, motion_joint_pos_flat], axis=0
        ).astype(np.float32)

    def _compute_anchor_ori(self) -> np.ndarray:
        """Build 30-dim anchor orientation from 5 future motion steps (rot6d × 5)."""
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(self._ANCHOR_ORI_DIM, dtype=np.float32)

        sp = self.state_processor
        robot_anchor_quat = sp.root_quat_w.copy()

        self._refresh_motion_indices()
        anchor_idx = self.resolved_anchor_body_index

        ref_quat_all = motion_data.body_quat_w[0, :, anchor_idx]
        ref_to_robot_init_batch = np.tile(
            self.ref_to_robot_quat_init[None, :], (self._NUM_FUTURE_STEPS, 1)
        )
        future_anchor_quat_w = quat_mul(ref_to_robot_init_batch, ref_quat_all)  # [5, 4]

        robot_anchor_quat_batch = np.tile(
            robot_anchor_quat[None, :], (self._NUM_FUTURE_STEPS, 1)
        )
        ori_b = quat_mul(
            quat_conjugate(robot_anchor_quat_batch), future_anchor_quat_w
        )  # [5, 4]

        r, i, j, k = ori_b[:, 0], ori_b[:, 1], ori_b[:, 2], ori_b[:, 3]
        norm2 = r * r + i * i + j * j + k * k
        # Guard against zero / near-zero quaternions (e.g. before the low state
        # is ready, or the no-pico default fallback): a zero norm2 yields an
        # identity rot6d below (two_s=0) instead of a NaN from 2/0.
        safe = norm2 > 1e-8
        two_s = np.divide(2.0, np.maximum(norm2, 1e-8), out=np.zeros_like(norm2), where=safe)
        ii = i * i; jj = j * j; kk = k * k
        ij = i * j; kr = k * r; ik = i * k
        jr = j * r; jk = j * k; ir = i * r
        rot6d = np.stack(
            [
                1.0 - two_s * (jj + kk),
                two_s * (ij - kr),
                two_s * (ij + kr),
                1.0 - two_s * (ii + kk),
                two_s * (ik - jr),
                two_s * (jk + ir),
            ],
            axis=-1,
        ).reshape(-1)  # [5, 6] → 30
        # Replace any non-finite (shouldn't occur after the guard) with identity.
        rot6d = np.where(np.isfinite(rot6d), rot6d, np.tile(np.array([1, 0, 0, 1, 0, 0], np.float32), self._NUM_FUTURE_STEPS)).reshape(-1)
        return rot6d.astype(np.float32)

    def compute(self) -> np.ndarray:
        command = self._compute_command()       # 155
        anchor_ori = self._compute_anchor_ori()  # 30

        stacked = np.array(list(self._frame_buffer), dtype=np.float32)  # [5, 87]
        parts = [command, anchor_ori]
        for start, end in self._SLICES:
            parts.append(stacked[:, start:end].reshape(-1))
        return np.concatenate(parts, axis=0)  # 620


__all__ = ["jaka_frame_stack_mf"]
