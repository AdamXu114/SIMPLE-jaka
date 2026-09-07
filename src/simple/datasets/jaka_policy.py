"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka RL policy closed-loop inference for replay.

Loads a sim2real-jaka ONNX policy (MF v1 620-dim or MF v2 835-dim, detected
from the config's ``future_steps`` length) and drives the MuJoCo robot each
step by building the observation (command + anchor_ori + 5-frame history),
inferring an action, and returning a PD position target (q_target) in MuJoCo
joint order.

Reference motion (motion_data) is rebuilt from the recorded episode's
``observation.body_poses`` + ``observation.state`` so the policy tracks the
recorded teleop trajectory closed-loop (keeps the biped balanced).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion helpers (batch-capable, wxyz), ported from sim2real-jaka utils/math
# ---------------------------------------------------------------------------


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return (a - b + c).reshape(shape)


def _quat_inv_single(q: np.ndarray) -> np.ndarray:
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    norm_sq = max(np.sum(q**2), 1e-9)
    return conj / norm_sq


def _yaw_quat_single(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


# ---------------------------------------------------------------------------
# Jaka policy replay
# ---------------------------------------------------------------------------


class JakaPolicyReplay:
    """Closed-loop replay driver for a Jaka MF RL policy (v1 620-dim / v2 835-dim)."""

    # 0bs layout (dim depends on version: 620 for MF v1, 835 for MF v2)
    _FRAME_DIM = 87
    _STACK_SIZE = 5
    _SLICES = [(0, 3), (3, 6), (6, 33), (33, 60), (60, 87)]
    _ANCHOR_IDX = 3  # waist_yaw_Link in npz_body_names

    def __init__(self, model_path: str, yaml_path: str):
        import yaml

        with open(yaml_path) as f:
            self.cfg = yaml.safe_load(f)

        # joint names
        sim_joint_names = list(self.cfg["joint_names_simulation"])          # MuJoCo order
        policy_joint_names = list(self.cfg["policy_joint_names"])           # policy/npz order
        npz_joint_names = list(self.cfg["motion"]["npz_joint_names"])       # = policy order
        self.n_joints = len(policy_joint_names)

        # reindex mujoco -> npz (same array used for obs and action scatter)
        self.reindex = np.array(
            [sim_joint_names.index(n) for n in npz_joint_names], dtype=int
        )
        # controlled_joint_indices: npz/policy index -> MuJoCo position
        self.controlled_joint_indices = np.array(
            [sim_joint_names.index(n) for n in policy_joint_names], dtype=int
        )

        # action scale / clip
        self.action_scale = float(self.cfg.get("action_scale", 0.5))
        self.action_clip = self.cfg.get("action_clip")

        # default joint angles in MuJoCo order
        self.default_dof_angles = np.zeros(self.n_joints, dtype=np.float32)
        default_dict = self.cfg.get("default_joint_pos", {})
        for jname, jval in default_dict.items():
            self.default_dof_angles[sim_joint_names.index(jname)] = float(jval)
        # default angles in npz order (for obs dof_pos)
        self.default_angles_npz = np.zeros(self.n_joints, dtype=np.float32)
        for jname, jval in default_dict.items():
            self.default_angles_npz[npz_joint_names.index(jname)] = float(jval)

        self.future_steps = np.array(self.cfg["motion"]["future_steps"], dtype=int)
        # Command/anchor_ori step count and obs version. MF v1 uses 5 future
        # steps (command 155 / anchor_ori 30 / obs 620); MF v2 uses 10
        # (command 340 / anchor_ori 60 / obs 835). Detect from the config so
        # switching policies only requires swapping the ONNX + YAML.
        self.n_cmd_steps = len(self.future_steps)
        self._obs_version = "v1" if self.n_cmd_steps <= 5 else "v2"

        # per-joint PD gains (MuJoCo order) — policy was trained with these
        self.joint_kp = self._resolve_joint_params(self.cfg.get("joint_kp", {}), sim_joint_names)
        self.joint_kd = self._resolve_joint_params(self.cfg.get("joint_kd", {}), sim_joint_names)

        # ONNX session
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        in_name = self.session.get_inputs()[0].name
        out_name = self.session.get_outputs()[0].name
        self._in_name = in_name
        self._out_name = out_name

        # per-episode motion + state
        self._motion: Optional[Dict[str, np.ndarray]] = None
        self.motion_t = 0
        self.motion_length = 0
        self._history: deque = deque(maxlen=self._STACK_SIZE)
        self._is_first = True
        self.ref_to_robot_quat_init = np.array([1.0, 0, 0, 0], dtype=np.float32)
        self._prev_action = np.zeros(self.n_joints, dtype=np.float32)

    @staticmethod
    def _resolve_joint_params(param_dict, joint_names):
        import re
        out = np.ones(len(joint_names), dtype=np.float32)
        for jname in joint_names:
            for pattern, value in param_dict.items():
                if re.fullmatch(pattern, jname) or pattern == ".*":
                    out[joint_names.index(jname)] = float(value)
                    break
        return out

    # ------------------------------------------------------------------
    # Per-episode setup
    # ------------------------------------------------------------------
    def set_motion(
        self,
        joint_pos: np.ndarray,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        body_lin_vel_w: np.ndarray,
        body_ang_vel_w: np.ndarray,
    ) -> None:
        """Set the reference trajectory (npz layout) from the recorded episode."""
        self._motion = {
            "joint_pos": np.asarray(joint_pos, dtype=np.float32),           # [T,27]
            "body_pos_w": np.asarray(body_pos_w, dtype=np.float32),         # [T,28,3]
            "body_quat_w": np.asarray(body_quat_w, dtype=np.float32),       # [T,28,4]
            "body_lin_vel_w": np.asarray(body_lin_vel_w, dtype=np.float32), # [T,28,3]
            "body_ang_vel_w": np.asarray(body_ang_vel_w, dtype=np.float32), # [T,28,3]
        }
        self.motion_length = self._motion["joint_pos"].shape[0]
        self.motion_t = 0

    def reset(self, initial_root_quat_w: np.ndarray) -> None:
        """Reset per-episode state (history, ref init, prev action)."""
        self._history.clear()
        for _ in range(self._STACK_SIZE):
            self._history.append(np.zeros(self._FRAME_DIM, dtype=np.float32))
        self._is_first = True
        self._prev_action = np.zeros(self.n_joints, dtype=np.float32)

        if self._motion is not None:
            ref_anchor_quat = self._motion["body_quat_w"][0, self._ANCHOR_IDX]
            ref_init_yaw = _yaw_quat_single(ref_anchor_quat)
            ref_init_yaw_inv = _quat_inv_single(ref_init_yaw)
            robot_init_yaw = _yaw_quat_single(np.asarray(initial_root_quat_w, dtype=np.float32))
            self.ref_to_robot_quat_init = (
                _quat_mul_single(robot_init_yaw, ref_init_yaw_inv)
            ).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation (motion window)
    # ------------------------------------------------------------------
    def _motion_window(self) -> Dict[str, np.ndarray]:
        m = self._motion
        t = self.motion_t
        idx = np.clip(t + self.future_steps, 0, self.motion_length - 1)  # [n]
        return {
            "joint_pos": m["joint_pos"][idx],                 # [n,27]
            "body_quat_w": m["body_quat_w"][idx, self._ANCHOR_IDX],   # [n,4]
            "body_pos_w": m["body_pos_w"][idx, self._ANCHOR_IDX],     # [n,3]
            "body_lin_vel_w": m["body_lin_vel_w"][idx, self._ANCHOR_IDX],  # [n,3]
            "body_ang_vel_w": m["body_ang_vel_w"][idx, self._ANCHOR_IDX],  # [n,3]
        }

    def _compute_command(self) -> np.ndarray:
        w = self._motion_window()
        n = self.n_cmd_steps
        anchor_quat = w["body_quat_w"]        # [n,4]
        anchor_pos = w["body_pos_w"]          # [n,3]
        if self._obs_version == "v1":
            # 155-dim: root_pos_diff_b(15) + root_z_mf(5) + ref_joint_pos(135).
            # Position diff relative to the current frame, expressed in the
            # current-frame anchor body frame (MF v1 formula, jaka_mf.py).
            ref_anchor_quat_cur = anchor_quat[0]              # current frame [4]
            ref_anchor_quat_batch = np.tile(ref_anchor_quat_cur[None, :], (n, 1))
            diff_b = quat_rotate_inverse(ref_anchor_quat_batch, anchor_pos - anchor_pos[0:1])
            root_pos_diff_b = diff_b.reshape(-1)              # 15
            root_z_mf = anchor_pos[:, 2:3].reshape(-1)        # 5
            joint_pos = w["joint_pos"]                        # [n,27] npz order
            return np.concatenate([
                root_pos_diff_b, root_z_mf, joint_pos.reshape(-1),
            ], axis=0).astype(np.float32)                     # 155
        # v2 (MF v2 formula, jaka_mf_v2.py): vel_xy + height + gravity +
        # ang_vel_z + joint_pos.
        ref_root_vel = quat_rotate_inverse(anchor_quat, w["body_lin_vel_w"])
        ref_root_ang_vel = quat_rotate_inverse(anchor_quat, w["body_ang_vel_w"])
        height = anchor_pos[:, 2:3]
        gravity_vec = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (n, 1))
        ref_projected_gravity = quat_rotate_inverse(anchor_quat, gravity_vec)
        joint_pos = w["joint_pos"]            # [n,27]
        return np.concatenate([
            ref_root_vel[:, :2].reshape(-1),      # 20
            height.reshape(-1),                   # 10
            ref_projected_gravity.reshape(-1),    # 30
            ref_root_ang_vel[:, 2:3].reshape(-1), # 10
            joint_pos.reshape(-1),                # 270
        ], axis=0).astype(np.float32)             # 340

    def _compute_anchor_ori(self, root_quat_w: np.ndarray) -> np.ndarray:
        w = self._motion_window()
        n = self.n_cmd_steps
        ref_quat_all = w["body_quat_w"]        # [n,4]
        ref_to_robot_init_batch = np.tile(self.ref_to_robot_quat_init, (n, 1))
        future_anchor_quat_w = quat_mul(ref_to_robot_init_batch, ref_quat_all)  # [n,4]
        robot_anchor_quat_batch = np.tile(np.asarray(root_quat_w, dtype=np.float32), (n, 1))
        ori_b = quat_mul(
            quat_conjugate(robot_anchor_quat_batch), future_anchor_quat_w
        )  # [n,4]
        r, i, j, k = ori_b[:, 0], ori_b[:, 1], ori_b[:, 2], ori_b[:, 3]
        two_s = 2.0 / (r * r + i * i + j * j + k * k)
        ii = i * i; jj = j * j; kk = k * k
        ij = i * j; kr = k * r; ik = i * k
        jr = j * r; jk = j * k; ir = i * r
        rot6d = np.stack([
            1.0 - two_s * (jj + kk),
            two_s * (ij - kr),
            two_s * (ij + kr),
            1.0 - two_s * (ii + kk),
            two_s * (ik - jr),
            two_s * (jk + ir),
        ], axis=-1).reshape(-1).astype(np.float32)  # n*6
        return rot6d

    def _compute_frame(
        self, root_quat_w: np.ndarray, root_ang_vel_b: np.ndarray,
        joint_pos_mujoco: np.ndarray, joint_vel_mujoco: np.ndarray,
    ) -> np.ndarray:
        obs = np.zeros(self._FRAME_DIM, dtype=np.float32)
        qw, qx, qy, qz = root_quat_w
        obs[0] = 2 * (-qz * qx + qw * qy)
        obs[1] = -2 * (qz * qy + qw * qx)
        obs[2] = 1 - 2 * (qw * qw + qz * qz)
        obs[3:6] = root_ang_vel_b * 0.25
        jp = np.asarray(joint_pos_mujoco, dtype=np.float32)
        jv = np.asarray(joint_vel_mujoco, dtype=np.float32)
        obs[6:33] = jp[self.reindex] - self.default_angles_npz
        obs[33:60] = jv[self.reindex] * 0.05
        obs[60:87] = self._prev_action[: self.n_joints]
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        root_quat_w: np.ndarray,
        root_ang_vel_b: np.ndarray,
        joint_pos_mujoco: np.ndarray,
        joint_vel_mujoco: np.ndarray,
    ) -> np.ndarray:
        """Build the version-dependent obs (620 for MF v1, 835 for MF v2), run
        ONNX, return q_target (MuJoCo order)."""
        frame = self._compute_frame(root_quat_w, root_ang_vel_b, joint_pos_mujoco, joint_vel_mujoco)
        if self._is_first:
            for _ in range(self._STACK_SIZE):
                self._history.append(frame.copy())
            self._is_first = False
        else:
            self._history.append(frame.copy())

        command = self._compute_command()                       # 155 (v1) / 340 (v2)
        anchor_ori = self._compute_anchor_ori(root_quat_w)      # 30 (v1) / 60 (v2)
        stacked = np.array(list(self._history), dtype=np.float32)  # [5,87]
        parts = [command, anchor_ori]
        for start, end in self._SLICES:
            parts.append(stacked[:, start:end].reshape(-1))
        obs = np.concatenate(parts, axis=0).astype(np.float32)  # 620 (v1) / 835 (v2)

        outputs = self.session.run([self._out_name], {self._in_name: obs[None, :]})
        action = np.asarray(outputs[0], dtype=np.float32).squeeze(0)  # (27,)
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        self._prev_action = action

        q_target = self.default_dof_angles.copy()
        q_target[self.controlled_joint_indices] += action * self.action_scale
        return q_target

    # ------------------------------------------------------------------
    # Motion advance
    # ------------------------------------------------------------------
    def advance_motion(self) -> None:
        self.motion_t = (self.motion_t + 1) % max(self.motion_length, 1)


def _quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


__all__ = ["JakaPolicyReplay"]
