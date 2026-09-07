"""Realtime motion buffer that subscribes to the pico retarget motion stream.

Ported from sim2real-jaka ``rl_policy/utils/motion_buffer.py``. Subscribes to
a ZMQ PUB socket (``tcp://127.0.0.1:28701``) carrying JSON payloads from
``pico_retarget_pub``, buffers them by timestamp, and interpolates the
requested ``future_steps`` window on ``get_obs()``.

The only change vs sim2real is the default-posture source: instead of
resolving the MJCF path through a ``RobotCfg``, the buffer accepts explicit
``joint_names`` / ``body_names`` and an optional pre-built ``mj_model`` /
``mj_data`` (when provided, no FK is repeated).
"""

from __future__ import annotations

import json
import threading
import time
from bisect import bisect_right
from typing import Any, Iterable

import numpy as np
import mujoco
import zmq

from loguru import logger

from simple.jaka_rl.math import quat_normalize, quat_slerp_batch
from simple.jaka_rl.motion import MotionData

_normalize_quat_batch = quat_normalize
_quat_slerp_batch = quat_slerp_batch


def _ensure_np(value: Any, ndim: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"Expected ndim={ndim}, got shape={arr.shape}")
    return arr


class RealtimeMotionBuffer:
    """Timestamped motion buffer with interpolation over a future window."""

    def __init__(
        self,
        joint_names: Iterable[str],
        body_names: Iterable[str],
        future_steps: Iterable[int],
        *,
        mj_model: mujoco.MjModel | None = None,
        mj_data: mujoco.MjData | None = None,
        default_qpos: np.ndarray | None = None,
        motion_zmq_connect: str | None = "tcp://127.0.0.1:28701",
        motion_zmq_hwm: int = 1,
        dt_s: float = 0.02,
        tolerance_s: float = 0.04,
    ):
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.joint_names: list[str] = list(joint_names)
        self.body_names: list[str] = list(body_names)
        self._num_joints = len(self.joint_names)
        self._num_bodies = len(self.body_names)
        self.future_steps = np.asarray(list(future_steps), dtype=int)
        if self.future_steps.ndim != 1:
            raise ValueError(f"future_steps must be 1D, got {self.future_steps.shape}")
        self.dt_s = float(dt_s)
        self.tolerance_s = float(tolerance_s)
        self.min_future_step = (
            int(np.min(self.future_steps)) if self.future_steps.size else 0
        )
        self.max_future_step = (
            int(np.max(self.future_steps)) if self.future_steps.size else 0
        )
        self._dt_ns = int(self.dt_s * 1e9)
        self._tolerance_ns = int(self.tolerance_s * 1e9)
        self._future_steps_ns = self.future_steps.astype(np.int64, copy=False) * self._dt_ns
        self._delay_ns = self.max_future_step * self._dt_ns + self._tolerance_ns
        self._history_ns = self._delay_ns + abs(self.min_future_step) * self._dt_ns
        self.delay_s = float(self._delay_ns / 1e9)
        self._identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._lock = threading.Lock()
        self._timestamps_ns: list[int] = []
        self._joint_pos_frames: list[np.ndarray] = []
        self._body_pos_w_frames: list[np.ndarray] = []
        self._body_quat_w_frames: list[np.ndarray] = []

        self._init_default_posture(
            mj_model=mj_model,
            mj_data=mj_data,
            default_qpos=default_qpos,
        )

        self._motion_id_template = np.zeros((1, self.future_steps.shape[0]), dtype=np.int64)
        self._step_template = self.future_steps.reshape(1, -1)
        self._zmq_context = zmq.Context.instance()
        self._motion_zmq_connect = motion_zmq_connect
        self._motion_zmq_hwm = int(motion_zmq_hwm)
        self._motion_stream_socket = None
        self._motion_stream_thread: threading.Thread | None = None
        self._motion_stream_stop = threading.Event()
        if self._motion_zmq_connect:
            self._start_motion_stream()

    def _init_default_posture(self, *, mj_model, mj_data, default_qpos):
        """Resolve a default posture (used when the buffer is empty).

        Preference order: caller-supplied ``mj_model``+``mj_data``; otherwise a
        supplied ``default_qpos`` (built into a fresh model on the fly). Falls
        back to zeros if none are usable.
        """
        try:
            if mj_model is not None:
                # Use the caller's MODEL to FK the default posture, but on a
                # SCRATCH MjData so we never clobber the live robot's qpos. We
                # write `default_qpos` into `d.qpos` below to define the stand
                # pose; if we reused the live `mj_data` this would teleport the
                # robot from its task-specified spawn position (e.g. x=-1.45 in
                # front of a table) to the neutral-stand origin (x=0), dropping
                # it into the table.
                m = mj_model
                d = mujoco.MjData(m)
            elif default_qpos is not None:
                # Build a minimal free-base + hinge model to FK the default qpos.
                spec = mujoco.MjSpec()
                spec.add_joint("base_joint", type="free")
                for jname in self.joint_names:
                    spec.add_joint(jname, type="hinge", axis=[0, 0, 1])
                m = spec.compile()
                d = mujoco.MjData(m)
            else:
                raise ValueError("No default-posture source provided")
            # Overwrite the robot slice with the fallback default pose BEFORE FK,
            # so the default posture reflects `default_qpos` (e.g. the static
            # ZMQ_DEFAULT_QPOS stand), not whatever the robot is posed at now
            # (which may be the spawn DEFAULT_QPOS / motion frame0).
            if default_qpos is not None:
                d.qpos[: len(default_qpos)] = np.asarray(default_qpos, dtype=float)
            mujoco.mj_forward(m, d)

            joint_qpos_indices = []
            for joint_name in self.joint_names:
                joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise ValueError(f"Failed to resolve joint name in MJCF: {joint_name}")
                joint_qpos_indices.append(int(m.jnt_qposadr[joint_id]))

            body_ids = []
            for body_name in self.body_names:
                body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise ValueError(f"Failed to resolve body name in MJCF: {body_name}")
                body_ids.append(int(body_id))

            self._default_joint_pos = np.asarray(
                d.qpos[joint_qpos_indices], dtype=np.float32
            )
            self._default_body_pos_w = np.asarray(d.xpos[body_ids], dtype=np.float32)
            self._default_body_quat_w = np.asarray(d.xquat[body_ids], dtype=np.float32)
            logger.info("Initialized default standing posture via MuJoCo FK successfully.")
        except Exception as exc:
            logger.warning(
                f"Failed to initialize default posture via MuJoCo FK: {exc}. "
                "Falling back to zeros."
            )
            self._default_joint_pos = np.zeros(self._num_joints, dtype=np.float32)
            self._default_body_pos_w = np.zeros((self._num_bodies, 3), dtype=np.float32)
            self._default_body_quat_w = np.tile(
                self._identity_quat[None, :], (self._num_bodies, 1)
            )

    def _start_motion_stream(self) -> None:
        import zmq

        if self._motion_stream_thread is not None:
            return

        sock = self._zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, self._motion_zmq_hwm)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self._motion_zmq_connect)
        self._motion_stream_socket = sock

        def _stream_loop() -> None:
            while not self._motion_stream_stop.is_set():
                try:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                except Exception as exc:
                    logger.warning(f"Motion subscriber error: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    self.__append_payload(raw)
                except Exception as exc:
                    logger.warning(f"Failed to decode motion payload: {exc}")

        self._motion_stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._motion_stream_thread.start()

    def __append_payload(
        self,
        payload: dict[str, Any] | str | bytes,
        recv_time_ns: int | None = None,
    ) -> None:
        import json as _json

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = _json.loads(payload.strip())
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        timestamp_ns = (
            payload.get("smplx_t_ns") or recv_time_ns or time.time_ns()
        )
        timestamp_ns = int(timestamp_ns)

        joint_pos = payload.get("joint_pos", payload.get("dof_pos", payload.get("qpos", None)))
        if joint_pos is None:
            raise ValueError("Payload missing joint_pos/dof_pos/qpos")
        joint_pos = _ensure_np(joint_pos, 1)
        if joint_pos.shape[0] >= 7 + self._num_joints and payload.get("joint_pos") is None:
            joint_pos = joint_pos[7 : 7 + self._num_joints]
        if joint_pos.shape[0] != self._num_joints:
            raise ValueError(
                f"Expected {self._num_joints} joint positions, got {joint_pos.shape[0]}"
            )

        body_pos_w = payload.get("body_pos_w", None)
        body_quat_w = payload.get("body_quat_w", None)
        if body_pos_w is None or body_quat_w is None:
            raise ValueError("Payload missing body_pos_w/body_quat_w")

        body_pos_w = _ensure_np(body_pos_w, 2)
        body_quat_w = _ensure_np(body_quat_w, 2)
        if body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w[..., 3], got {body_pos_w.shape}")
        if body_quat_w.shape[-1] != 4:
            raise ValueError(f"Expected body_quat_w[..., 4], got {body_quat_w.shape}")
        if body_pos_w.shape[-2] != self._num_bodies:
            raise ValueError(
                f"Expected {self._num_bodies} body positions, got {body_pos_w.shape[-2]}"
            )
        if body_quat_w.shape[-2] != self._num_bodies:
            raise ValueError(
                f"Expected {self._num_bodies} body quaternions, got {body_quat_w.shape[-2]}"
            )

        joint_pos_frame = joint_pos.astype(np.float32, copy=True)
        body_pos_w_frame = body_pos_w.astype(np.float32, copy=True)
        body_quat_w_frame = _normalize_quat_batch(
            body_quat_w.astype(np.float32, copy=False),
            eps=1e-8,
        ).astype(np.float32, copy=True)

        with self._lock:
            if not self._timestamps_ns or timestamp_ns >= self._timestamps_ns[-1]:
                self._timestamps_ns.append(timestamp_ns)
                self._joint_pos_frames.append(joint_pos_frame)
                self._body_pos_w_frames.append(body_pos_w_frame)
                self._body_quat_w_frames.append(body_quat_w_frame)
            else:
                insert_idx = bisect_right(self._timestamps_ns, timestamp_ns)
                self._timestamps_ns.insert(insert_idx, timestamp_ns)
                self._joint_pos_frames.insert(insert_idx, joint_pos_frame)
                self._body_pos_w_frames.insert(insert_idx, body_pos_w_frame)
                self._body_quat_w_frames.insert(insert_idx, body_quat_w_frame)

    @property
    def latest_timestamp_ns(self) -> int | None:
        with self._lock:
            return self._timestamps_ns[-1] if self._timestamps_ns else None

    def _fill_sample_frames_locked(
        self,
        target_times_ns: np.ndarray,
        joint_pos_out: np.ndarray,
        body_pos_w_out: np.ndarray,
        body_quat_w_out: np.ndarray,
    ) -> None:
        if not self._timestamps_ns:
            joint_pos_out[:] = self._default_joint_pos[None, :]
            body_pos_w_out[:] = self._default_body_pos_w[None, :, :]
            body_quat_w_out[:] = self._default_body_quat_w[None, :, :]
            return

        if len(self._timestamps_ns) == 1:
            joint_pos_out[:] = self._joint_pos_frames[0]
            body_pos_w_out[:] = self._body_pos_w_frames[0]
            body_quat_w_out[:] = self._body_quat_w_frames[0]
            return

        timestamps_ns = np.asarray(self._timestamps_ns, dtype=np.int64)
        clamped_times_ns = np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1])
        right = np.searchsorted(timestamps_ns, clamped_times_ns, side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        t0 = timestamps_ns[left]
        t1 = timestamps_ns[right]
        alpha = np.divide(
            clamped_times_ns - t0,
            t1 - t0,
            out=np.zeros_like(clamped_times_ns, dtype=np.float32),
            where=t1 > t0,
        ).astype(np.float32, copy=False)

        joint_pos_left = np.stack([self._joint_pos_frames[idx] for idx in left], axis=0)
        joint_pos_right = np.stack([self._joint_pos_frames[idx] for idx in right], axis=0)
        alpha_joint = alpha[:, None]
        joint_pos_out[:] = joint_pos_left + alpha_joint * (joint_pos_right - joint_pos_left)

        body_pos_left = np.stack([self._body_pos_w_frames[idx] for idx in left], axis=0)
        body_pos_right = np.stack([self._body_pos_w_frames[idx] for idx in right], axis=0)
        alpha_body = alpha[:, None, None]
        body_pos_w_out[:] = body_pos_left + alpha_body * (body_pos_right - body_pos_left)

        body_quat_left = np.stack([self._body_quat_w_frames[idx] for idx in left], axis=0)
        body_quat_right = np.stack([self._body_quat_w_frames[idx] for idx in right], axis=0)
        body_quat_w_out[:] = _quat_slerp_batch(
            body_quat_left,
            body_quat_right,
            alpha,
            normalize_inputs=False,
            eps=1e-8,
        ).astype(np.float32, copy=False)

    def cleanup(self, cutoff_ns: int) -> None:
        with self._lock:
            while len(self._timestamps_ns) > 1 and self._timestamps_ns[1] < cutoff_ns:
                self._timestamps_ns.pop(0)
                self._joint_pos_frames.pop(0)
                self._body_pos_w_frames.pop(0)
                self._body_quat_w_frames.pop(0)

    def get_obs(self) -> MotionData:
        current_time_ns = time.time_ns()
        num_steps = self.future_steps.shape[0]
        retain_cutoff_ns = current_time_ns - self._history_ns
        self.cleanup(retain_cutoff_ns)

        target_base_ns = current_time_ns - self._delay_ns
        target_times_ns = target_base_ns + self._future_steps_ns

        joint_pos = np.zeros((1, num_steps, self._num_joints), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_quat_w = np.empty((1, num_steps, self._num_bodies, 4), dtype=np.float32)
        body_ang_vel_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)

        with self._lock:
            self._fill_sample_frames_locked(
                target_times_ns,
                joint_pos[0],
                body_pos_w[0],
                body_quat_w[0],
            )

        return MotionData(
            motion_id=self._motion_id_template,
            step=self._step_template,
            timestamps_ns=target_times_ns.reshape(1, -1),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
        )

    def close(self) -> None:
        self._motion_stream_stop.set()
        if self._motion_stream_socket is not None:
            self._motion_stream_socket.close(0)


__all__ = ["RealtimeMotionBuffer"]
