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
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import mujoco
import zmq

from loguru import logger

from simple.jaka_rl.config import TOGGLE_DATA_COLLECTION_KEY
from simple.jaka_rl.math import quat_normalize, quat_slerp_batch
from simple.jaka_rl.motion import MotionData

_normalize_quat_batch = quat_normalize
_quat_slerp_batch = quat_slerp_batch


def _ensure_np(value: Any, ndim: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"Expected ndim={ndim}, got shape={arr.shape}")
    return arr


@dataclass
class LatestMotionFrame:
    """The most recent reference motion frame, for record-time ("now") sampling.

    Unlike :meth:`RealtimeMotionBuffer.get_obs`, this carries **no** control-loop
    look-back (``_delay_ns``): it is the reference *as it is right now*, so it
    sits on the same wall-clock basis as the live robot state/camera it is
    recorded against. This mirrors OpenHLM's "latest pico frame" semantics
    (``joint_pos[-1]``). ``prev_body_pos_w`` / ``prev_timestamp_ns`` expose the
    frame immediately before the latest so a reference linear velocity can be
    computed from the raw stream.

    ``body_quat_w`` is world-frame **wxyz** (MuJoCo ``xquat`` order).
    """

    joint_pos: np.ndarray  # (num_joints,), JAKA/robot order
    body_pos_w: np.ndarray  # (num_bodies, 3)
    body_quat_w: np.ndarray  # (num_bodies, 4), wxyz
    timestamp_ns: int
    prev_body_pos_w: Optional[np.ndarray]  # (num_bodies, 3) or None
    prev_timestamp_ns: Optional[int]  # or None


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
        # Latest ``toggle_data_collection`` level seen on the stream (see
        # :attr:`latest_toggle_data_collection`); None until the first payload.
        self._latest_toggle_data_collection: bool | None = None

        # ── Stream diagnostics (see :attr:`diagnostics`) ──────────────────
        # All are cheap counters refreshed by the ingest thread / get_obs; they
        # exist to tell a *live* pico stream apart from a replay-like one, since
        # the policy only sees this buffer.
        self._diag_payloads = 0            # payloads accepted
        self._diag_ts_regressions = 0      # payload ts older than the newest -> bisect insert
        self._diag_last_payload_ts_ns: int | None = None
        self._diag_last_recv_ns: int | None = None   # local wall clock at receipt
        self._diag_last_smplx_ts_ns: int | None = None  # the payload's raw smplx_t_ns
        self._diag_last_dt_ns: int | None = None     # ts gap vs the previous payload
        self._diag_max_jump_m = 0.0        # largest per-frame body translation jump (all time)
        # Rolling (2 s bucket) max jump, so a one-off transition (e.g. the idle
        # stand -> live motion frame) does not mask jumps that recur while tracking.
        self._jump_bucket_m = 0.0
        self._jump_bucket_prev_m = 0.0
        self._jump_bucket_t0 = time.monotonic()
        self._diag_clamped_low = False     # get_obs: window starts before the oldest frame
        self._diag_clamped_high = False    # get_obs: window ends after the newest frame
        self._diag_window_frames = 0       # distinct buffered frames the window touched
        self._diag_buffered = 0
        self._collapsed_warned = False
        self._collapsed_warned_at = 0.0

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
        """Resolve the default posture used when the buffer holds no frames.

        Mirrors sim2real-jaka, which FKs the robot MJCF at the default qpos. The
        caller here hands us its already-compiled ``mj_model`` (instead of a
        ``robot_cfg`` MJCF path), so the same FK runs on a SCRATCH ``MjData``.
        ``mj_data`` is accepted for API compatibility only and is deliberately
        unused: writing ``default_qpos`` into the LIVE ``mj_data`` would re-pose
        the robot from its task spawn (e.g. x=-1.45 in front of a table) to the
        neutral-stand origin (x=0), dropping it into the table.

        A model is required — there is no synthetic-model fallback. The old
        ``mujoco.MjSpec()``-built placeholder model raised on mujoco>=3.3
        (``MjSpec.add_joint`` does not exist), and the ``except`` below turned
        that into a silent all-zero posture.
        """
        if mj_model is None:
            logger.error(
                "RealtimeMotionBuffer was given no mj_model, so the default posture "
                "cannot be FK'd. The empty-buffer fallback (before the first stream "
                "frame, and right after clear()) will be an ALL-ZERO reference "
                "posture — anchor at z=0 instead of the default stand (~0.874)."
            )
            self._set_zero_default_posture()
            return

        try:
            m = mj_model
            d = mujoco.MjData(m)
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
            logger.error(
                f"Failed to initialize the default posture via MuJoCo FK: {exc}. "
                "Falling back to ALL ZEROS — the empty-buffer reference (before the "
                "first stream frame, and right after clear()) will be wrong: anchor "
                "at z=0 instead of the default stand."
            )
            self._set_zero_default_posture()

    def _set_zero_default_posture(self) -> None:
        """Last-resort default posture: zero joints, all-identity body quats.

        Only correct if a zero posture is genuinely what the reference should be;
        normally this means the FK route failed and the empty-buffer fallback is
        wrong (see :meth:`_init_default_posture`).
        """
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

        # Which clock to buffer on. ``get_obs`` places its window with the LOCAL
        # ``time.time_ns()``, so the stored timestamps must be on that same clock or
        # ``cleanup``/``np.clip`` silently collapse the look-ahead (see
        # :meth:`diagnostics`). ``publish_t_ns`` is the publisher's own wall clock —
        # pico_retarget_pub sets it from ``time.time_ns()`` on this same machine, so
        # it is directly comparable. ``smplx_t_ns`` is the HEADSET's timestamp (and
        # the controller stream's while paused), i.e. a different clock; using it
        # (as the sim2real reference does) is what breaks a live pico stream while
        # a replay-style publisher — which stamps both fields with the wall clock —
        # works fine. Falls back to smplx / arrival time for payloads without it.
        smplx_ts_ns = payload.get("smplx_t_ns")
        timestamp_ns = (
            payload.get("publish_t_ns") or smplx_ts_ns or recv_time_ns or time.time_ns()
        )
        timestamp_ns = int(timestamp_ns)

        # Record-control toggle (level, not edge) — read before the frame
        # validation below so a malformed motion frame still updates it.
        self._update_toggle_data_collection(payload)

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
            self._diag_payloads += 1
            prev_ts = self._timestamps_ns[-1] if self._timestamps_ns else None
            if prev_ts is not None:
                self._diag_last_dt_ns = timestamp_ns - prev_ts
                if timestamp_ns < prev_ts:
                    # Older than the newest frame: this is the bisect-insert path,
                    # i.e. the stream's clock went backwards (mixed clock domains,
                    # or a paused publisher stamping a foreign clock). The timeline
                    # then interleaves two epochs, which corrupts the window.
                    self._diag_ts_regressions += 1
                if self._body_pos_w_frames:
                    jump = float(
                        np.abs(body_pos_w_frame - self._body_pos_w_frames[-1]).max()
                    )
                    if np.isfinite(jump):
                        self._diag_max_jump_m = max(self._diag_max_jump_m, jump)
                        mono = time.monotonic()
                        if mono - self._jump_bucket_t0 >= 2.0:
                            self._jump_bucket_prev_m = self._jump_bucket_m
                            self._jump_bucket_m = 0.0
                            self._jump_bucket_t0 = mono
                        self._jump_bucket_m = max(self._jump_bucket_m, jump)
            self._diag_last_payload_ts_ns = timestamp_ns
            self._diag_last_recv_ns = time.time_ns()
            self._diag_last_smplx_ts_ns = None if smplx_ts_ns is None else int(smplx_ts_ns)

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

    def diagnostics(self) -> dict[str, Any]:
        """Health of the incoming reference stream — for telling a live pico stream
        apart from a replay-like one. All values are cheap counters, no behaviour change.

        Read as a group:

        ``ts_offset_ms``  buffered timestamp minus the LOCAL wall clock at receipt.
                          Should stay near 0: the buffer must run on this process's
                          clock (it places the window with ``time.time_ns()``).
        ``smplx_offset_ms`` the payload's raw ``smplx_t_ns`` minus that same local
                          clock. A live pico hub stamps the HEADSET's clock here
                          (``body["timeStampNs"]`` via the XRoboToolkit service), so
                          a huge value is expected and harmless *as long as*
                          ``ts_offset_ms`` is ~0 — it just means the two clocks are
                          different domains, which is exactly why ``smplx_t_ns`` must
                          not be used as the buffer's timeline.
        ``clamped``       the requested window extended past the buffered range, i.e.
                          the look-ahead was silently cut.
        ``window_frames`` distinct buffered frames the last window touched. 1 means
                          every future step was the same pose → ``root_pos_diff_b``
                          is 0 → the policy sees no feed-forward at all.
        ``buffered``      frames currently in the buffer (~``_history_ns`` worth).
        ``ts_regressions``payloads that arrived with a timestamp older than the newest
                          one. >0 means the stream's clock went backwards (mixed
                          domains, or a paused publisher stamping another clock) and
                          those frames were inserted mid-timeline.
        ``jump_recent_m`` largest per-frame body translation jump within the last ~2 s
                          (``max_jump_m`` is the all-time one, which a single
                          transition can pin, hiding jumps that recur while tracking).
        """
        with self._lock:
            offset = (
                None
                if self._diag_last_payload_ts_ns is None or self._diag_last_recv_ns is None
                else (self._diag_last_payload_ts_ns - self._diag_last_recv_ns) / 1e6
            )
            smplx_offset = (
                None
                if self._diag_last_smplx_ts_ns is None or self._diag_last_recv_ns is None
                else (self._diag_last_smplx_ts_ns - self._diag_last_recv_ns) / 1e6
            )
            return {
                "payloads": self._diag_payloads,
                "buffered": self._diag_buffered,
                "window_frames": self._diag_window_frames,
                "clamped": self._diag_clamped_low or self._diag_clamped_high,
                "clamped_lo": self._diag_clamped_low,
                "clamped_hi": self._diag_clamped_high,
                "ts_offset_ms": offset,
                "smplx_offset_ms": smplx_offset,
                "last_dt_ms": (
                    None if self._diag_last_dt_ns is None else self._diag_last_dt_ns / 1e6
                ),
                "ts_regressions": self._diag_ts_regressions,
                "max_jump_m": self._diag_max_jump_m,
                "jump_recent_m": max(self._jump_bucket_m, self._jump_bucket_prev_m),
                "delay_ms": self._delay_ns / 1e6,
                "history_ms": self._history_ns / 1e6,
            }

    def _update_toggle_data_collection(self, payload: dict[str, Any]) -> None:
        """Store the payload's ``toggle_data_collection`` level (if present).

        The publisher (``pico_retarget_pub``) sends ``[bool]`` on every frame: a
        persistent level it flips on the PICO A button. Accepts a list/array/bool.
        """
        raw = payload.get(TOGGLE_DATA_COLLECTION_KEY, None)
        if raw is None:
            return
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return
            raw = raw[0]
        with self._lock:
            self._latest_toggle_data_collection = bool(raw)

    @property
    def latest_toggle_data_collection(self) -> bool | None:
        """Latest record-control level from the stream, or ``None`` before the first frame."""
        with self._lock:
            return self._latest_toggle_data_collection

    # ------------------------------------------------------------------
    # Live sessions (pico paused → live)
    # ------------------------------------------------------------------
    def _fill_sample_frames_locked(
        self,
        target_times_ns: np.ndarray,
        joint_pos_out: np.ndarray,
        body_pos_w_out: np.ndarray,
        body_quat_w_out: np.ndarray,
    ) -> None:
        self._diag_buffered = len(self._timestamps_ns)
        if not self._timestamps_ns:
            self._diag_clamped_low = self._diag_clamped_high = True
            self._diag_window_frames = 0
            joint_pos_out[:] = self._default_joint_pos[None, :]
            body_pos_w_out[:] = self._default_body_pos_w[None, :, :]
            body_quat_w_out[:] = self._default_body_quat_w[None, :, :]
            return

        if len(self._timestamps_ns) == 1:
            # One frame replicated across the whole window: the policy gets NO
            # look-ahead (root_pos_diff_b == 0), so it cannot anticipate motion.
            self._diag_clamped_low = self._diag_clamped_high = True
            self._diag_window_frames = 1
            joint_pos_out[:] = self._joint_pos_frames[0]
            body_pos_w_out[:] = self._body_pos_w_frames[0]
            body_quat_w_out[:] = self._body_quat_w_frames[0]
            return

        timestamps_ns = np.asarray(self._timestamps_ns, dtype=np.int64)
        # The window is expressed on the CONSUMER's wall clock while the buffer is
        # keyed on the publisher's timestamp: if those are different clock domains
        # (or the stream stalled), the clip below silently collapses the look-ahead.
        self._diag_clamped_low = bool(target_times_ns.min() < timestamps_ns[0])
        self._diag_clamped_high = bool(target_times_ns.max() > timestamps_ns[-1])
        clamped_times_ns = np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1])
        right = np.searchsorted(timestamps_ns, clamped_times_ns, side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        self._diag_window_frames = int(len(np.unique(np.stack([left, right]))))
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

    def clear(self) -> None:
        """Drop all buffered frames, so the reference falls back to the default posture.

        With no frames buffered, :meth:`get_obs` fills the window with
        ``_default_joint_pos`` / ``_default_body_*`` (the FK'd ``default_qpos`` stand,
        i.e. ``ZMQ_DEFAULT_QPOS``) — so right after a reset the policy tracks the
        **default joint positions** instead of whatever the reference stream held
        before (or is publishing now). Mirrors the "start from the default pose"
        behaviour of a fresh episode.

        The next payload from the stream re-fills the buffer as usual, and the
        record-control toggle is deliberately left untouched (it is not motion data).
        """
        with self._lock:
            self._timestamps_ns.clear()
            self._joint_pos_frames.clear()
            self._body_pos_w_frames.clear()
            self._body_quat_w_frames.clear()

    def _warn_if_lookahead_collapsed(self) -> None:
        """Warn (rate-limited) when the look-ahead window has collapsed to one frame.

        That means every future step resolves to the same pose, so the policy sees
        ``root_pos_diff_b == 0``: it still tracks, but purely reactively, with no
        feed-forward — the robot lags and easily goes unstable instead of failing
        outright. Worth surfacing because nothing else in the pipeline signals it.
        """
        s = self.diagnostics()
        if s["payloads"] == 0 or s["window_frames"] > 1:
            self._collapsed_warned = False
            return
        now = time.monotonic()
        if self._collapsed_warned and now - self._collapsed_warned_at < 5.0:
            return
        self._collapsed_warned, self._collapsed_warned_at = True, now
        logger.warning(
            "Reference look-ahead collapsed: all {} future step(s) resolve to ONE "
            "buffered frame (buffered={}, clamped={}, ts_offset={} ms, "
            "smplx_offset={} ms). The policy now sees root_pos_diff_b == 0 and "
            "cannot anticipate motion. Usual cause: the stream is keyed on a "
            "timestamp from another clock domain, or the publisher stalled.",
            len(self.future_steps), s["buffered"], s["clamped"],
            None if s["ts_offset_ms"] is None else round(s["ts_offset_ms"], 1),
            None if s["smplx_offset_ms"] is None else round(s["smplx_offset_ms"], 1),
        )

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

        self._warn_if_lookahead_collapsed()

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

    def get_latest_frame(self) -> Optional[LatestMotionFrame]:
        """Return the most recent reference frame (no look-back), or ``None``.

        Record-time counterpart of :meth:`get_obs`: it reads the newest buffered
        frame directly and applies **no** ``_delay_ns`` look-back, so the
        reference is sampled at the present wall-clock instant and therefore
        aligns with the live robot state/camera it is recorded against. Falls
        back to ``None`` (rather than a metadata-only frame) so callers can
        distinguish "no reference data yet" from a real frame.
        """
        with self._lock:
            n = len(self._timestamps_ns)
            if n == 0:
                return None
            return LatestMotionFrame(
                joint_pos=self._joint_pos_frames[-1],
                body_pos_w=self._body_pos_w_frames[-1],
                body_quat_w=self._body_quat_w_frames[-1],
                timestamp_ns=self._timestamps_ns[-1],
                prev_body_pos_w=(self._body_pos_w_frames[-2] if n >= 2 else None),
                prev_timestamp_ns=(self._timestamps_ns[-2] if n >= 2 else None),
            )

    def close(self) -> None:
        self._motion_stream_stop.set()
        if self._motion_stream_socket is not None:
            self._motion_stream_socket.close(0)


__all__ = ["RealtimeMotionBuffer", "LatestMotionFrame"]
