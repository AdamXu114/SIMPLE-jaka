"""ZMQ publishers for the Jaka MF teleop (provider side of the record split).

``teleop_jaka_mf.py`` runs the MuJoCo sim + the whole-body MF policy **in one
process** and, when recording, reads state + camera from ``mjData``/``render``
in-process. To support a **standalone** OpenHLM-style record script
(``simple.cli.record_jaka_zmq``) that runs in its own process, this class
re-publishes exactly what the record script needs over two ZMQ PUB sockets:

  - ``state``   (default ``tcp://*:28711``) — JSON, pico-compatible schema:
    ``publish_t_ns / smplx_t_ns / paused / seq / joint_pos[27] /
    body_pos_w[28,3] / body_quat_w[28,4] / qpos``, all in ``JAKA_*`` order so the
    record script can re-use the same parser as the :28701 motion hub.
  - ``camera``  (default ``tcp://*:28712``) — ``[w,h,c]`` int32 header + RGB
    payload (matches ``HeadZMQClient._decode_message``); the ``head_stereo_left`` render.
    Rendered on the sim thread every control tick (50 Hz) but **sent by a dedicated
    publisher thread on its own clock at ``camera_hz``** (default **30 Hz**), so the
    stream is a steady 30 Hz, decoupled from the control loop (OpenHLM's 50 Hz state /
    30 Hz image split).

Only state + camera are published here. The **reference motion / action** label
is already published by ``pico_retarget_pub`` on ``:28701`` and the pico handle
buttons on ``:5592`` — the record script subscribes to those directly and never
needs this process.
"""

from __future__ import annotations

import json
import struct
import time
from typing import Optional

import numpy as np
import zmq

from simple.datasets.jaka_lerobot import JAKA_BODY_NAMES, JAKA_JOINT_NAMES
from simple.jaka_rl.config import (
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_POS_KEY,
    PUBLISH_T_NS_KEY,
    SEQ_KEY,
    SMPLX_T_NS_KEY,
)


class JakaTeleopZmqPublisher:
    """Publishes Jaka body state + head camera over ZMQ.

    Place one call to :meth:`publish` in the teleop loop (after ``env.step``).
    It reads the robot state straight from ``mjData`` (the same in-process source
    the teleop uses for its own recording) and renders the head camera, so the
    standalone recorder observes the *same* wall-clock state as the teleop.

    Cadence is split, OpenHLM-style (50 Hz state / 30 Hz images), but the two rates are
    produced by **different clocks**:

      * **state** — published by :meth:`publish` every control tick (``rl_rate``, 50 Hz);
      * **camera** — **rendered on every control tick** (50 Hz, on the sim thread — MuJoCo
        render is NOT thread-safe) into a lock-protected "latest frame" slot, and **sent by a
        dedicated publisher thread on its own monotonic clock at a fixed ``camera_hz``**
        (default 30 Hz). Nothing is sent from the control loop, so the camera stream is exactly
        30 Hz regardless of the loop's tick grid. (OpenHLM uses a separate *process* + shared
        memory for the same decoupling; a thread is enough here since the socket is
        single-owner and the payload is small.)

    With ``camera_hz <= 0`` the throttle/thread is disabled and the camera is rendered+sent
    inline on every call (legacy behaviour).
    """

    def __init__(
        self,
        mujoco_sim,
        robot,
        *,
        state_zmq_bind: str = "tcp://*:28711",
        camera_zmq_bind: str = "tcp://*:28712",
        camera_name: str = "head_stereo_left",
        camera_hz: float = 30.0,
    ):
        import threading

        self._sim = mujoco_sim
        self._robot = robot
        self._camera_name = camera_name
        # camera_hz > 0: render at camera_hz, SEND at exactly camera_hz from a dedicated
        # thread. camera_hz <= 0: no throttle/thread, render+send inline every call.
        self._camera_period = (1.0 / camera_hz) if camera_hz > 0 else 0.0
        self._cam_lock = threading.Lock()
        self._cam_frame: Optional[bytes] = None  # latest packed frame (header + payload)
        self._cam_stop = threading.Event()
        self._cam_thread: Optional[threading.Thread] = None

        # Resolve the 28 JAKA body ids (MuJoCo name → id).
        self._body_ids: list[int] = []
        self._resolve_body_ids()

        self._ctx = zmq.Context.instance()
        self._state_sock = self._ctx.socket(zmq.PUB)
        self._state_sock.setsockopt(zmq.LINGER, 0)
        self._state_sock.setsockopt(zmq.SNDHWM, 1)
        self._state_sock.bind(state_zmq_bind)

        # PUB socket for the camera. Owned/used by the camera thread only (ZMQ sockets are
        # not thread-safe); with camera_hz <= 0 it is used inline by publish().
        self._cam_sock = self._ctx.socket(zmq.PUB)
        self._cam_sock.setsockopt(zmq.LINGER, 0)
        self._cam_sock.setsockopt(zmq.SNDHWM, 1)
        self._cam_sock.bind(camera_zmq_bind)

        self._seq = 0
        # Publish counters (for the teleop's periodic rate report). Incremented by the
        # control thread (state) and the camera thread (camera); read via stats().
        self._state_frames = 0
        self._camera_frames = 0

        if self._camera_period > 0.0:
            self._cam_thread = threading.Thread(
                target=self._camera_loop, name="jaka-camera-pub", daemon=True
            )
            self._cam_thread.start()

    def _resolve_body_ids(self) -> None:
        """Map ``JAKA_BODY_NAMES`` to body ids in the sim's *current* mjModel."""
        import mujoco as _mj

        mj = self._sim.mjModel
        ids: list[int] = []
        for name in JAKA_BODY_NAMES:
            bid = _mj.mj_name2id(mj, _mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"body '{name}' not found in MuJoCo model")
            ids.append(int(bid))
        self._body_ids = ids

    def refresh(self) -> None:
        """Re-resolve body ids after the scene was rebuilt (``env.reset()``).

        ``MujocoSimulator._setup_scene`` recompiles ``mjModel``, so the ids cached at
        construction can point at different bodies (or out of range) in the new model —
        which would scramble the published state stream. The ZMQ sockets are untouched.
        """
        self._resolve_body_ids()

    def publish(self) -> Optional[np.ndarray]:
        """Publish one state frame (always) and refresh the camera buffer.

        State is sent here, every call (control rate, 50 Hz). The camera is **rendered here
        every call** (on the sim thread — MuJoCo render is not thread-safe) into the
        lock-protected latest-frame slot; the dedicated camera thread *sends* it at the fixed
        ``camera_hz``, so nothing camera-related leaves the process from this call. Returns
        the rendered head image (or ``None`` on render failure).

        With ``camera_hz <= 0`` the camera is both rendered and sent inline (legacy).
        """
        self._publish_state()
        if self._camera_period <= 0.0:
            return self._send_camera_inline()
        return self._render_camera()

    # ------------------------------------------------------------------
    def _publish_state(self) -> None:
        d = self._sim.mjData

        qpos = {k: float(v) for k, v in self._robot.get_robot_qpos().items()}
        joint_pos = np.array([qpos.get(jn, 0.0) for jn in JAKA_JOINT_NAMES], dtype=np.float32)

        # Body arrays in JAKA_BODY_NAMES order; base_link == index 0.
        body_pos_w = np.asarray(d.xpos[self._body_ids], dtype=np.float32)
        body_quat_w = np.asarray(d.xquat[self._body_ids], dtype=np.float32)

        payload = {
            PUBLISH_T_NS_KEY: int(time.time_ns()),
            SMPLX_T_NS_KEY: int(d.time * 1e9),
            "paused": False,
            SEQ_KEY: self._seq,
            JOINT_POS_KEY: joint_pos.tolist(),
            BODY_POS_W_KEY: body_pos_w.tolist(),
            BODY_QUAT_W_KEY: body_quat_w.tolist(),
            "qpos": np.asarray(d.qpos, dtype=np.float32).tolist(),
        }
        self._seq += 1
        try:
            self._state_sock.send_string(
                json.dumps(payload, separators=(",", ":")), flags=zmq.NOBLOCK
            )
            self._state_frames += 1
        except zmq.Again:
            pass

    def _render_camera(self) -> Optional[np.ndarray]:
        """Render the head camera (sim thread) and store the packed frame for the camera thread.

        Does **not** send — the camera thread owns the socket. Returns the rendered image.
        """
        try:
            images = self._sim.render(camera_names=[self._camera_name])
        except Exception:
            return None
        img = images.get(self._camera_name)
        if img is None or img.size == 0:
            return None
        img = np.ascontiguousarray(img.astype(np.uint8))
        h, w, c = img.shape[:3]
        buf = struct.pack("iii", w, h, c) + img.tobytes()
        with self._cam_lock:
            self._cam_frame = buf
        return img

    def _send_camera_inline(self) -> Optional[np.ndarray]:
        """Legacy path (``camera_hz <= 0``): render + send on the calling thread."""
        img = self._render_camera()
        if img is None:
            return None
        self._send_latest_frame()
        return img

    def _send_latest_frame(self) -> bool:
        """Send the latest stored frame on the camera socket (non-blocking). Returns sent?"""
        with self._cam_lock:
            buf = self._cam_frame
        if buf is None:
            return False
        try:
            self._cam_sock.send(buf, flags=zmq.NOBLOCK)
            self._camera_frames += 1
            return True
        except zmq.Again:
            return False  # SNDHWM=1 and no subscriber ready: drop, next tick will retry

    def _camera_loop(self) -> None:
        """Dedicated camera publisher: send the latest frame at a FIXED ``camera_hz``.

        Runs on its own ``time.monotonic()`` clock, independent of the control loop, so the
        stream is a steady 30 Hz rather than snapped to the loop's 50 Hz tick grid. Deadline
        is accumulated (``+=``), not ``now + period`` — the latter would drift a slot every
        time the send runs slightly late. If it falls behind (machine stall) it resyncs
        instead of bursting catch-up frames.
        """
        period = self._camera_period
        next_t = time.monotonic() + period
        while not self._cam_stop.is_set():
            now = time.monotonic()
            if next_t > now:
                # Sleep most of the wait, then busy-wait the last ~1 ms for a precise tick
                # (same pacing trick the control loop uses).
                self._cam_stop.wait(max(0.0, (next_t - now) - 0.001))
                if self._cam_stop.is_set():
                    break
                while time.monotonic() < next_t and not self._cam_stop.is_set():
                    pass
            self._send_latest_frame()
            next_t += period
            now = time.monotonic()
            if next_t < now:  # fell behind: resync
                next_t = now + period

    def stats(self) -> tuple[int, int]:
        """Cumulative ``(state_frames, camera_frames)`` published since construction.

        Sample twice over a known interval to get the actual publish rates (the teleop prints
        them periodically). State is written by the control thread, camera by the camera thread.
        """
        return self._state_frames, self._camera_frames

    def close(self) -> None:
        self._cam_stop.set()
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=1.0)
            self._cam_thread = None
        for sock in (self._state_sock, self._cam_sock):
            try:
                sock.close(0)
            except Exception:
                pass


__all__ = ["JakaTeleopZmqPublisher"]
