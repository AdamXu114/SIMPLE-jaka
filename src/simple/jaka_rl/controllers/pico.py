"""Pico controller (handle buttons :5592) for the Jaka MF policy.

Ported from sim2real-jaka ``rl_policy/controllers/pico.py``. The pico hub
publishes a ``PicoControllerStateMessage`` binary packet; A/B button combos
(yaw) map to init / zero / policy modes over ZMQ.

Port: from sim2real ``utils/common.PORTS['pico_controller']`` = 5592.
"""

from __future__ import annotations

import struct
from copy import deepcopy

import zmq
from loguru import logger

from simple.jaka_rl.control_mode import PicoButtonState, resolve_pico_control_mode
from simple.jaka_rl.controllers.base import ControllerBase

PICO_CONTROLLER_PORT = 5592


class PicoControllerStateMessage:
    """Binary message containing PICO controller button states.

    Layout: ``<QBBBB`` → timestamp_ns (u64) + A/B/X/Y bytes.
    """

    _STRUCT = struct.Struct("<QBBBB")

    def __init__(
        self,
        timestamp_ns: int = 0,
        A: bool = False,
        B: bool = False,
        X: bool = False,
        Y: bool = False,
    ):
        self.timestamp_ns = int(timestamp_ns)
        self.A = bool(A)
        self.B = bool(B)
        self.X = bool(X)
        self.Y = bool(Y)

    def to_bytes(self) -> bytes:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        return self._STRUCT.pack(
            self.timestamp_ns,
            int(self.A),
            int(self.B),
            int(self.X),
            int(self.Y),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PicoControllerStateMessage":
        expected_size = cls._STRUCT.size
        if len(data) != expected_size:
            raise ValueError(
                f"invalid size: expected {expected_size} bytes, got {len(data)}"
            )
        timestamp_ns, a, b, x, y = cls._STRUCT.unpack(data)
        return cls(
            timestamp_ns=timestamp_ns,
            A=bool(a),
            B=bool(b),
            X=bool(x),
            Y=bool(y),
        )


class PicoController(ControllerBase):
    name = "pico"

    def __init__(
        self,
        connect: str = f"tcp://127.0.0.1:{PICO_CONTROLLER_PORT}",
        hwm: int = 1,
    ) -> None:
        self._pico_msg = PicoButtonState()
        self._last_pico_msg = PicoButtonState()
        self._available = True
        self._connect = connect

        self._zmq_context = zmq.Context.instance()
        self._socket = self._zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, int(hwm))
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            self._socket.connect(connect)
        except Exception as exc:
            self._available = False
            self._socket.close(0)
            logger.warning(
                f"PICO controller ZMQ subscriber unavailable, continuing without it: {exc}"
            )
        else:
            logger.info("PICO controller ZMQ subscriber connected to {}", connect)

    def _receive_pico_controller(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            try:
                decoded = PicoControllerStateMessage.from_bytes(raw)
            except Exception as exc:
                logger.debug(f"PICO controller ZMQ decode error: {exc}")
                continue

            self._pico_msg = PicoButtonState(
                A=decoded.A,
                B=decoded.B,
            )

    def get_control_mode(self):
        if not self._available:
            return None

        try:
            self._receive_pico_controller()
        except Exception as exc:
            logger.debug(
                f"PICO controller ZMQ receive error from {self._connect}: {exc}"
            )
            return None

        pico_local = deepcopy(self._pico_msg)
        mode = resolve_pico_control_mode(pico_local, self._last_pico_msg)
        self._last_pico_msg = pico_local
        return mode

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug(f"Failed to stop PICO listener cleanly: {exc}")


__all__ = ["PicoController", "PicoControllerStateMessage", "PICO_CONTROLLER_PORT"]
