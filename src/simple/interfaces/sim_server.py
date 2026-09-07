"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

ZMQ-based simulation server for external policy control.

Protocol (JSON over ZMQ REP/REQ):

  Request →  {"type": "step", "action": {"type": "position", "target_qpos": {...}}}
  Response ← {"joint_qpos": {...}, "joint_qvel": {...}, "info": {...}}

  Request →  {"type": "reset"}
  Response ← {"joint_qpos": {...}, "joint_qvel": {...}, "info": {...}}

  Request →  {"type": "get_obs"}
  Response ← {"joint_qpos": {...}, "joint_qvel": {...}, "info": {...}}

  Request →  {"type": "close"}
  Response ← {"status": "closed"}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class SimulationServer:
    """ZMQ REP server wrapping a SIMPLE simulation environment.

    Usage::

        import gymnasium as gym
        server = SimulationServer(env_id="simple/JakaWholebodySim-v0", port=5555)
        server.run()
    """

    def __init__(
        self,
        env_id: str = "simple/JakaWholebodySim-v0",
        host: str = "127.0.0.1",
        port: int = 5555,
        sim_mode: str = "mujoco",
        headless: bool = True,
        render_hz: int = 30,
    ) -> None:
        self.env_id = env_id
        self.host = host
        self.port = port
        self.sim_mode = sim_mode
        self.headless = headless
        self.render_hz = render_hz

        self._env = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._running = False
        self._step_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Create the environment and bind the ZMQ socket."""
        import gymnasium as gym

        # Trigger env registration
        import simple.envs as _  # noqa: F401

        logger.info("Creating environment: %s", self.env_id)
        self._env = gym.make(
            self.env_id,
            sim_mode=self.sim_mode,
            headless=self.headless,
            render_hz=self.render_hz,
        )
        self._env.reset()
        logger.info("Environment ready.")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{self.host}:{self.port}"
        self._socket.bind(addr)
        logger.info("ZMQ server listening on %s", addr)
        self._running = True

    def stop(self) -> None:
        """Close environment and ZMQ resources."""
        self._running = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        if self._env is not None:
            self._env.close()
            self._env = None
        logger.info("Server stopped.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the server loop (blocking)."""
        self.start()
        try:
            while self._running:
                try:
                    raw = self._socket.recv_json()  # type: ignore[union-attr]
                    response = self._handle(raw)
                    self._socket.send_json(response, cls=NumpyEncoder)  # type: ignore[union-attr]
                except zmq.ZMQError as e:
                    logger.error("ZMQ error: %s", e)
                    break
                except Exception as e:
                    logger.exception("Error handling request: %s", e)
                    self._socket.send_json(  # type: ignore[union-attr]
                        {"error": str(e)}, cls=NumpyEncoder
                    )
        finally:
            self.stop()

    def run_once(self) -> Dict[str, Any] | None:
        """Process a single request (non-blocking wait with timeout).

        Returns the response dict, or None on timeout.
        """
        if self._socket is None:
            self.start()
        try:
            if self._socket.poll(timeout=100):  # type: ignore[union-attr]
                raw = self._socket.recv_json()  # type: ignore[union-attr]
                response = self._handle(raw)
                self._socket.send_json(response, cls=NumpyEncoder)  # type: ignore[union-attr]
                return response
        except zmq.ZMQError as e:
            logger.error("ZMQ error: %s", e)
        return None

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------
    def _handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_type = request.get("type", "step")

        if req_type == "reset":
            return self._handle_reset(request)
        elif req_type == "step":
            return self._handle_step(request)
        elif req_type == "get_obs":
            return self._handle_get_obs()
        elif req_type == "close":
            self._running = False
            return {"status": "closed"}
        else:
            return {"error": f"Unknown request type: {req_type}"}

    def _handle_reset(self, request: Dict[str, Any]) -> Dict[str, Any]:
        seed = request.get("seed")
        options = request.get("options")
        obs, info = self._env.reset(seed=seed, options=options)  # type: ignore[union-attr]
        self._step_count = 0
        return {"joint_qpos": obs.get("joint_qpos", []),
                "joint_qvel": obs.get("joint_qvel", []),
                "info": info}

    def _handle_step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action", {})
        # Support both flat format and typed format
        if "type" not in action:
            action = {"type": "position", "parameters": {"target_qpos": action}}

        obs, reward, terminated, truncated, info = self._env.step(action)  # type: ignore[union-attr]
        self._step_count += 1

        response: Dict[str, Any] = {
            "joint_qpos": obs.get("joint_qpos", []),
            "joint_qvel": obs.get("joint_qvel", []),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self._step_count,
            "info": info,
        }
        return response

    def _handle_get_obs(self) -> Dict[str, Any]:
        obs = self._env.unwrapped._get_obs()  # type: ignore[union-attr]
        info = self._env.unwrapped._get_info()  # type: ignore[union-attr]
        return {"joint_qpos": obs.get("joint_qpos", []),
                "joint_qvel": obs.get("joint_qvel", []),
                "info": info}
