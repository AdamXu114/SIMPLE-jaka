"""Standalone OpenHLM-style data record script for the Jaka MF teleop.

This is THE recording path for Jaka (the teleop no longer records in-process):
run ``teleop_jaka_mf.py`` in one terminal and this script in another (the teleop always
publishes state + head camera over ZMQ; see ``state_zmq_bind`` / ``camera_zmq_bind``
in ``data/jaka_mf/teleop_jaka_mf.yaml``).
Instead of holding the sim/control loop, this **separate process** subscribes over ZMQ
to the *latest* frame of each stream and writes the openhlm LeRobot format (30-dim
state + 40-dim actions + RAW-resolution head image — the training side resizes).

Streams it reads (all "latest-only", drained each loop so state/action/image are
from the same wall-clock instant):

  - ``state``     : Jaka body state from the teleop provider (``JakaTeleopZmqPublisher``)
  - ``action``    : reference motion from the pico hub (``pico_retarget_pub`` :28701)
                    — this is the REFERENCE the frozen policy tracks, not the policy output.
                    Subscribed DIRECTLY (``LatestMotionClient``, mirroring OpenHLM's
                    ``run_openhlm_data_record.py``): no ``RealtimeMotionBuffer``, just the
                    newest frame each loop.
  - ``camera``    : head_stereo_left RGB from the teleop provider (~30 Hz, matching this
                    script's ``--frequency`` default; the provider publishes state at 50 Hz
                    but throttles the camera to 30 Hz)
  - ``trigger``   : record start/stop — 默认 ``both``:跟随 pico **A 按键**在运动流里广播的
                    ``toggle_data_collection`` 电平(与 teleop 的「录制结束 reset」同一
                    个信号,读的是同一个 :28701 订阅,不额外开 socket),同时键盘
                    **回车** 也能手动开/停、``q`` 退出。可显式指定
                    ``keyboard`` / ``pico``。

Records at ``--frequency`` (default 30 Hz), taking only the latest frame of each stream.

Usage (run the sim/control in another terminal)::

    python -m simple.cli.record_jaka_zmq \\
        --save-dir data/teleop_jaka_mf_zmq \\
        --desc "put the bottle on the mouse pad" \\
        --frequency 30 \\
        --policy-config data/jaka_mf/latest56k_pico_dr.yaml

Output: ``<save-dir>/<run-id>/level-<dr-level>``, where ``run-id`` is ``--run-name``
if given, else a timestamp (``20260915-134512``). An existing dataset is **never
wiped or reused** — a colliding run id gets a numeric suffix instead, so every run
starts a new directory (point ``--save-dir`` at another root to move it elsewhere).
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np
import typer
import zmq
from typing_extensions import Annotated

from simple.datasets.jaka_lerobot import (
    JAKA_BODY_NAMES,
    JAKA_JOINT_NAMES,
    OPENHLM_ACTION_DIM,
    OPENHLM_STATE_DIM,
    OpenHLMRootVel,
    build_openhlm_frame,
    finalize_openhlm_episode,
    init_openhlm_exporter,
    reference_action_live_latest,
)
from simple.jaka_rl.config import (
    JOINT_POS_KEY,
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    PUBLISH_T_NS_KEY,
    SMPLX_T_NS_KEY,
    TOGGLE_DATA_COLLECTION_KEY,
)

_DEFAULT_POLICY_CONFIG = "data/jaka_mf/latest56k_pico_dr.yaml"


# ---------------------------------------------------------------------------
# Latest-only ZMQ clients (CONFLATE + keep the freshest frame under a lock)
# ---------------------------------------------------------------------------
class LatestStateClient:
    """Subscribe to the provider's JSON state stream; keep only the newest frame.

    Exposes the record-relevant fields directly: ``joint_pos`` (27, JAKA order)
    and ``base_quat_wxyz`` (the robot base / ``base_link`` world orientation).
    """

    def __init__(self, connect: str, hwm: int = 1):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, int(hwm))
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(connect)
        self._lock = threading.Lock()
        self._joint_pos: Optional[np.ndarray] = None
        self._base_quat_wxyz: Optional[np.ndarray] = None
        self._ts_ns: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            joint_pos = np.asarray(payload.get(JOINT_POS_KEY), dtype=np.float32)
            body_quat_w = np.asarray(payload.get(BODY_QUAT_W_KEY), dtype=np.float32)
            if joint_pos.shape[0] != 27 or body_quat_w.ndim != 2:
                continue
            with self._lock:
                self._joint_pos = joint_pos.copy()
                self._base_quat_wxyz = body_quat_w[0].copy()  # base_link
                self._ts_ns = int(payload.get(PUBLISH_T_NS_KEY, time.time_ns()))

    def latest(self):
        with self._lock:
            return self._joint_pos, self._base_quat_wxyz, self._ts_ns

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close(0)
        except Exception:
            pass


class LatestImageClient:
    """Subscribe to the provider's packed RGB camera stream; keep only the newest frame.

    Wire format matches ``HeadZMQClient._decode_message``: ``[w,h,c]`` int32 header
    + RGB payload.
    """

    def __init__(self, connect: str, hwm: int = 1):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, int(hwm))
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(connect)
        self._lock = threading.Lock()
        self._image: Optional[np.ndarray] = None
        self._ts_ms: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            if len(msg) < 12:
                continue
            w, h, c = struct.unpack("iii", msg[:12])
            payload = msg[12:]
            if len(payload) != w * h * c or c not in (3, 4):
                continue
            img = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))
            if c == 4:  # BGRA -> BGR
                img = img[..., :3]
            with self._lock:
                self._image = img.copy()
                self._ts_ms = int(time.time() * 1000)

    def latest(self):
        with self._lock:
            return self._image

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close(0)
        except Exception:
            pass


class LatestMotionClient:
    """Direct SUB to the pico retarget motion stream; keep only the newest frame.

    Mirrors OpenHLM's direct pico subscription (``run_openhlm_data_record.py``): a plain
    SUB socket, **no** ``RealtimeMotionBuffer`` and no interpolation — the background
    thread just overwrites the latest decoded payload each time. It exposes the minimal
    interface ``reference_action_live_latest`` needs (``joint_names`` / ``body_names`` /
    ``get_latest_frame``) and returns a ``LatestMotionFrame`` that also carries the
    previous frame so a reference linear velocity can be finite-differenced.

    Payload (JSON, from ``pico_retarget_pub._build_payload``): ``joint_pos``,
    ``body_pos_w``, ``body_quat_w`` (wxyz), ``publish_t_ns`` / ``smplx_t_ns``.
    """

    def __init__(self, connect: str, joint_names, body_names, hwm: int = 1):
        from simple.jaka_rl.motion_buffer import LatestMotionFrame

        self._frame_cls = LatestMotionFrame
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, int(hwm))
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(connect)
        self._lock = threading.Lock()
        self._joint_pos: Optional[np.ndarray] = None
        self._body_pos_w: Optional[np.ndarray] = None
        self._body_quat_w: Optional[np.ndarray] = None
        self._ts: Optional[int] = None
        self._prev_body_pos_w: Optional[np.ndarray] = None
        self._prev_ts: Optional[int] = None
        # Record-control level broadcast alongside the motion frames (None = no data yet).
        self._toggle_data_collection: Optional[bool] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            joint_pos = payload.get(JOINT_POS_KEY)
            body_pos_w = payload.get(BODY_POS_W_KEY)
            body_quat_w = payload.get(BODY_QUAT_W_KEY)
            if joint_pos is None or body_pos_w is None or body_quat_w is None:
                continue
            ts = int(
                payload.get(PUBLISH_T_NS_KEY)
                or payload.get(SMPLX_T_NS_KEY)
                or time.time_ns()
            )
            # PICO A-button record toggle: a persistent [bool] level in every payload.
            toggle = payload.get(TOGGLE_DATA_COLLECTION_KEY)
            if isinstance(toggle, (list, tuple, np.ndarray)):
                toggle = toggle[0] if len(toggle) else None
            with self._lock:
                # 只保留最新帧; 上一帧留着算参考线速度(差分)。
                self._prev_body_pos_w = self._body_pos_w
                self._prev_ts = self._ts
                self._joint_pos = np.asarray(joint_pos, dtype=np.float32)
                self._body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
                self._body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
                self._ts = ts
                if toggle is not None:
                    self._toggle_data_collection = bool(toggle)

    @property
    def latest_toggle_data_collection(self) -> Optional[bool]:
        """Latest record-control level from the stream, or ``None`` before the first frame."""
        with self._lock:
            return self._toggle_data_collection

    def get_latest_frame(self):
        """最新一帧(无则 None),字段与 ``RealtimeMotionBuffer.get_latest_frame`` 一致。"""
        with self._lock:
            if self._joint_pos is None or self._body_quat_w is None:
                return None
            return self._frame_cls(
                joint_pos=self._joint_pos,
                body_pos_w=self._body_pos_w,
                body_quat_w=self._body_quat_w,
                timestamp_ns=self._ts,
                prev_body_pos_w=self._prev_body_pos_w,
                prev_timestamp_ns=self._prev_ts,
            )

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close(0)
        except Exception:
            pass


class MotionToggleTrigger:
    """Record start/stop from the pico ``toggle_data_collection`` level on :28701.

    ``pico_retarget_pub`` flips a **persistent level** on the PICO A button and ships
    it (``[bool]``) in every motion payload. We **follow the level** rather than
    edge-detect one frame: a dropped payload then can't desync the recorder, and the
    level always states the operator's actual intent.

    Reads it off the already-subscribed :class:`LatestMotionClient` — no extra socket
    or thread. Returns ``None`` until the first payload ("no toggle data yet").
    """

    def __init__(self, motion_client):
        self._client = motion_client
        self._level: Optional[bool] = None

    def poll(self) -> Optional[bool]:
        level = self._client.latest_toggle_data_collection
        if level is not None:
            self._level = level
        return self._level


# ---------------------------------------------------------------------------
def _load_policy_motion_config(policy_config: str) -> dict:
    import yaml

    with open(policy_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    motion = dict(cfg.get("motion", {}))
    return {
        "joint_names": list(cfg.get("joint_names_simulation", JAKA_JOINT_NAMES)),
        "body_names": list(cfg.get("body_names_simulation", JAKA_BODY_NAMES)),
        "future_steps": np.asarray(motion.get("future_steps", [0]), dtype=int),
        "motion_zmq_connect": motion.get("motion_zmq_connect", "tcp://127.0.0.1:28701"),
        "anchor_body_name": motion.get("anchor_body_name", "waist_yaw_Link"),
    }


def _resolve_run_dir(save_dir: str, dr_level: int, run_name: str = "") -> str:
    """Pick this run's dataset dir: ``<save-dir>/<run-id>/level-<dr-level>``.

    The run id is ``run_name`` when given, else a timestamp. An existing dataset is
    never reused (lerobot's ``create`` would refuse the existing root anyway) — a
    colliding run id gets a ``-2``/``-3`` suffix, so old data is always left alone.
    """
    run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
    root = os.path.abspath(save_dir)
    run_save_dir = f"{root}/{run_id}/level-{dr_level}"
    suffix = 2
    while os.path.exists(run_save_dir):
        run_save_dir = f"{root}/{run_id}-{suffix}/level-{dr_level}"
        suffix += 1
    return run_save_dir


def main(
    # dataset
    save_dir: Annotated[str, typer.Option()] = "data/teleop_jaka_mf",
    run_name: Annotated[str, typer.Option(help="Leaf dir under --save-dir; default: timestamp")] = "",
    desc: Annotated[str, typer.Option()] = "close the trash can",
    num_episodes: Annotated[int, typer.Option()] = 100,
    frequency: Annotated[int, typer.Option()] = 30,
    dr_level: Annotated[int, typer.Option()] = 0,
    # ZMQ connects
    state_zmq_connect: Annotated[str, typer.Option()] = "tcp://127.0.0.1:28711",
    camera_zmq_connect: Annotated[str, typer.Option()] = "tcp://127.0.0.1:28712",
    motion_zmq_connect: Annotated[str, typer.Option()] = "tcp://127.0.0.1:28701",
    # policy (mirrors the control-side motion buffer so the action label matches)
    policy_config: Annotated[str, typer.Option()] = _DEFAULT_POLICY_CONFIG,
    # trigger: keyboard (Enter) | pico (A-button toggle_data_collection on the motion
    # stream) | both. 默认 `both`:pico 的录制开关**一直生效**(和 teleop 的
    # reset 判断保持一致),同时键盘回车也能手动控制。With `both`, a pico level
    # change wins over a keyboard press.
    trigger: Annotated[str, typer.Option()] = "both",
):
    assert trigger in ("keyboard", "pico", "both"), f"Invalid trigger {trigger}"
    use_keyboard = trigger in ("keyboard", "both")
    use_pico = trigger in ("pico", "both")

    motion_cfg = _load_policy_motion_config(policy_config)

    print(f"[record] Connecting state→{state_zmq_connect} camera→{camera_zmq_connect} "
          f"motion→{motion_zmq_connect}")
    state_client = LatestStateClient(state_zmq_connect)
    image_client = LatestImageClient(camera_zmq_connect)

    # 直连 pico motion 流(对齐 OpenHLM): 不用 RealtimeMotionBuffer, 只保留最新帧。
    motion_client = LatestMotionClient(
        connect=motion_zmq_connect,
        joint_names=motion_cfg["joint_names"],
        body_names=motion_cfg["body_names"],
    )

    # Owns no socket (it reads the already-subscribed motion client), so it exists
    # unconditionally and is only *consulted* when ``use_pico``.
    pico_toggle = MotionToggleTrigger(motion_client)

    run_save_dir = _resolve_run_dir(save_dir, dr_level, run_name)

    # The LeRobot dataset is created LAZILY on the first received camera frame, using that
    # frame's actual HWC shape as the declared `head_image_left` shape. The camera resolution
    # lives in the sim's camera cfg (tasks/jaka_vla_cameras.py), not in this process, so
    # hard-coding it here would break every time the camera cfg changes.
    exporter = None
    head_shape: tuple[int, ...] | None = None
    print(f"[record] New dataset will be created at {run_save_dir} on the first camera "
          f"frame (--run-name picks the subdir; existing datasets are never touched)")

    state_root_calc = OpenHLMRootVel()   # 3 维 (roll,pitch,yaw_vel) — state 30 / action 也用这个
    action_root_calc = OpenHLMRootVel()  # 3 维; 绝对 yaw 不再单独存(可从原始 quat 恢复)

    # Keyboard trigger: a daemon thread reads stdin (Enter=record toggle, q=quit).
    rec_flags = {"toggle": False, "quit": False, "pico_level": None}

    def _stdin_listener():
        try:
            for line in sys.stdin:
                k = line.strip().lower()
                if k in ("q", "quit", "exit"):
                    rec_flags["quit"] = True
                    return
                if k == "" or k in ("s", "start", "r", "record"):
                    rec_flags["toggle"] = True
                if k in ("d", "discard"):
                    rec_flags["toggle"] = True
        except Exception:
            pass

    stdin_thread = None
    if use_keyboard:
        stdin_thread = threading.Thread(target=_stdin_listener, daemon=True)
        stdin_thread.start()
        print("[record] Keyboard: ENTER=record toggle, 'q'=quit")
    if use_pico:
        print("[record] Pico: A button (toggle_data_collection on the motion stream) starts/stops")

    recording = False
    frames_in_episode = 0
    episodes_saved = 0
    control_dt = 1.0 / frequency
    step = 0

    print("[record] Waiting for data. Press ENTER "
          + ("/A on pico " if use_pico else "")
          + "to start recording.\n")
    try:
        while not rec_flags["quit"]:
            start = time.time()

            # Resolve the desired record state from whichever triggers are armed.
            # pico (a persistent *level*) is applied last so it wins on conflict.
            desired: Optional[bool] = None
            if use_keyboard and rec_flags["toggle"]:
                rec_flags["toggle"] = False
                desired = not recording
            if use_pico:
                level = pico_toggle.poll()
                if level is not None:
                    if rec_flags["pico_level"] is None:
                        # First payload: adopt the level as the baseline. Follow it, so a
                        # toggle flipped before the recorder started is honoured.
                        rec_flags["pico_level"] = level
                        desired = level
                        print(f"[record] pico toggle_data_collection = {level}")
                    elif level != rec_flags["pico_level"]:
                        rec_flags["pico_level"] = level
                        desired = level
                        print(f"[record] pico toggle_data_collection -> {level}")

            if desired is not None and desired != recording:
                recording = desired
                if recording:
                    state_root_calc.reset()
                    action_root_calc.reset()
                    frames_in_episode = 0
                    print(f"[record] EPISODE {episodes_saved}: RECORDING started")
                else:
                    if exporter is not None and frames_in_episode > 0:
                        exporter.save_episode()
                        # parquet already holds the image bytes; turn images/ into a compact mp4.
                        # Never fatal: the episode is already on disk (parquet is the source of truth).
                        try:
                            video_path = finalize_openhlm_episode(exporter)
                        except Exception as exc:
                            video_path = None
                            print(f"[record] WARN: mp4 encode failed ({exc}); episode parquet is still valid")
                        episodes_saved += 1
                        print(f"[record] Episode {episodes_saved} saved"
                              + (f" (video {video_path})" if video_path else ""))
                        if episodes_saved >= num_episodes:
                            break
                    else:
                        print("[record] Stop toggled with no frames — episode skipped")

            if recording:
                joint_pos, base_quat_wxyz, _ts = state_client.latest()
                image = image_client.latest()
                if image is None or joint_pos is None:
                    time.sleep(control_dt)
                    continue

                # Create the dataset on the first frame, declaring the stream's real shape.
                shape = tuple(int(v) for v in np.asarray(image).shape[:3])
                if exporter is None:
                    head_shape = shape
                    exporter = init_openhlm_exporter(
                        run_save_dir, fps=frequency, head_image_shape=head_shape
                    )
                    print(f"[record] Dataset created at {run_save_dir} "
                          f"(head_image_left {head_shape}, state {OPENHLM_STATE_DIM}, "
                          f"actions {OPENHLM_ACTION_DIM}, fps {frequency})")
                elif shape != head_shape:
                    print(f"[record] head image shape changed {head_shape} -> {shape}; "
                          f"skipping frame (restart the recorder if the camera cfg changed)")
                    time.sleep(control_dt)
                    continue

                # ONE timestamp for both state and action root, so roll/pitch/yaw_vel
                # for the robot state and the reference share the same clock basis.
                t_ns = int(time.time() * 1e9)
                state = np.concatenate([
                    np.asarray(joint_pos, dtype=np.float32),
                    state_root_calc(base_quat_wxyz, t_ns),
                ])
                
                ref_joints, ref_anchor_quat,ref_anchor_lin_vel,ref_anchor_pos_w \
                    = reference_action_live_latest(motion_client, anchor_name=motion_cfg["anchor_body_name"]
                )
                # action(40) = [27 dof | roll,pitch,yaw_vel | lin_vel xyz |
                #               anchor_pos_w xyz | anchor_quat_w wxyz]
                # 末尾 7 维是 pico 原始的世界系 anchor 位姿(DEBUG,便于直接对比 motion_mf);
                # 绝对 yaw 不再单独存, 可从 quat 恢复。
                actions = np.concatenate([
                    ref_joints,
                    action_root_calc(ref_anchor_quat, t_ns),
                    ref_anchor_lin_vel,
                    ref_anchor_pos_w,
                    ref_anchor_quat,
                ])
                exporter.add_frame(
                    build_openhlm_frame(image, state, actions), task=desc
                )
                frames_in_episode += 1
                step += 1
                if step % (frequency * 5) == 0:
                    print(f"[record] step {step} saved (state{state.dtype} action{actions.shape})")

            elapsed = time.time() - start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[record] Interrupted by user.")
    finally:
        # Finalize an in-progress episode on Ctrl+C / quit (exporter exists iff a frame arrived).
        if exporter is not None and recording and frames_in_episode > 0:
            try:
                exporter.save_episode()
                episodes_saved += 1
            except Exception:
                pass
            try:
                finalize_openhlm_episode(exporter)
            except Exception:
                pass
        print(f"[record] Done. {episodes_saved} episodes saved")
        # LeRobotDataset needs no explicit close; save_episode() finalizes on disk.
        state_client.close()
        image_client.close()
        motion_client.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
