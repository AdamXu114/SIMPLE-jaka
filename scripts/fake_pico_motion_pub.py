#!/usr/bin/env python3
"""Simulate ``pico_retarget_hub``: replay a recorded NPZ motion over ZMQ :28701.

Publishes the payload format of sim2real's ``pico_retarget_pub.py`` so SIMPLE's
whole-body MF policy can be tested on its real **live zmq** path
(``motion_backend=zmq`` -> ``RealtimeMotionBuffer``) without the Pico hub.
Joints/bodies are reordered from the NPZ (IsaacLab/policy order) to the canonical
MuJoCo (JAKA) order the hub publishes (``robot_cfg.joint_names`` / ``body_names``).

State machine — matches the real hub, and is ONE flag (``live``):

  idle   republishes ONE frozen pose every frame with ``paused: true``. At boot that
         pose is the default stand (FK of ``ZMQ_DEFAULT_QPOS``) — the same pose the
         real hub's ``paused_qpos`` holds and the same one ``RealtimeMotionBuffer``
         falls back to with no frames, so starting this script does NOT move the
         reference. (``--live-at N`` spends N frames here first.)
  live   advances one motion frame per publish with ``paused: false``.
  X      live->idle freezes at the pose just published (continuous, no jump);
         idle->live continues from that same frame.
  A      flips ``toggle_data_collection``, a persistent level sent on every payload:
         the record start/stop marker the recorder (``--trigger pico``) and the
         teleop's record-end reset both follow.

It also publishes the hub's **PICO button stream on :5592**
(``--controller-bind``, same ``<QBBBB>`` = ts + A + B + X + Y as
``pico_retarget_pub.PicoControllerStateMessage``), so SIMPLE's ``PicoController``
— which reads A/B for the init/zero/policy mode switch — can be exercised too.

Button map (identical to the hub; ``--keyboard`` presses them from the keyboard):

  A = RightController key_one -> toggle_data_collection (record marker)
  X = LeftController  key_one -> paused (freeze / resume)
  B = RightController key_two -> no hub-side effect, only broadcast on :5592
  Y = LeftController  key_two -> no hub-side effect, only broadcast on :5592

Usage (terminal A — the "hub"):
    python scripts/fake_pico_motion_pub.py                  # live right away
    python scripts/fake_pico_motion_pub.py --live-at 50     # idle, then live
    python scripts/fake_pico_motion_pub.py --pause-at 200 --pause-frames 100
    python scripts/fake_pico_motion_pub.py --keyboard       # a/x/b/y press buttons

Terminal B (SIMPLE, live zmq):
    python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 --no-headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simple.jaka_rl.config import (  # noqa: E402
    BODY_NAMES,
    JOINT_NAMES,
    NPZ_BODY_NAMES,
    NPZ_JOINT_NAMES,
    TOGGLE_DATA_COLLECTION_KEY,
    ZMQ_DEFAULT_QPOS,
)
from simple.jaka_rl.controllers.pico import PicoControllerStateMessage  # noqa: E402

ANCHOR = "waist_yaw_Link"
# (joint_pos[27] JAKA order, body_pos_w[28,3], body_quat_w[28,4] wxyz)
Pose = tuple[np.ndarray, np.ndarray, np.ndarray]

# The hub's PICO buttons and what each one does, mirroring pico_retarget_pub.py:
#   A = RightController key_one -> toggle_data_collection (record start/stop marker)
#   X = LeftController  key_one -> paused (freeze / resume the reference)
#   B = RightController key_two -> (no hub-side effect; only broadcast on :5592)
#   Y = LeftController  key_two -> (no hub-side effect; only broadcast on :5592)
# Keyboard stands in for the buttons; the hub-side effects are applied on the RISING
# EDGE, exactly like the hub's ``x_pressed and not self._x_button_was_pressed``.
BUTTON_KEYS = {"a": "A", "b": "B", "x": "X", "y": "Y"}


def anchor_yaw_deg(quat_wxyz: np.ndarray) -> float:
    """Yaw (deg) of a wxyz quaternion, for the edge log lines."""
    w, x, y, z = quat_wxyz
    return float(np.rad2deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def default_stand_pose() -> Pose | None:
    """FK the neutral stand the hub republishes while idle (``ZMQ_DEFAULT_QPOS``).

    Returns None (caller falls back to the motion's start frame) if the robot MJCF
    cannot be loaded.
    """
    try:
        import mujoco

        from simple.robots.jaka import Jaka
        from simple.utils import resolve_data_path

        m = mujoco.MjModel.from_xml_path(resolve_data_path(Jaka.mjcf_path))
        d = mujoco.MjData(m)
        d.qpos[: len(ZMQ_DEFAULT_QPOS)] = np.asarray(ZMQ_DEFAULT_QPOS, dtype=float)
        mujoco.mj_forward(m, d)
        jid = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
               for n in JOINT_NAMES]
        bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODY_NAMES]
        return (
            np.asarray(d.qpos[jid], dtype=np.float32),
            np.asarray(d.xpos[bid], dtype=np.float32),
            np.asarray(d.xquat[bid], dtype=np.float32),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot FK the default stand ({exc}); "
              "the idle pose will be the motion's start frame instead")
        return None


def load_motion(path: str) -> tuple[Pose, int]:
    """Load the NPZ and reorder IsaacLab/policy order -> JAKA order."""
    npz = np.load(path)
    joint_idx = [NPZ_JOINT_NAMES.index(n) for n in JOINT_NAMES]
    body_idx = [NPZ_BODY_NAMES.index(n) for n in BODY_NAMES]
    return (
        (
            npz["joint_pos"][:, joint_idx].astype(np.float32),      # [T,27]
            npz["body_pos_w"][:, body_idx].astype(np.float32),      # [T,28,3]
            npz["body_quat_w"][:, body_idx].astype(np.float32),     # [T,28,4]
        ),
        int(npz["fps"][0]),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--motion-npz", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "motion", "motion_mf.npz"))
    p.add_argument("--bind", default="tcp://*:28701")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--start-frame", type=int, default=0,
                   help="motion frame the live phase starts at")
    p.add_argument("--log-every", type=int, default=250)
    # ── 确定性脚本相位(不按键也能复现同一条时间线;帧号都是 1-based 的发布帧)──
    p.add_argument("--live-at", type=int, default=None,
                   help="先发 N 帧 idle(冻结站姿)再转 live;0=立刻 live;-1=永不自动转 live"
                        "(只发冻结站姿/等键盘 x,复现「hub 没数据」)。"
                        "默认:普通模式 1(先发 1 帧空闲快照再 live),--keyboard 模式 -1"
                        "(同真实 hub:开机 paused=True,按 x 才 live)。")
    p.add_argument("--pause-at", type=int, default=-1,
                   help="live 满 N 帧时暂停(同按 X);-1=不自动暂停")
    p.add_argument("--pause-frames", type=int, default=50,
                   help="--pause-at 触发后保持暂停多少帧,然后自动恢复 live")
    p.add_argument("--record-start", type=int, default=-1,
                   help="第 N 帧起 toggle_data_collection=True(同按 A);-1=不自动开")
    p.add_argument("--record-stop", type=int, default=-1,
                   help="第 N 帧起 toggle_data_collection=False(同再按 A);-1=一直保持")
    p.add_argument("--keyboard", action="store_true",
                   help="按行读 stdin 模拟 pico 手柄按键:a=A 键(录制开关)、x=X 键(暂停/恢复)、"
                        "b=B 键、y=Y 键(后两个真实 hub 只转发不自用,这里同样只播发)、q=退出。"
                        "开启后 --record-start/stop 失效(电平交给键盘)。")
    # ── 手柄按键流(:5592,与真实 hub 的 PicoControllerStateMessage 同格式)──────
    p.add_argument("--controller-bind", default="tcp://*:5592",
                   help="同时按真实 hub 的格式播发 A/B/X/Y 按键状态(SIMPLE 的 PicoController "
                        "订阅这里做模式切换);空字符串=不播发。")
    p.add_argument("--key-press-ms", type=float, default=200.0,
                   help="键盘模拟按键时把该键压住多久(ms)。真实手柄是人手按住 ~百 ms,"
                        "订阅端(~50Hz)才来得及看到那个上升沿;太短会被 CONFLATE 吞掉。")
    args = p.parse_args()
    # 让日志在重定向/管道下也即时可见(否则 `| tee` 时全部堆在缓冲区里)。
    sys.stdout.reconfigure(line_buffering=True)
    # idle 帧数;-1 = 永不自动转 live(只等键盘 x)。真实 hub 开机就是 paused=True。
    live_at = args.live_at if args.live_at is not None else (-1 if args.keyboard else 1)
    motion, npz_fps = load_motion(args.motion_npz)
    T = motion[0].shape[0]
    anchor_idx = list(BODY_NAMES).index(ANCHOR)
    start_i = args.start_frame % T

    # 冻结站姿 = 真实 hub 空闲时重发的 paused_qpos(= ZMQ_DEFAULT_QPOS 的 FK),也正是
    # RealtimeMotionBuffer 无帧时的回退姿态 —— 所以本脚本起来时参考不会跳。注意不能用
    # motion[start_frame] 代替:那段运动的手臂(0.69 rad)和站姿(1.57 rad)差 ~87°,
    # 会在 idle→live 那一刻给 policy 一个大的 ref_joint_pos 阶跃。首次 FK 约 3s。
    fk_stand = default_stand_pose()
    stand = fk_stand if fk_stand is not None else (
        motion[0][start_i], motion[1][start_i], motion[2][start_i])

    def scripted_toggle(n: int) -> bool:
        """Scripted A-key level: True over [record_start, record_stop). -1 disables."""
        if args.record_start <= 0:
            return False
        if args.record_stop <= 0:
            return n >= args.record_start
        return args.record_start <= n < args.record_stop

    print(f"motion: {T} frames (npz fps={npz_fps}), joints={motion[0].shape} "
          f"bodies={motion[1].shape} -> JAKA order")
    print(f"idle pose: {'default stand (FK of ZMQ_DEFAULT_QPOS)' if fk_stand is not None else 'motion start frame (FK unavailable)'}"
          f"  anchor z={stand[1][anchor_idx][2]:.3f}")
    print(f"live starts at motion frame {start_i}")
    print(f"[live] {live_at} idle frame(s) first, then live"
          if live_at > 0 else
          ("[live] live right away" if live_at == 0 else
           "[live] never auto-live: stays idle until 'x' (keyboard)"))
    if args.pause_at >= 0:
        print(f"[pause] after {args.pause_at} live frames, pause for "
              f"{args.pause_frames} frames")
    if not args.keyboard and args.record_start > 0:
        print(f"[toggle] frame {args.record_start} -> True"
              + (f", frame {args.record_stop} -> False" if args.record_stop > 0
                 else " (stays True; --record-stop not set)"))
    print(f"publishing to {args.bind} @ {args.fps} Hz (Ctrl+C to stop)")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)

    ctrl_sock = None
    if args.controller_bind:
        ctrl_sock = ctx.socket(zmq.PUB)
        ctrl_sock.setsockopt(zmq.LINGER, 0)
        ctrl_sock.setsockopt(zmq.SNDHWM, 1)
        ctrl_sock.setsockopt(zmq.CONFLATE, 1)
        ctrl_sock.bind(args.controller_bind)
        print(f"pico controller state -> {args.controller_bind}  (same <QBBBB> as the hub)")

    # ── 键盘(可选):线程只把"按下"记成截止时间,主循环做边沿检测 ────────────
    keys = {"quit": False}
    press_until = dict.fromkeys("ABXY", 0.0)   # monotonic deadline per button

    def _stdin_listener() -> None:
        for line in sys.stdin:
            for ch in line.strip().lower():
                if ch in BUTTON_KEYS:
                    press_until[BUTTON_KEYS[ch]] = time.monotonic() + args.key_press_ms / 1e3
                elif ch == "q":
                    keys["quit"] = True
            if keys["quit"]:
                return

    if args.keyboard:
        threading.Thread(target=_stdin_listener, daemon=True).start()
        print("[keys] a = A 键 (录制开关)      x = X 键 (暂停/恢复)")
        print("[keys] b = B 键 / y = Y 键 (真实 hub 只转发不自用)      q = 退出")

    # ── 状态 ──────────────────────────────────────────────────────────────
    live = live_at == 0
    auto_live_pending = live_at > 0   # one-shot: only the INITIAL idle -> live edge
    pause_pending = args.pause_at >= 0  # one-shot: schedule exactly one pause
    next_idx = start_i              # motion frame the next live publish will use
    frozen = stand                  # pose republished while idle
    frozen_idx = -1                 # its motion frame; -1 = the default stand
    live_frames = 0                 # live publishes so far (pause time not counted)
    published = 0
    toggle = False
    resume_at: int | None = None    # scripted pause: published-frame to resume at
    buttons = dict.fromkeys("ABXY", False)       # current (emulated) press state
    last_log = time.monotonic()
    frame_period = 1.0 / args.fps

    def frozen_label() -> str:
        return "default stand" if frozen_idx < 0 else f"motion frame {frozen_idx}"

    def freeze_on_last() -> None:
        """Enter idle on the pose last PUBLISHED, so the pause edge is continuous.

        That is the stand until the first live frame goes out (pressing x twice
        before any live frame must not jump to ``motion[start_frame - 1]``).
        """
        nonlocal frozen, frozen_idx
        if live_frames == 0:
            frozen, frozen_idx = stand, -1
        else:
            frozen_idx = (next_idx - 1) % T
            frozen = (motion[0][frozen_idx], motion[1][frozen_idx], motion[2][frozen_idx])

    try:
        while not keys["quit"]:
            n = published + 1

            # ── 手柄按键流(:5592):复刻 hub 每轮 _poll_pause_toggle() 的播发 ──
            now_mono = time.monotonic()
            prev_buttons, buttons = buttons, {b: now_mono < press_until[b] for b in "ABXY"}
            if ctrl_sock is not None:
                ctrl_sock.send(
                    PicoControllerStateMessage(
                        timestamp_ns=time.time_ns(), **buttons
                    ).to_bytes(),
                    flags=zmq.NOBLOCK,
                )

            # ── A 键:toggle_data_collection 持久电平(上升沿,同 hub)────────
            if args.keyboard:
                if buttons["A"] and not prev_buttons["A"]:
                    toggle = not toggle
                    print(f"[keys] frame {n}: A -> toggle_data_collection={toggle} "
                          f"({'开始录制' if toggle else '结束录制'})", flush=True)
            else:
                new = scripted_toggle(n)
                if new != toggle:
                    print(f"[toggle] frame {n}: toggle_data_collection -> {new} "
                          f"({'开始录制' if new else '结束录制'})")
                    toggle = new

            # ── X 键:暂停 / 恢复(上升沿,同 hub)───────────────────────────
            if buttons["X"] and not prev_buttons["X"]:
                live = not live
                resume_at = None
                if live:
                    auto_live_pending = False       # keyboard took over the live state
                else:
                    freeze_on_last()
                print(f"[keys] frame {n}: X -> {'恢复 live' if live else '暂停'} "
                      f"(frozen on {frozen_label()}, anchor yaw "
                      f"{anchor_yaw_deg(frozen[2][anchor_idx]):+.1f} deg)", flush=True)
            elif live and pause_pending and live_frames >= args.pause_at:
                live, pause_pending = False, False
                freeze_on_last()
                resume_at = n + args.pause_frames
                print(f"[live] frame {n}: live -> paused after {live_frames} live "
                      f"frames (frozen on {frozen_label()}, resume at frame {resume_at})")
            elif not live and resume_at is not None and n >= resume_at:
                live, resume_at = True, None
                print(f"[live] frame {n}: paused -> live "
                      f"(froze on {frozen_label()}, resume at motion frame {next_idx})")
            elif not live and auto_live_pending and published >= live_at:
                live, auto_live_pending = True, False
                print(f"[live] frame {n}: idle -> live (start at motion frame {next_idx})")

            # ── 组帧 ──────────────────────────────────────────────────────
            if live:
                joint, pos, quat = motion[0][next_idx], motion[1][next_idx], motion[2][next_idx]
                next_idx = (next_idx + 1) % T
                live_frames += 1
            else:
                joint, pos, quat = frozen

            now = time.time_ns()
            sock.send_string(json.dumps({
                "publish_t_ns": now,
                "smplx_t_ns": now,
                "paused": not live,
                TOGGLE_DATA_COLLECTION_KEY: [bool(toggle)],
                "joint_pos": joint.tolist(),
                "body_pos_w": pos.tolist(),
                "body_quat_w": quat.tolist(),
            }), flags=zmq.NOBLOCK)

            published += 1
            if published % args.log_every == 0:
                delta = time.monotonic() - last_log
                print(f"published {published} frames ({'live' if live else 'paused'}, "
                      f"live_frames={live_frames}), {args.log_every / delta:.1f} Hz")
                last_log = time.monotonic()

            time.sleep(frame_period)
    except KeyboardInterrupt:
        print(f"\nstopped after {published} frames")
    finally:
        sock.close(0)


if __name__ == "__main__":
    main()
