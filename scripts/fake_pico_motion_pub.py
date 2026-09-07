#!/usr/bin/env python3
"""Simulate ``pico_retarget_hub``: replay a recorded NPZ motion over ZMQ :28701.

Publishes the exact payload format used by sim2real's ``pico_retarget_pub.py``
so the SIMPLE whole-body MF policy can be tested on its real **live zmq** path
(``motion_backend=zmq`` -> ``RealtimeMotionBuffer``) without the Pico hub.

Joints/bodies are reordered from the NPZ (IsaacLab / policy) order to the
canonical MuJoCo (JAKA) order that the hub publishes, matching what
``pico_retarget_pub`` sends (``robot_cfg.joint_names`` / ``body_names``).

Usage (terminal A — the "hub"):
    python scripts/fake_pico_motion_pub.py

Then terminal B (SIMPLE, live zmq):
    python -m simple.cli.teleop_jaka_mf simple/JakaOpenTrashCanTeleop-v0 ...
    # or the offline test but with motion_backend=zmq
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simple.jaka_rl.config import (  # noqa: E402
    BODY_NAMES,
    JOINT_NAMES,
    NPZ_BODY_NAMES,
    NPZ_JOINT_NAMES,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion-npz",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "motion", "motion_mf.npz"),
    )
    p.add_argument("--bind", default="tcp://*:28701")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    npz = np.load(args.motion_npz)
    joint_pos = npz["joint_pos"].astype(np.float32)      # [T, 27] IsaacLab order
    body_pos_w = npz["body_pos_w"].astype(np.float32)    # [T, 28, 3]
    body_quat_w = npz["body_quat_w"].astype(np.float32)  # [T, 28, 4]

    # Reorder from NPZ/IsaacLab order -> canonical MuJoCo (JAKA) order.
    joint_idx = np.array([NPZ_JOINT_NAMES.index(name) for name in JOINT_NAMES], dtype=int)
    body_idx = np.array([NPZ_BODY_NAMES.index(name) for name in BODY_NAMES], dtype=int)
    joint_jaka = joint_pos[:, joint_idx]                   # [T, 27] JAKA order
    body_pos_jaka = body_pos_w[:, body_idx]                # [T, 28, 3]
    body_quat_jaka = body_quat_w[:, body_idx]              # [T, 28, 4]

    T = joint_jaka.shape[0]
    print(f"motion: {T} frames, fps={npz['fps'][0]}, joint_pos{joint_jaka.shape}, "
          f"body_pos{body_pos_jaka.shape}, body_quat{body_quat_jaka.shape}")
    print(f"joint order -> JAKA (reindex size {joint_idx.size}), "
          f"body order -> JAKA (reindex size {body_idx.size})")
    print(f"publishing to {args.bind} @ {args.fps} Hz (Ctrl+C to stop)")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)

    frame_period = 1.0 / args.fps
    i = args.start_frame
    published = 0
    last_log = time.monotonic()
    try:
        while True:
            now = time.time_ns()
            payload = {
                "publish_t_ns": now,
                "smplx_t_ns": now,
                "paused": False,
                "joint_pos": joint_jaka[i % T].tolist(),
                "body_pos_w": body_pos_jaka[i % T].tolist(),
                "body_quat_w": body_quat_jaka[i % T].tolist(),
            }
            sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)

            published += 1
            i += 1
            if published % args.log_every == 0:
                delta = time.monotonic() - last_log
                print(f"published {published} frames (frame {i % T}), "
                      f"{published / delta:.1f} Hz")
                last_log = time.monotonic()

            time.sleep(frame_period)
    except KeyboardInterrupt:
        print(f"\nstopped after {published} frames")
    finally:
        sock.close(0)


if __name__ == "__main__":
    main()
