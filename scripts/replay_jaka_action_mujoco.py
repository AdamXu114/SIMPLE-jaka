#!/usr/bin/env python3
"""Replay a recorded Jaka OpenHLM action trunk in MuJoCo — hard-set qpos, no policy.

Reads the recorded ``actions`` (frame x 33 = [27 dof, roll, pitch, yaw_vel, lin_vel xyz])
straight from the LeRobot parquet, reconstructs the full robot ``qpos`` trajectory
(base free-joint + 27 joints) from the action via ``reconstruct_base_qpos``, then
teleports the robot to each frame and plays it in a MuJoCo passive viewer — the same
sim2real-style window ``teleop_jaka_mf`` uses (tracking camera on ``base_link``, side UIs
hidden) — so you can eyeball whether the positions / orientations reconstructed from the
action are correct. Nothing is run through the frozen MF policy.

No ``--headless`` (default): opens the interactive MuJoCo viewer, loops the episode at 50 Hz.
``--headless``: instead render frames to PNG / a GIF (no window).

Usage::

    python scripts/replay_jaka_action_mujoco.py \
        --data-dir data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0 \
        --episode 000000
    # headless -> save every 5th frame as PNG + a GIF:
    python scripts/replay_jaka_action_mujoco.py ... --headless --step 5 --gif /tmp/jaka.gif
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _tracking_cam(model, viewer, body: str) -> None:
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(bid)


def _run_interactive(model, data, qpos, fps: float) -> None:
    """Loop the reconstructed motion in a sim2real-style passive MuJoCo viewer."""
    import mujoco
    from mujoco import viewer

    v = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    _tracking_cam(model, v, "base_link")
    period = 1.0 / float(fps)
    has_run = callable(getattr(v, "is_running", None))
    loops = 0
    try:
        while True:
            loops += 1
            T = qpos.shape[0]
            for t in range(T):
                t0 = time.monotonic()
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                v.sync()
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
                if has_run and not v.is_running():
                    break
            if has_run and not v.is_running():
                break
            if not has_run and loops >= 3:  # fallback: play the episode 3x then stop
                break
    except KeyboardInterrupt:
        pass
    finally:
        v.close()
    print("viewer closed")


def _run_headless(model, data, qpos, args) -> None:
    """Render frames to PNG (and an optional GIF) via mujoco.Renderer."""
    import mujoco

    renderer = mujoco.Renderer(model, args.height, args.width)  # signature: (model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1  # base_link (root body)
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    os.makedirs(args.out_dir, exist_ok=True)
    frames_for_gif = []
    idx = list(range(0, args.n_frames, args.step))
    if args.max_frames and args.max_frames > 0:
        idx = idx[: args.max_frames]

    from PIL import Image

    for t in idx:
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()  # (H, W, 3) uint8 RGB
        if args.gif:
            frames_for_gif.append(img)
        Image.fromarray(img).save(os.path.join(args.out_dir, f"frame_{t:06d}.png"))

    print(f"wrote {len(idx)} PNG frames to {args.out_dir}")
    if args.gif:
        import imageio

        imageio.mimsave(args.gif, frames_for_gif, fps=args.fps, loop=0)
        print(f"wrote GIF to {args.gif}")


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default="data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0",
        help="LeRobot dataset root (contains data/chunk-000/*.parquet)",
    )
    p.add_argument("--episode", default="000000", help="episode id, e.g. 000000")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--headless", action="store_true", help="render PNG/GIF instead of opening a viewer")
    p.add_argument("--base-z", type=float, default=0.65, help="start the base at this height (m)")
    # headless-only options
    p.add_argument("--step", type=int, default=1, help="headless: save every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="headless: 0 = all frames")
    p.add_argument("--out-dir", default="/tmp/jaka_replay_frames", help="headless PNG output dir")
    p.add_argument("--gif", default="", help="headless: also write an animated GIF here (optional)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--distance", type=float, default=2.8, help="tracking camera distance")
    p.add_argument("--azimuth", type=float, default=130.0, help="tracking camera azimuth (deg)")
    p.add_argument("--elevation", type=float, default=-18.0, help="tracking camera elevation (deg)")
    args = p.parse_args()

    import mujoco
    import pyarrow.parquet as pq
    from simple.jaka_rl.action_trunk_reconstruct import reconstruct_base_qpos

    # 1. Load the recorded action trunk [T, 33].
    files = sorted(glob.glob(os.path.join(args.data_dir, "data", "chunk-*", f"episode_{args.episode}.parquet")))
    if not files:
        raise SystemExit(f"no parquet for episode {args.episode} under {args.data_dir}")
    actions = np.array(pq.read_table(files[0], columns=["actions"]).column("actions").to_pylist(), np.float32)
    args.n_frames = actions.shape[0]
    print(f"episode {args.episode}: {args.n_frames} frames, actions {actions.shape}")

    # 2. Reconstruct the full qpos trajectory from the action (base + 27 joints).
    qpos, dof, model = reconstruct_base_qpos(
        actions, fps=args.fps, start_pos=np.array([0.0, 0.0, args.base_z])
    )
    data = mujoco.MjData(model)
    print(f"reconstructed qpos {qpos.shape} (model nq={model.nq}), finite={bool(np.all(np.isfinite(qpos)))}")

    # 3. Play it.
    if args.headless:
        _run_headless(model, data, qpos, args)
    else:
        _run_interactive(model, data, qpos, args.fps)


if __name__ == "__main__":
    _main()
