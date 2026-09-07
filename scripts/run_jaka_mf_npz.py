#!/usr/bin/env python3
"""Offline test: drive the Jaka whole-body MF policy from a recorded NPZ motion.

Substitutes the pico motion stream with a recorded ``.npz`` (from sim2real-jaka)
so the migrated policy stack can be validated end-to-end in SIMPLE without the
Pico hub. The policy tracks the recorded motion closed-loop and the robot stays
balanced under real physics.

Usage:
    python scripts/run_jaka_mf_npz.py \
        --policy-config data/jaka_mf/latest56k_pico_dr.yaml \
        --motion-npz data/motion/motion_mf.npz \
        --steps 600 --no-headless
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _default_policy_config() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data", "jaka_mf", "latest56k_pico_dr.yaml")


def _default_motion_npz() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data", "motion", "motion_mf.npz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="simple/JakaOpenTrashCanTeleop-v0")
    p.add_argument("--target", default="graspnet1b:0")
    p.add_argument("--policy-config", default=_default_policy_config())
    p.add_argument("--policy-model", default=None, help="ONNX; default = yaml with .onnx")
    p.add_argument("--motion-npz", default=_default_motion_npz())
    p.add_argument("--motion-backend", choices=["raw_npz", "zmq"], default="raw_npz",
                   help="raw_npz=replay npz in-process; zmq=receive from fake_pico_motion_pub on :28701")
    p.add_argument("--motion-zmq-connect", default="tcp://127.0.0.1:28701")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--headless", action="store_true", default=True, help="no MuJoCo viewer")
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--band", action="store_true", help="enable elastic band (hang robot)")
    p.add_argument("--controller", default="keyboard")
    p.add_argument("--inference-backend", default="onnx-cpu")
    p.add_argument("--rl-rate", type=float, default=50.0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--save-dir", default="data/jaka_mf_npz")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import gymnasium as gym
    import simple.envs as _  # noqa: F401  (trigger env registration)

    # Load sim2real config; for raw_npz override the motion source, for zmq keep
    # the yaml as-is (motion_path unused) and just point at the fake hub.
    import yaml

    with open(args.policy_config) as f:
        cfg = yaml.safe_load(f)
    if args.motion_backend == "raw_npz":
        cfg["motion"]["motion_backend"] = "raw_npz"
        cfg["motion"]["motion_path"] = os.path.abspath(args.motion_npz)
    else:
        cfg["motion"]["motion_backend"] = "zmq"
        cfg["motion"]["motion_zmq_connect"] = args.motion_zmq_connect
    tmp_yaml = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    policy_model = args.policy_model or args.policy_config.replace(".yaml", ".onnx")

    print(f"env={args.env_id} target={args.target}")
    print(f"policy_config={args.policy_config}")
    print(f"motion_backend={args.motion_backend} connect={args.motion_zmq_connect}")
    if args.motion_backend == "raw_npz":
        print(f"motion_npz={os.path.abspath(args.motion_npz)}")

    env = gym.make(
        args.env_id,
        sim_mode="mujoco",
        render_hz=args.rl_rate,
        physics_dt=0.002,
        headless=args.headless,
        target=args.target,
        dr_level=0,
        success_criteria=0.9,
    )
    obs, info = env.reset()
    jaka_env = env.unwrapped
    task = jaka_env.task
    robot = task.robot

    from simple.agents.jaka_mf_agent import JakaMFAgent

    num_substeps = max(1, int(round(1.0 / (robot.sim_dt * args.rl_rate))))
    agent = JakaMFAgent(
        robot,
        policy_config=tmp_yaml,
        policy_model=policy_model,
        controller=args.controller,
        inference_backend=args.inference_backend,
        rl_rate=args.rl_rate,
        motion_backend=args.motion_backend,
        motion_zmq_connect=args.motion_zmq_connect,
        num_substeps=num_substeps,
    )
    if args.band:
        agent._enable_band()

    # Engage policy from the start (offline validation).
    agent.policy.set_policy_mode(source="npz-test")

    obs_obs = agent.policy.observations["obs"].funcs["jaka_obs"]
    reindex = np.asarray(obs_obs.mujoco_to_isaaclab_reindex, dtype=int)

    if args.record:
        import shutil

        from simple.cli.teleop_jaka_mf import _init_exporter

        root = f"{os.path.abspath(args.save_dir)}/{env.spec.id}/level-0"
        shutil.rmtree(root, ignore_errors=True)
        exporter = _init_exporter(
            root, task, robot, list(jaka_env.mujoco.mj_objects.keys()), jaka_env.mujoco
        )

    base_z = []
    track_err = []
    finite = []
    for i in range(args.steps):
        action = agent.get_action(
            obs, instruction=task.instruction, privileged_info=info
        )
        obs, reward, terminated, truncated, info = env.step(action)

        base_z.append(float(robot.mjData.qpos[2]))
        finite.append(bool(np.all(np.isfinite(robot.mjData.qpos[:7]))))

        # Tracking error: robot joints vs reference joint at the current frame.
        sp = agent.policy.state_processor
        if sp.motion_data is not None:
            ref = sp.motion_data.joint_pos[0, 0]  # [27] isaaclab order (step 0)
            robot_iso = np.asarray(list(robot.get_robot_qpos().values()))[reindex]
            track_err.append(float(np.mean(np.abs(robot_iso - ref))))

        if args.record:
            images = jaka_env.mujoco.render(
                camera_names=["head_stereo_left", "front_stereo_left"]
            )
            from simple.cli.teleop_jaka_mf import _build_frame

            frame = _build_frame(
                images, jaka_env.mujoco, agent._bridge,
                agent.policy.action_manager.cmd_q,
            )
            exporter.add_frame(frame)

        if i % max(1, args.steps // 10) == 0:
            print(
                f"step {i:5d}  base_z={base_z[-1]:.3f}  "
                f"track_err={track_err[-1]:.3f} rad  finite={finite[-1]}"
            )

    if args.record:
        exporter.save_episode()
        print(f"[record] saved to {root}")

    bz = np.array(base_z)
    te = np.array(track_err)
    print("\n=== summary ===")
    print(f"steps={args.steps}")
    print(f"base_z: min={bz.min():.3f} max={bz.max():.3f} final={bz[-1]:.3f}")
    print(f"tracking error: mean={te.mean():.3f} rad max={te.max():.3f} rad")
    print(f"all finite: {all(finite)}")
    survived = bz.min() > 0.2
    print(f"SURVIVED (min base_z > 0.2): {survived}")
    print("cleanup")
    env.close()
    agent.close()


if __name__ == "__main__":
    main()
