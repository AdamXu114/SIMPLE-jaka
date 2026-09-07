"""Teleoperation using the Jaka whole-body MF RL policy (in-process).

Single terminal: SIMPLE runs the MuJoCo scene AND the Jaka MF strategy (from
sim2real-jaka) in one process. The only sim2real process still running is
``pico_retarget_hub`` (publishes motion :28701 + handle :5592).

Usage (mirror teleop_decoupled_wbc)::

    export TASK_NAME=JakaOpenTrashCanTeleop-v0
    python -m simple.cli.teleop_jaka_mf \\
        simple/$TASK_NAME --target=graspnet1b:0 --sim-mode=mujoco \\
        --record --no-headless --success-criteria=0.9 \\
        --policy-config data/jaka_mf/latest56k_pico_dr.yaml
"""

from __future__ import annotations

import enum
import os
import time

import gymnasium as gym
import numpy as np
import typer
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401  (trigger env registration)

from simple.agents.jaka_mf_agent import JakaMFAgent
from simple.jaka_rl.config import DEFAULT_QPOS

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"


class RecordingState(enum.Enum):
    WAITING_FOR_LANDING = "waiting_for_landing"
    RECORDING = "recording"
    EPISODE_DONE = "episode_done"


def _save_episode_env_config(exporter, task, episode_index: int):
    from simple.utils import NumpyArrayEncoder
    import json

    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    lines[episode_index]["environment_config"] = json.dumps(
        env_conf, cls=NumpyArrayEncoder
    )
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _init_exporter(save_dir, task, robot, obj_names, mujoco_sim, data_format: str = "openhlm"):
    """Create the LeRobot exporter for the chosen ``data_format``.

    ``data_format``:
      - ``"openhlm"`` (default): final train-consumed format — one head image
        (per-frame PNG), 30-dim float32 state + 33-dim actions, per-episode task. No
        scene/replay fields (see ``build_openhlm_*`` / ``init_openhlm_exporter``).
      - ``"jaka"`` (kept for replay/video): the Jaka-replay format with
        ``body_poses``/``scene_qpos`` + mp4 videos (see ``init_jaka_exporter``).
    """
    if data_format != "openhlm":
        from simple.datasets.jaka_lerobot import (
            JAKA_VIDEO_SHAPES,
            VIDEO_SHAPE,
            init_jaka_exporter,
        )

        scene_qpos_shape = mujoco_sim.mjModel.nq - (7 + robot.dof)
        return init_jaka_exporter(
            str(save_dir),
            task_prompt=task.instruction,
            obj_names=obj_names,
            video_shape=VIDEO_SHAPE,
            video_shapes=JAKA_VIDEO_SHAPES,
            scene_qpos_shape=scene_qpos_shape,
        )

    from simple.datasets.jaka_lerobot import init_openhlm_exporter

    return init_openhlm_exporter(str(save_dir), fps=50)


def _build_frame(images, mujoco_sim, bridge, action):
    """Assemble one LeRobot frame from rendered images + sim state.

    ``mujoco_sim`` is the env's ``MujocoSimulator`` (``task.mujoco`` does not
    expose the simulator; the env does).
    """
    from simple.datasets.jaka_lerobot import build_jaka_frame

    qpos = bridge.get_joint_positions()
    return build_jaka_frame(
        images,
        mujoco_sim.mjData,
        np.asarray(action, dtype=np.float32),
        mujoco_sim.mj_objects,
        qpos,
        mj_model=mujoco_sim.mjModel,
    )


def _reference_action_jaka(state_processor):
    """OpenHLM ``actions`` label: the pico/ZMQ reference motion's current-frame joint
    target (reordered to ``JAKA_JOINT_NAMES``), its base quaternion (wxyz), and its
    per-frame base LINEAR VELOCITY (xyz, m/s, in the current base frame).

    The linear velocity is computed at collection time from the reference motion window:
        vel = R(current_base_quat)^T (next_base_pos - current_base_pos) / dt
    where ``next/current`` are the reference base_link position one motion step apart and
    ``dt`` comes from the motion timestamps. This is the base velocity the frozen lower-level
    MF policy tracks — so the VLA is trained to output it from ``state``. Falls back to the
    neutral stand / identity quat / zero velocity when no reference motion is available.

    Returns:
        (ref_joints_jaka[27], ref_base_quat[4] wxyz, ref_base_lin_vel[3] m/s)
    """
    from simple.datasets.jaka_lerobot import JAKA_JOINT_NAMES
    from simple.jaka_rl.math import quat_rotate_inverse_numpy

    md = state_processor.motion_data
    if md is None:
        return (
            np.zeros(len(JAKA_JOINT_NAMES), dtype=np.float32),
            np.array([1, 0, 0, 0], np.float32),
            np.zeros(3, dtype=np.float32),
        )
    future_steps = np.atleast_1d(np.asarray(state_processor.motion_future_steps, dtype=int))
    idx_zero = 0
    try:
        idx_zero = int(list(future_steps).index(0))
    except ValueError:
        pass
    # Next reference motion step (first future step after the current one).
    idx_next = None
    for i, fs in enumerate(future_steps):
        if int(fs) > int(future_steps[idx_zero]):
            idx_next = i
            break

    ref = np.asarray(md.joint_pos[0, idx_zero], dtype=np.float32)  # motion-order [27]
    mj_names = list(state_processor.motion_joint_names)
    if list(mj_names) == list(JAKA_JOINT_NAMES):
        ref_joints = ref.astype(np.float32)
    else:
        ref_joints = ref[[mj_names.index(n) for n in JAKA_JOINT_NAMES]].astype(np.float32)

    body_names = list(state_processor.motion_body_names)
    base_idx = body_names.index("base_link") if "base_link" in body_names else 0
    ref_base_quat = np.asarray(md.body_quat_w[0, idx_zero, base_idx], dtype=np.float32)

    # Per-frame base linear velocity (m/s) in the current base (root) frame.
    ref_base_lin_vel = np.zeros(3, dtype=np.float32)
    if idx_next is not None:
        pos_cur = np.asarray(md.body_pos_w[0, idx_zero, base_idx], dtype=np.float32)
        pos_next = np.asarray(md.body_pos_w[0, idx_next, base_idx], dtype=np.float32)
        ts = getattr(md, "timestamps_ns", None)
        dt = (float(ts[0, idx_next]) - float(ts[0, idx_zero])) * 1e-9 if ts is not None and len(ts) > 0 else 0.0
        if dt > 1e-6:
            vel_body = quat_rotate_inverse_numpy(
                ref_base_quat[None, :], (pos_next - pos_cur)[None, :]
            )[0]
            ref_base_lin_vel = np.clip(vel_body / np.float32(dt), -2.0, 2.0).astype(np.float32)
    return ref_joints, ref_base_quat, ref_base_lin_vel


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/JakaOpenTrashCanTeleop-v0",
    target: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    save_dir: Annotated[str, typer.Option()] = "data/teleop_jaka_mf",
    num_episodes: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False,
    success_criteria: Annotated[float, typer.Option()] = 5,
    # data format written when recording:
    #   "openhlm" = final train-consumed LeRobot (head PNG 224x224 + float32 state 30/actions 33 + task)
    #   "jaka"    = Jaka-replay/mp4 format (body_poses/scene_qpos, mp4) — kept for replay/video
    data_format: Annotated[str, typer.Option()] = "openhlm",
    # policy
    policy_config: Annotated[str, typer.Option()] = "data/jaka_mf/latest56k_pico_dr.yaml",
    policy_model: Annotated[str, typer.Option()] = "",
    controller: Annotated[str, typer.Option()] = "pico",
    inference_backend: Annotated[str, typer.Option()] = "onnx-cpu",
    motion_backend: Annotated[str, typer.Option()] = "",
    motion_zmq_connect: Annotated[str, typer.Option()] = "tcp://127.0.0.1:28701",
    pico_zmq_connect: Annotated[str, typer.Option()] = "tcp://127.0.0.1:5592",
    rl_rate: Annotated[int, typer.Option()] = 50,
    # viewer
    enable_elastic_band: Annotated[bool, typer.Option(help="Enable elastic band")] = True,
    # recording
    record_continuous: Annotated[bool, typer.Option()] = False,
    record_fps: Annotated[int, typer.Option()] = 50,
    # diagnostics: write a per-step JSONL snapshot of the policy/sim state
    debug_log: Annotated[str, typer.Option()] = "",
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    if not policy_config:
        raise typer.BadParameter("--policy-config is required")

    import glfw

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=0.002,
        headless=headless,
        max_episode_steps=max_episode_steps,
        target=target,
        dr_level=dr_level,
        success_criteria=success_criteria,
    )
    jaka_env = env.unwrapped
    task = jaka_env.task
    robot = task.robot

    # Viewer key callback — MUST be set before reset so launch_passive uses it.
    # Keys: 7/8 = band length, 9 = toggle band. Band state lives on the robot,
    # which the agent configures after reset.
    # Recorder toggle: R key in the MuJoCo window starts/ends an episode.
    rec_flags = {"toggle": False}

    def _band_key_callback(key: int):
        if robot.elastic_band is not None:
            if key == glfw.KEY_7:
                robot.elastic_band.length -= 0.1
            elif key == glfw.KEY_8:
                robot.elastic_band.length += 0.1
            elif key == glfw.KEY_9:
                robot.elastic_band.enable = not robot.elastic_band.enable
        if key == glfw.KEY_R:
            rec_flags["toggle"] = True  # consumed by the recording state machine

    if not headless:
        # sim2real-style MuJoCo window: hide the left/right panels and track the
        # robot base with the camera (config read by launch_passive at (re)build).
        jaka_env.mujoco.viewer_key_callback = _band_key_callback
        jaka_env.mujoco.viewer_show_left_ui = False
        jaka_env.mujoco.viewer_show_right_ui = False
        jaka_env.mujoco.viewer_track_body = "base_link"

    observation, privileged_info = env.reset()

    # ── Init (no room-rebuild: that would close+re-launch the MuJoCo viewer).
    #    Explicitly reset the robot to the default standing pose (DEFAULT_QPOS),
    #    exactly as run_jaka_sim_server does, and pin the physics timestep.
    import mujoco

    robot_qpos_size = 7 + robot.dof
    mj_data = jaka_env.mujoco.mjData
    mj_data.qpos[7:robot_qpos_size] = DEFAULT_QPOS[7:robot_qpos_size]
    jaka_env.mujoco.mjModel.opt.timestep = 0.002
    mujoco.mj_forward(jaka_env.mujoco.mjModel, mj_data)

    num_substeps = max(1, int(round(1.0 / (robot.sim_dt * rl_rate))))
    agent = JakaMFAgent(
        robot,
        policy_config=policy_config,
        policy_model=policy_model,
        motion_zmq_connect=motion_zmq_connect,
        controller=controller,
        inference_backend=inference_backend,
        rl_rate=rl_rate,
        pico_zmq_connect=pico_zmq_connect,
        motion_backend=motion_backend or "zmq",
        num_substeps=num_substeps,
    )
    control_dt = num_substeps * robot.sim_dt

    if enable_elastic_band:
        agent._enable_band()

    # Stateful root [roll,pitch,yaw_vel] calculators for the OpenHLM recording
    # (replicate gear_sonic QuatProcessor). One per stream (robot state / pico reference);
    # reset at the start of each recorded episode.
    state_root_calc = None
    action_root_calc = None
    if data_format == "openhlm":
        from simple.datasets.jaka_lerobot import OpenHLMRootVel

        state_root_calc = OpenHLMRootVel()
        action_root_calc = OpenHLMRootVel()

    obj_names = list(jaka_env.mujoco.mj_objects.keys())

    exporter = None
    rec_state = RecordingState.WAITING_FOR_LANDING
    episodes_saved = 0
    if record:
        run_save_dir = f"{os.path.abspath(save_dir)}/{env.spec.id}/level-{dr_level}"
        import shutil

        if os.path.exists(run_save_dir):
            shutil.rmtree(run_save_dir)
        exporter = _init_exporter(
            run_save_dir, task, robot, obj_names, jaka_env.mujoco, data_format
        )
        print(f"\n[Record] Exporter initialized ({data_format}), saving to {run_save_dir}")

    # Diagnostic snapshot writer (JSONL: one dict per step).
    import json

    debug_f = open(debug_log, "w") if debug_log else None

    def _snap(step: int, event: str = "step", base_before=None):
        import numpy as _np

        sp = agent.policy.state_processor
        mf0 = getattr(sp, "motion_frame0", None)
        md = sp.motion_data
        idx_zero = 0
        ref_joint = md.joint_pos[0, idx_zero] if md is not None else None
        snap = {
            "step": step,
            "event": event,
            "control_mode": agent.policy.state_dict.get("control_mode"),
            "paused": agent.policy.state_dict.get("paused"),
            "base_pos": _np.round(robot.mjData.qpos[:3], 5).tolist(),
            "base_quat": _np.round(robot.mjData.qpos[3:7], 5).tolist(),
            "joint_qpos": _np.round(_np.asarray(list(robot.get_robot_qpos().values()), dtype=float), 4).tolist(),
            "cmd_q": _np.round(_np.asarray(agent.policy.action_manager.cmd_q, dtype=float), 4).tolist(),
            "ref_joint": (None if ref_joint is None else _np.round(ref_joint.astype(float), 4).tolist()),
            "ref_base_pos": (None if mf0 is None else _np.round(mf0["base_pos"].astype(float), 5).tolist()),
            "align_target": _np.round(agent.policy.align_target_joint_pos.astype(float), 4).tolist(),
            "band_enable": (robot.elastic_band.enable if robot.elastic_band is not None else None),
            "bd_attach": robot.band_attached_link,
            "need_reset": getattr(agent.policy, "need_reset_simulation", False),
            "finite": bool(_np.all(_np.isfinite(robot.mjData.qpos[:7]))),
        }
        if base_before is not None:
            snap["base_before"] = _np.round(base_before[:7], 5).tolist()
        if debug_f is not None:
            debug_f.write(json.dumps(snap) + "\n")
            debug_f.flush()

    try:
        sim_cnt = 0
        _snap(sim_cnt, event="init")
        while True:
            step_start = time.monotonic()

            base_before = robot.mjData.qpos[:7].copy()
            action = agent.get_action(
                observation, instruction=task.instruction, privileged_info=privileged_info
            )
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            _snap(sim_cnt, event="step", base_before=base_before)

            # Recording — key-driven: press R (MuJoCo window) to START and END
            # an episode. 7/8/9 (band) and the terminal mode keys are unaffected.
            # The exporter auto-creates an episode on add_frame; save_episode
            # finalizes it and advances to the next episode index.
            if exporter is not None:
                toggle = rec_flags["toggle"]
                rec_flags["toggle"] = False  # edge-triggered, consume once
                if rec_state == RecordingState.WAITING_FOR_LANDING:
                    if toggle:
                        rec_state = RecordingState.RECORDING
                        if state_root_calc is not None:
                            state_root_calc.reset()
                            action_root_calc.reset()
                        print(f"[Record] Episode {episodes_saved}: RECORDING started (R)")
                elif rec_state == RecordingState.RECORDING:
                    if data_format != "openhlm":
                        images = jaka_env.mujoco.render(
                            camera_names=["head_stereo_left", "front_stereo_left"]
                        )
                        frame = _build_frame(
                            images, jaka_env.mujoco, agent._bridge,
                            agent.policy.action_manager.cmd_q,
                        )
                        exporter.add_frame(frame)
                    else:
                        from simple.datasets.jaka_lerobot import build_openhlm_frame

                        head = jaka_env.mujoco.render(
                            camera_names=["head_stereo_left"]
                        )["head_stereo_left"]
                        t_ns = int(time.time() * 1e9)
                        d = jaka_env.mujoco.mjData
                        # state = [27 Jaka joints (robot/MuJoCo), root roll,pitch,yaw_vel (robot)]
                        state = np.concatenate([
                            np.asarray(list(robot.get_robot_qpos().values()), np.float32),
                            state_root_calc(d.qpos[3:7], t_ns),
                        ])
                        # VLA action label = the pico/ZMQ REFERENCE action the frozen
                        # policy tracks (NOT the policy output / cmd_q):
                        #   [27 joints (reference), roll,pitch,yaw_vel (reference),
                        #    pelvis linear velocity xyz, m/s (reference, current base frame)]
                        ref_joints, ref_base_quat, ref_base_lin_vel = _reference_action_jaka(
                            agent.policy.state_processor
                        )
                        actions = np.concatenate([
                            ref_joints,
                            action_root_calc(ref_base_quat, t_ns),
                            ref_base_lin_vel,
                        ])
                        exporter.add_frame(
                            build_openhlm_frame(head, state, actions),
                            task=task.instruction,
                        )
                    if toggle or terminated or truncated:
                        rec_state = RecordingState.EPISODE_DONE
                elif rec_state == RecordingState.EPISODE_DONE:
                    ep_idx = exporter.episode_buffer["episode_index"]
                    exporter.save_episode()
                    if data_format != "openhlm":
                        # OpenHLM format carries no replay environment_config.
                        _save_episode_env_config(exporter, task, ep_idx)
                    episodes_saved += 1
                    print(f"[Record] Episode {episodes_saved} saved")
                    if episodes_saved >= num_episodes:
                        break
                    # Keep teleoperating: do NOT reset the env/policy after the
                    # save. The exporter auto-creates a fresh episode on the next
                    # add_frame, and the robot keeps its current pose/obs history,
                    # so the user can press R again to record another continuous
                    # segment without the robot being teleported back.
                    rec_state = RecordingState.WAITING_FOR_LANDING
                    continue

            # The MuJoCo viewer is already synced inside MujocoSimulator.step();
            # LocoManipulationEnv has no update_viewer (that's G1-specific).
            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
    finally:
        if debug_f is not None:
            debug_f.close()
            print(f"[debug] wrote {debug_log}")
        if exporter is not None:
            if data_format != "openhlm":
                exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved")
        env.close()
        agent.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
