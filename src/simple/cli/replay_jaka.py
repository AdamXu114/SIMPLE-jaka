"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Stage 2 — replay of a Stage-1 recorded Jaka episode into a LeRobot dataset
with rendered camera images (MuJoCo or IsaacSim).

Two replay drivers:
- ``--direct`` (default): hard replay — snap the robot AND the whole scene
  (objects, articulated hinges) to the recorded pose every frame, no physics.
  Never falls; reproduces object interaction exactly. For rendering VLA videos.
- ``--policy-model/--policy-config``: closed-loop — run the Jaka RL policy
  each step (tracking the recorded episode as reference) so the biped stays
  balanced under real physics.

Usage:
    replay-jaka simple/JakaOpenTrashCanTeleop-v0 \\
        --data-dir data/jaka_teleop/jaka_open_trash_can_teleop/level-0 \\
        --save-dir data/replay_jaka \\
        --sim-mode mujoco_isaac --save-all --direct
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")


def _ensure_lula_lib_on_path() -> None:
    """Prepend the lula prebundle lib dir so liblula_kinematics.so resolves its
    urdfdom dependency. Some pip IsaacSim installs have a conflicting urdfdom
    earlier on LD_LIBRARY_PATH that lacks the ``urdf::parseURDF`` symbol, which
    makes the lula extension fail to load (undefined symbol). This is a no-op
    when the prebundle is absent.
    """
    prebundle = (
        Path("/home/xu/code/SIMPLE/.venv/lib/python3.10/site-packages/isaacsim")
        / "exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs"
    )
    candidates = [
        Path(os.environ.get("VIRTUAL_ENV", "/home/xu/code/SIMPLE/.venv"))
        / "lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs",
        prebundle,
    ]
    lib_dir = next((p for p in candidates if (p / "liblula_kinematics.so").exists()), None)
    if lib_dir is not None:
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in cur:
            os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + cur if cur else "")


_ensure_lula_lib_on_path()

import numpy as np
import typer
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401
from simple.datasets.jaka_lerobot import (
    JAKA_VIDEO_SHAPES,
    VIDEO_SHAPE,
    build_jaka_frame,
    fix_hssd_room_pose,
    init_jaka_exporter,
    save_episode_env_config,
)
from simple.datasets.jaka_policy import JakaPolicyReplay
from simple.utils import NumpyArrayEncoder


def _load_episodes(data_dir: str):
    """Load all episodes from a LeRobot dataset directory."""
    import pyarrow.parquet as pq

    data_path = Path(data_dir) / "data"
    parquet_files = sorted(data_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    episodes = {}
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        for ep_idx in df["episode_index"].unique():
            episodes[int(ep_idx)] = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    return episodes


def _load_episode_configs(data_dir: str):
    """Load environment_config per episode from meta/episodes.jsonl."""
    meta_file = Path(data_dir) / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return {}

    configs = {}
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            ep_idx = entry.get("episode_index", None)
            env_conf_str = entry.get("environment_config", None)
            if ep_idx is not None and env_conf_str is not None:
                configs[int(ep_idx)] = json.loads(env_conf_str)
    return configs


def _save_episode_env_config_impl(exporter, saved_env_conf: dict, episode_index: int) -> None:
    """Back-fill the replayed environment_config into the new dataset's episodes.jsonl."""
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    lines[episode_index]["environment_config"] = json.dumps(saved_env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


class JakaReplayAgent:
    """Wraps a recorded episode DataFrame (used for ``_replay_episode``)."""

    def __init__(self, episode_df):
        self._df = episode_df


def _episode_to_motion(df, reindex: np.ndarray) -> dict[str, np.ndarray]:
    """Build motion_data (npz layout) from a recorded episode.

    Returns dict with joint_pos[T,27] (npz order), body_pos_w/body_quat_w/
    body_lin_vel_w/body_ang_vel_w each [T,28,...] — from the recorded
    ``observation.state`` and ``observation.body_poses``.
    """
    T = len(df)
    joint_pos_mujoco = np.stack([np.asarray(r, dtype=np.float32) for r in df["observation.state"]])  # [T,27]
    joint_pos_npz = joint_pos_mujoco[:, reindex]
    body_poses = np.stack([np.asarray(r, dtype=np.float32) for r in df["observation.body_poses"]])  # [T,364]
    bp = body_poses.reshape(T, 28, 13)
    return {
        "joint_pos": joint_pos_npz,
        "body_pos_w": bp[:, :, 0:3].copy(),
        "body_quat_w": bp[:, :, 3:7].copy(),
        "body_lin_vel_w": bp[:, :, 7:10].copy(),
        "body_ang_vel_w": bp[:, :, 10:13].copy(),
    }


def _find_imu_sensors(mj_model, site_name: str = "waist_imu"):
    """Return (gyro_adr, framequat_adr) for the IMU site."""
    import mujoco

    gyro_adr = quat_adr = None
    for i in range(mj_model.nsensor):
        s = mj_model.sensor(i)
        if s.objtype.item() == mujoco.mjtObj.mjOBJ_SITE:
            if mj_model.site(s.objid.item()).name == site_name:
                if s.type.item() == mujoco.mjtSensor.mjSENS_GYRO:
                    gyro_adr = int(mj_model.sensor_adr[i])
                elif s.type.item() == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                    quat_adr = int(mj_model.sensor_adr[i])
    return gyro_adr, quat_adr


def _replay_episode(
    env,
    agent: JakaReplayAgent,
    exporter,
    policy: Optional["JakaPolicyReplay"] = None,
    direct: bool = False,
) -> bool:
    """Replay one episode in MuJoCo + (optionally) IsaacSim, recording frames.

    With ``sim_mode="mujoco_isaac"`` the Isaac renderer is used for images;
    with ``sim_mode="mujoco"`` the MuJoCo renderer is used (no GPU needed).

    Replay driver is chosen by ``direct`` / ``policy``:
    - ``direct=True`` (hard replay): snap the robot AND the whole scene
      (objects + articulated hinges) to each recorded frame's pose, no physics
      — nothing falls and objects move exactly as recorded. Purely renders a
      video of the recorded trajectory.
    - ``policy`` given: the Jaka RL policy runs closed-loop each step
      (tracking the recorded episode as reference) so the biped stays balanced.
    """
    import mujoco

    jaka_env = env.unwrapped
    task = jaka_env.task
    mujoco_robot = jaka_env.mujoco
    isaac = getattr(jaka_env, "isaac", None)
    mj_model = mujoco_robot.mjModel
    mj_data = mujoco_robot.mjData

    # Per-substep PD controller (no ZMQ sockets): recompute the PD torque every
    # physics substep from the current joint state — identical to the recording
    # server's bridge so the closed-loop replay tracks exactly like the real
    # robot / recording did (500 Hz PD on top of 50 Hz policy commands).
    pd_bridge = None
    if policy is not None:
        from simple.interfaces.zmq_bridge import ZMQSimBridge
        pd_bridge = ZMQSimBridge.create_without_zmq(
            mj_model, mj_data, list(task.robot.joint_names)
        )
        pd_bridge.has_received_command = True  # cmd_kp/cmd_kd default to config gains

    # 1. Initialize to the recorded first frame — base pose (pos+quat wxyz) and
    #    joint state — so replay starts exactly where recording began.
    #    Replaying from the env's default init starts with a different pose and
    #    straight legs, and the robot falls before the first action stabilizes it.
    first_row = agent._df.iloc[0]
    base_pose = np.asarray(first_row["observation.base_pose"], dtype=np.float64)
    first_state = np.asarray(first_row["observation.state"], dtype=np.float64)
    robot_dof = task.robot.dof
    mj_data.qpos[:7] = base_pose[:7]                      # base pos + quat (wxyz)
    mj_data.qpos[7 : 7 + robot_dof] = first_state[:robot_dof]  # joints
    mj_data.qvel[:] = 0.0                                 # start from rest
    mujoco.mj_forward(mj_model, mj_data)

    # 2. IMU sensors (waist_imu) for the policy's root orientation/ang-vel.
    gyro_adr, quat_adr = _find_imu_sensors(mj_model)
    waist_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_Link")

    # 3. Closed-loop policy setup (if provided): rebuild motion_data from the
    #    recorded episode and reset the policy to the first frame.
    if policy is not None:
        motion = _episode_to_motion(agent._df, policy.reindex)
        policy.set_motion(**motion)
        policy.reset(mj_data.xquat[waist_body])

    # Step physics at 50 Hz (nstep=10 at 500 Hz sim_dt), syncing Isaac each step.
    # env.step() can't be used: with need_gravity=True + robot.command is None,
    # MujocoSimulator.step() advances only one 500 Hz substep.
    frames = 0
    terminated = False
    n_frames = len(agent._df)
    while frames < n_frames:
        row = agent._df.iloc[frames]

        if direct:
            # Direct (hard) replay: snap the robot AND the whole scene exactly
            # to the recorded frame's pose (base + joints + object free joints
            # + articulated hinges) — no physics, so nothing falls and objects
            # move exactly as recorded. Used purely to render a video.
            base_pose = np.asarray(row["observation.base_pose"], dtype=np.float64)
            state = np.asarray(row["observation.state"], dtype=np.float64)
            mj_data.qpos[:7] = base_pose[:7]
            mj_data.qpos[7 : 7 + robot_dof] = state[:robot_dof]
            if "observation.scene_qpos" in row:
                scene_qpos = np.asarray(row["observation.scene_qpos"], dtype=np.float64)
                mj_data.qpos[
                    7 + robot_dof : 7 + robot_dof + scene_qpos.size
                ] = scene_qpos
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            action = np.asarray(row["action"], dtype=np.float64)
        elif policy is not None:
            # Closed loop: read sim state, infer action from Jaka policy.
            root_quat_w = (
                mj_data.sensordata[quat_adr : quat_adr + 4]
                if quat_adr is not None else mj_data.xquat[waist_body]
            ).copy()
            root_ang_vel_b = (
                mj_data.sensordata[gyro_adr : gyro_adr + 3]
                if gyro_adr is not None else mj_data.cvel[waist_body, 3:6]
            ).copy()
            joint_pos = np.asarray(list(mujoco_robot.get_robot_qpos().values()), dtype=np.float32)
            joint_vel = np.asarray(
                list(task.robot.get_robot_qvel().values()), dtype=np.float32
            )
            action = policy.step(root_quat_w, root_ang_vel_b, joint_pos, joint_vel)
            policy.advance_motion()

        if not direct:
            # Step 10 physics substeps (500 Hz) with the PD torque recomputed
            # from the current joint state each substep — matches the recording
            # server's bridge (50 Hz policy command, 500 Hz PD).
            for _ in range(10):
                pd_bridge.cmd_q[:] = action
                pd_bridge.apply_pd()
                mujoco.mj_step(mj_model, mj_data)
        if mujoco_robot.viewer is not None:
            mujoco_robot.viewer.sync()  # update on-screen viewer if enabled
        if isaac is not None:
            isaac.step(mujoco_robot)
            images = isaac.render()
        else:
            images = mujoco_robot.render()
        qpos = np.asarray(list(mujoco_robot.get_robot_qpos().values()), dtype=np.float64)
        frame = build_jaka_frame(
            images,
            mujoco_robot.mjData,
            action,
            mujoco_robot.mj_objects,
            qpos,
            mj_model=mujoco_robot.mjModel,
        )
        exporter.add_frame(frame)
        frames += 1

        info = jaka_env._get_info()
        terminated = bool(task.check_success(info, mujoco_env=mujoco_robot))
        if terminated:
            break

    return terminated


def main(
    env_id: Annotated[str, typer.Argument()],
    data_dir: Annotated[str, typer.Option()],
    save_dir: Annotated[str, typer.Option()] = "data/replay_jaka",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    render_hz: Annotated[int, typer.Option()] = 50,
    dr_level: Annotated[int, typer.Option(help="Output level dir suffix only; "
                                             "replay always restores the full recorded env_conf")] = 0,
    num_episodes: Annotated[int, typer.Option()] = -1,
    save_all: Annotated[bool, typer.Option()] = False,
    policy_model: Annotated[str, typer.Option()] = "",
    policy_config: Annotated[str, typer.Option()] = "",
    direct: Annotated[bool, typer.Option()] = False,
):
    import gymnasium as gym

    # Choose the replay driver: hard snapshot (--direct) or closed-loop policy.
    policy = None
    if direct:
        print("Direct (hard) replay: snapping robot + scene to recorded poses, no physics")
    elif policy_model or policy_config:
        if not (policy_model and policy_config):
            raise typer.BadParameter("Both --policy-model and --policy-config required")
        policy = JakaPolicyReplay(policy_model, policy_config)
        print(f"Closed-loop policy loaded from {policy_model}")
    else:
        raise typer.BadParameter(
            "Choose a replay driver: --direct (hard snapshot) or "
            "--policy-model/--policy-config (closed-loop policy)"
        )

    episodes = _load_episodes(data_dir)
    episode_configs = _load_episode_configs(data_dir)
    episode_ids = sorted(episodes.keys())
    if num_episodes > 0:
        episode_ids = episode_ids[:num_episodes]
    if not episode_ids:
        print(f"No episodes found in {data_dir}")
        raise typer.Exit(1)

    print(f"Replaying {len(episode_ids)} episodes from {data_dir}")

    env = gym.make(env_id, sim_mode=sim_mode, render_hz=render_hz, headless=headless)
    jaka_env = env.unwrapped
    task = jaka_env.task

    out_root = Path(save_dir) / env.spec.id / f"level-{dr_level}"
    print(f"Saving to {out_root}")

    # Start fresh: remove any partially-initialized dataset from a previous run.
    # Gr00tDataExporter resumes an existing dir, and a dir left incomplete by an
    # interrupted run (e.g. missing meta/tasks.jsonl) crashes the resume path.
    import shutil
    if out_root.exists():
        shutil.rmtree(out_root)
        print(f"Removed existing {out_root} (fresh replay)")

    episodes_saved = 0
    results = {}
    exporter = None
    for i, ep_idx in enumerate(episode_ids):
        print(f"--- Replaying episode {ep_idx} ({i+1}/{len(episode_ids)}) ---")
        env_conf = episode_configs.get(ep_idx)
        if env_conf is None:
            print(f"[skip] episode {ep_idx}: no environment_config")
            continue

        # Load the full recorded env_conf (dr_level=None restores material/lighting
        # too, not just pose) and fix the HSSD room pose (yaw=0, centered) so the
        # rendered room lines up with the world-frame robot/objects.
        obs, info = env.reset(options={"state_dict": env_conf, "dr_level": None})
        fix_hssd_room_pose(task)

        obj_names = list(jaka_env.mujoco.mj_objects.keys())
        if exporter is None:
            scene_qpos_shape = jaka_env.mujoco.mjModel.nq - (7 + task.robot.dof)
            exporter = init_jaka_exporter(
                str(out_root),
                task_prompt=task.instruction,
                obj_names=obj_names,
                video_shape=VIDEO_SHAPE,
                video_shapes=JAKA_VIDEO_SHAPES,
                scene_qpos_shape=scene_qpos_shape,
            )

        agent = JakaReplayAgent(episodes[ep_idx])
        ep_idx_out = exporter.episode_buffer["episode_index"]
        success = _replay_episode(
            env, agent, exporter,
            policy=policy,
            direct=direct,
        )

        if success or save_all:
            exporter.save_episode()
            _save_episode_env_config_impl(exporter, env_conf, ep_idx_out)
            episodes_saved += 1
            status = "success" if success else "saved(force)"
            print(f"[ok]   episode {ep_idx} saved ({status})")
        else:
            exporter.skip_and_start_new_episode()
            print(f"[fail] episode {ep_idx} not saved")
        results[ep_idx] = success

    # Print results BEFORE closing — IsaacSim shutdown is slow and would
    # otherwise make the run look hung.
    sr = sum(results.values()) / len(results) if results else 0.0
    print(f"Done. {episodes_saved} episodes saved. Success rate: {sr:.2%}")
    print("Closing environment (IsaacSim shutdown may take a minute)...")
    env.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
