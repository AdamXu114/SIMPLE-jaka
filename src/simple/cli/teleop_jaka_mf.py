"""Teleoperation using the Jaka whole-body MF RL policy (in-process).

Single terminal: SIMPLE runs the MuJoCo scene AND the Jaka MF strategy (from
sim2real-jaka) in one process. The only sim2real process still running is
``pico_retarget_hub`` (publishes motion :28701 + handle :5592).

CLI 只留「每次运行会变」的开关,其余参数全在配置文件里(默认
``data/jaka_mf/teleop_jaka_mf.yaml``),日常调参改那个文件即可。

Usage::

    export TASK_NAME=JakaOpenTrashCanTeleop-v0
    python -m simple.cli.teleop_jaka_mf simple/$TASK_NAME \\
        --target=graspnet1b:0 --no-headless

Recording is handled by the standalone ``simple.cli.record_jaka_zmq`` process: this
teleop always re-publishes state + head camera over ZMQ (``state_zmq_bind`` /
``camera_zmq_bind`` in the config), so just run the recorder in another terminal.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import gymnasium as gym
import typer
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401  (trigger env registration)

from simple.agents.jaka_mf_agent import JakaMFAgent
from simple.jaka_rl.config import BODY_NAMES, DEFAULT_QPOS

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

DEFAULT_CONFIG = "data/jaka_mf/teleop_jaka_mf.yaml"

# 配置文件缺省值(缺键时用这些;见 data/jaka_mf/teleop_jaka_mf.yaml 的注释)。
_DEFAULTS: dict = {
    "policy_config": "data/jaka_mf/latest56k_pico_dr.yaml",
    "policy_model": "",
    "inference_backend": "onnx-cpu",
    "motion_backend": "zmq",
    "motion_zmq_connect": "tcp://127.0.0.1:28701",
    "pico_zmq_connect": "tcp://127.0.0.1:5592",
    "sim_mode": "mujoco",
    "rl_rate": 50,
    "render_hz": 50,
    "dr_level": 0,
    "max_episode_steps": 30000,
    "success_criteria": 5,
    "obs_visual": False,
    "viewer_hz": 0,
    # 窗口视角:留空(默认)= 保持原来的第三人称,跟踪 base_link。
    # 想切成机器人头部第一人称就填 "head_stereo_left"(模型里的相机名)。
    "viewer_camera": "",
    "init_base_z": 0.596,
    "enable_elastic_band": False,
    "state_zmq_bind": "tcp://*:28711",
    "camera_zmq_bind": "tcp://*:28712",
    "camera_name": "head_stereo_left",
    "camera_hz": 30,
    "reset_on_record_end": True,
}


def _fmt_opt(value, unit: str = "", spec: str = ".1f", scale: float = 1.0) -> str:
    """Format an optional diagnostic number, ``n/a`` when None.

    The reference-stream diagnostics are None until the first frame arrives (and
    some stay None with no publisher at all), so status printing must never assume
    a number — a bare ``:.1f`` on None raises and would kill the control loop.
    """
    return "n/a" if value is None else f"{value * scale:{spec}}{unit}"


def load_config(path: str) -> dict:
    """读取 teleop 配置 YAML,缺失的键用 :data:`_DEFAULTS` 补齐。"""
    import yaml

    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(yaml.load(f, Loader=yaml.FullLoader) or {})
    else:
        print(f"[config] {path} 不存在,使用内置缺省值")
    return cfg


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/JakaOpenTrashCanTeleop-v0",
    config: Annotated[str, typer.Option(help="teleop 配置文件")] = DEFAULT_CONFIG,
    target: Annotated[str | None, typer.Option(help="目标物体 id")] = None,
    controller: Annotated[str, typer.Option(help="模式来源 pico|keyboard")] = "pico",
    headless: Annotated[bool, typer.Option()] = False,
    debug_log: Annotated[str, typer.Option(help="每步 JSONL 诊断快照")] = "",
):
    cfg = load_config(config)
    sim_mode = cfg["sim_mode"]
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    if not cfg["policy_config"]:
        raise AssertionError("config: policy_config is required")

    import glfw

    print(f"Creating environment: {env_id}  (config: {config})")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=cfg["render_hz"],
        physics_dt=0.002,
        headless=headless,
        max_episode_steps=cfg["max_episode_steps"],
        target=target,
        dr_level=cfg["dr_level"],
        success_criteria=cfg["success_criteria"],
        obs_visual=cfg["obs_visual"],
    )
    jaka_env = env.unwrapped
    task = jaka_env.task
    robot = task.robot

    # Viewer key callback — MUST be set before reset so launch_passive uses it.
    # Keys: 7/8 = band length, 9 = toggle band. Band state lives on the robot,
    # which the agent configures after reset.
    def _band_key_callback(key: int):
        if robot.elastic_band is not None:
            if key == glfw.KEY_7:
                robot.elastic_band.length -= 0.1
            elif key == glfw.KEY_8:
                robot.elastic_band.length += 0.1
            elif key == glfw.KEY_9:
                robot.elastic_band.enable = not robot.elastic_band.enable

    if not headless:
        # sim2real-style MuJoCo window: hide the left/right panels and track the
        # robot base with the camera (config read by launch_passive at (re)build).
        jaka_env.mujoco.viewer_key_callback = _band_key_callback
        jaka_env.mujoco.viewer_show_left_ui = False
        jaka_env.mujoco.viewer_show_right_ui = False
        # 视角:固定到模型里的头相机(第一人称)。留空则跟踪 base_link。
        # 注意 env.reset() 会重建模型 + 重开窗口,该设置会在每次重开后重新应用,
        # 所以 reset 之后视角不会跳回默认。
        jaka_env.mujoco.viewer_fixed_camera = cfg["viewer_camera"] or None
        jaka_env.mujoco.viewer_track_body = "base_link"
        if cfg["viewer_hz"] > 0:
            # sync the on-screen viewer every rl_rate/viewer_hz steps (physics stays at rl_rate).
            jaka_env.mujoco.viewer_sync_interval = max(
                1, round(cfg["rl_rate"] / cfg["viewer_hz"])
            )
            print(f"[viewer] throttled display to {cfg['viewer_hz']} Hz (sync every "
                  f"{jaka_env.mujoco.viewer_sync_interval} step(s)), "
                  f"control at {cfg['rl_rate']} Hz")

    observation, privileged_info = env.reset()

    # ── Init (no room-rebuild: that would close+re-launch the MuJoCo viewer).
    #    Explicitly reset the robot to the default standing pose (DEFAULT_QPOS),
    #    exactly as run_jaka_sim_server does, and pin the physics timestep.
    import mujoco

    robot_qpos_size = 7 + robot.dof
    mj_data = jaka_env.mujoco.mjData
    mj_data.qpos[7:robot_qpos_size] = DEFAULT_QPOS[7:robot_qpos_size]
    mj_data.qpos[2] = cfg["init_base_z"]  # 直接用站立平衡高度起步,避免开局下坠
    jaka_env.mujoco.mjModel.opt.timestep = 0.002
    mujoco.mj_forward(jaka_env.mujoco.mjModel, mj_data)

    num_substeps = max(1, int(round(1.0 / (robot.sim_dt * cfg["rl_rate"]))))
    agent = JakaMFAgent(
        robot,
        policy_config=cfg["policy_config"],
        policy_model=cfg["policy_model"],
        motion_zmq_connect=cfg["motion_zmq_connect"],
        controller=controller,
        inference_backend=cfg["inference_backend"],
        rl_rate=cfg["rl_rate"],
        pico_zmq_connect=cfg["pico_zmq_connect"],
        motion_backend=cfg["motion_backend"] or "zmq",
        num_substeps=num_substeps,
    )
    control_dt = num_substeps * robot.sim_dt

    # Standalone recorder publisher: re-publish state + head camera over ZMQ each
    # loop so ``simple.cli.record_jaka_zmq`` (a separate process) can read latest
    # state/action/camera without holding the sim/control loop.
    from simple.interfaces.jaka_zmq_pub import JakaTeleopZmqPublisher

    zmq_pub = JakaTeleopZmqPublisher(
        jaka_env.mujoco,
        robot,
        state_zmq_bind=cfg["state_zmq_bind"],
        camera_zmq_bind=cfg["camera_zmq_bind"],
        camera_name=cfg["camera_name"],
        camera_hz=cfg["camera_hz"],
    )
    print(f"[publish] streaming state@{cfg['rl_rate']}Hz→{cfg['state_zmq_bind']} "
          f"camera@{cfg['camera_hz']}Hz→{cfg['camera_zmq_bind']}")

    # ── 弹力带(config: enable_elastic_band,默认 false = 暂时禁用)──────────
    # 恢复启用只需把配置改成 true。注意必须显式设 enable:
    # JakaMFAgent.__init__ 内部已经 _enable_band() 建好弹力带(enable=True),
    # 且 env.reset() -> task.reset() -> robot.reset() 还会把它置回 True。
    # 原始写法:`if enable_elastic_band: agent._enable_band()`
    if cfg["enable_elastic_band"]:
        agent._enable_band()
    if robot.elastic_band is not None:
        robot.elastic_band.enable = bool(cfg["enable_elastic_band"])

    # ── 启动即进入 policy 模式(关节用 `[` 的数值)─────────────────────────
    # 一开始就让机器人站住:set_align_mode() 把对齐目标设为 motion frame0 的关节值
    # (= ZMQ_DEFAULT_QPOS[7:],JAKA 顺序),并置 need_reset_simulation=True,于是第一
    # 个 policy.step() 会把关节段吸附到该数值;set_policy_mode() 随即进入 policy。
    agent.policy.set_align_mode(source="startup")
    agent.policy.set_policy_mode(source="startup")

    # ── Record-end reset ────────────────────────────────────────────────────
    # ``toggle_data_collection`` (published by the pico hub alongside the motion
    # frames) is the record start/stop marker. The standalone recorder ends the
    # episode; this teleop only handles the END — on the high→low edge it resets the
    # environment so the next episode starts from a clean, freshly randomized scene.
    # Mirrors teleop_decoupled_wbc's EPISODE_DONE branch + _on_episode_reset().
    motion_buffer = getattr(agent.policy.state_processor, "motion_buffer", None)
    prev_toggle: bool | None = None

    # 手动触发同样一套 reset:终端里输入 ``r`` + 回车(与 pico 的下降沿并列,
    # 方便没接 hub 时测试)。daemon 线程逐行读 stdin,主循环消费这个标志。
    reset_flags = {"keyboard": False}

    def _stdin_listener() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() == "r":
                    reset_flags["keyboard"] = True
        except Exception:
            pass

    threading.Thread(target=_stdin_listener, daemon=True).start()

    def _reset_episode() -> None:
        """Reset the env + policy after an episode ends (pico toggle went high→low)."""
        nonlocal mj_data, observation, privileged_info
        print("[reset] Record end (toggle_data_collection 1→0): resetting environment")
        # Full env.reset(): re-samples DR and rebuilds the scene -- and with it the
        # MuJoCo model (mjModel/mjData are recompiled by _setup_scene).
        observation, privileged_info = env.reset()
        # Re-fetch and re-pose exactly like the startup init (the old mjData is dead).
        mj_data = jaka_env.mujoco.mjData
        mj_data.qpos[7:robot_qpos_size] = DEFAULT_QPOS[7:robot_qpos_size]
        mj_data.qpos[2] = cfg["init_base_z"]  # 同上:用站立平衡高度起步
        jaka_env.mujoco.mjModel.opt.timestep = 0.002
        mujoco.mj_forward(jaka_env.mujoco.mjModel, mj_data)
        # The model was recompiled: re-point the PD bridge (otherwise apply_pd writes
        # into the dead MjData and the new sim receives zero torque -> robot limp) and
        # re-resolve the publisher's body ids (ids can shift in the new model).
        agent.rebind_bridge()
        if zmq_pub is not None:
            zmq_pub.refresh()
        # Elastic band: re-resolve the attach body + re-anchor to the new base, then
        # restore the configured enable state (env.reset() re-enables it via
        # robot.reset(), so a stale anchor could otherwise drag the robot).
        if robot.elastic_band is not None:
            robot.band_attached_link = -1
            robot._init_band_locally(mj_data.qpos[:2])
            robot.elastic_band.enable = bool(cfg["enable_elastic_band"])
        # 清空 motion buffer:buffer 空时 get_obs() 回退到 FK 出的 default posture
        # (ZMQ_DEFAULT_QPOS 站姿),于是**新回合一开始跟踪的是 default joint pos**,
        # 而不是上一段残留 / 当前 live 的参考。随后流里的新帧会照常重新填满 buffer。
        # (录制开关 toggle 不属于运动数据,clear() 不动它,避免误触发。)
        if motion_buffer is not None:
            motion_buffer.clear()
        agent.policy.state_processor.reset()  # 丢弃缓存的 motion_data 窗口,立即改用空 buffer

        # Stand at the confirmed pose (`[` align target) and go straight to tracking.
        agent.policy.set_align_mode(source="record_end")
        agent.policy.set_policy_mode(source="record_end")

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
        # Real-time loop-rate monitor: confirms whether we're truly at rl_rate (50 Hz).
        fps_t0 = time.monotonic()
        fps_n0 = 0
        pub_n0 = (0, 0)  # last (state_frames, camera_frames) sample from the ZMQ publisher
        # Per-stage timing (ms) accumulators for the loop bottleneck trace.
        time_get = time_env = time_data = 0.0
        time_phys = time_sync = time_obj = 0.0
        env_dbg = {"apply": 0.0, "obs": 0.0, "info": 0.0, "reward": 0.0, "success": 0.0}
        n_steps = 0
        while True:
            step_start = time.monotonic()

            base_before = robot.mjData.qpos[:7].copy()
            t_ga = time.monotonic()
            action = agent.get_action(
                observation, instruction=task.instruction, privileged_info=privileged_info
            )
            t_es = time.monotonic()
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            _snap(sim_cnt, event="step", base_before=base_before)
            if zmq_pub is not None:
                zmq_pub.publish()

            # Reset trigger (a): pico ``toggle_data_collection`` going high→low (the
            # recorder stops on the same edge). Only the falling edge acts; the first
            # observation just adopts the current level as the baseline.
            if cfg["reset_on_record_end"] and motion_buffer is not None:
                level = motion_buffer.latest_toggle_data_collection
                if level is not None:
                    if prev_toggle is True and level is False:
                        _reset_episode()
                    prev_toggle = level

            # Reset trigger (b): 终端键盘 ``r`` + 回车(手动,随时可用)。
            if reset_flags["keyboard"]:
                reset_flags["keyboard"] = False
                print("[reset] 终端键盘 'r' 触发")
                _reset_episode()
            time_get += (t_es - step_start) * 1000.0
            time_env += (time.monotonic() - t_es) * 1000.0
            _ms = jaka_env.mujoco
            time_phys += _ms.dbg_ms_physics
            time_sync += _ms.dbg_ms_sync
            time_obj += _ms.dbg_ms_obj
            for _k, _v in jaka_env.dbg_ms.items():
                if _k in env_dbg:
                    env_dbg[_k] += _v

            # The MuJoCo viewer is already synced inside MujocoSimulator.step();
            # LocoManipulationEnv has no update_viewer (that's G1-specific).
            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                # Precise pacing: sleep most of the budget, then busy-wait the last ~1 ms so the
                # loop lands exactly on control_dt (0.02 s). time.sleep granularity on Linux
                # overshoots by ~1 ms, which would otherwise cap the rate at ~49 Hz.
                time.sleep(max(0.0, sleep_time - 0.001))
                target = step_start + control_dt
                while time.monotonic() < target:
                    pass

            sim_cnt += 1
            now = time.monotonic()
            if now - fps_t0 >= 2.0:
                n_steps = sim_cnt - fps_n0
                measured = n_steps / (now - fps_t0)
                loop_ms = 1000.0 * (now - fps_t0) / n_steps
                time_data = max(0.0, loop_ms * n_steps - (time_get + time_env))
                # [fps]/[time] per-stage trace kept for debugging but muted: un-comment to
                # bring back the loop-rate + bottleneck breakdown.
                # print(f"[fps] {measured:.1f} Hz over last {n_steps} steps "
                #       f"(target {rl_rate} Hz, {loop_ms:.2f} ms/step)", flush=True)
                # _env_rest = max(0.0, time_env - time_phys - time_sync - time_obj)
                # print(f"[time] get_action={time_get / n_steps:.2f} ms  env.step={time_env / n_steps:.2f} ms "
                #       f"[physics={time_phys / n_steps:.2f}  viewer_sync={time_sync / n_steps:.2f}  objects={time_obj / n_steps:.2f}  "
                #       f"apply={env_dbg['apply'] / n_steps:.2f}  obs={env_dbg['obs'] / n_steps:.2f}  info={env_dbg['info'] / n_steps:.2f}  "
                #       f"reward={env_dbg['reward'] / n_steps:.2f}  success={env_dbg['success'] / n_steps:.2f}  rest={_env_rest / n_steps:.2f}]  "
                #       f"other/data={time_data / n_steps:.2f}", flush=True)
                # Actual publish rates (0 when --publish-zmq is off): state goes out every
                # control tick (target rl_rate), the camera from its own thread (target camera_hz).
                if zmq_pub is not None:
                    ns, nc = zmq_pub.stats()
                    dt = now - fps_t0
                    state_hz = (ns - pub_n0[0]) / dt
                    cam_hz = (nc - pub_n0[1]) / dt
                    pub_n0 = (ns, nc)
                    print(f"[zmq] state {state_hz:.1f} Hz (target {cfg['rl_rate']}) "
                          f"-> {cfg['state_zmq_bind']}   "
                          f"camera {cam_hz:.1f} Hz (target {cfg['camera_hz']}) "
                          f"-> {cfg['camera_zmq_bind']}   "
                          f"[loop {measured:.1f} Hz, {loop_ms:.2f} ms/step]", flush=True)
                # Reference-stream health. `frames` = distinct buffered frames the
                # 5-step window resolved to: 1 means NO look-ahead (the policy then
                # has root_pos_diff_b == 0 and cannot anticipate motion — it still
                # tracks, but reactively and easily destabilises). `clamped` is the
                # same condition from the window's side; `ts_off` should sit near 0 ms
                # (the buffer must run on this process's clock) while `smplx_off` is
                # the publisher's raw XR-clock offset and can be huge. `ref_z` is the
                # reference anchor height the policy is asked to hold.
                # if motion_buffer is not None:
                #     # Never let a *diagnostic* take down the control loop: every
                #     # optional field is None until the first reference frame arrives
                #     # (and some stay None with no publisher at all), so format them
                #     # through _fmt_opt and guard the whole block.
                #     try:
                #         rd = motion_buffer.diagnostics()
                #         md = agent.policy.state_processor.motion_data
                #         ref_z = ""
                #         if md is not None:
                #             try:
                #                 anchor = (agent.policy.policy_config.get("motion", {}) or {}).get(
                #                     "anchor_body_name", "waist_yaw_Link")
                #                 z = md.body_pos_w[0, :, BODY_NAMES.index(anchor)][:, 2]
                #                 ref_z = (f"  ref_z={float(z.min()):.3f}..{float(z.max()):.3f}")
                #             except (ValueError, IndexError, TypeError):
                #                 pass
                #         print(f"[ref] frames={rd['window_frames']}/{rd['buffered']} "
                #               f"clamped={rd['clamped']} "
                #               f"dt={_fmt_opt(rd['last_dt_ms'], 'ms')} "
                #               f"ts_off={_fmt_opt(rd['ts_offset_ms'], 'ms')} "
                #               f"smplx_off={_fmt_opt(rd['smplx_offset_ms'], 's', '.0f', 1e-3)} "
                #               f"regress={rd['ts_regressions']} "
                #               f"jump={_fmt_opt(rd['jump_recent_m'], 'm', '.2f')} "
                #               f"align={_fmt_opt(rd['align_yaw_deg'], 'deg')}"
                #               f"{'' if rd['aligned'] else ' (identity)'}{ref_z}",
                #               flush=True)
                #     except Exception as exc:  # noqa: BLE001
                #         print(f"[ref] diagnostics unavailable: {exc}", flush=True)
                fps_t0, fps_n0 = now, sim_cnt
                time_get = time_env = time_data = time_phys = time_sync = time_obj = 0.0
                for _k in env_dbg:
                    env_dbg[_k] = 0.0
                n_steps = 0
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
    finally:
        if debug_f is not None:
            debug_f.close()
            print(f"[debug] wrote {debug_log}")
        if zmq_pub is not None:
            zmq_pub.close()
        env.close()
        agent.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
