"""Jaka MF whole-body policy (in-process).

Ported from sim2real-jaka ``rl_policy/base_policy.py``. Runs the control-mode
state machine, builds the MF observation (620-dim v1), runs the ONNX policy,
and writes the resulting PD position target to an in-process bridge.

The policy is *stateful*: it owns the observation history stacks and the motion
window. In SIMPLE it is driven at 50 Hz by ``JakaMFAgent.get_action()``
(which calls ``step()``), while the robot applies PD torque at 500 Hz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Type

import numpy as np
import yaml
from loguru import logger

from simple.jaka_rl.action_manager import ActionManager
from simple.jaka_rl.config import (
    DEFAULT_QPOS,
    JOINT_KD,
    JOINT_KP,
    JOINT_POS_LOWER_LIMIT,
    JOINT_POS_UPPER_LIMIT,
)
from simple.jaka_rl.controllers import ControllerBase, KeyboardController, PicoController
from simple.jaka_rl.inference import build_inference_module
from simple.jaka_rl.observations import Observation, ObsGroup
from simple.jaka_rl.state_processor import StateProcessor
from simple.jaka_rl.strings import resolve_matching_names_values


class BasePolicy:
    def __init__(
        self,
        args: "BasePolicyArgs",
        *,
        state_getter=None,
        bridge=None,
        mj_model=None,
        mj_data=None,
        default_qpos=None,
    ):
        self.args = args
        with open(args.policy_config) as file:
            policy_config = yaml.load(file, Loader=yaml.FullLoader)
        policy_config = self.prepare_policy_config(policy_config)
        model_path = args.policy_model or args.policy_config.replace(".yaml", ".onnx")
        self.policy_config = policy_config
        self.model_path = model_path
        self.default_qpos = np.asarray(DEFAULT_QPOS, dtype=np.float32)
        # initialize robot related processes
        self.joint_names_simulation = list(policy_config.get("joint_names_simulation"))
        self.body_names_simulation = list(policy_config.get("body_names_simulation"))
        self.state_processor = StateProcessor(
            self.joint_names_simulation,
            policy_config,
            body_names=self.body_names_simulation,
            state_getter=state_getter,
            mj_model=mj_model,
            mj_data=mj_data,
            default_qpos=default_qpos,
        )
        self.state_processor.env = self
        self.action_manager = ActionManager(
            self.state_processor.joint_names,
            policy_config,
            bridge=bridge,
        )
        self.rl_dt = 1.0 / float(args.rl_rate)
        self.inference_backend = args.inference_backend

        self.num_dofs = len(self.joint_names_simulation)

        default_joint_pos_dict = policy_config.get("default_joint_pos", {})
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.action_manager.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.joint_names_simulation))
        self.default_dof_angles[joint_indices] = default_joint_pos

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.action_manager.joint_names.index(name)
            for name in self.policy_joint_names
        ]

        action_scale_cfg = policy_config.get("action_scale", 0.5)
        self.action_scale = np.ones((self.num_actions))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.policy_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        elif isinstance(action_scale_cfg, list):
            if len(action_scale_cfg) != self.num_actions:
                raise ValueError(
                    f"Action scale list length {len(action_scale_cfg)} does not "
                    f"match num actions {self.num_actions}"
                )
            self.action_scale[:] = np.array(action_scale_cfg)
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.init_count = 0
        self.align_count = 0
        self.align_target_joint_pos = self.default_dof_angles.copy()
        self.perf_dict: Dict[str, float] = {}
        self.key_pressed: set[str] = set()
        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }

        joint_indices, joint_names, joint_pos_lower_limit = resolve_matching_names_values(
            JOINT_POS_LOWER_LIMIT,
            self.joint_names_simulation,
            preserve_order=True,
            strict=False,
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = resolve_matching_names_values(
            JOINT_POS_UPPER_LIMIT,
            self.joint_names_simulation,
            preserve_order=True,
            strict=False,
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        self.controller_type = args.controller
        self.controller = self._build_controller()
        self.use_joystick = self.controller_type == "joystick"
        self.wc_msg = None
        self.need_reset_simulation = False

        self.setup_policy(model_path)
        self.setup_observations(policy_config.get("observation"))

    def prepare_policy_config(self, policy_config):
        """Inject runtime motion-backend options into the yaml config.

        Mirrors sim2real's ``Tracking`` subclass: applies ``motion_backend``,
        trims ``future_steps`` to ``max_future``, and fills ZMQ connect/hwm/
        dt/tolerance when running the live ``zmq`` backend.
        """
        from copy import deepcopy

        policy_config = deepcopy(policy_config)
        motion_cfg = policy_config.setdefault("motion", {})

        mb = self.args.motion_backend
        if mb is not None:
            motion_cfg["motion_backend"] = mb
        if self.args.max_future is not None and "future_steps" in motion_cfg:
            max_future = int(self.args.max_future)
            original = [int(s) for s in motion_cfg["future_steps"]]
            trimmed = [s for s in original if s <= max_future]
            if trimmed != original:
                motion_cfg["future_steps"] = trimmed
                logger.info(
                    "Trimmed motion.future_steps with max_future=%d from %s to %s",
                    max_future, original, trimmed,
                )
        if mb == "zmq":
            motion_cfg["motion_zmq_connect"] = self.args.motion_zmq_connect
            motion_cfg["motion_zmq_hwm"] = self.args.motion_zmq_hwm
            motion_cfg["motion_dt_s"] = 1.0 / float(self.args.rl_rate)
            motion_cfg["motion_tolerance_s"] = self.args.motion_tolerance_s
        return policy_config

    def _build_controller(self) -> ControllerBase:
        self.keyboard_controller = None
        self.joystick_controller = None
        self.pico_controller = None

        controller_type = self.controller_type
        if controller_type == "keyboard":
            print("Using keyboard")
            self.keyboard_controller = KeyboardController()
            self.key_pressed = self.keyboard_controller.key_pressed
            return self.keyboard_controller
        if controller_type == "pico":
            self.pico_controller = PicoController(connect=self.args.pico_zmq_connect)
            return self.pico_controller
        raise ValueError(f"Unsupported controller_type: {controller_type}")

    def setup_policy(self, model_path):
        runtime_module = build_inference_module(model_path, self.inference_backend)
        runtime_label = self.inference_backend
        logger.info("Using policy inference backend {}", runtime_label)

        action_clip = self.policy_config.get("action_clip", None)

        def policy(input_dict):
            output_dict = runtime_module(input_dict)
            if "action" in output_dict:
                action = np.asarray(output_dict["action"], dtype=np.float32)
            elif "actions" in output_dict:
                action = np.asarray(output_dict["actions"], dtype=np.float32)
            else:
                raise KeyError(
                    f"Expected 'action' or 'actions' in output, got: "
                    f"{list(output_dict.keys())}"
                )
            next_state_dict = {
                k[1]: v
                for k, v in output_dict.items()
                if isinstance(k, tuple) and len(k) == 2 and k[0] == "next"
            }
            input_dict.update(next_state_dict)

            if action_clip is not None:
                action = np.clip(action, -action_clip, action_clip)

            q_target = self.default_dof_angles.copy()
            q_target[self.controlled_joint_indices] += action * self.action_scale

            return action, q_target, input_dict

        self.policy = policy

        # Warmup the model (mirrors sim2real reference): run a few zero-input
        # forward passes so the first real `]`(policy) step isn't cold.
        logger.info("Warming up policy model...")
        try:
            warmup_dict = {}
            for key, shape in zip(runtime_module.in_keys, runtime_module.input_shapes):
                warmup_dict[key] = np.zeros(shape, dtype=np.float32)
            for _ in range(5):
                _ = runtime_module(warmup_dict)
            logger.info("Warmup complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to warmup policy model: {exc}")

    def setup_observations(self, obs_cfg):
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []

        self.reset_callbacks.append(self.state_processor.reset)
        self.update_callbacks.append(self.state_processor.update)

        for obs_group, obs_items in (obs_cfg or {}).items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                print(f"\t{obs_name}: {obs_config}")
                obs_config = dict(obs_config)
                obs_key = obs_config.pop("_target_", obs_name)
                if "." in obs_key:
                    obs_key = obs_key.split(".")[-1]
                obs_class: Type[Observation] = Observation.registry[obs_key]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        self.state_dict["paused"] = True
        [reset_callback() for reset_callback in self.reset_callbacks]

    def update(self):
        [update_callback(self.state_dict) for update_callback in self.update_callbacks]

    def prepare_obs_for_rl(self):
        obs_dict: Dict[str, np.ndarray] = {}
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs.astype(np.float32)
        return obs_dict

    def get_align_target(self):
        if self.align_count > 100:
            self.align_count = 100
        dof_pos = self.state_processor.joint_pos
        progress = self.align_count / 100
        q_target = dof_pos + (self.align_target_joint_pos - dof_pos) * progress
        self.align_count += 1
        return q_target

    def get_init_target(self):
        if self.init_count > 100:
            self.init_count = 100
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 100
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def set_init_mode(self, *, source: str) -> None:
        self.init_count = 0
        self.state_dict["control_mode"] = "init"
        logger.info(f"Control mode set to init via {source}")

    def set_align_mode(self, *, source: str) -> None:
        self.align_count = 0
        self.state_processor.reset()

        # Prefer the deterministic motion first frame (motion_frame0, JAKA order)
        # so `[` aligns to frame0 even if the live zmq buffer has advanced.
        mf0 = getattr(self.state_processor, "motion_frame0", None)
        if mf0 is not None:
            self.align_target_joint_pos = mf0["joint_pos"].copy()
            self.state_dict["control_mode"] = "align"
            self.need_reset_simulation = True
            logger.info(f"Control mode set to align via {source} (motion frame0)")
            return

        # Fallback: current motion window (live stream without an npz).
        if self.state_processor.motion_data is not None:
            try:
                idx_zero = list(self.state_processor.motion_future_steps).index(0)
                motion_joint_pos = self.state_processor.motion_data.joint_pos[0, idx_zero]

                motion_joint_names = list(self.state_processor.motion_joint_names)
                sim_joint_names = list(self.state_processor.joint_names)
                align_target = self.default_dof_angles.copy()
                for i, sim_name in enumerate(sim_joint_names):
                    if sim_name in motion_joint_names:
                        motion_idx = motion_joint_names.index(sim_name)
                        if motion_idx < len(motion_joint_pos):
                            align_target[i] = motion_joint_pos[motion_idx]
                self.align_target_joint_pos = align_target
            except Exception as exc:
                logger.warning(
                    f"Failed to get reference motion first frame joints: {exc}. "
                    "Falling back to default pose."
                )
                self.align_target_joint_pos = self.default_dof_angles.copy()
        else:
            self.align_target_joint_pos = self.default_dof_angles.copy()

        self.state_dict["control_mode"] = "align"
        self.need_reset_simulation = True
        logger.info(f"Control mode set to align via {source}")

    def set_zero_mode(self, *, source: str) -> None:
        self.state_dict["control_mode"] = "zero"
        logger.info(f"Control mode set to zero via {source}")

    def set_policy_mode(self, *, source: str) -> None:
        self.reset()
        self.state_dict["control_mode"] = "policy"
        logger.info(f"Control mode set to policy via {source}")

    def toggle_paused(self, *, source: str) -> None:
        """Toggle playback of the reference motion (mirrors sim2real Tracking)."""
        paused = not bool(self.state_dict.get("paused", False))
        self.state_dict["paused"] = paused
        logger.info(f"Paused state toggled to {paused} via {source}")

    def process_controllers(self) -> None:
        if self.joystick_controller is not None:
            self.wc_msg = self.joystick_controller.state
        mode = self.controller.get_control_mode()
        if mode == "policy":
            self.set_policy_mode(source=self.controller.name)
        elif mode == "zero":
            self.set_zero_mode(source=self.controller.name)
        elif mode == "init":
            self.set_init_mode(source=self.controller.name)
        elif mode == "align":
            self.set_align_mode(source=self.controller.name)

        # Space toggles reference playback (matches sim2real Tracking).
        extra_keys = self.controller.get_extra_keys()
        if extra_keys and isinstance(self.controller, KeyboardController):
            if "space" in extra_keys:
                self.toggle_paused(source="keyboard:space")

    def step(self):
        # 1. read low state (in-process)
        if not self.state_processor._prepare_low_state():
            return None
        # 2. controllers → control mode
        self.process_controllers()
        # 3. observations
        self.update()
        obs_dict = self.prepare_obs_for_rl()
        self.state_dict.update(obs_dict)
        self.state_dict["is_init"] = np.zeros(1, dtype=bool)
        # 4. policy inference — only in policy mode (mirrors the sim2real
        # reference, which zeroes action/q_target for init/zero/align so the
        # ONNX never runs until `]` is pressed).
        control_mode = self.state_dict.get("control_mode", "zero")
        if control_mode == "policy":
            try:
                action, q_target, self.state_dict = self.policy(self.state_dict)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.warning(f"Error in policy inference: {e}")
                self.state_dict["action"] = np.zeros(self.num_actions)
                return self.state_dict
            self.state_dict["action"] = action
            self.state_dict["q_target"] = q_target
        else:
            self.state_dict["action"] = np.zeros(self.num_actions, dtype=np.float32)
            self.state_dict["q_target"] = self.default_dof_angles.copy()
        # 5. rule-based control flow → select q target
        if control_mode == "init":
            q_target = self.get_init_target()
        elif control_mode == "align":
            q_target = self.get_align_target()
        elif control_mode == "zero":
            q_target = self.state_processor.joint_pos
        elif control_mode == "policy":
            q_target = self.state_dict["q_target"]
        else:
            raise ValueError(f"Invalid control mode: {control_mode}")

        cmd_q = q_target
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

        reset_qpos = None
        reset_qvel = None
        if getattr(self, "need_reset_simulation", False):
            self.need_reset_simulation = False
            if self.state_processor.motion_data is not None:
                try:
                    motion_data = self.state_processor.motion_data
                    # Base pos/quat from the deterministic motion frame0 (if the
                    # npz is available), else the motion data's first body frame.
                    mf0 = getattr(self.state_processor, "motion_frame0", None)
                    if mf0 is not None:
                        base_pos = mf0["base_pos"]
                        base_quat = mf0["base_quat"]
                    else:
                        base_pos = motion_data.body_pos_w[0, 0, 0]
                        base_quat = motion_data.body_quat_w[0, 0, 0]

                    reset_qpos = np.zeros(self.num_dofs + 7, dtype=np.float32)
                    reset_qpos[:3] = base_pos
                    reset_qpos[3:7] = base_quat
                    reset_qpos[7:] = self.align_target_joint_pos

                    reset_qvel = np.zeros(self.num_dofs + 6, dtype=np.float32)
                    logger.info(
                        f"Simulation reset requested: pos={base_pos}, quat={base_quat}"
                    )
                except Exception as exc:
                    logger.warning(f"Failed to prepare reset pose: {exc}")

        self.action_manager.send_command(
            np.asarray(cmd_q, dtype=np.float32),
            np.asarray(cmd_dq, dtype=np.float32),
            np.asarray(cmd_tau, dtype=np.float32),
            reset_qpos,
            reset_qvel,
        )
        return self.state_dict

    def close(self):
        try:
            self.controller.close()
        except Exception:
            pass


@dataclass
class BasePolicyArgs:
    """Robot."""

    policy_config: str = ""
    policy_model: str = ""
    robot: str = "jaka"
    rl_rate: float = 50.0
    inference_backend: Literal["onnx-gpu", "onnx-cpu", "tensorrt"] = "onnx-cpu"
    controller: Literal["keyboard", "joystick", "pico"] = "keyboard"
    pico_zmq_connect: str = "tcp://127.0.0.1:5592"
    motion_backend: Literal["npz", "zmq", "raw_npz"] | None = None
    max_future: int | None = None
    motion_zmq_connect: str = "tcp://127.0.0.1:28701"
    motion_zmq_hwm: int = 1
    motion_tolerance_s: float = 0.04


__all__ = ["BasePolicy", "BasePolicyArgs"]
