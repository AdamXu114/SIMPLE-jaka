"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka Khan Mini LeRobot-format dataset helpers for VLA training.

Shared by Stage 1 (teleop recording in run_jaka_sim_server.py) and
Stage 2 (replay + IsaacSim rendering in simple.cli.replay_jaka).

Uses the robot-agnostic ``Gr00tDataExporter`` from decoupled_wbc. No hands
on Jaka, so no ``observation.eef_state`` / ``action.eef`` features.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from simple.robots.jaka import ALL_JOINTS, Jaka

# 27 joints in MuJoCo order
JAKA_JOINT_NAMES: list[str] = list(ALL_JOINTS)

# 28 bodies in npz_body_names order (matches sim2real-jaka policy config).
# Used to record per-body pose/velocity so replay can rebuild motion_data.
JAKA_BODY_NAMES: list[str] = [
    "base_link", "Left_hip_pitch_Link", "Right_hip_pitch_Link", "waist_yaw_Link",
    "Left_hip_roll_Link", "Right_hip_roll_Link", "Left_shoulder_pitch_Link",
    "Right_shoulder_pitch_Link", "Neck_yaw_Link", "Left_hip_yaw_Link",
    "Right_hip_yaw_Link", "Left_shoulder_roll_Link", "Right_shoulder_roll_Link",
    "Neck_pitch_Link", "Left_knee_Link", "Right_knee_Link",
    "Left_shoulder_yaw_Link", "Right_shoulder_yaw_Link", "Left_ankle_pitch_Link",
    "Right_ankle_pitch_Link", "Left_elbow_Link", "Right_elbow_Link",
    "Left_ankle_roll_Link", "Right_ankle_roll_Link", "Left_wrist_roll_Link",
    "Right_wrist_roll_Link", "Left_wrist_yaw_Link", "Right_wrist_yaw__Link",
]
BODY_FEATURE_SUFFIXES = (
    "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z",
    "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
)

# camera image shape (H, W, C) — matches the 640x360 camera cfgs
VIDEO_SHAPE: list[int] = [360, 640, 3]

# Per-camera video shapes (both Realsense 640x360).
JAKA_VIDEO_SHAPES: dict[str, list[int]] = {
    "ego_view": [360, 640, 3],
    "front_view": [360, 640, 3],
}


def build_jaka_features(
    obj_names: Optional[Iterable[str]] = None,
    video_shape: list[int] = VIDEO_SHAPE,
    video_shapes: Optional[dict[str, list[int]]] = None,
    scene_qpos_shape: Optional[int] = None,
) -> dict[str, dict]:
    """Build the LeRobot feature spec for Jaka VLA data.

    Args:
        obj_names: Names of tracked objects (target + distractors). If given,
            adds ``observation.object_poses`` with shape (N*7,).
        video_shape: HWC shape for all video features (fallback).
        video_shapes: Per-camera HWC shapes, keyed by ego_view/front_view.
        scene_qpos_shape: Size of the non-robot MuJoCo qpos slice
            (``mjModel.nq - (7 + robot.dof)``). If given, adds
            ``observation.scene_qpos`` — the full scene state (object free
            joints + articulated base + hinge joints) so Stage-2 hard replay
            can snap objects to the recorded pose exactly.
    """
    shapes = video_shapes or {}
    features: dict[str, dict] = {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": shapes.get("ego_view", video_shape),
            "names": ["height", "width", "channel"],
        },
        "observation.images.front_view": {
            "dtype": "video",
            "shape": shapes.get("front_view", video_shape),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (27,),
            "names": JAKA_JOINT_NAMES,
        },
        "action": {
            "dtype": "float64",
            "shape": (27,),
            "names": JAKA_JOINT_NAMES,
        },
        "observation.base_pose": {
            "dtype": "float64",
            "shape": (7,),
            "names": ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"],
        },
        "observation.base_vel": {
            "dtype": "float64",
            "shape": (6,),
            "names": ["lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"],
        },
        "observation.body_poses": {
            "dtype": "float64",
            "shape": (len(JAKA_BODY_NAMES) * 13,),
            "names": [
                f"{b}.{s}" for b in JAKA_BODY_NAMES for s in BODY_FEATURE_SUFFIXES
            ],
        },
    }
    if obj_names:
        obj_names = list(obj_names)
        names = [f"{n}.{s}" for n in obj_names for s in
                 ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        features["observation.object_poses"] = {
            "dtype": "float64",
            "shape": (len(obj_names) * 7,),
            "names": names,
        }
    if scene_qpos_shape is not None and int(scene_qpos_shape) > 0:
        n = int(scene_qpos_shape)
        features["observation.scene_qpos"] = {
            "dtype": "float64",
            "shape": (n,),
            "names": [f"qpos_{i}" for i in range(n)],
        }
    return features


def build_jaka_modality_config(
    video_keys: tuple[str, ...] = ("ego_view", "front_view"),
) -> dict[str, dict]:
    """Build the modality config (state/action joint slices + video/annotation)."""
    return {
        "state": {
            "joints": {"start": 0, "end": 27, "original_key": "observation.state"},
            "base_pos": {"start": 0, "end": 3, "original_key": "observation.base_pose"},
            "base_quat": {
                "start": 3, "end": 7,
                "original_key": "observation.base_pose",
                "rotation_type": "quaternion",
            },
            "base_lin_vel": {"start": 0, "end": 3, "original_key": "observation.base_vel"},
            "base_ang_vel": {"start": 3, "end": 6, "original_key": "observation.base_vel"},
        },
        "action": {
            "joints": {"start": 0, "end": 27, "original_key": "action"},
        },
        "video": {
            k: {"original_key": f"observation.images.{k}"} for k in video_keys
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _patch_lerobot_offline() -> None:
    """Make lerobot's version check work without a HuggingFace Hub connection.

    ``LeRobotDatasetMetadata.create`` calls ``get_safe_version`` -> ``get_repo_versions``
    which always queries the Hub. In offline environments (no network) that raises.
    We monkeypatch ``get_safe_version`` to return the target revision directly.
    """
    import lerobot.datasets.utils as lerobot_utils

    if getattr(lerobot_utils, "_simple_offline_patched", False):
        return

    def _offline_safe_version(repo_id: str, version) -> str:
        import packaging.version
        v = (
            packaging.version.parse(version)
            if not isinstance(version, packaging.version.Version)
            else version
        )
        return f"v{v}"

    lerobot_utils.get_safe_version = _offline_safe_version
    lerobot_utils._simple_offline_patched = True


def init_jaka_exporter(
    save_dir: str,
    task_prompt: str,
    obj_names: Optional[Iterable[str]] = None,
    video_shape: list[int] = VIDEO_SHAPE,
    video_shapes: Optional[dict[str, list[int]]] = None,
    scene_qpos_shape: Optional[int] = None,
):
    """Create a Gr00tDataExporter configured for Jaka VLA data.

    Returns a ``Gr00tDataExporter`` instance (from third_party/decoupled_wbc).
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_lerobot_offline()

    from decoupled_wbc.data.exporter import Gr00tDataExporter

    return Gr00tDataExporter.create(
        save_root=save_dir,
        fps=50,
        features=build_jaka_features(
            obj_names, video_shape, video_shapes, scene_qpos_shape
        ),
        modality_config=build_jaka_modality_config(),
        task=task_prompt,
    )


def build_jaka_frame(
    images: Dict[str, np.ndarray],
    mj_data,
    action: np.ndarray,
    obj_names: Optional[Dict[str, Any]] = None,
    joint_qpos: Optional[np.ndarray] = None,
    mj_model=None,
) -> dict[str, np.ndarray]:
    """Assemble one LeRobot frame for Jaka.

    Args:
        images: dict from MujocoSimulator.render() / IsaacSimSimulator.render(),
            keyed by camera name (head_stereo_left, front_stereo_left).
        mj_data: MuJoCo MjData (for base pose/vel + object poses + body poses).
        action: 27-dim joint position targets (low_cmd q_target).
        obj_names: dict {obj_type: mj_body} for object poses (target, distractor_*).
        joint_qpos: 27-dim joint positions in JAKA_JOINT_NAMES order. If None,
            uses ``mj_data.qpos[7:34]`` (only valid in bare/robot-only scenes).
        mj_model: MuJoCo MjModel (needed to resolve body ids for body_poses).
    """
    import mujoco

    # Non-robot qpos slice: object free joints + articulated base + hinge joints.
    # Stored so Stage-2 hard replay can snap the whole scene (objects included)
    # to the recorded pose — not just the robot.
    n_joint = len(joint_qpos) if joint_qpos is not None else 27
    frame: dict[str, np.ndarray] = {
        "observation.images.ego_view": images["head_stereo_left"],
        "observation.images.front_view": images["front_stereo_left"],
        "observation.state": np.asarray(joint_qpos, dtype=np.float64)
        if joint_qpos is not None
        else np.asarray(mj_data.qpos[7:34], dtype=np.float64),
        "action": np.asarray(action, dtype=np.float64),
        "observation.base_pose": np.asarray(mj_data.qpos[:7], dtype=np.float64),
        "observation.base_vel": np.concatenate(
            [mj_data.qvel[:3], mj_data.qvel[3:6]]
        ).astype(np.float64),
        "observation.scene_qpos": np.asarray(
            mj_data.qpos[7 + n_joint:], dtype=np.float64
        ),
    }
    if mj_model is not None:
        body_vals = []
        for name in JAKA_BODY_NAMES:
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"body '{name}' not found in MuJoCo model")
            body_vals.append(np.concatenate([
                mj_data.xpos[bid], mj_data.xquat[bid],
                mj_data.cvel[bid, :3], mj_data.cvel[bid, 3:6],
            ]))
        frame["observation.body_poses"] = np.concatenate(body_vals).astype(np.float64)
    if obj_names:
        obj_poses = np.concatenate(
            [np.concatenate([mj_obj.xpos, mj_obj.xquat]) for mj_obj in obj_names.values()]
        ).astype(np.float64)
        frame["observation.object_poses"] = obj_poses
    return frame


def fix_hssd_room_pose(task) -> None:
    """Fix the HSSD room pose to the DR-range middle (yaw=0, centered offset).

    Robot/table/objects are laid out in a fixed MuJoCo world frame, but Isaac
    renders the room USD transformed by the DR-sampled ``center_offset`` /
    ``center_orientation``. If those stay random, the room is rotated/translated
    relative to the robot/objects in the render, so the two don't line up.

    Called after ``task.reset()`` on both the recording server and the replay so
    the rendered room stays fixed (yaw=0) and matches the world-frame robot.
    """
    dr = getattr(task, "dr", None)
    if dr is None:
        return
    scene_dr = dr.get_randomizer("scene")
    inner = getattr(scene_dr, "_inner_state", None)
    if inner is None or not hasattr(inner, "middle"):
        return
    inner.middle()  # center_offset/center_orientation → middle of DR ranges


def save_episode_env_config(exporter, task, episode_index: int) -> None:
    """Write the task state_dict as environment_config into meta/episodes.jsonl.

    Mirrors _save_episode_env_config in teleop_decoupled_wbc.py.
    """
    import json

    from simple.utils import NumpyArrayEncoder

    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


# ───────────────────── OpenHLM π0.5 (final train-consumed format) ─────────────────────
# The LeRobot dataset consumed directly by OpenHLM training. Unlike the Jaka-replay format
# (build_jaka_features / build_jaka_frame / init_jaka_exporter above), this carries ONLY what
# the model reads: one head image (per-frame PNG), 30-dim float32 state + 33-dim actions, and
# a per-episode task string. The Jaka-replay/mp4 helpers are intentionally kept above for
# replay & video workflows but are NOT used here.
OPENHLM_HEAD_IMAGE_SHAPE = [224, 224, 3]
# OpenHLM layouts (see also build_openhlm_features / build_openhlm_frame / teleop_jaka_mf):
#   state  (30) = [27 Jaka joints, root roll/pitch/yaw_vel]  — robot, read from MuJoCo.
#   action (33) = state layout + [pelvis linear velocity xyz] — pico/ZMQ reference base
#                 velocity (m/s, in the current base frame), computed per frame at collection.
OPENHLM_ROOT_DIM = 3  # roll, pitch, yaw_vel (robot state or reference base)
OPENHLM_LIN_VEL_DIM = 3  # reference pelvis linear velocity xyz (m/s, action only)
OPENHLM_STATE_DIM = len(JAKA_JOINT_NAMES) + OPENHLM_ROOT_DIM  # 30 = 27 dof + 3 rpy
OPENHLM_ACTION_DIM = OPENHLM_STATE_DIM + OPENHLM_LIN_VEL_DIM  # 33 = 27 dof + 3 rpy + 3 lin vel


class OpenHLMRootVel:
    """Replicates gear_sonic ``QuatProcessor.process_output_yaw_vel`` -> [roll, pitch, yaw_vel].

    Stateful: feed each frame's base quaternion (wxyz) + timestamp_ns; returns 3-dim root —
    roll/pitch from the quat (xyz intrinsic euler), yaw_vel = EMA'd yaw rate, clipped to +-2.
    """

    def __init__(self, yaw_vel_ema_alpha: float = 0.1):
        from scipy.spatial.transform import Rotation as _R

        self._R = _R
        self._alpha = float(yaw_vel_ema_alpha)
        self.reset()

    def reset(self) -> None:
        self._prev_raw_yaw: float | None = None
        self._prev_ts: int | None = None
        self._yaw_vel_ema: float | None = None

    def __call__(self, quat_wxyz: np.ndarray, timestamp_ns: int) -> np.ndarray:
        w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
        roll, pitch, raw_yaw = self._R.from_quat([x, y, z, w]).as_euler("xyz", degrees=False)
        if self._prev_raw_yaw is None:
            self._prev_ts = timestamp_ns
            self._prev_raw_yaw = raw_yaw
            self._yaw_vel_ema = 0.0
            yaw_vel = 0.0
        else:
            dt = (timestamp_ns - self._prev_ts) * 1e-9 if self._prev_ts is not None else 0.0
            raw = ((raw_yaw - self._prev_raw_yaw + np.pi) % (2 * np.pi) - np.pi) / dt if dt > 1e-6 else 0.0
            self._yaw_vel_ema = (
                raw
                if self._yaw_vel_ema is None
                else self._alpha * raw + (1 - self._alpha) * self._yaw_vel_ema
            )
            yaw_vel = float(np.clip(self._yaw_vel_ema, -2.0, 2.0))
            self._prev_ts = timestamp_ns
            self._prev_raw_yaw = raw_yaw
        return np.array([roll, pitch, yaw_vel], dtype=np.float32)


def _openhlm_resize_pad(image: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize to 224x224 with zero padding (tf.image.resize_with_pad replica)."""
    from openpi_client.image_tools import resize_with_pad

    img = np.asarray(image, dtype=np.uint8)[None, ...]
    return resize_with_pad(img, 224, 224)[0]


_OPENHLM_STATE_NAMES = list(JAKA_JOINT_NAMES) + ["root_roll", "root_pitch", "yaw_vel"]
_OPENHLM_ACTION_NAMES = _OPENHLM_STATE_NAMES + ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]


def build_openhlm_features() -> dict[str, dict]:
    """Feature spec the OpenHLM data loader expects (Jaka-adapted: head image + 30-dim state + 33-dim actions)."""
    return {
        "head_image_left": {
            "dtype": "image",
            "shape": OPENHLM_HEAD_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (OPENHLM_STATE_DIM,), "names": _OPENHLM_STATE_NAMES},
        "actions": {"dtype": "float32", "shape": (OPENHLM_ACTION_DIM,), "names": _OPENHLM_ACTION_NAMES},
    }


def build_openhlm_frame(head_image: np.ndarray, state, actions) -> dict[str, np.ndarray]:
    """One OpenHLM frame: padded 224x224 head image + float32 state/actions.

    ``state``   (``OPENHLM_STATE_DIM``, 30)  = [27 Jaka joints, root roll/pitch/yaw_vel] (robot).
    ``actions`` (``OPENHLM_ACTION_DIM``, 33) = state layout + [pelvis lin vel xyz] (reference).
    """
    return {
        "head_image_left": _openhlm_resize_pad(head_image),  # (224,224,3) uint8
        "state": np.asarray(state, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def init_openhlm_exporter(save_root: str, fps: int = 50):
    """Create a lerobot LeRobotDataset with use_videos=False -> images/ PNG (OpenHLM layout).

    ``exporter.add_frame(frame, task=<task string>)`` then ``exporter.save_episode()``.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_lerobot_offline()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id="tmp/tmp_dataset",
        fps=fps,
        features=build_openhlm_features(),
        root=save_root,
        use_videos=False,  # per-frame PNG under images/ (OpenHLM-consumed layout)
    )


__all__ = [
    "JAKA_JOINT_NAMES",
    "VIDEO_SHAPE",
    "JAKA_VIDEO_SHAPES",
    "build_jaka_features",
    "build_jaka_modality_config",
    "build_jaka_frame",
    "init_jaka_exporter",
    "save_episode_env_config",
    "fix_hssd_room_pose",
    "OPENHLM_HEAD_IMAGE_SHAPE",
    "OPENHLM_ROOT_DIM",
    "OPENHLM_LIN_VEL_DIM",
    "OPENHLM_STATE_DIM",
    "OPENHLM_ACTION_DIM",
    "OpenHLMRootVel",
    "build_openhlm_features",
    "build_openhlm_frame",
    "init_openhlm_exporter",
]
