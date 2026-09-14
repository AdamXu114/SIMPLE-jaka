"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Jaka Khan Mini LeRobot-format dataset helpers for VLA training.

Shared by Stage 1 (teleop recording in run_jaka_sim_server.py) and
Stage 2 (replay + IsaacSim rendering in simple.cli.replay_jaka).

Uses the robot-agnostic ``Gr00tDataExporter`` from decoupled_wbc. No hands
on Jaka, so no ``observation.eef_state`` / ``action.eef`` features.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

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


def _patch_lerobot_save_episode_video_assert() -> None:
    """Tolerate LeRobot's post-save ``*.mp4`` count assert when there are no video keys.

    ``LeRobotDataset.save_episode`` ends with two sanity asserts::

        assert len(list(self.root.rglob("*.parquet"))) == self.num_episodes
        assert len(list(self.root.rglob("*.mp4"))) == (
            self.num_episodes - self.episodes_since_last_encoding
        ) * len(self.meta.video_keys)

    With ``use_videos=False`` — our head image is an ``image``-dtype feature whose PNG bytes
    are embedded in the parquet — ``meta.video_keys`` is empty, so the right-hand side is
    always 0. But :func:`finalize_openhlm_episode` deliberately keeps a browsable mp4 per
    episode under ``videos/``, so from the SECOND episode on the earlier episodes' mp4s get
    counted and the assert fires (the first episode passes only because its own mp4 is written
    *after* ``save_episode`` returns). That is not a data problem — the assert sits after the
    parquet write and the meta update, so the episode is fully on disk — but the exception
    kills the recorder, and the line it skips (resetting the episode buffer) would wedge the
    exporter for every later episode.

    This patch swallows exactly that case and performs the buffer reset the original would have
    done, re-raising everything else: a failing *parquet* count (a real integrity problem) and
    any assert raised while video keys DO exist (a real video-encoding problem). Drop it if the
    pipeline ever switches to ``use_videos=True``.
    """
    import logging

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    if getattr(lerobot_dataset, "_simple_video_assert_patched", False):
        return

    original_save_episode = lerobot_dataset.LeRobotDataset.save_episode
    warned = False

    def save_episode(self, episode_data: dict | None = None) -> None:
        nonlocal warned
        try:
            return original_save_episode(self, episode_data)
        except AssertionError:
            if list(self.meta.video_keys) or len(list(self.root.rglob("*.parquet"))) != self.num_episodes:
                raise  # a real problem: keep LeRobot's own error
            if not warned:  # expected from episode 1 on -- say it once, don't spam the log
                warned = True
                logging.getLogger(__name__).warning(
                    "save_episode: tolerating LeRobot's '*.mp4'-count sanity assert for the "
                    "rest of this session (use_videos=False, but a browsable mp4 per episode "
                    "lives under videos/); episodes are still fully written (parquet + meta + "
                    "episodes.jsonl + videos/*.mp4)."
                )
            if not episode_data:  # what the original does right after those asserts
                self.episode_buffer = self.create_episode_buffer()
            return None

    lerobot_dataset.LeRobotDataset.save_episode = save_episode
    lerobot_dataset._simple_video_assert_patched = True


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
        # front (3rd-person) camera has been dropped from recording; fall back to head.
        "observation.images.front_view": images.get("front_stereo_left", images["head_stereo_left"]),
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
# the model reads: one head image (per-episode mp4 under videos/), 30-dim float32 state +
# 40-dim actions, and a per-episode task string. The Jaka-replay helpers are intentionally kept
# above for replay & video workflows but are NOT used here.
#
# The head image is stored at the RAW camera resolution (no resize/pad baked in) — matching
# OpenHLM's record script, which saves the full-res frames and lets the training side resize.
# openpi's ``ResizeImages(h, w)`` calls ``image_tools.resize_with_pad`` (aspect-preserving
# + zero pad), NOT a stretch, so any stored aspect works; keeping full res avoids permanently
# throwing away resolution.
#
# ``record_jaka_zmq`` does NOT rely on this constant: it declares the shape it actually sees
# on the first camera frame (the resolution lives in the sim's camera cfg, e.g.
# ``tasks/jaka_vla_cameras.py``), so changing that cfg can't desync the dataset. This value is
# only the fallback default for direct callers of ``init_openhlm_exporter``.
OPENHLM_HEAD_IMAGE_SHAPE = [224, 224, 3]
# OpenHLM layouts (see also build_openhlm_features / build_openhlm_frame / teleop_jaka_mf):
#   state  (30) = [27 Jaka joints, root roll/pitch/yaw_vel]  — robot, read from MuJoCo.
#   action (40) = [27 Jaka joints, root roll/pitch/yaw_vel,
#                  anchor linear velocity xyz, anchor_pos_w xyz, anchor_quat_w wxyz]
#                 — pico/ZMQ reference ANCHOR (waist_yaw_Link, the MF tracker command frame).
#                 DEBUG: the RAW world-frame anchor pos/quat are appended verbatim so the
#                 verification can compare them directly against motion_mf (should be ~0
#                 deviation), and measure how far integrating yaw_vel drifts over 1 s.
#                 The absolute yaw is NOT stored separately — it is recoverable from the quat.
OPENHLM_ROOT_DIM = 3  # roll, pitch, yaw_vel (robot state root or reference base)
OPENHLM_LIN_VEL_DIM = 3  # reference anchor (waist_yaw_Link) linear velocity xyz (m/s, action only)
OPENHLM_ANCHOR_POSE_DIM = 7  # RAW world anchor pose: pos xyz (3) + quat wxyz (4) (action-only, DEBUG)
OPENHLM_STATE_DIM = len(JAKA_JOINT_NAMES) + OPENHLM_ROOT_DIM  # 30 = 27 dof + 3 rpy
OPENHLM_ACTION_DIM = (
    len(JAKA_JOINT_NAMES) + OPENHLM_ROOT_DIM + OPENHLM_LIN_VEL_DIM + OPENHLM_ANCHOR_POSE_DIM
)  # 40 = 27 dof + 3 rpy + 3 lin vel + 7 raw anchor pose


class OpenHLMRootVel:
    """Replicates gear_sonic ``QuatProcessor.process_output_yaw_vel`` -> [roll, pitch, yaw_vel].

    Stateful: feed each frame's base quaternion (wxyz) + timestamp_ns; returns 3-dim root —
    roll/pitch from the quat (xyz intrinsic euler), yaw_vel = wrapped per-frame yaw rate
    ``Δyaw / dt`` using the ACTUAL wall-clock frame interval (``timestamp_ns``). This is the true
    angular velocity (rad/s). Note: re-integrating it at the fixed control dt (1/fps) drifts
    ~2.6 deg/s because the record loop's real frame interval jitters around 50 Hz (mean ~0.0196s,
    not exactly 1/50). The absolute yaw is NOT returned — it is recoverable from the raw anchor
    quaternion that the action trunk also carries.
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
        else:
            dt = (timestamp_ns - self._prev_ts) * 1e-9 if self._prev_ts is not None else 0.0
            dyaw = (raw_yaw - self._prev_raw_yaw + np.pi) % (2 * np.pi) - np.pi  # 去掉 ±π 跳变
            self._yaw_vel_ema = dyaw / dt if dt > 1e-6 else 0.0
            self._prev_ts = timestamp_ns
            self._prev_raw_yaw = raw_yaw
        return np.array([roll, pitch, self._yaw_vel_ema], dtype=np.float32)


def _openhlm_resize_pad(image: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize to 224x224 with zero padding (resize_with_pad replica, fast cv2).

    NOTE: **no longer used by the record path.** ``build_openhlm_frame`` now stores the RAW
    camera image and the training side applies openpi's ``ResizeImages`` →
    ``image_tools.resize_with_pad`` itself (which would, for our 640x360 head camera, letterbox
    to 224x126 + 44% black bars). Kept as a fast, faithful ``resize_with_pad`` reference.

    The openpi ``resize_with_pad`` is PIL-based and costs ~5 ms/frame; this cv2 version (~0.3 ms)
    was what kept the record loop at 50 Hz. Output is RGB uint8 (224, 224, 3).
    """
    import cv2

    img = np.asarray(image, dtype=np.uint8)
    if img.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {img.shape}")
    h, w = img.shape[:2]
    scale = min(224.0 / h, 224.0 / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = img if (nh, nw) == (h, w) else cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224 - nh) // 2, (224 - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


_OPENHLM_STATE_NAMES = list(JAKA_JOINT_NAMES) + ["root_roll", "root_pitch", "yaw_vel"]
_OPENHLM_ACTION_NAMES = (
    list(JAKA_JOINT_NAMES)
    + ["root_roll", "root_pitch", "yaw_vel"]
    + ["anchor_lin_vel_x", "anchor_lin_vel_y", "anchor_lin_vel_z"]
    + ["anchor_pos_w_x", "anchor_pos_w_y", "anchor_pos_w_z"]  # DEBUG: 原始世界 pos
    + ["anchor_quat_w_w", "anchor_quat_w_x", "anchor_quat_w_y", "anchor_quat_w_z"]  # DEBUG: 原始世界 quat (含 yaw)
)


def build_openhlm_features(head_image_shape: Optional[Sequence[int]] = None) -> dict[str, dict]:
    """Feature spec the OpenHLM data loader expects (Jaka-adapted: head video + 30-dim state + 40-dim actions).

    ``head_image_shape`` overrides the declared HWC head-image shape (defaults to
    ``OPENHLM_HEAD_IMAGE_SHAPE``, the raw camera resolution). It MUST match what
    :func:`build_openhlm_frame` actually writes, or LeRobot's add_frame will reject it.

    ``dtype="image"`` keeps the frames embedded in the parquet (LeRobot's ``embed_images``
    reads the per-frame PNGs back into ``data/*.parquet``). ``record_jaka_zmq`` additionally
    writes a compact mp4 per episode under ``videos/`` and deletes the redundant
    ``images/`` PNG folder afterwards (see :func:`finalize_openhlm_episode`), so the dataset
    keeps the self-contained parquet *and* a browsable video — without two full copies of
    the frames.
    """
    shape = list(head_image_shape) if head_image_shape is not None else list(OPENHLM_HEAD_IMAGE_SHAPE)
    return {
        "head_image_left": {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (OPENHLM_STATE_DIM,), "names": _OPENHLM_STATE_NAMES},
        "actions": {"dtype": "float32", "shape": (OPENHLM_ACTION_DIM,), "names": _OPENHLM_ACTION_NAMES},
    }


def build_openhlm_frame(head_image: np.ndarray, state, actions) -> dict[str, np.ndarray]:
    """One OpenHLM frame: RAW head image + float32 state/actions.

    The head image is stored **as captured** (HWC uint8, no resize/pad) — the training side
    applies openpi's ``ResizeImages`` → ``resize_with_pad`` itself, so baking the letterbox in
    here would only throw away resolution. Shape must equal ``OPENHLM_HEAD_IMAGE_SHAPE``
    (override via :func:`build_openhlm_features` if the camera resolution differs).

    ``state``   (``OPENHLM_STATE_DIM``, 30)  = [27 Jaka joints, root roll/pitch/yaw_vel] (robot).
    ``actions`` (``OPENHLM_ACTION_DIM``, 40) = [27 joints, roll/pitch/yaw_vel, anchor lin vel xyz,
                raw anchor world pos xyz + quat wxyz] (reference).
    """
    return {
        "head_image_left": np.ascontiguousarray(np.asarray(head_image, dtype=np.uint8)),
        "state": np.asarray(state, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def reference_action_live_latest(
    motion_buffer,
    *,
    anchor_name: str = "waist_yaw_Link",
    neutral: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Live-ZMQ *record-time* reference action from a latest-frame motion source.

    ``motion_source`` may be a ``RealtimeMotionBuffer`` or the standalone
    ``record_jaka_zmq.LatestMotionClient`` (a direct, latest-only pico subscription) —
    anything exposing ``joint_names`` / ``body_names`` / ``get_latest_frame``. It reads the
    source's **latest** frame (no control-loop look-back) and reorders it to
    ``JAKA_JOINT_NAMES``. Returns::

        (ref_joints[27], anchor_quat_wxyz[4], anchor_lin_vel[3], anchor_pos_w[3])

    — the anchor is ``anchor_name`` (the MF tracker command frame, e.g. ``waist_yaw_Link``).
    Its linear velocity (m/s, current anchor frame) is the finite difference over the two most
    recent reference frames; ``anchor_pos_w`` is the **raw world-frame** anchor position from
    the pico stream (and ``anchor_quat_wxyz`` is likewise the raw world quat), saved verbatim
    for verification against motion_mf. Falls back to a neutral stand when no reference motion
    is available yet.
    """
    from simple.jaka_rl.math import quat_rotate_inverse_numpy

    if neutral is None:
        neutral = (
            np.zeros(len(JAKA_JOINT_NAMES), dtype=np.float32),
            np.array([1, 0, 0, 0], np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    body_names = list(motion_buffer.body_names)
    anchor_idx = body_names.index(anchor_name) if anchor_name in body_names else 0

    latest = motion_buffer.get_latest_frame()
    if latest is None:
        return neutral

    ref = np.asarray(latest.joint_pos, dtype=np.float32)
    mj_names = list(motion_buffer.joint_names)
    if list(mj_names) == list(JAKA_JOINT_NAMES):
        ref_joints = ref.copy().astype(np.float32)
    else:
        ref_joints = ref[[mj_names.index(n) for n in JAKA_JOINT_NAMES]].astype(np.float32)

    # Raw world-frame anchor pose (verbatim from the pico stream) + body-frame lin velocity.
    ref_anchor_pos_w = np.asarray(latest.body_pos_w[anchor_idx], dtype=np.float32)
    ref_anchor_quat = np.asarray(latest.body_quat_w[anchor_idx], dtype=np.float32)

    ref_anchor_lin_vel = np.zeros(3, dtype=np.float32)
    if latest.prev_body_pos_w is not None and latest.prev_timestamp_ns is not None:
        pos_prev = np.asarray(latest.prev_body_pos_w[anchor_idx], dtype=np.float32)
        dt = (float(latest.timestamp_ns) - float(latest.prev_timestamp_ns)) * 1e-9
        if dt > 1e-6:
            vel_body = quat_rotate_inverse_numpy(
                ref_anchor_quat[None, :], (ref_anchor_pos_w - pos_prev)[None, :]
            )[0]
            ref_anchor_lin_vel = np.clip(vel_body / np.float32(dt), -2.0, 2.0).astype(np.float32)
    return ref_joints, ref_anchor_quat, ref_anchor_lin_vel, ref_anchor_pos_w


def init_openhlm_exporter(
    save_root: str,
    fps: int = 50,
    head_image_shape: Optional[Sequence[int]] = None,
):
    """Create a lerobot LeRobotDataset with use_videos=False -> image bytes embedded in parquet.

    ``exporter.add_frame(frame, task=<task string>)`` then ``exporter.save_episode()`` — call
    :func:`finalize_openhlm_episode` after each ``save_episode()`` to also write the per-episode
    mp4 under ``videos/`` and drop the redundant ``images/`` PNG folder.
    ``head_image_shape`` (HWC) overrides the declared head-image shape — it must match the
    raw camera resolution actually written by :func:`build_openhlm_frame`.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_lerobot_offline()
    _patch_lerobot_save_episode_video_assert()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id="tmp/tmp_dataset",
        fps=fps,
        features=build_openhlm_features(head_image_shape),
        root=save_root,
        use_videos=False,  # keep the frames embedded in data/*.parquet (see build_openhlm_features)
    )


def finalize_openhlm_episode(exporter) -> Optional[str]:
    """After ``exporter.save_episode()``: encode the episode's PNGs to ``videos/...mp4`` and delete them.

    LeRobot writes every frame of an ``image``-dtype feature to
    ``images/<key>/episode_XXXXXX/*.png`` and only then embeds those bytes into
    ``data/*.parquet`` (``embed_images`` in ``_save_episode_table``) — so once ``save_episode``
    has returned, the parquet is self-contained and the PNG folder is a redundant second copy.

    This encodes that folder into the standard ``videos/chunk-XXX/<key>/episode_XXXXXX.mp4``
    and removes it. NOTE: the mp4 is a browsable companion only — LeRobot's reader serves this
    key from the parquet bytes (the key stays ``image``-dtype, so it is not registered in
    ``meta.video_keys``); do not rely on it instead of the parquet.

    Keeping those mp4s under the dataset root trips LeRobot's own post-save ``*.mp4``-count
    assert as soon as a second episode is saved — see
    :func:`_patch_lerobot_save_episode_video_assert`, which ``init_openhlm_exporter`` applies
    for exactly this reason.

    Returns the mp4 path, or ``None`` if there was nothing to encode.
    """
    import shutil
    from pathlib import Path

    from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
    from lerobot.datasets.video_utils import encode_video_frames

    root = Path(exporter.root)
    episode_index = int(exporter.meta.total_episodes) - 1  # the episode just saved
    chunk = episode_index // int(exporter.meta.chunks_size)
    written = None
    for key in exporter.meta.camera_keys:
        img_dir = exporter._get_image_file_path(episode_index, key, 0).parent
        if not img_dir.is_dir():
            continue
        video_path = root / DEFAULT_VIDEO_PATH.format(
            episode_chunk=chunk, video_key=key, episode_index=episode_index
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video_frames(img_dir, video_path, int(exporter.fps), overwrite=True)
        shutil.rmtree(img_dir)
        # Prune the now-empty images/<key>/ (and images/) so no image folder is left behind.
        parent = img_dir.parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        written = str(video_path)
    return written


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
    "finalize_openhlm_episode",
    "reference_action_live_latest",
]
