#!/usr/bin/env python3
"""VLA 参考流发送端仿真:用录制数据复现 openpi-eval 的下发行为(→ :28701)。

用**录制的 teleop 数据**(``record_jaka_zmq`` 落盘的 parquet)喂进 ``zmq_vla``
参考缓冲链路,用来在**没有 VLA 服务器、没有机器人**的情况下把整条链路跑起来。

为什么需要它
------------
``zmq_vla`` 后端的语义(见 ``docs/openhlm_vla/implementation_analysis.md``)只在
"30Hz 下发 + 周期性阻塞式推理停顿"这个特定数据模式下才可验证。真实链路里那个
停顿来自 π0.5 推理(约 120ms,期间一帧都不发),本机没有 VLA 服务器,所以这里用
录制数据把**下发模式**重放出来。

复现的下发逻辑(= ``openpi-eval/main.py`` 主循环 + ``docs/openhlm_vla/jaka_tabletop_env.py``):
  1. 30Hz 控制拍(``--control-hz``),每拍把一个 33 维策略 action 变成一帧协议 v1 数据;
  2. **每 ``--open-loop-horizon``(=25)拍重推理一次**,那一拍阻塞 ``--inference-ms``
     (默认 120ms)且**一帧都不发** —— 这就是参考缓冲要处理的"停顿";
  3. 2 帧滑动窗口 ``[i-1, i]`` 打包成一条 ZMQ 消息 PUB 出去,``frame_index`` 是
     **发送计数器**(全局单调,与数据内容无关,回放循环也不会归零)。

⚠️ 因此**实际下发速率低于 ``--control-hz``**:每 25 帧里有 1 拍要阻塞推理时间,那张单子
算进帧率里 —— ``25 / (24×33.3ms + 120ms) ≈ 27.2 Hz``。``openpi-eval`` 的循环结构完全
相同(推理拍那次 ``env.step()`` 花掉 120ms+ 后不再补 sleep),**真机也到不了 30 Hz**。
看到 ``[send] 27.3 Hz`` 是正常的,不是丢帧;每次运行结束会打印这个上限。

33 维 action 的构造(录制数据 → 策略 action)::

    action[0:27]  = 录制 actions[0:27] 按**名字**重排到 OpenPI 策略序(手臂在前)
    action[27:30] = 录制 actions[27:30]  (anchor roll 绝对值 / pitch 绝对值 / yaw_vel)
    action[30:33] = 录制 actions[30:33]  (anchor 机体系线速度)

⚠️ ``actions[0:27]`` 的关节顺序**随数据集而变**,所以是按 ``meta/info.json`` 里的
``features.actions.names`` **按名字**判的,不靠位置假设:

* ``data/simple/JakaTabletopPickTeleop-v0/level-0``(合并后的规范集)——
  **OpenPI 策略序**(手臂在前),正是 VLA 的输出空间;
* ``data/teleop_jaka_mf/<run>/level-0``(``record_jaka_zmq`` 直出)——
  **JAKA/MuJoCo 序**(腿在前)。

搞错这个不会报错,只会让参考关节整体错位、机器人直接摔 —— 所以脚本启动时会把判定结果
打出来(``[data] actions 关节序: ...``)。

⚠️ 上面那份规范集是**处理后的数据、直接喂 VLA 训练**的 —— 所以它的 ``actions`` 就是
VLA 的输出空间,33 维 action **逐位等于 ``actions[:, 0:33]``**,不做任何重排(脚本会断言
这一点)。``actions[33:40]``(anchor 世界位姿)是 DEBUG 列,**VLA 不输出**。

anchor(``waist_yaw_Link``)世界位姿的来源见 ``--anchor-source``:

* ``recorded``(默认)—— 直接把 ``[33:40]`` 的 anchor 真值位姿发出去,参考轨迹 ===
  录制轨迹,没有积分漂移。但用了一条 VLA **产生不了**的通道,属理想化参考;
  **单纯验证缓冲时序时用这个**(排除漂移这个变量)。
* ``integrated`` —— 只用 VLA 真会输出的 ``[30:33]``,按 ``JakaTabletopEnv.step()``
  那样积分出 anchor 位姿,**与真实部署逐位一致**。代价是引入实测
  **mean 5.6 cm / max 8.0 cm** 的漂移(线速度按 pico 发布周期算出、却按 1/30 积分,
  且被 clip 到 ±2.0),脚本会把漂移量打出来。想知道"VLA 真会看到什么"就用这个。

用法
----
默认数据源是 ``data/simple/JakaTabletopPickTeleop-v0/level-0``(47 集,3.4~18.7s),
``--session`` 会递归找 ``episode_XXXXXX.parquet``,所以下面两种落盘布局都能直接指:

* ``<session>/data/chunk-000/episode_000000.parquet`` —— 规范数据集
* ``<session>/<run-id>/level-0/data/chunk-000/episode_000000.parquet`` —— ``record_jaka_zmq``

终端 A —— 本脚本(仿真 openpi-eval 下发)::

    python scripts/test_vla_motion_stream.py --episode 0

终端 B —— 接收端(``motion_backend`` 已设为 ``zmq_vla``)::

    python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 --headless

⚠️ **先后顺序很重要**:PUB 不缓存 —— 订阅者没连上之前发的消息直接丢弃。而 teleop 启动
到进入控制循环要约 15s,一段 5s 的数据在它连上之前就发完了。所以:

* **推荐:先 B 后 A**。等 B 打印出 ``[zmq] state ... Hz``(≈15s),再运行 A。
* 或**先 A 后 B**,但 A 要加 ``--loop``(数据循环回放,B 随时加入都能收到)。

另外 A 的 ``--hold-s``(默认 3s)决定数据放完后还保持多久,跑长实验就加大它。

延时/异常场景的扫法::

    ... --inference-ms 0                    # 无停顿的理想流(基线)
    ... --inference-ms 400                  # 停顿比 delay(120ms)还长(§7 "长停顿语义")
    ... --inference-ms 120 --inference-jitter-ms 60
    ... --drop-every 10                     # 每 10 条消息丢 1 条(验 2 帧滑动窗抗丢帧)
    ... --final-stall-s 5                   # VLA 服务挂掉:数据流彻底停住
    ... --anchor-source integrated          # 复现 openpi-eval 的积分路径 + 漂移报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import zmq

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from simple.jaka_rl.config import JOINT_NAMES  # noqa: E402
from simple.jaka_rl.motion_buffer import _decode_binary_v1  # noqa: E402

# ---------------------------------------------------------------------------
# 关节序:openpi-eval 的「策略序」(手臂在前) <-> SIMPLE 的 JAKA/MuJoCo 序(腿在前)
# ---------------------------------------------------------------------------

# ⚠️ 这是 **openpi-eval / OpenPI 的 action 顺序**(= jaka_tabletop_env.POLICY_JOINT_NAMES),
# 与 ``simple.jaka_rl.config.POLICY_JOINT_NAMES`` **不是一回事** —— 后者是 IsaacLab/npz
# 序(左右腿交替排列)。两者只共享 "policy order" 这个名字,别混用。
OPENPI_JOINT_NAMES = [
    # left arm (6)
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
    # right arm (6)
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
    # left leg (6)
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    # right leg (6)
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
    # waist (1), neck (2)
    "waist_yaw_joint", "Neck_yaw_joint", "Neck_pitch_joint",
]

# wire 上跑的是 SIM/JAKA 序 = config.JOINT_NAMES = jaka_tabletop_env.SIM_JOINT_NAMES
SIM_JOINT_NAMES = list(JOINT_NAMES)

# sim = policy[PERM_POLICY_TO_SIM] / policy = sim[PERM_SIM_TO_POLICY]
PERM_POLICY_TO_SIM = np.array(
    [OPENPI_JOINT_NAMES.index(n) for n in SIM_JOINT_NAMES], dtype=np.int64
)
PERM_SIM_TO_POLICY = np.argsort(PERM_POLICY_TO_SIM)
assert sorted(OPENPI_JOINT_NAMES) == sorted(SIM_JOINT_NAMES), "两套关节名不一致"
assert np.array_equal(
    PERM_SIM_TO_POLICY[PERM_POLICY_TO_SIM], np.arange(len(SIM_JOINT_NAMES))
), "关节序置换不可逆"

# 默认数据集:47 集真录 teleop 数据(3.4~18.7s),VLA 训练用的就是它
DEFAULT_SESSION = "data/simple/JakaTabletopPickTeleop-v0/level-0"

# ---------------------------------------------------------------------------
# 协议 v1 打包 —— 与 openpi-eval ``sonic_g1_env.pack_pose_message`` 逐字节等价
# (那份文件 import 了 gear_sonic,本机 import 不了,所以在此复制实现;
#  正确性由 ``check_wire_roundtrip()`` 用接收端的真解码器 ``_decode_binary_v1`` 校验)
# ---------------------------------------------------------------------------

_HEADER_SIZE = 1280
_DTYPE_NAMES = {
    np.float32: "f32",
    np.float64: "f64",
    np.int32: "i32",
    np.int64: "i64",
    np.bool_: "bool",
}

# 单条消息里的字段顺序(接收端按头里的顺序算偏移,所以顺序必须与头一致)
_WIRE_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "frame_index")


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 1) -> bytes:
    """[topic][1280B JSON 头(null 填充)][按头中字段顺序拼接的二进制负载]。"""
    fields, parts = [], []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_str = _DTYPE_NAMES.get(value.dtype.type)
        if dtype_str is None:
            value = value.astype(np.float32)
            dtype_str = "f32"
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        parts.append(value.tobytes())

    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > _HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header)} > {_HEADER_SIZE}")
    return topic.encode("utf-8") + header.ljust(_HEADER_SIZE, b"\x00") + b"".join(parts)


def pack_window(frames: list[dict], frame_indices: list[int]) -> bytes:
    """把滑动窗口里的若干帧按协议 v1 打包成一条消息。"""
    data = {
        "joint_pos": np.stack([f["joint_pos"] for f in frames], axis=0),    # (N,27) f32
        "joint_vel": np.stack([f["joint_vel"] for f in frames], axis=0),    # (N,27) f32
        "body_pos_w": np.stack([f["body_pos_w"] for f in frames], axis=0),  # (N,3)  f32
        "body_quat_w": np.stack([f["body_quat_w"] for f in frames], axis=0),  # (N,4) f32
        "frame_index": np.asarray(frame_indices, dtype=np.int64),           # (N,)   i64
    }
    assert tuple(data.keys()) == _WIRE_FIELDS
    return pack_pose_message(data, topic="pose", version=1)


# ---------------------------------------------------------------------------
# 四元数小工具(与 jaka_tabletop_env / sonic_g1_env 同语义)
# ---------------------------------------------------------------------------


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """内旋 xyz 欧拉角 → wxyz 四元数(scipy 约定,与 sonic_g1_env 一致)。"""
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


def quat_rotate_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``R(q) · v``(active rotation),对应 jaka_tabletop_env._quat_rotate_wxyz。"""
    w, x, y, z = (float(c) for c in quat_wxyz)
    qv = np.array([x, y, z], dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    t = 2.0 * np.cross(qv, vec)
    return (vec + w * t + np.cross(qv, t)).astype(np.float32)


def yaw_of_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """wxyz 四元数的 yaw(弧度),用于给积分路径播种。"""
    w, x, y, z = (float(c) for c in quat_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# 录制数据读取
# ---------------------------------------------------------------------------


def read_dataset_joint_order(parquet_path: str) -> list[str] | None:
    """从数据集 ``meta/info.json`` 读 ``actions[0:27]`` 的关节名顺序。

    两种落盘格式的顺序**不一样**,必须按名字判、不能靠位置假设:

    * ``record_jaka_zmq`` 直出(``data/teleop_jaka_mf/<run>/level-0``)——
      **JAKA/MuJoCo 序**(腿在前),即 ``config.JOINT_NAMES``;
    * 合并后的规范数据集(``data/simple/JakaTabletopPickTeleop-v0/level-0``)——
      **OpenPI 策略序**(手臂在前),即 ``OPENPI_JOINT_NAMES``,
      也就是 VLA 实际输出的那个序。

    找不到 ``meta/info.json`` 时返回 None(调用方回退到 JAKA 序并告警)。
    """
    d = Path(parquet_path).resolve().parent
    for _ in range(4):  # chunk-000 → data → level-0 → 数据集根
        info = d / "meta" / "info.json"
        if info.exists():
            try:
                feats = json.loads(info.read_text()).get("features", {})
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 解析 {info} 失败({exc}),按 JAKA 序解释")
                return None
            for key in ("actions", "action"):
                names = (feats.get(key) or {}).get("names")
                if names and len(names) >= len(SIM_JOINT_NAMES):
                    return [str(n) for n in names[: len(SIM_JOINT_NAMES)]]
            return None
        if d.parent == d:
            break
        d = d.parent
    return None


class Episode:
    """一集录制数据里发送端要用的那几列。"""

    def __init__(self, path: str):
        import pyarrow.parquet as pq

        raw = np.asarray(
            pq.read_table(path, columns=["actions"]).column("actions").to_pylist(),
            dtype=np.float32,
        )
        if raw.ndim != 2 or raw.shape[1] < 40:
            raise SystemExit(f"[err] actions 列形状异常: {raw.shape},期望 (T, 40)")

        self.path = path
        self.num_frames = raw.shape[0]

        # actions[0:27] 的关节顺序:优先信数据集自己的 meta,其次按既定 JAKA 序
        order = read_dataset_joint_order(path)
        if order is None:
            order = SIM_JOINT_NAMES
            print("[warn] 没找到 meta/info.json —— 按 JAKA/MuJoCo 序(腿在前)解释 actions[0:27]")
        elif sorted(order) != sorted(SIM_JOINT_NAMES):
            raise SystemExit(
                f"[err] meta/info.json 里的 actions 关节名与预期不符\n"
                f"      拿到: {order[:6]} ...\n      预期是这 27 个的某个排列: {SIM_JOINT_NAMES[:6]} ..."
            )
        self.joint_order = list(order)
        # 数据集本身就是策略序(手臂在前)= VLA 训练用的规范集 —— 此时 33 维 action 就是
        # 录制 actions[:, 0:33] **原样**,零变换,这正是要模拟的 VLA 输出空间。
        self.is_vla_native = self.joint_order == list(OPENPI_JOINT_NAMES)
        self.order_label = (
            "OpenPI 策略序(手臂在前)= VLA 输出空间"
            if self.is_vla_native
            else "JAKA/MuJoCo 序(腿在前),喂 VLA 前需重排"
            if self.joint_order == SIM_JOINT_NAMES
            else "自定义序"
        )

        # 33 维策略 action = [27 关节(策略序)] + [roll, pitch, yaw_vel] + [anchor 机体系线速度]
        to_policy = [self.joint_order.index(n) for n in OPENPI_JOINT_NAMES]
        self.actions = np.empty((self.num_frames, 33), dtype=np.float32)
        self.actions[:, 0:27] = raw[:, 0:27][:, to_policy]
        self.actions[:, 27:33] = raw[:, 27:33]
        # 往返自检:策略序 → SIM 序 应还原成"原始列按 SIM 序排列"
        back = [OPENPI_JOINT_NAMES.index(n) for n in SIM_JOINT_NAMES]
        assert np.array_equal(
            self.actions[:, 0:27][:, back],
            raw[:, 0:27][:, [self.joint_order.index(n) for n in SIM_JOINT_NAMES]],
        ), "策略序 → SIM 序 往返不一致"
        if self.is_vla_native:
            # 钉住这个恒等关系:哪天有人在上面加了一步重排,这里立刻炸
            assert np.array_equal(self.actions, raw[:, 0:33]), (
                "策略序数据集的 33 维 action 必须等于录制 actions[:, 0:33] 原样"
            )

        # anchor 真值位姿(录制数据 §4 的 DEBUG 列,原始世界系)
        self.anchor_pos_w = raw[:, 33:36].copy()
        self.anchor_quat_w = raw[:, 36:40].copy()

        # pico 未就绪时录到的全零 action 行(§4 已知坑),统计一下提醒使用者
        self.zero_rows = int(np.sum(np.abs(raw).sum(axis=1) == 0.0))

    def describe(self) -> str:
        z = self.anchor_pos_w[:, 2]
        return (
            f"{self.num_frames} 帧 @30Hz ≈ {self.num_frames / 30.0:.1f}s | "
            f"anchor z {z.min():.3f}..{z.max():.3f} m | 全零 action 行 {self.zero_rows}"
        )


def resolve_parquet(args) -> str:
    """``--parquet`` 直给,或 ``--session`` + ``--episode`` 拼。

    递归找 ``episode_XXXXXX.parquet``,这样下面两种落盘布局都能直接用:

    * ``<session>/data/chunk-000/episode_000000.parquet``
      —— 规范数据集(如 ``data/simple/JakaTabletopPickTeleop-v0/level-0``)
    * ``<session>/<run-id>/level-0/data/chunk-000/episode_000000.parquet``
      —— ``record_jaka_zmq`` 的 ``--save-dir <run 目录>`` 形态
    """
    if args.parquet:
        if not os.path.exists(args.parquet):
            raise SystemExit(f"[err] parquet 不存在: {args.parquet}")
        return args.parquet
    if not args.session:
        raise SystemExit("[err] 需要 --parquet 或 --session")
    if not os.path.isdir(args.session):
        raise SystemExit(f"[err] session 目录不存在: {args.session}")

    want = f"episode_{args.episode:06d}.parquet"
    hits = sorted(str(p) for p in Path(args.session).rglob(want))
    if not hits:
        avail = sorted(p.stem for p in Path(args.session).rglob("episode_*.parquet"))
        raise SystemExit(
            f"[err] {args.session} 下没找到 {want}\n"
            f"      可选: {', '.join(avail[:20])}{' ...' if len(avail) > 20 else ''}"
        )
    if len(hits) > 1:
        print(f"[data] 匹配到 {len(hits)} 个 {want},取第一个: {hits[0]}")
    return hits[0]


# ---------------------------------------------------------------------------
# 帧构造 —— 逐行对应 JakaTabletopEnv.step()
# ---------------------------------------------------------------------------


class FrameBuilder:
    """把 33 维策略 action 变成一帧协议 v1 数据。

    对应 ``docs/openhlm_vla/jaka_tabletop_env.py:267`` 的 ``step()``:发布的永远是
    **区间起点**的位姿(先出帧、再推进 anchor 积分状态)。
    """

    def __init__(self, dt, anchor_source, initial_anchor_pos, initial_anchor_rpy):
        self.dt = float(dt)
        self.anchor_source = anchor_source
        self._pos = np.asarray(initial_anchor_pos, dtype=np.float32).copy()
        self._yaw = float(initial_anchor_rpy[2])
        self._prev_joint_pos: np.ndarray | None = None
        self._drift: list[np.ndarray] = []

    def seed_from(self, anchor_pos_w, anchor_quat_w) -> None:
        """用录制首帧的 anchor 位姿给积分链播种。

        位置本身无所谓(策略只用相对位移),但 **yaw 必须播种**:积分路径用
        ``R(yaw_built)`` 把机体系线速度转回世界系,而录制线速度是在录制 anchor 的
        绝对 yaw 下算的 —— 不播种就会让整条位移轨迹凭空转一个 yaw 偏置。
        """
        if self.anchor_source != "integrated":
            return
        self._pos = np.asarray(anchor_pos_w, dtype=np.float32).copy()
        self._yaw = yaw_of_quat_wxyz(anchor_quat_w)

    def build(self, action33, anchor_pos_gt, anchor_quat_gt) -> dict:
        """返回一帧帧字典;副作用是推进内部 anchor 积分状态(供下一帧用)。"""
        # 1) 27 关节:策略序 → SIM 序
        joint_pos = np.asarray(action33[0:27], dtype=np.float32)[PERM_POLICY_TO_SIM]

        # 2) 关节速度:有限差分(接收端解析后丢弃,仅为与 openpi-eval 线格式对齐)
        if self._prev_joint_pos is None:
            joint_vel = np.zeros(27, dtype=np.float32)
        else:
            joint_vel = ((joint_pos - self._prev_joint_pos) / self.dt).astype(np.float32)
        self._prev_joint_pos = joint_pos.copy()

        # 3) anchor 姿态:roll/pitch 取绝对值,yaw 由 yaw_vel 累积
        roll = float(action33[27])
        pitch = float(action33[28])
        yaw_vel = float(action33[29])
        quat_built = euler_xyz_to_quat_wxyz(roll, pitch, self._yaw)

        if self.anchor_source == "recorded":
            # 直接用录制真值 —— 参考轨迹 === 录制轨迹,无积分漂移
            body_pos_w = np.asarray(anchor_pos_gt, dtype=np.float32).copy()
            body_quat_w = np.asarray(anchor_quat_gt, dtype=np.float32).copy()
        else:
            # openpi-eval 原样路径:先按当前位姿出帧,随后再积分推进
            body_pos_w = self._pos.copy()
            body_quat_w = quat_built
            self._drift.append(
                body_pos_w - np.asarray(anchor_pos_gt, dtype=np.float32)
            )

        # 4) 推进 anchor 状态(下一帧用):机体系线速度先转进世界系再积分
        world_vel = quat_rotate_wxyz(quat_built, np.asarray(action33[30:33], np.float32))
        self._pos = (self._pos + world_vel * self.dt).astype(np.float32)
        self._yaw += yaw_vel * self.dt

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
        }

    def drift_summary(self) -> str | None:
        if not self._drift:
            return None
        n = np.linalg.norm(np.asarray(self._drift, dtype=np.float32), axis=1) * 100.0
        return (
            f"integrated anchor 相对录制真值的偏差: mean {n.mean():.2f} cm, "
            f"max {n.max():.2f} cm, 末帧 {n[-1]:.2f} cm"
        )


# ---------------------------------------------------------------------------
# 发送端:复现 openpi-eval 主循环
# ---------------------------------------------------------------------------


class Publisher:
    """30Hz 下发 + 2 帧滑动窗口 + 周期性阻塞停顿。"""

    def __init__(self, args):
        self.args = args
        self.window: deque = deque(maxlen=args.num_frames_to_send)
        self.frame_index = 0          # 发送计数器(全局单调,回放循环也不归零)
        self.sent_msgs = 0            # 真正发出去的消息数
        self.attempts = 0             # 组装出来的消息数(含被丢弃的)
        self.dropped = 0
        self.published_wall: list[tuple[int, float]] = []
        self._t0 = time.time()

        ctx = zmq.Context()
        self._ctx = ctx
        self.sock = ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(args.bind)

    def emit(self, frame: dict) -> None:
        """把一帧塞进滑动窗口;窗口满了就发一条消息。"""
        self.window.append(frame)
        self.published_wall.append((self.frame_index, time.time() - self._t0))
        self.frame_index += 1
        if len(self.window) < self.args.num_frames_to_send:
            return
        # 丢包模拟:整条消息不发。窗口里的旧帧会在下一条消息里重发,
        # 所以单条丢失不会造成帧空洞(接收端 frame_index 去重兜住)。
        # ⚠️ 计数用 attempts 而不是 sent_msgs —— 后者只在真发出去时自增,
        # 会卡在同一个值上导致"一直丢"。
        self.attempts += 1
        if self.args.drop_every and self.attempts % self.args.drop_every == 0:
            self.dropped += 1
            return
        idx = list(range(self.frame_index - len(self.window), self.frame_index))
        self.sock.send(pack_window(list(self.window), idx))
        self.sent_msgs += 1

    def close(self) -> None:
        self.sock.close(0)
        self._ctx.term()


def run_publish(args) -> int:
    parquet = resolve_parquet(args)
    ep = Episode(parquet)
    print(f"[data] {parquet}")
    print(f"[data] {ep.describe()}")
    print(f"[data] actions 关节序: {ep.order_label}  (按 meta/info.json 的名字判定)")
    if ep.is_vla_native:
        print(
            "[data] 33 维 policy action == 录制 actions[:, 0:33] 原样(逐位相等),"
            "即 VLA 的输出;debug 列 [33:40] 不参与发送"
        )
    if ep.zero_rows:
        print(
            f"[warn] 有 {ep.zero_rows} 行全零 action(pico 未就绪时录到的)",
            "会让参考瞬间塌到零位姿 —— 建议换一集"
            if args.anchor_source == "recorded"
            else "会让积分链塌到零位姿 —— 建议换一集或改用 --anchor-source recorded",
        )

    builder = FrameBuilder(
        dt=1.0 / args.control_hz,
        anchor_source=args.anchor_source,
        initial_anchor_pos=args.initial_anchor_pos,
        initial_anchor_rpy=args.initial_anchor_rpy,
    )
    builder.seed_from(ep.anchor_pos_w[0], ep.anchor_quat_w[0])

    if args.check_wire:
        check_wire_roundtrip(ep, args, builder)

    pub = Publisher(args)
    print(
        f"[send] ZMQ PUB bind {args.bind}  (control {args.control_hz:g} Hz, "
        f"窗口 {args.num_frames_to_send} 帧, anchor={args.anchor_source})"
    )

    # ZMQ 慢订阅者:SUB 还没连上时 PUB 发的消息会直接丢,所以先等一下再开流
    if args.connect_settle_s > 0:
        print(f"[send] 等待 {args.connect_settle_s:.1f}s 让订阅者连上(ZMQ slow-joiner)…")
        time.sleep(args.connect_settle_s)

    print_teleop_hint()

    control_dt = 1.0 / args.control_hz
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    msg_t0, msg_n0 = time.time(), 0
    step_idx = 0
    actions_from_chunk_completed = 0
    last_frame: dict | None = None

    try:
        while True:
            tick_start = time.time()

            # === 重推理的那一拍:阻塞,期间一帧都不发 ===
            if (
                actions_from_chunk_completed == 0
                or actions_from_chunk_completed >= args.open_loop_horizon
            ):
                pause = float(args.inference_ms)
                if args.inference_jitter_ms > 0:
                    pause = max(
                        0.0,
                        pause
                        + float(
                            rng.uniform(-args.inference_jitter_ms, args.inference_jitter_ms)
                        ),
                    )
                if pause > 0:
                    time.sleep(pause / 1000.0)
                actions_from_chunk_completed = 0

            if step_idx >= ep.num_frames:
                if not args.loop:
                    break
                step_idx = 0  # 循环回放:frame_index 仍单调,但 anchor 会跳一次

            last_frame = builder.build(
                ep.actions[step_idx], ep.anchor_pos_w[step_idx], ep.anchor_quat_w[step_idx]
            )
            pub.emit(last_frame)
            step_idx += 1
            actions_from_chunk_completed += 1

            # 维持控制频率(与 openpi-eval 一致:算上本拍已经花掉的时间)
            elapsed = time.time() - tick_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

            now = time.time()
            if now - msg_t0 >= 2.0:
                print(
                    f"[send] {(pub.sent_msgs - msg_n0) / (now - msg_t0):4.1f} Hz  "
                    f"msgs={pub.sent_msgs} frames={pub.frame_index} "
                    f"dropped={pub.dropped} step={step_idx}/{ep.num_frames}",
                    flush=True,
                )
                msg_t0, msg_n0 = now, pub.sent_msgs

            if args.max_seconds and now - t_start >= args.max_seconds:
                print(f"[send] 到达 --max-seconds={args.max_seconds:g},停止下发")
                break
    except KeyboardInterrupt:
        print("\n[send] Ctrl+C,停止下发")

    # === 数据放完:原地保持末帧,把最后 ~delay 的尾巴播完,进入稳态 ===
    if args.hold_s > 0 and last_frame is not None:
        print(f"[send] 保持末帧 {args.hold_s:.1f}s(参考应播完尾巴后位级冻结)")
        hold_until = time.time() + args.hold_s
        while time.time() < hold_until:
            tick_start = time.time()
            pub.emit(last_frame)
            elapsed = time.time() - tick_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    # === 模拟 VLA 服务挂掉:彻底停发 ===
    if args.final_stall_s > 0:
        print(
            f"[send] 模拟 VLA 服务挂掉:停发 {args.final_stall_s:.1f}s "
            f"(观察 stale_ms 上升、而参考应位级冻结)"
        )
        time.sleep(args.final_stall_s)

    print(f"\n[send] 结束: {pub.sent_msgs} 条消息 / {pub.frame_index} 帧 / 丢弃 {pub.dropped} 条")
    drift = builder.drift_summary()
    if drift:
        print(f"[send] {drift}")
    report_pause_stats(pub.published_wall, args)
    pub.close()
    return 0


def check_wire_roundtrip(ep: Episode, args, builder: FrameBuilder) -> None:
    """自检 1:关节序置换 + 本脚本打包的字节能否被接收端真解码器原样解回。

    ⚠️ 用一个**独立的** FrameBuilder,免得把待会儿真正要发的那条积分链推进掉。
    """
    probe = FrameBuilder(
        dt=1.0 / args.control_hz,
        anchor_source=args.anchor_source,
        initial_anchor_pos=args.initial_anchor_pos,
        initial_anchor_rpy=args.initial_anchor_rpy,
    )
    probe.seed_from(ep.anchor_pos_w[0], ep.anchor_quat_w[0])
    f = probe.build(ep.actions[0], ep.anchor_pos_w[0], ep.anchor_quat_w[0])

    idx = [11, 12]
    msg = pack_window([f, f], idx)
    jp, bp, bq, fi = _decode_binary_v1(msg)

    checks = {
        "joint_pos": np.array_equal(jp, np.stack([f["joint_pos"]] * 2)),
        "body_pos_w": np.array_equal(bp, np.stack([f["body_pos_w"]] * 2)),
        "body_quat_w": np.array_equal(bq, np.stack([f["body_quat_w"]] * 2)),
        "frame_index": np.array_equal(fi, np.asarray(idx, dtype=np.int64)),
        # 关节序:发出去的 SIM 序关节 == 录制首帧关节(SIM 序)
        "joint order": np.allclose(jp[0], ep.actions[0][0:27][PERM_POLICY_TO_SIM], atol=1e-6),
    }
    bad = [k for k, v in checks.items() if not v]
    size = len(msg)
    print(
        f"[wire] 打包/解码往返自检: {'OK' if not bad else '失败 ' + ','.join(bad)} "
        f"({size} B = topic 4 + header {_HEADER_SIZE} + payload "
        f"{size - 4 - _HEADER_SIZE})"
    )
    if bad:
        raise SystemExit(f"[err] 线格式/关节序自检失败: {bad}")


def effective_publish_hz(args) -> float | None:
    """阻塞推理把节拍吃掉一拍后的**有效下发速率上限**。

    每个 chunk 循环 = (N-1) 个正常拍 + 1 个推理拍;推理拍那一拍要花
    ``max(节拍, 推理时长)``(推理比节拍短就照常被节拍兜住,不额外扣),
    所以 25 帧 @30Hz + 120ms 推理 → ``25 / (24×33.3ms + 120ms) ≈ 27.2 Hz``。
    **openpi-eval 的循环结构完全相同**(推理拍那次 ``env.step()`` 花掉 120ms+ 后
    ``elapsed > target_dt`` 不再补 sleep),所以真机也到不了 30 Hz ——
    这正是 RealtimeMotionBufferVla 要处理的数据模式。
    """
    if args.inference_ms <= 0 or args.open_loop_horizon <= 1:
        return None
    tick_s = 1.0 / args.control_hz
    cycle_s = (args.open_loop_horizon - 1) * tick_s + max(
        tick_s, args.inference_ms / 1000.0
    )
    return args.open_loop_horizon / cycle_s


def report_pause_stats(published_wall: list, args) -> None:
    """从发送侧的帧时刻反推停顿长度,和设定值对账。"""
    if len(published_wall) < 4:
        return
    ts = np.array([t for _, t in published_wall], dtype=np.float64)
    nominal_ms = 1000.0 / args.control_hz
    big = np.diff(ts) * 1000.0
    big = big[big > 3 * nominal_ms]
    if big.size:
        print(
            f"[send] 停顿事件 {big.size} 次,下发间隔 {big.min():.0f}..{big.max():.0f} ms "
            f"(设定 推理 {args.inference_ms:g} ms + 1 拍 {nominal_ms:.0f} ms)"
        )
    else:
        print("[send] 未观察到 >3 拍的下发间隔(本次流里没有停顿)")

    eff = effective_publish_hz(args)
    if eff is not None:
        print(
            f"[send] 有效下发速率上限 ≈ {eff:.1f} Hz"
            f" ({args.open_loop_horizon} 帧 / ({(args.open_loop_horizon - 1)}×{nominal_ms:.1f}ms"
            f" + {args.inference_ms:g}ms 推理))"
            f" —— 低于 {args.control_hz:g} Hz 是正常的:推理拍阻塞挤占了一整个节拍,"
            f"openpi-eval 同样如此"
        )


def print_teleop_hint() -> None:
    print(
        "\n[recv] 接收端(另开终端):\n"
        "       1) 把 data/jaka_mf/teleop_jaka_mf.yaml 的 motion_backend 改成 zmq_vla\n"
        "          (motion_zmq_connect 应保持 tcp://127.0.0.1:28701)\n"
        "       2) python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 "
        "--no-headless\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="仿真 openpi-eval 的 Jaka VLA 参考流下发(录制数据回放)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="用法与场景示例见脚本头部 docstring。",
    )

    g = p.add_argument_group("数据源")
    g.add_argument("--parquet", default="", help="直接给 episode parquet 路径")
    g.add_argument("--session", default=DEFAULT_SESSION,
                   help="数据集目录(会在其下递归找 episode_XXXXXX.parquet)")
    g.add_argument("--episode", type=int, default=0,
                   help=f"episode 序号(配合 --session;{DEFAULT_SESSION} 下有 47 集)")

    g = p.add_argument_group("下发模式(对标 openpi-eval)")
    g.add_argument("--bind", default="tcp://*:28701", help="action PUB 绑定地址")
    g.add_argument("--control-hz", type=float, default=30.0, help="下发频率")
    g.add_argument("--num-frames-to-send", type=int, default=2,
                   help="单条消息里的帧数(2 = 滑动窗口 [i-1,i])")
    g.add_argument("--open-loop-horizon", type=int, default=25,
                   help="每多少拍重推理一次(openpi-eval 默认 25)")
    g.add_argument("--inference-ms", type=float, default=120.0,
                   help="重推理的阻塞时长,期间一帧都不发")
    g.add_argument("--inference-jitter-ms", type=float, default=0.0, help="推理耗时抖动 ±ms")
    g.add_argument("--drop-every", type=int, default=0,
                   help="每 N 条消息丢 1 条(模拟丢帧,0=不丢)")
    g.add_argument("--loop", action="store_true", help="数据放完后从头循环回放")
    g.add_argument("--hold-s", type=float, default=3.0,
                   help="数据放完后原地保持末帧的秒数(让尾巴播完、进入稳态)")
    g.add_argument("--final-stall-s", type=float, default=0.0,
                   help="最后彻底停发这么多秒(模拟 VLA 服务挂掉)")
    g.add_argument("--max-seconds", type=float, default=0.0, help="总时长上限(0=不限)")
    g.add_argument("--connect-settle-s", type=float, default=1.0,
                   help="开流前等待订阅者连上的秒数(ZMQ slow-joiner)")
    g.add_argument("--seed", type=int, default=0, help="抖动/丢帧的随机种子")

    g = p.add_argument_group("anchor 位姿来源")
    g.add_argument("--anchor-source", choices=("recorded", "integrated"), default="recorded",
                   help="recorded=直接用录制 anchor 真值(参考轨迹===录制轨迹);"
                        "integrated=走 openpi-eval step() 的速度积分路径(有漂移)")
    g.add_argument("--initial-anchor-pos", type=float, nargs=3, default=(0.0, 0.0, 0.83),
                   help="integrated 模式的 anchor 初始世界位置(仅 z 有语义)")
    g.add_argument("--initial-anchor-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   help="integrated 模式的 anchor 初始 rpy")

    g = p.add_argument_group("杂项")
    g.add_argument("--check-wire", action="store_true", default=True,
                   help="发布前做一次协议打包/解码往返自检")
    g.add_argument("--no-check-wire", dest="check_wire", action="store_false")
    return p


def main(argv=None) -> int:
    return run_publish(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
