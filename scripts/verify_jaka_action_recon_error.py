#!/usr/bin/env python3
"""校验录制好的 Jaka OpenHLM action trunk 的重建误差（一秒内）。

从录制的 ``[T, 33]`` action trunk 重建参考 anchor（``waist_yaw_Link``）的世界位姿轨迹，
然后与真值参考运动（``motion_mf.npz``）做对比。

流程（对应需求）:
  1. **随机抽取一些起始帧**:在当前录制区间 ``[min_t, n_frames-horizon]`` 里随机抽
     ``--num-samples`` 个起始帧 ``t0``。
  2. **按 dof 对齐时间戳**:对窗口内 **每一帧** 都用 27 个关节位置在参考运动里找最接近的那一帧
     ``m_idx``（关节位置是极强对齐签名，无歧义）。
  3. **取接下来 ``--horizon`` 帧（默认 50 = 1 秒）**:重建取 ``recon[t0:t0+H+1]``，参考取
     ``ref[m_idx]``（逐帧 dof 匹配，不用固定 ``m0+k`` 的 1:1 对应 —— 这样即使录制过程
     fps 掉帧，后段的参考帧也不会错位）。
  4. **把第一帧的 pos 和 rotation 对齐到一起**:把两条轨迹各自减去第 0 帧位置、并转进
     各自第 0 帧机体坐标系。这样两条轨迹"第一帧重合"了（约去 ``--start-pos`` 的常数
     平移、以及重建与参考之间的初始 yaw 差）。
  5. **在第 0 帧机器人参考系下比较后续位移和旋转**:位移用 ``R0^T·(pos[k]-pos0)`` 表达
     在机体系，再对逐帧 ``(recon-ref)`` 求模长；旋转用 ``conj(q0)·q[k]`` 得到相对第 0 帧
     的转角增量，再比较两者的增减。

报告三个基于"第一帧对齐"的指标:

  * ``POS 1s body``        -- 第 1 秒末,机体系位移 |recon_disp - ref_disp| (cm)。
  * ``POS max window``     -- 窗口内逐帧机体系位移误差(第 0 帧对齐后)的最大值 (cm)。
  * ``ROT 1s body``        -- 第 1 秒末,两机器人相对第 0 帧转角增量之间的夹角 (deg)。
  * ``YAW 1s body``        -- 其中 yaw 分量的误差 (deg),反映 yaw_vel 积分漂移。
  * ``DOF match resid``    -- dof 对齐时 27 关节距离 (cm)。≈0 说明对齐无歧义;若此时误差
    仍大,则不是错帧,而是重建本身在快速转动段的漂移 (yaw_vel 被 clamp)。

录制并保存一个 episode 后运行::

    python scripts/verify_jaka_action_recon_error.py
    python scripts/verify_jaka_action_recon_error.py --episode 000000 \
        --motion-npz data/motion/motion_mf.npz --num-samples 250 --seed 0 --plot-stats
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# action trunk 列下标（40 维 DEBUG 布局: dof(0:27) | roll(27) | pitch(28) | yaw_vel(29) |
#   lin_vel(30:33) | anchor_pos_w(33:36) | anchor_quat_w(36:40)）
_R_ROLL, _R_PITCH, _R_YAWVEL = 27, 28, 29
_R_POS_W = slice(33, 36)   # 原始世界系 anchor pos (x,y,z)
_R_QUAT_W = slice(36, 40)  # 原始世界系 anchor quat (w,x,y,z) —— 绝对 yaw 从这里取


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def heading_of(quat_wxyz: np.ndarray) -> float:
    """取 ``wxyz`` 四元数的偏航角 yaw（弧度），用于增量朝向指标。"""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap_angles(a) -> np.ndarray:
    """把弧度数组归一化到 (-pi, pi]。"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def quat_angle_deg(quat: np.ndarray) -> float:
    """单位 ``wxyz`` 四元数表示的旋转角（度），取绝对值 w。"""
    w = np.clip(abs(float(quat[0])), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(w)))


def load_action(parquet_path: str) -> np.ndarray:
    """读取录制的 ``actions`` 列，返回 float32 的 ``[T, 33]`` 数组。"""
    import pyarrow.parquet as pq

    actions = pq.read_table(parquet_path, columns=["actions"]).column("actions").to_pylist()
    return np.asarray(actions, dtype=np.float32)


def load_reference(motion_npz_path: str, anchor_body: str):
    """读取真值参考的 anchor 位姿 + 关节（已重排为 MuJoCo 顺序）。

    返回 ``(body_pos_w, body_quat_w, joint_jaka)``:
      - ``body_pos_w``  [M, 3]  anchor 世界位置,
      - ``body_quat_w`` [M, 4]  anchor 世界四元数(wxyz),
      - ``joint_jaka``  [M, 27] 参考关节，重排到 MuJoCo 顺序。
    """
    from simple.jaka_rl.config import JOINT_NAMES, NPZ_BODY_NAMES, NPZ_JOINT_NAMES

    npz = np.load(motion_npz_path)
    anchor_idx = NPZ_BODY_NAMES.index(anchor_body)

    body_pos_w = npz["body_pos_w"][:, anchor_idx].astype(np.float32)
    body_quat_w = npz["body_quat_w"][:, anchor_idx].astype(np.float32)

    # NPZ(IsaacLab/policy) 关节顺序 -> MuJoCo(JAKA) 顺序，以便与 action 的 dof 对齐。
    reindex = [NPZ_JOINT_NAMES.index(name) for name in JOINT_NAMES]
    joint_jaka = npz["joint_pos"][:, reindex].astype(np.float32)

    return body_pos_w, body_quat_w, joint_jaka


def nearest_reference_frame(dof_row: np.ndarray, motion_joint_jaka: np.ndarray) -> int:
    """在参考运动里找 27 个关节最接近 ``dof_row`` 的那一帧的下标。"""
    distances = np.linalg.norm(motion_joint_jaka - dof_row, axis=1)
    return int(np.argmin(distances))


def align_dof(dof_row: np.ndarray, motion_joint_jaka: np.ndarray) -> tuple[int, float]:
    """按 27 关节位置对齐，返回 ``(参考帧下标 m0, 该帧的关节距离 dmin)``。

    ``dmin`` 越小对齐越可信；它接近 0 说明 action 的关节特征在参考里能找到一个几乎
    完全一致的姿态——对齐是可靠的，此时如果后续误差仍很大，就**不是**错帧造成。
    """
    distances = np.linalg.norm(motion_joint_jaka - dof_row, axis=1)
    m0 = int(np.argmin(distances))
    return m0, float(distances[m0])


def align_dof_batch(dof_rows: np.ndarray, motion_joint_jaka: np.ndarray):
    """批量逐帧 dof 对齐：为每个 action 行找最近参考帧。

    Args:
        dof_rows: ``[W, 27]`` 一个窗口里的 dof 行。
        motion_joint_jaka: ``[M, 27]`` 参考关节。

    Returns:
        ``(m_idx[W] int, dmin[W] float)`` —— 每个 action 帧对应的参考帧下标及其关节距离。

    逐帧重找（而非固定 ``m0 + k``）：如果录制过程 fps 有波动，窗口后段的参考帧不再与
    action 帧 1:1 对应，固定 offset 会错位；逐帧匹配能把每个 action 帧对齐到它此刻真正
    的参考姿态。``dmin`` 逐帧给出，可用 ``max(dmin)`` 标记对齐不靠谱的样本。
    """
    dists = np.linalg.norm(motion_joint_jaka[None, :, :] - dof_rows[:, None, :], axis=2)  # [W, M]
    m_idx = np.argmin(dists, axis=1).astype(int)
    dmin = dists[np.arange(len(dof_rows)), m_idx]
    return m_idx, dmin


# --------------------------------------------------------------------------- #
# 核心：第一帧对齐后，在各自第一帧机体坐标系里比较
# --------------------------------------------------------------------------- #
def align_to_first_frame(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """把一段位姿轨迹表达在**第 0 帧机体坐标系**里。

    ``pos`` / ``quat`` 是 ``[w+1, 3]`` / ``[w+1, 4]``（帧序号 0..w）。返回 ``[w+1, 3]``
    的机体系位置：``R(q0)^T · (pos[k] - pos0)``。第 0 帧对应原点，初始朝向已被去掉。
    """
    from simple.jaka_rl.math import quat_rotate_inverse_numpy

    return quat_rotate_inverse_numpy(
        quat[0][None, :], pos - pos[0][None, :]
    )


def rel_rot_increment(quat: np.ndarray) -> np.ndarray:
    """第 0 帧到最后一帧的相对旋转 ``conj(q0)·q_w``（wxyz）。

    把旋转表达在"第 0 帧机体坐标系"，这样不同初始朝向下可公平地比较转角增量。
    """
    from simple.jaka_rl.math import quat_conjugate, quat_mul

    return quat_mul(quat_conjugate(quat[0]), quat[-1])


def compute_aligned_errors(t0, m_idx, dmin, horizon, recon_pos, recon_quat, ref_pos, ref_quat):
    """单个样本(窗口内逐帧 dof 对齐好) -> (pos_incr_cm, pos_max_cm, rot_incr_deg, yaw_incr_deg, dof_resid_max_cm)。\n"
             "\n"
             "两个窗口: 重建``recon[t0:t0+H+1]`` 与 参考``ref[m_idx]``(逐帧 dof 匹配到的参考帧)。\n"
             "各自先把第 0 帧对齐(减第 0 帧位置、转进第 0 帧机体坐标系),再逐帧比较位移与旋转:\n"
             "\n"
             "  - ``pos_incr``: 第 1 秒末,机体系恢复位移与参考位移之差 |-|。\n"
             "  - ``pos_max`` : 窗口内逐帧(第 0 帧对齐后)机体系位移误差的最大值。\n"
             "  - ``rot_incr``: 两机器人相对第 0 帧转角增量之间的夹角(度)。\n"
             "  - ``yaw_incr``: 该夹角的 yaw 分量误差(度,反映 yaw_vel 积分漂移)。\n"
             "  - ``dof_resid_max``: 窗口内逐帧 dof 对齐残差的最大值(米),标记对齐不靠谱的样本。"
    """
    from simple.jaka_rl.math import quat_mul, quat_conjugate, quat_rotate_inverse_numpy

    recon_win_pos = recon_pos[t0:t0 + horizon + 1]
    recon_win_quat = recon_quat[t0:t0 + horizon + 1]
    ref_win_pos = ref_pos[m_idx]     # 逐帧 dof 匹配 -> 参考窗口(时间轴不一定 1:1)
    ref_win_quat = ref_quat[m_idx]

    # —— 位置:各自在第 0 帧机体系里表达整段轨迹 ——
    rp = quat_rotate_inverse_numpy(
        recon_win_quat[0][None, :], recon_win_pos - recon_win_pos[0][None, :]
    )  # (H+1, 3)
    fp = quat_rotate_inverse_numpy(
        ref_win_quat[0][None, :], ref_win_pos - ref_win_pos[0][None, :]
    )  # (H+1, 3)

    pos_incr = float(np.linalg.norm(rp[horizon] - fp[horizon]))   # 第 1 秒末
    pos_max = float(np.max(np.linalg.norm(rp - fp, axis=1)))       # 窗口内最大

    # —— 旋转:相对第 0 帧的转角增量,再比较两者的偏差 ——
    recon_rot = quat_mul(quat_conjugate(recon_win_quat[0]), recon_win_quat[horizon])
    ref_rot = quat_mul(quat_conjugate(ref_win_quat[0]), ref_win_quat[horizon])
    err_q = quat_mul(quat_conjugate(recon_rot), ref_rot)
    rot_incr = quat_angle_deg(err_q)

    yaw_recon = heading_of(recon_rot)
    yaw_ref = heading_of(ref_rot)
    yaw_incr = abs(np.degrees(wrap_angles([yaw_recon - yaw_ref])[0]))

    dof_resid_max = float(np.max(dmin))  # 窗口内对齐残差最大值,标记不可靠样本
    return pos_incr, pos_max, rot_incr, yaw_incr, dof_resid_max


def compute_yaw_errors(t0, m_idx, horizon, actions, ref_quat, fps):
    """单一样本的 yaw 专项误差(度)。

      - ``saved_vs_ref``: **记录的 yaw(取自保存的原始 anchor quat 第 36:40 列)** vs
        **motion_mf 的 yaw**。参考 yaw 用 ``ref_quat[m_idx]`` 逐帧对齐取到。两者各按窗口起点
        归零后看最大 |Δ|。因为 quat 是直接保存(不积分),应当接近 0;若不为 0,说明记录时
        参考与 motion_mf 有偏差。绝对 yaw 不再单独存列,而是从 quat 现算(等价)。
      - ``yaw_vel_recon``: 用保存的 ``yaw_vel`` 从记录的 yaw 起点积分 1 秒,再和记录 yaw 比,
        取**窗口内最大 |Δ|**,反映整段的最大漂移。
      - ``yaw_vel_end``: **1 秒端点**处,积分 yaw 减记录 yaw 的**有符号**差(度)。正值=积分
        超了,负值=滞后。这个就是"积分 yaw 和 yaw 真实变化在 1s 后的差",用于看误差分布。
    """
    # 记录 yaw: 从保存的原始世界 quat 现算(与 motion_mf 同用 arctan2 heading, 完全等价)。
    yaw_sav = np.unwrap(_heading_array(actions[t0:t0 + horizon + 1, _R_QUAT_W].astype(np.float64)))
    yaw_ref = np.unwrap(_heading_array(ref_quat[m_idx]))                          # motion_mf yaw (rad)

    d = yaw_sav - yaw_ref
    d -= d[0]
    saved_vs_ref_deg = float(np.degrees(np.max(np.abs(d))))

    dt = 1.0 / float(fps)
    yaw_vel = actions[t0:t0 + horizon, _R_YAWVEL].astype(np.float64)  # rad/s
    yaw_int = yaw_sav[0] + np.concatenate([[0.0], np.cumsum(yaw_vel * dt)])
    yaw_vel_recon_deg = float(np.degrees(np.max(np.abs(yaw_int - yaw_sav))))
    yaw_vel_end_deg = float(np.degrees(yaw_int[-1] - yaw_sav[-1]))  # 1s 端点有符号差
    return saved_vs_ref_deg, yaw_vel_recon_deg, yaw_vel_end_deg


def _heading_array(quat_wxyz: np.ndarray) -> np.ndarray:
    """向量化取 ``(..., 4)`` wxyz 四元数的偏航角，返回 ``(...)`` 弧度数组。"""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def plot_stats(data: np.ndarray, fps: float, out_path: str) -> None:
    """画重建误差统计 dashboard（2x3 面板），保存到 ``out_path``。

    ``data`` 是 ``[n, 7]``，列 = ``(t0, m0, pos_incr, pos_max, rot_incr, yaw_incr, dof_resid)``。
    面板:
      * 0,0 ``POS 1s`` 分布直方图 (cm)
      * 0,1 ``ROT/YAW 1s`` 分布直方图 (deg)
      * 0,2 ``DOF 对齐残差`` 分布 (cm) —— 若集中≈0，说明 dof 对齐无歧义
      * 1,0 误差随时间(采样起点 t0)的散点 —— 离群点集中在一小段时间
      * 1,1 yaw 误差 vs rot 误差 散点 + 45° 参考线
      * 1,2 文本面板: 样本数 / 相关性 / 最差样本
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = data[:, 2] * 100.0        # cm
    rot = data[:, 4]                # deg
    yaw = data[:, 5]                # deg
    dof_resid = data[:, 6] * 100.0  # cm
    ts = data[:, 0] / fps           # 采样起点对应的时间轴(s)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 0,0 POS 直方图
    ax = axes[0, 0]
    ax.hist(pos, bins=32, color="tab:blue", alpha=0.8)
    ax.axvline(np.median(pos), color="k", ls="--", label=f"median {np.median(pos):.2f} cm")
    ax.set_xlabel("POS 1s body error (cm)")
    ax.set_ylabel("samples")
    ax.set_title("1s position error (first-frame body frame)")
    ax.legend()

    # 0,1 ROT / YAW 直方图
    ax = axes[0, 1]
    ax.hist(rot, bins=32, alpha=0.6, label=f"ROT (median {np.median(rot):.2f})", color="tab:red")
    ax.hist(yaw, bins=32, alpha=0.6, label=f"YAW (median {np.median(yaw):.2f})", color="tab:green")
    ax.set_xlabel("ROT / YAW 1s error (deg)")
    ax.set_ylabel("samples")
    ax.set_title("1s rotation error (first-frame body frame)")
    ax.legend()

    # 0,2 DOF 对齐残差
    ax = axes[0, 2]
    ax.hist(dof_resid, bins=24, color="tab:purple", alpha=0.8)
    ax.axvline(np.median(dof_resid), color="k", ls="--", label=f"median {np.median(dof_resid):.4f} cm")
    ax.set_xlabel("dof alignment residual (cm)")
    ax.set_ylabel("samples")
    ax.set_title("dof alignment residual (≈0 -> alignment unambiguous)")
    ax.legend()

    # 1,0 误差 vs 时间
    ax = axes[1, 0]
    ax.scatter(ts, rot, s=10, alpha=0.6, color="tab:red", label="ROT")
    ax.scatter(ts, pos, s=10, alpha=0.6, color="tab:blue", label="POS")
    ax.set_xlabel("sample start time t0 (s)")
    ax.set_ylabel("error (deg / cm)")
    ax.set_title("error vs sample start time (outlier cluster = one fast-turn segment)")
    ax.legend()

    # 1,1 yaw vs rot
    ax = axes[1, 1]
    ax.scatter(rot, yaw, s=10, alpha=0.6)
    lim = max(rot.max(), yaw.max())
    ax.plot([0, lim], [0, lim], ls=":", color="k")
    ax.set_xlabel("ROT 1s (deg)")
    ax.set_ylabel("YAW 1s (deg)")
    ax.set_title("yaw vs overall rotation error")

    # 1,2 文本面板
    ax = axes[1, 2]
    ax.axis("off")
    n = len(data)
    iw = int(np.argmax(data[:, 4]))
    lines = [
        f"samples  = {n}",
        f"POS p50  = {np.median(pos):6.2f} cm",
        f"ROT p50  = {np.median(rot):6.2f} deg",
        f"YAW p50  = {np.median(yaw):6.2f} deg",
        f"dof_resid p50 = {np.median(dof_resid):.3f} cm",
        "",
        f"SAVE-yaw vs MOTION p50 = {np.median(data[:, 7]):6.2f} deg",
        f"YAWVEL integ p50       = {np.median(data[:, 8]):6.2f} deg",
        "",
        f"worst ROT sample:",
        f"  t0={int(data[iw, 0])}  m0={int(data[iw, 1])}",
        f"  pos={data[iw, 2] * 100:5.1f} cm  rot={data[iw, 4]:5.1f} deg",
        f"  saved-ref={data[iw, 7]:5.1f}  yawvel={data[iw, 8]:5.1f}",
        "",
        "SAVED yaw vs MOTION should be ~0",
        "(directly saved, not integrated).",
        "YAWVEL integ vs SAVED is the real",
        "1 s reconstruction drift.",
    ]
    if data[:, 6].std() > 1e-9:
        c_rot = np.corrcoef(data[:, 6] * 100, data[:, 4])[0, 1]
        lines.insert(0, f"corr(dof_resid, ROT) = {c_rot:+.3f}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", transform=ax.transAxes,
            fontsize=9, family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] error stats saved -> {out_path}")


def plot_yaw_comparison(
    dof,
    recon_quat,
    ref_quat,
    ref_joint_jaka,
    fps,
    out_path,
    window_s: float = 5.0,
    t0: int | None = None,
) -> None:
    """画"积分 yaw 变化" vs "真实 yaw 变化（motion_mf）"的对比图，保存到 ``out_path``。

    先取一个对齐起点帧 ``t0``（用 dof 匹配把 action 帧对齐到参考帧 ``m0``），然后从该帧向前看
    ``window_s`` 秒（时间轴 1:1），画两条曲线：``recon``（积分 yaw_vel 得到的 yaw 变化）与
    ``true``（motion_mf anchor 四元数的 yaw 变化）。两条都从各自起点归零，差值 = 积分漂移。
    """
    import matplotlib
    matplotlib.use("Agg")  # headless 保存
    import matplotlib.pyplot as plt

    if t0 is None:
        t0 = 0
    window = int(window_s * fps)
    window = min(window, recon_quat.shape[0] - t0)
    if window <= 1:
        raise SystemExit("plot window too short -- record a longer episode")

    idx = np.arange(window)
    # 与主流程一致：窗口内每一帧都重新按 dof 对齐（抗 fps 波动，不依赖固定 m0+k）。
    m_idx, _ = align_dof_batch(dof[t0:t0 + window], ref_joint_jaka)

    yaw_recon = np.unwrap(_heading_array(recon_quat[t0 + idx]))
    yaw_recon -= yaw_recon[0]
    yaw_true = np.unwrap(_heading_array(ref_quat[m_idx]))
    yaw_true -= yaw_true[0]

    time_s = idx / fps
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s, np.degrees(yaw_recon), label="recon (integrated yaw)", lw=2)
    ax.plot(time_s, np.degrees(yaw_true), label="true (motion_mf)", lw=2, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw change (deg)")
    ax.set_title(f"yaw: integrated vs true  (start frame t0={t0} -> ref m0={int(m_idx[0])})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] yaw comparison saved -> {out_path}")


def plot_yaw_integration(actions, fps, out_path, window_s: float = 5.0, t0: int | None = None):
    """画"积分 yaw_vel 得到的 yaw" vs "记录的绝对 yaw(action 第 30 列)"的对比。

    与 ``plot_yaw_comparison`` 不同:这里**只用 action 自身**(不动 motion_mf),所以直接看
    yaw_vel 积分回去和记录的 yaw 差多少 —— 也就是积分漂移。两者都从窗口起点归零。
    """
    import matplotlib
    matplotlib.use("Agg")  # headless 保存
    import matplotlib.pyplot as plt

    if t0 is None:
        t0 = 0
    window = int(window_s * fps)
    window = min(window, actions.shape[0] - t0)
    if window <= 1:
        raise SystemExit("plot window too short -- record a longer episode")

    dt = 1.0 / float(fps)
    # 记录 yaw 从保存的原始世界 quat 现算(绝对 yaw 不再单独存列)。
    yaw_rec = np.unwrap(_heading_array(actions[t0:t0 + window, _R_QUAT_W].astype(np.float64)))
    yaw_vel = actions[t0:t0 + window - 1, _R_YAWVEL].astype(np.float64)                # 每帧速率
    yaw_int = yaw_rec[0] + np.concatenate([[0.0], np.cumsum(yaw_vel * dt)])            # 积分 yaw
    yaw_rec -= yaw_rec[0]
    yaw_int -= yaw_int[0]

    time_s = np.arange(window) / fps
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s, np.degrees(yaw_int), label="integrated (from yaw_vel)", lw=2)
    ax.plot(time_s, np.degrees(yaw_rec), label="recorded (absolute yaw)", lw=2, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw change (deg)")
    ax.set_title(f"yaw: integrated vs recorded  (start frame t0={t0}; "
                 f"end drift {np.degrees(yaw_int[-1] - yaw_rec[-1]):+.2f} deg)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] yaw integration comparison saved -> {out_path}")


def plot_yaw_dist(data, out_path, unit: str = "deg") -> None:
    """画"yaw_vel 积分 1s vs 记录 yaw"的有符号误差分布(直方图)。

    ``data`` 列: ``col9`` = 1s 端点有符号差(积分-记录,度)。正面直方图看分布/偏置。
    同时给出 mean/std/中位/各幅度百分位的文本。
    """
    import matplotlib
    matplotlib.use("Agg")  # headless 保存
    import matplotlib.pyplot as plt

    e = data[:, 9]  # 1s 端点有符号误差(度)
    lim = np.max(np.abs(e)) * 1.05
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 直方图
    ax = axes[0]
    ax.hist(e, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.median(e), color="r", ls="--", label=f"median {np.median(e):+.2f}")
    ax.axvline(e.mean(), color="g", ls=":", label=f"mean {e.mean():+.2f}")
    ax.set_xlabel(f"integrated − recorded yaw over 1 s ({unit})")
    ax.set_ylabel("samples")
    ax.set_title("yaw 1s endpoint integration error")
    ax.legend()

    # 累积分布(带符号)
    ax = axes[1]
    xs = np.sort(e)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys * 100, color="tab:blue")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(f"integrated − recorded yaw over 1 s ({unit})")
    ax.set_ylabel("cumulative %")
    ax.set_title(f"CDF\nmean {e.mean():+.2f}, std {e.std():.2f}, "
                 f"p95|.| {np.percentile(np.abs(e), 95):.2f}, max|.| {np.abs(e).max():.2f}")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] yaw 1s integ error distribution saved -> {out_path}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/teleop_jaka_mf_zmq/level-0",
        help="LeRobot 数据集根目录（含 data/chunk-000/*.parquet）",
    )
    parser.add_argument("--episode", default="000000", help="episode 编号")
    parser.add_argument("--motion-npz", default="data/motion/motion_mf.npz")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=50, help="向前看多少帧 = fps 下的 1 秒")
    parser.add_argument("--num-samples", type=int, default=200, help="随机抽取多少个起始帧样本")
    parser.add_argument("--seed", type=int, default=0, help="起始帧随机抽样的种子")
    parser.add_argument("--min-t", type=int, default=200, help="跳过前 N 帧（pre-roll）")
    parser.add_argument("--start-pos", default="0,0,0.72", help="重建起点 start_pos (x,y,z)")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="重建起点 start_yaw")
    parser.add_argument("--plot-yaw", action="store_true", help="额外画一张积分 yaw vs 真实(motion_mf) yaw 对比图")
    parser.add_argument("--plot-out", default="/tmp/yaw_compare.png", help="yaw 对比图保存路径")
    parser.add_argument("--plot-yaw-integ", action="store_true", help="额外画一张 积分 yaw vs 记录 yaw 对比图")
    parser.add_argument("--yaw-integ-out", default="/tmp/yaw_integration.png", help="积分 vs 记录 yaw 图保存路径")
    parser.add_argument("--plot-yaw-dist", action="store_true", help="额外画 1s 端点积分误差分布直方图")
    parser.add_argument("--yaw-dist-out", default="/tmp/yaw_integ_dist.png", help="积分误差分布图保存路径")
    parser.add_argument("--plot-stats", action="store_true", help="额外画一张重建误差统计图（4 面板）")
    parser.add_argument("--stats-out", default="/tmp/recon_error_stats.png", help="统计图保存路径")
    parser.add_argument("--plot-window", type=float, default=5.0, help="yaw 图显示的秒数")
    parser.add_argument("--plot-t0", type=int, default=None, help="yaw 图起点 action 帧（默认用 dof 对齐第一个采样）")
    args = parser.parse_args()

    from simple.jaka_rl.action_trunk_reconstruct import reconstruct_anchor_motion
    from simple.jaka_rl.config import ANCHOR_BODY

    # 1. 读取录制的 action。
    parquet = sorted(glob.glob(
        os.path.join(args.data_dir, "data", "chunk-*", f"episode_{args.episode}.parquet")
    ))
    if not parquet:
        raise SystemExit(f"no parquet for episode {args.episode} under {args.data_dir}")
    actions = load_action(parquet[0])
    n_frames = actions.shape[0]
    print(f"actions:    {actions.shape}")
    if actions.shape[1] != 40:
        raise SystemExit(
            f"expected 40-dim actions (DOF+roll/pitch/yaw_vel+lin_vel+原始anchor世界pos/quat), "
            f"got {actions.shape[1]}. 请用 record_jaka_zmq.py 重新录制（现在会保存 40 维）。"
        )

    # 2. 从 action 重建 anchor 位姿（对保存的 yaw_vel 积分）。
    start_pos = np.array([float(v) for v in args.start_pos.split(",")], np.float32)
    dof, recon_pos, recon_quat = reconstruct_anchor_motion(
        actions, fps=args.fps, start_pos=start_pos, start_yaw=args.start_yaw
    )

    # 3. 真值参考。
    ref_pos, ref_quat, ref_joint_jaka = load_reference(args.motion_npz, ANCHOR_BODY)
    n_ref = ref_joint_jaka.shape[0]
    print(f"reference:  {n_ref} frames")

    # 4. 随机抽取起始帧，逐帧用 dof 对齐到参考，再量 1 秒误差。
    rng = np.random.default_rng(args.seed)
    max_t = n_frames - 1 - args.horizon                      # t0 + horizon <= n_frames-1
    avail = np.arange(args.min_t, max_t + 1)
    if avail.size <= 0:
        raise SystemExit(
            f"no valid start frame range [{args.min_t}, {max_t}] -- lower --min-t or --horizon"
        )
    sel = rng.choice(avail, size=min(args.num_samples, avail.size), replace=False)
    if args.num_samples > avail.size:
        print(f"[warn] only {avail.size} valid start frames; using all of them")

    samples = []
    for t0 in sel:
        t0 = int(t0)
        # 窗口内每一帧都重新按 dof 对齐(不依赖固定的 m0+k 1:1 对应,抗 fps 波动)
        m_idx, dmin = align_dof_batch(dof[t0:t0 + args.horizon + 1], ref_joint_jaka)
        cols = compute_aligned_errors(
            t0, m_idx, dmin, args.horizon, recon_pos, recon_quat, ref_pos, ref_quat
        )
        yaw_cols = compute_yaw_errors(t0, m_idx, args.horizon, actions, ref_quat, args.fps)
        # 列: t0, m0=起始参考帧, pos_incr, pos_max, rot_incr, yaw_incr, dof_resid_max,
        #     saved_vs_ref, yaw_vel_recon(max), yaw_vel_end(1s端点有符号)
        samples.append((t0, int(m_idx[0]), *cols, *yaw_cols))

    if not samples:
        raise SystemExit("no aligned samples -- record a longer episode, or lower --min-t/--horizon")

    data = np.asarray(samples)
    print(f"samples:    {len(data)}   (random, seed={args.seed}; horizon {args.horizon} frames "
          f"= {args.horizon / args.fps:.2f} s)")

    # 5. 汇报（第 0 帧对齐后,机体系）。重尾分布，给 p50/mean/p90/p95/max。
    def report(label, col, unit, scale):
        vals = data[:, col] * scale
        p50, p90, p95 = np.percentile(vals, [50, 90, 95])
        print(f"  {label:24s}: p50 {p50:7.3f} {unit}   mean {vals.mean():7.3f}   "
              f"p90 {p90:7.3f}   p95 {p95:7.3f}   max {vals.max():7.3f}")

    print("\n-- 1 秒重建误差（第 0 帧对齐,机体系,积分 yaw_vel）--")
    report("POS 1s body", 2, "cm", 100.0)    # 第 1 秒末机体系位移误差
    report("POS max window", 3, "cm", 100.0)  # 窗口内（对齐后）最大位移误差
    report("ROT 1s body", 4, "deg", 1.0)     # 相对第 0 帧转角增量之间的夹角
    report("YAW 1s body", 5, "deg", 1.0)     # 其中 yaw 分量的误差
    report("DOF match resid", 6, "cm", 100.0)  # 窗口内逐帧 dof 对齐残差最大值（≈0 = 对齐无歧义）
    report("SAVED yaw vs MOTION", 7, "deg", 1.0)  # 直接保存的 yaw 与 motion_mf 的偏差（应≈0）
    report("YAWVEL integ max", 8, "deg", 1.0)     # yaw_vel 积分 1s 窗口内最大 |漂移|
    report("YAWVEL integ end", 9, "deg", 1.0)     # 1s 端点有符号差(积分-记录)

    # 1s 端点误差分布(有符号)专门统计: 偏置 / std / 幅度百分位。
    e = data[:, 9]
    print(f"\n-- yaw_vel 1s 端点积分误差分布(有符号,度) --")
    print(f"  mean(偏置) {e.mean():+.3f}   std {e.std():.3f}   "
          f"median {np.median(e):+.3f}   p50|.| {np.median(np.abs(e)):.3f}   "
          f"p95|.| {np.percentile(np.abs(e), 95):.3f}   max|.| {np.abs(e).max():.3f}")

    print("\n-- 逐样本明细 (t0, m0, pos_incr_cm, pos_max_cm, rot_incr_deg, yaw_incr_deg, "
          "dof_resid_cm, saved_vs_ref_deg, yv_recon_max_deg, yv_end_deg) --")
    for row in data:
        t0, m0, pos_incr, pos_max, rot_incr, yaw_incr, dof_resid, saved_vs_ref, yv_recon, yv_end = row
        print(f"  t0={int(t0):5d}  m0={int(m0):5d}  incr {pos_incr * 100:6.2f}  "
              f"max {pos_max * 100:6.2f}  rot {rot_incr:5.2f}  yaw {yaw_incr:5.2f}  "
              f"res {dof_resid * 100:6.4f}  sav_ref {saved_vs_ref:5.2f}  "
              f"yv_max {yv_recon:5.2f}  yv_end {yv_end:+6.2f}")

    print("\n说明：每个样本都把各自轨迹的第 0 帧（pos + rotation）对齐到一起——减第 0 帧位置、"
          "转进第 0 帧机体坐标系——所以约去了 --start-pos/--start-yaw 带来的任意常数偏差"
          "以及两者初始 yaw 的差别，才是真正的一秒精度。")

    # 2. 离群点诊断：最差的样本是否仍是 dof 完美对齐（是 -> 真信号，不是错帧）。
    idx_worst = int(np.argmax(data[:, 4]))
    print("\n-- 离群点诊断 --")
    print(f"  worst ROT: t0={int(data[idx_worst, 0])}  m0={int(data[idx_worst, 1])}  "
          f"pos={data[idx_worst, 2] * 100:.2f}cm  rot={data[idx_worst, 4]:.2f}deg  "
          f"dof_resid={data[idx_worst, 6] * 100:.3f}cm")
    if data[:, 6].std() > 1e-9:
        c_pos = np.corrcoef(data[:, 6] * 100, data[:, 2] * 100)[0, 1]
        c_rot = np.corrcoef(data[:, 6] * 100, data[:, 4])[0, 1]
        print(f"  corr(dof_resid, POS) = {c_pos:+.3f}   corr(dof_resid, ROT) = {c_rot:+.3f}")
    else:
        print("  dof_resid 恒定(≈0) -> dof 对齐无歧义，大误差不是错帧造成的")
    top = data[np.argsort(-data[:, 4])][:5]
    print("  top-5 by ROT:  t0      m0    pos_cm   rot_deg  dof_resid_cm")
    for r in top:
        print(f"            t0={int(r[0]):5d}  m0={int(r[1]):5d}  "
              f"pos={r[2] * 100:7.2f}  rot={r[4]:7.1f}  res={r[6] * 100:8.3f}")

    # 6. 可选：画"积分 yaw vs 真实 yaw(motion_mf)"变化曲线。
    if args.plot_yaw:
        t0 = args.plot_t0 if args.plot_t0 is not None else int(data[0, 0])
        plot_yaw_comparison(
            dof, recon_quat, ref_quat, ref_joint_jaka, args.fps,
            args.plot_out, window_s=args.plot_window, t0=t0,
        )

    # 7. 可选：画误差统计图（含离群点时段）。
    if args.plot_stats:
        plot_stats(data, float(args.fps), args.stats_out)

    # 8. 可选：画 积分 yaw vs 记录 yaw（只看 action 自身的积分漂移）。
    if args.plot_yaw_integ:
        t0 = args.plot_t0 if args.plot_t0 is not None else int(data[0, 0])
        plot_yaw_integration(actions, float(args.fps), args.yaw_integ_out,
                             window_s=args.plot_window, t0=t0)

    # 9. 可选：画 1s 端点积分误差分布。
    if args.plot_yaw_dist:
        plot_yaw_dist(data, args.yaw_dist_out)


if __name__ == "__main__":
    main()
