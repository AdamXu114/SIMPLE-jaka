#!/usr/bin/env python3
"""从已录好的 LeRobot v2.1 数据集中删除若干 episode，并把剩下的重新编号。

这个 venv 里的 lerobot 是 0.3.3，**没有** ``delete_episodes`` API（``lerobot.datasets.
dataset_tools`` 不存在，也没有 ``lerobot-edit-dataset`` 脚本），所以这里直接按磁盘布局改。

一个 episode 在磁盘上占三处（``record_jaka_zmq`` 写出来的形态）::

    data/chunk-XXX/episode_XXXXXX.parquet          真正的数据(图像字节内嵌在 parquet 里)
    videos/chunk-XXX/<key>/episode_XXXXXX.mp4      只是给人看的伴随视频(不在 meta.video_keys 里)
    images/<key>/episode_XXXXXX/*.png              只在录制被打断时残留
    meta/episodes.jsonl / episodes_stats.jsonl     每行一个 episode，按 episode_index 索引
    meta/info.json                                 total_episodes / total_frames / splits

**为什么必须重排**:``LeRobotDataset`` 定位每一集的行区间靠 ``meta/episodes.jsonl`` 里各集
``length`` 的累加(``get_episode_data_index``),文件名靠 ``meta.get_data_file_path(ep_idx)``
拼。所以只删文件不改 meta、或者删中间一集却不重排后面的编号,都会让后续 episode 全部错位。
本脚本把 ``episode_index`` 列、parquet 里的全局 ``index`` 列、parquet/mp4/images 的文件名、
以及 meta 里的编号一起改成连续的(与重新录一遍得到的布局一致)。

用法::

    # 先看这个数据集里有哪些 episode、多长
    python scripts/delete_episodes.py --dataset data/teleop_jaka_mf/20260915-145329/level-0 --list

    # 默认 dry-run,只打印计划
    python scripts/delete_episodes.py --dataset <level-0 目录> --episodes 3,7,11

    # 真删(--yes 才落盘);动手前建议 cp -r 一份备份
    python scripts/delete_episodes.py --dataset <level-0 目录> --episodes 3,7,11 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """原子替换,避免写到一半失败把 meta 弄坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _data_path(root: Path, ep_idx: int, chunks_size: int) -> Path:
    return root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"


def _video_paths(root: Path, ep_idx: int, chunks_size: int) -> list[Path]:
    vdir = root / "videos" / f"chunk-{ep_idx // chunks_size:03d}"
    return sorted(vdir.glob(f"*/episode_{ep_idx:06d}.mp4")) if vdir.is_dir() else []


def _image_dirs(root: Path, ep_idx: int) -> list[Path]:
    idir = root / "images"
    return sorted(idir.glob(f"*/episode_{ep_idx:06d}")) if idir.is_dir() else []


def _renumber_parquet(old_path: Path, new_path: Path, new_ep_idx: int, new_start: int) -> int:
    """改写一个 episode 的 parquet:episode_index 列 + 全局 index 列,并挪到新文件名。

    录制时 ``index = 该集起始全局帧号 + frame_index``(已在本仓库数据上验证过),删集之后
    后续集的起始偏移变小,所以按 ``new_start + frame_index`` 重算即可。返回帧数。
    """
    table = pq.read_table(old_path)
    if table.num_rows == 0:
        raise ValueError(f"{old_path} 是空表")

    frame_index = np.asarray(table["frame_index"])
    if not (frame_index == np.arange(table.num_rows)).all():
        raise ValueError(f"{old_path} 的 frame_index 不是 0..N-1,不敢自动重排")

    table = table.set_column(
        table.schema.get_field_index("episode_index"), "episode_index",
        pa.array(np.full(table.num_rows, new_ep_idx, dtype=np.int64)),
    )
    table = table.set_column(
        table.schema.get_field_index("index"), "index",
        pa.array(new_start + frame_index, type=pa.int64()),
    )

    new_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = new_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, new_path)
    if old_path != new_path and old_path.exists():
        old_path.unlink()
    return table.num_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="删除 LeRobot v2.1 数据集里的若干 episode 并重排编号")
    ap.add_argument("--dataset", required=True, help="数据集根目录,即 <save-dir>/<run-id>/level-N")
    ap.add_argument("--episodes", default="", help="要删的 episode 编号,逗号分隔,如 3,7,11")
    ap.add_argument("--list", action="store_true", help="只列出 episode 及长度")
    ap.add_argument("--yes", action="store_true", help="真的落盘(dry-run)")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    meta_dir = root / "meta"
    if not (meta_dir / "info.json").exists():
        raise SystemExit(f"{root} 看起来不是数据集根目录(缺 meta/info.json)")

    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = sorted(_read_jsonl(meta_dir / "episodes.jsonl"), key=lambda e: e["episode_index"])
    stats = sorted(_read_jsonl(meta_dir / "episodes_stats.jsonl"), key=lambda e: e["episode_index"])
    by_idx = {int(e["episode_index"]): e for e in episodes}

    # 没写进 meta 的残留(录制中途被杀,只留下 images/... 或半个 parquet)
    orphans = [
        p.name for p in (root / "images").glob("*/episode_*") if
        int(p.name.split("_")[1]) not in by_idx
    ] if (root / "images").is_dir() else []

    if args.list or not args.episodes:
        print(f"{root}\n  fps={info['fps']}  total_episodes={info['total_episodes']}  "
              f"total_frames={info['total_frames']}  (meta/info.json 里的值)")
        for e in episodes:
            n_frames = _data_path(root, int(e['episode_index']), chunks_size).exists()
            print(f"  episode {e['episode_index']:>6}  length={e['length']:>5}  "
                  f"parquet={'有' if n_frames else '缺失!'}  tasks={e.get('tasks')}")
        if orphans:
            print(f"  残留(不在 meta 里,可单独删): images/.../episode_{orphans}")
        if args.list:
            return
        raise SystemExit("没给 --episodes,什么都没做")

    drop = sorted({int(x) for x in args.episodes.replace(" ", "").split(",") if x})
    missing = [i for i in drop if i not in by_idx]
    if missing:
        raise SystemExit(f"这些 episode 不在 meta/episodes.jsonl 里: {missing}")
    keep = [e for e in episodes if int(e["episode_index"]) not in set(drop)]
    if not keep:
        raise SystemExit("会把所有 episode 都删光,拒绝执行(整个数据集直接 rm -rf 更直接)")

    # 计划:重排后每集的 新编号 -> (旧编号, 新起始全局帧号)
    offsets, acc = {}, 0
    for new_idx, e in enumerate(keep):
        offsets[new_idx] = (int(e["episode_index"]), acc, int(e["length"]))
        acc += int(e["length"])

    print(f"{root}\n删 {len(drop)} 集: {drop}   留 {len(keep)} 集, "
          f"帧数 {info['total_frames']} -> {acc}")
    for new_idx, (old_idx, start, length) in offsets.items():
        mark = "" if new_idx == old_idx else f"  (原 {old_idx} -> {new_idx})"
        print(f"  keep episode {new_idx:>6}  frames {start:>6}..{start + length - 1:<6}{mark}")
    for i in drop:
        print(f"  drop episode {i:>6}  length={by_idx[i]['length']}")

    if not args.yes:
        print("\n[dry-run] 加 --yes 才会真的改盘(建议先 cp -r 备份)")
        return

    # 1) 先删要删的,腾出文件名(这样后面升序重命名不会撞车)
    for i in drop:
        for p in [_data_path(root, i, chunks_size), *_video_paths(root, i, chunks_size),
                  *_image_dirs(root, i)]:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"  rm -r {p.relative_to(root)}")
            elif p.exists():
                p.unlink()
                print(f"  rm    {p.relative_to(root)}")

    # 2) 升序重排保留的 episode(new_idx <= old_idx,目标名此时已空出)
    total = 0
    for new_idx, (old_idx, start, _length) in offsets.items():
        old_pq = _data_path(root, old_idx, chunks_size)
        if not old_pq.exists():
            print(f"  WARN: 第 {old_idx} 集的 parquet 不存在,跳过(meta 里仍会有这一条)")
            continue
        rows = _renumber_parquet(old_pq, _data_path(root, new_idx, chunks_size), new_idx, start)
        if rows != offsets[new_idx][2]:
            print(f"  WARN: 第 {old_idx} 集 parquet 行数 {rows} != meta 里的 length "
                  f"{offsets[new_idx][2]}")
        if old_idx != new_idx:
            for p in _video_paths(root, old_idx, chunks_size):
                new_p = p.with_name(f"episode_{new_idx:06d}.mp4")
                os.replace(p, new_p)
            for p in _image_dirs(root, old_idx):
                os.replace(p, p.with_name(f"episode_{new_idx:06d}"))
            print(f"  重排 episode {old_idx} -> {new_idx}")
        total += rows

    # 3) meta:编号跟着重排,顺序不变
    _write_jsonl(meta_dir / "episodes.jsonl", [
        {**e, "episode_index": new_idx} for new_idx, e in enumerate(keep)
    ])
    if stats:
        stats_by_idx = {int(s["episode_index"]): s for s in stats}
        _write_jsonl(meta_dir / "episodes_stats.jsonl", [
            {**stats_by_idx[old_idx], "episode_index": new_idx}
            for new_idx, (old_idx, _s, _l) in offsets.items() if old_idx in stats_by_idx
        ])

    # 4) info.json
    info["total_episodes"] = len(keep)
    info["total_frames"] = total
    info["splits"] = {"train": f"0:{len(keep)}"}
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    print(f"\n完成: {len(keep)} 集 / {total} 帧。episodes.jsonl、episodes_stats.jsonl、"
          f"info.json 已同步更新")
    if orphans:
        print(f"提示: images/.../episode_{orphans} 是残留,不在 meta 里,可直接 rm -r 删掉")


if __name__ == "__main__":
    main()
