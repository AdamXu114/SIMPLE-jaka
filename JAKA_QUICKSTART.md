# Jaka Khan Mini — 单终端全身 MF 策略遥操作速查

SIMPLE 中 Jaka 全身 MF 策略（从 sim2real 整合）的单终端遥操作、离线验证、录制命令。

## 环境准备

```bash
source .venv/bin/activate

# IsaacSim 渲染需要（回放 `replay-jaka` 会自动设置）
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs:$LD_LIBRARY_PATH"
```

## 1. 单终端全身 MF 策略遥操作（核心）

仿真 + 全身 MF 策略 + 运动缓冲**都在 SIMPLE 一个终端**跑（无需再用 sim2real 的 `base_sim`/`tracking`）。策略与 sim2real 完全一致（单个全身 ONNX 输出 27 维动作，620 维观测），部署也用这个策略。参考运动可来自 pico 头显（通过 `motion_backend=zmq` 的 `:28701` 流），或离线 npz。

```bash
export TASK_NAME=JakaTabletopPickTeleop-v0
python -m simple.cli.teleop_jaka_mf \
  simple/$TASK_NAME \
  --target=objaverse:1128 \
  --sim-mode=mujoco \
  --no-headless \
  --record \
  --success-criteria=0.9 \
  --policy-config data/jaka_mf/latest56k_pico_dr.yaml \
  --controller keyboard
```

> `**JakaTabletopPickTeleop-v0**`：桌上**固定一个物体**（`objaverse:1128` 箱子，改 `--target` 换物体），**随机干扰物已关闭**（`tasks/jaka_tabletop_pick_teleop.py` 里 `number_of_distractors=0`）。目标物体在
> `src/simple/tasks/jaka_tabletop_pick_teleop.py` 的 `TargetDRCfg(asset_id="objaverse:1128")` 硬编码，改这一行或传 `--target` 即可换物体。

**操作流程**：启动（机器人已按 motion 帧0 站姿就位）→ 按 `**]`**（keyboard）或 pico **A+B** 进入 policy 模式 → 按 `**9`** 释放弹力带落地 → 机器人稳定站立并跟踪参考。

**常用参数**


| 参数                             | 说明                                           |
| ------------------------------ | -------------------------------------------- |
| `--policy-config <yaml>`       | 策略配置，必填                                      |
| `--policy-model <onnx>`        | ONNX；默认由 yaml 推导（yaml 同名 .onnx）              |
| `--target <id>`                | 目标物体（默认 `objaverse:1128`）                    |
| `--controller pico/keyboard`   | 模式来源：pico 手柄 `:5592` 或键盘 `i/o/[/]` + `space` |
| `--motion-backend zmq/raw_npz` | `zmq`(默认,订阅 `:28701`) / `raw_npz`(离线 npz)    |
| `--no-headless`                | 打开 MuJoCo 窗口（无左右面板 + 跟踪相机）                   |
| `--record`                     | 录制 LeRobot VLA 数据                            |
| `--success-criteria`           | 任务成功判定阈值                                     |
| `--debug-log <path>`           | 每步 JSONL 诊断快照                                |


### 1.1 live 遥操作：用真实的 sim2real pico hub（而非 fake）

上面 `--motion-backend zmq`（默认）的数据源可以是**真实的** sim2real pico 重定向 hub（`cxr/sim2real-jaka` 仓库），
不戴头显时才用 `scripts/fake_pico_motion_pub.py` 模拟。真实 hub 同时发运动流 `:28701` 和手柄按钮 `:5592`，
因此与 SIMPLE 的 wire 协议完全一致，可直接对接（fake 就是照它写的）。

**终端 A——启动真实 hub（在 `/home/xu/code/cxr/sim2real-jaka`，conda `teleop` 环境）**：

```bash
cd /home/xu/code/cxr/sim2real-jaka
# 必须显式 --robot jaka（默认 g1），否则 joint/body 顺序不是 JAKA 序，SIMPLE 校验会失败
# publish_hz 建议 50，与 SIMPLE 的 50 Hz 策略对齐（默认 30 也能用，buffer 会插值）
conda run -n teleop python sim2real/teleop/pico_retarget_pub.py \
  --robot jaka \
  --bind tcp://*:28701 \
  --controller_bind tcp://*:5592 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

**终端 B——SIMPLE 单终端遥操作（不变，默认 `motion-backend zmq` 即订阅 `:28701`）**：

```bash
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 \
  --target=objaverse:1128 --sim-mode=mujoco --no-headless --record \
  --controller pico --success-criteria=0.9 \
  --policy-config data/jaka_mf/latest56k_pico_dr.yaml
```

要点：

- **必须 `--robot jaka`**：真实 hub 默认 `g1`，不传会发布 G1 序，SIMPLE 校验 joint/body 长度与顺序会失败。
- **两个端口都不用改**：`--bind`(`:28701` motion) / `--controller_bind`(`:5592` 手柄) 是参考实现默认值，正好是 SIMPLE 订阅的地址。
- **真实链路需要硬件**：pico 头显 + GMR 重定向模型（真实动捕）。不戴头显时参考回退到中立站姿 `ZMQ_DEFAULT_QPOS`（机器人原地站住），或用 fake 播 npz 代替。
- `**--controller pico`**：用真实 hub 时建议 pico（按钮走 `:5592`）；没有手柄就用 `--controller keyboard`（`i/o/[/]` + `space`）。
- 与 fake 的唯一差异是数据来源（真实动捕 vs npz 重放），SIMPLE 侧无需任何改动。上线前建议先跑第 3 节的 fake 验证，再切真实 hub。

## 2. 键盘键位

**MuJoCo 窗口**（弹力带/虚拟吊架）：`7` 拉高、`8` 放低、`9` 释放/启用。

**终端键盘**（策略控制模式）：
`i` init、`o` zero、`[` align（对齐到 motion 帧0 关节）、`]` policy、`space` 暂停/播放参考。

## 3. 离线验证（无需 pico / hub）

**方式 A：offline replayer（npz 直喂策略）**

```bash
python scripts/run_jaka_mf_npz.py --motion-backend raw_npz --steps 600 --headless
```

**方式 B：模拟 pico hub（zmq 流，走真实 json 路径）**

```bash
# 终端 A：模拟 pico_retarget_hub，播 data/motion 的 motion
python scripts/fake_pico_motion_pub.py
# 终端 B：策略走真实 zmq 路径
python scripts/run_jaka_mf_npz.py --motion-backend zmq --steps 300 --headless
```

> 若不开 hub，`motion_backend=zmq` 会自动重放本地 `data/motion/motion_mf.npz`（参考=motion 帧0），机器人仍能站住。

## 4. 回放渲染（Stage 2，`replay-jaka`）

```bash
python -m simple.cli.replay_jaka simple/JakaTabletopPickTeleop-v0 \
  --data-dir data/jaka_teleop/jaka_tabletop_pick_teleop/level-0 \
  --save-dir data/replay_jaka --sim-mode mujoco --save-all --direct
```

## 5. 涉及的代码（本次整合）

- `src/simple/jaka_rl/` — 迁移的全身 MF 策略栈（`base_policy`/`state_processor`/`motion_buffer`/`observations`/`action_manager`/`config` 等）
- `src/simple/agents/jaka_mf_agent.py`、`src/simple/cli/teleop_jaka_mf.py` — 单终端 agent + CLI
- `src/simple/robots/jaka.py` — 弹力带/子步 PD/默认位姿
- `src/simple/interfaces/zmq_bridge.py`、`src/simple/engines/mujoco.py` — 复用
- `scripts/fake_pico_motion_pub.py`、`scripts/run_jaka_mf_npz.py` — 离线验证
- `data/motion/motion_mf.npz` — 参考运动

