# Jaka Khan Mini — 单终端全身 MF 策略遥操作速查

SIMPLE 中 Jaka 全身 MF 策略（从 sim2real 整合）的遥操作、录制、校验命令。

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
  --controller keyboard \
  --no-headless
```

> CLI 只保留「每次运行会变」的开关：`env_id`、`--config`、`--target`、`--controller`、
> `--headless/--no-headless`、`--debug-log`。**其余参数全部在配置文件
> `data/jaka_mf/teleop_jaka_mf.yaml`**（策略、频率、初始高度、ZMQ 端口、相机频率…），
> 日常调参改那个文件即可。state + 相机**始终发布**（不再需要 `--publish-zmq`）。

> `**JakaTabletopPickTeleop-v0`**：桌上**固定一个物体**（`objaverse:1128` 箱子，改 `--target` 换物体），**随机干扰物已关闭**（`tasks/jaka_tabletop_pick_teleop.py` 里 `number_of_distractors=0`）。目标物体在
> `src/simple/tasks/jaka_tabletop_pick_teleop.py` 的 `TargetDRCfg(asset_id="objaverse:1128")` 硬编码，改这一行或传 `--target` 即可换物体。

**操作流程**：启动后**自动进入 policy 模式**（`init_base_z`=0.596 站立高度，关节用 `[` 对齐值），机器人直接站住并跟踪参考 —— 不再需要手动按 `]` 或按 `9` 释放弹力带（弹力带默认已在配置里关闭）。

> 本步骤只是**遥操作**。要录数据，`--publish-zmq` 会把 state + 相机发到 ZMQ（见第 3 节），录制由独立进程完成。

**常用参数**


| 参数                           | 说明                                         |
| ---------------------------- | ------------------------------------------ |
| `--target <id>`              | 目标物体（默认 `objaverse:1128`）                  |
| `--controller pico/keyboard` | 输入源（只影响 `space` 暂停等；**模式切换已禁用**，见第 2 节）    |
| `--no-headless`              | 打开 MuJoCo 窗口（无左右面板 + 跟踪相机）                 |
| `--config <yaml>`            | 运行配置，默认 `data/jaka_mf/teleop_jaka_mf.yaml` |
| `--debug-log <path>`         | 每步 JSONL 诊断快照                              |


**配置文件 `data/jaka_mf/teleop_jaka_mf.yaml`** 常用键：


| 键                                    | 默认                  | 说明                                                                |
| ------------------------------------ | ------------------- | ----------------------------------------------------------------- |
| `init_base_z`                        | `0.596`             | 初始站立高度（base 的 z）。场景 spawn 在 ~0.793、policy 平衡点在 ~0.596，用平衡高度起步才不下坠 |
| `policy_config`                      | `data/jaka_mf/...`  | 策略 yaml（ONNX 默认取其同名 `.onnx`）                                      |
| `motion_backend`                     | `zmq`               | `zmq`(订阅 `:28701`) / `raw_npz`(离线 npz)                            |
| `rl_rate` / `camera_hz`              | `50` / `30`         | 控制频率 / 相机发布频率（state 按 `rl_rate` 发）                                |
| `enable_elastic_band`                | `false`             | 弹力带（虚拟吊带）。当前默认关闭，用于验证 policy 自主站立                                 |
| `reset_on_record_end`                | `true`              | 运动流 `toggle_data_collection` 由高变低（录制结束）时重置环境                      |
| `state_zmq_bind` / `camera_zmq_bind` | `:28711` / `:28712` | 供 `record_jaka_zmq` 订阅的发布端口                                       |


### 1.1 live 遥操作：用真实的 sim2real pico hub（而非 fake）

上面 `--motion-backend zmq`（默认）的数据源可以是**真实的** sim2real pico 重定向 hub（`cxr/sim2real-jaka` 仓库），
不戴头显时才用 `scripts/fake_pico_motion_pub.py` 模拟。真实 hub 同时发运动流 `:28701` 和手柄按钮 `:5592`，
因此与 SIMPLE 的 wire 协议完全一致，可直接对接（fake 就是照它写的）。

**终端 A——启动真实 hub（在 `/home/xu/code/cxr/sim2real-jaka`，conda `teleop` 环境）**：

```bash
cd /home/xu/code/cxr/sim2real-jaka
# 必须显式 --robot jaka（默认 g1），否则 joint/body 顺序不是 JAKA 序，SIMPLE 校验会失败
# publish_hz 建议 50；录制端 30Hz 只取最新帧，所以 hub >= 30 即可
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
  --target=objaverse:1128 --controller pico --no-headless
```

要点：

- **必须 `--robot jaka`**：真实 hub 默认 `g1`，不传会发布 G1 序，SIMPLE 校验 joint/body 长度与顺序会失败。
- **两个端口都不用改**：`--bind`(`:28701` motion) / `--controller_bind`(`:5592` 手柄) 是参考实现默认值，正好是 SIMPLE 订阅的地址。
- **真实链路需要硬件**：pico 头显 + GMR 重定向模型（真实动捕）。不戴头显时参考回退到中立站姿 `ZMQ_DEFAULT_QPOS`（机器人原地站住），或用 fake 播 npz 代替。
- `**--controller pico`**：用真实 hub 时建议 pico；没有手柄就用 `--controller keyboard`。注意两者**都不再切控制模式**（见第 2 节），机器人启动即锁定 policy。
- 与 fake 的唯一差异是数据来源（真实动捕 vs npz 重放），SIMPLE 侧无需任何改动。上线前建议先跑第 5 节的 fake 验证，再切真实 hub。

## 2. 键盘键位

**MuJoCo 窗口**（弹力带/虚拟吊架）：`7` 拉高、`8` 放低、`9` 释放/启用。

**终端键盘**：

- `space` 暂停/播放参考。
- `r` + 回车 —— **手动触发 reset**（重建场景 + 重随机化 DR + 回到 `init_base_z` 站立姿势并重新进入 policy）。
  与 pico 的 `toggle_data_collection` 下降沿（录制结束）是**两个并列的触发条件**；没接 hub 时用 `r` 就能测这套 reset。

> ⚠️ **模式切换已禁用**：`BasePolicy.process_controllers()` 里按 pico 手柄 / 键盘切换
> `init/zero/align/policy` 的调度已注释(`jaka_rl/base_policy.py`)。控制模式由调用方
> 直接设定并在启动 / 录制结束时固定为 **policy**,所以 `i` `o` `[` `]` 和 pico A/B
> **都不再切模式**(原因:pico A 键被录制开关占用,会顺带把机器人切出 policy)。
> 需要恢复应急切换时,取消那段注释即可。

## 3. 录制 VLA 数据（两终端，当前工作流）

录制由**独立进程** `record_jaka_zmq` 完成，遥操作进程只负责把 state + 相机发到 ZMQ。
两个进程用「只取最新帧」（latest-only）的方式对齐，无需共享控制环。

**终端 A——遥操作 + 发布 state/相机**（state 50Hz，相机 30Hz）：

```bash
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0  --no-headless
```

**终端 B——独立录制（30Hz）**：

```bash
python -m simple.cli.record_jaka_zmq \
  --save-dir data/teleop_jaka_mf_zmq \
  --desc "put the bottle on the mouse pad" \
  --frequency 30 \
  --policy-config data/jaka_mf/latest56k_pico_dr.yaml \
  --trigger pico      # 可省略:默认 both(pico A 键 + 键盘回车都行)
  # --run-name exp01  # 可省略:默认用时间戳当这一层的目录名
```

**数据来源（三路 ZMQ，全部 latest-only）**


| 流        | 端口       | 来源                                                     | 频率                 |
| -------- | -------- | ------------------------------------------------------ | ------------------ |
| `action` | `:28701` | pico hub（`pico_retarget_pub`）**直接订阅，不经 motion buffer** | hub `--publish_hz` |
| `state`  | `:28711` | 终端 A 的 `--publish-zmq`                                 | 50Hz               |
| `camera` | `:28712` | 终端 A 的 `--publish-zmq`（head_stereo_left）               | 30Hz               |


录制进程按 `--frequency 30` 采样：每次取三路各自的**最新一帧**，所以 action 是「最新时刻的 pico 参考数据」。
`action` 是**参考运动**（frozen MF 策略要跟踪的目标），不是策略输出。

**保存格式**（LeRobot，PNG 字节存在 `data/*.parquet`，另存每 episode 一个 mp4）

- `head_image_left`：**原始分辨率**头相机图（由 `tasks/jaka_vla_cameras.py` 的 `head_stereo`
决定，当前 224×224）—— 录制侧**不做** resize/pad，shape 由录制进程在收到第一帧时自动声明；
缩放交给训练侧的 openpi `ResizeImages` → `resize_with_pad`（保持比例 + 补零，不是拉伸）。
存储位置：
  - `**data/chunk-*/episode_*.parquet`** 内嵌 **PNG 字节**（LeRobot `embed_images`）——**数据格式不变**，
  读数据集只用它就够（`head_image_left` 是 `image` dtype，不依赖外挂文件）。
  - `**videos/chunk-*/head_image_left/episode_*.mp4`**：每 episode 一个 mp4，编码后**删掉
  `images/` 里的逐帧 PNG**（否则那是同一批图的第二份完整副本）。mp4 只是**方便浏览的副本**，
  训练不用读它。
- `state` (30) = `[27 关节, roll, pitch, yaw_vel]`（机器人，从 `:28711`）
- `actions` (40) = `[27 关节, roll, pitch, yaw_vel, anchor lin_vel xyz, anchor_pos_w xyz, anchor_quat_w wxyz]`
  - 前 33 维是经典 trunk；`anchor`（`waist_yaw_Link`）的线速度在 anchor 机体系（m/s）、
  后 7 维是**原始世界系** anchor 位姿（pos + quat wxyz，逐帧原样保存，便于校验）。绝对 yaw 不单独存，可从 quat 恢复。

**操作**：默认 `--trigger both` —— **pico A 键**（运动流里的 `toggle_data_collection` 电平，
与 teleop 的「录制结束 reset」是同一个信号）和**键盘回车**都能开始/结束一段 episode。

> ⚠️ 录制端是**跟随电平**的：若它启动时 pico 已经开始录制（电平已是 True），会**立即开始录**。
> 想测「第 N 帧那个开启沿」，要么先起录制端再起 hub，要么把 `--record-start` 调大。
> 另：`--trigger keyboard`（旧默认）**完全不看** pico 开关，会出现「teleop 重置了但录制端没反应」。
输出在 `<save-dir>/<run-id>/level-<dr-level>/`，`run-id` = `--run-name`（未指定则用时间戳
`20260915-134512`）。默认 `--save-dir` 时即 `data/teleop_jaka_mf_zmq/20260915-134512/level-0`。

> ✅ `record_jaka_zmq` **不会再删除已有数据**：每次运行都开一个新目录，`run-id` 撞名时自动加
> `-2`/`-3` 后缀，旧的 `level-0` 原地保留。想换存储位置直接 `--save-dir <别的根目录>`。

## 4. 校验录制的 action（可选）

对比「用 action 重建的 anchor 运动」与真值参考 `data/motion/motion_mf.npz`：

```bash
PYTHONPATH=src python scripts/verify_jaka_action_recon_error.py \
  --episode 000000 \
  --fps 30 --horizon 30 \
  --num-samples 300 --seed 0 \
  --plot-stats --plot-yaw
```

> `--fps` 必须与 `--frequency` 一致（30）；`--horizon 30` 才是 1 秒。
> 脚本默认是 `--fps 50 --horizon 50`，**所以这里必须显式传 30**。

输出：`POS / ROT / YAW 1s` 误差（第 0 帧对齐、机体系）、`SAVED yaw vs MOTION`（≈0 说明录制忠实）、
`YAWVEL integ vs SAVED`（yaw_vel 积分漂移）；配 `--plot-stats` / `--plot-yaw` / `--plot-yaw-integ` / `--plot-yaw-dist` 出图。

## 5. 离线验证（无需 pico / hub）

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

`fake_pico_motion_pub.py` 复刻真实 hub 的状态机，默认**先发 1 帧空闲快照再 live**
（空闲姿态 = `ZMQ_DEFAULT_QPOS` 的 FK，和策略无数据时的回退姿态一致，所以起脚本参考不会跳）。
常用的确定性相位（不用手按键就能复现同一条时间线）：

```bash
python scripts/fake_pico_motion_pub.py --live-at 50          # 先空闲 50 帧再 live
python scripts/fake_pico_motion_pub.py --live-at -1          # 永不 live(只发冻结站姿)
python scripts/fake_pico_motion_pub.py --pause-at 200 --pause-frames 100   # 第 200 个 live 帧暂停 100 帧
python scripts/fake_pico_motion_pub.py --record-start 100 --record-stop 400  # 同按 A 开/关录制电平
python scripts/fake_pico_motion_pub.py --keyboard            # 键盘当手柄(见下表)
```

> `--keyboard` 下开机是 idle（同真实 hub `paused=True`），按 `x` 才 live；`--record-start/stop` 失效。
> 暂停时冻结在**刚发出的那个姿态**上（连续、不跳），恢复时从它的下一帧继续。

**按键对应**（与 `pico_retarget_pub.py` 的手柄映射一致；脚本还会按真实 hub 的格式在
`:5592` 播发 `<QBBBB>` = ts+A+B+X+Y，所以 `PicoController` 的 A/B 模式切换也能测）：


| 键盘 | 手柄按键                  | hub 侧作用                | SIMPLE `PicoController` 解出的模式 |
| -- | --------------------- | ---------------------- | ---------------------------- |
| `a` | A = RightController key_one | 翻转 `toggle_data_collection`（录制开关） | `init`（单独按 A）                |
| `x` | X = LeftController key_one  | 翻转 `paused`（冻结/恢复参考）    | —（X 不映射模式）                   |
| `b` | B = RightController key_two | 无（真实 hub 也只转发）         | `zero`（单独按 B）                |
| `y` | Y = LeftController key_two  | 无（真实 hub 也只转发）         | —                            |
| `q` | —                     | 退出                     | —                            |

> `a` + `b` **同时**按住 = `A+B` 组合 → `policy`（`resolve_pico_control_mode` 的映射）。
> 按键会被压住 `--key-press-ms`（默认 200ms，模拟人手按住），所以两次同键要间隔超过这个时间才算两次边沿。

> 若不开 hub，`motion_backend=zmq` 的参考会回退到 **`ZMQ_DEFAULT_QPOS` 的 FK 站姿**（不是 motion 帧0：
> `_using_zmq_replay()` 已固定返回 False），机器人能站住但不会跟踪运动。

## 6. 回放渲染（Stage 2，`replay-jaka`）

> 仅适用于**旧 jaka-replay 格式**（含 `observation.body_poses` / `observation.scene_qpos` + mp4）。
> 第 3 节的录制写的是 openhlm 格式（仅头图 + state 30 + action 40），**没有** body_poses/scene_qpos，不能用此命令回放。

```bash
python -m simple.cli.replay_jaka simple/JakaTabletopPickTeleop-v0 \
  --data-dir data/jaka_teleop/jaka_tabletop_pick_teleop/level-0 \
  --save-dir data/replay_jaka --sim-mode mujoco --save-all --direct
```

## 7. 涉及的代码（本次整合）

- `src/simple/jaka_rl/` — 迁移的全身 MF 策略栈（`base_policy`/`state_processor`/`motion_buffer`/`observations`/`action_manager`/`config` 等）
- `src/simple/jaka_rl/action_trunk_reconstruct.py` — 从 action trunk 重建 anchor 位姿（校验用）
- `src/simple/agents/jaka_mf_agent.py`、`src/simple/cli/teleop_jaka_mf.py` — 单终端 agent + CLI（遥操作；`--publish-zmq` 发布 state/相机）
- `src/simple/cli/record_jaka_zmq.py` — 独立录制进程（直连 pico + state/相机，30Hz latest-only；`LatestMotionClient`/`LatestStateClient`/`LatestImageClient`）
- `src/simple/interfaces/jaka_zmq_pub.py` — 遥操作侧的 state/相机 ZMQ 发布（state 每 tick 50Hz；相机由独立线程按 `--camera-hz`=30Hz 发送）
- `src/simple/datasets/jaka_lerobot.py` — openhlm 数据格式（state 30 / action 40）+ `reference_action_live_latest`
- `scripts/verify_jaka_action_recon_error.py` — 录制 action 的重建误差校验
- `src/simple/robots/jaka.py` — 弹力带/子步 PD/默认位姿
- `src/simple/interfaces/zmq_bridge.py`、`src/simple/engines/mujoco.py` — 复用
- `scripts/fake_pico_motion_pub.py`、`scripts/run_jaka_mf_npz.py` — 离线验证
- `data/motion/motion_mf.npz` — 参考运动

