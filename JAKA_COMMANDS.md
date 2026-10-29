# JAKA 命令速查(整合版)

> 本文把仓库里所有 JAKA 相关文档中的**可执行命令**集中到一处,按使用场景组织,
> 并保留每条命令的关键上下文(终端顺序、端口、坑)。整合来源:
>
> - [JAKA_QUICKSTART.md](JAKA_QUICKSTART.md) —— 遥操作 / 录制 / 校验 / 离线验证 / 回放
> - [docs/openhlm_vla/implementation_analysis.md](docs/openhlm_vla/implementation_analysis.md) —— VLA 部署链路(`zmq_vla`)测试命令
> - [scripts/test_vla_motion_stream.py](scripts/test_vla_motion_stream.py) —— VLA 下发仿真的场景扫法
> - [JAKA_RECORDED_DATA_REFERENCE.md](JAKA_RECORDED_DATA_REFERENCE.md) —— 录制数据校验 / 重建自检
> - [MIGRATION_PLAN.md](MIGRATION_PLAN.md) —— 历史设计命令(已过时,见 §9)
>
> 数据格式细节(每一维怎么算、参考系)见 `JAKA_RECORDED_DATA_REFERENCE.md`;
> 训练侧格式见 `TRAINING_DATA_FORMAT.md`;`zmq_vla` 缓冲语义见
> `docs/openhlm_vla/implementation_analysis.md`。本文只管"怎么跑"。

## 场景导航

| 想做什么 | 去哪节 |
|---|---|
| 跑起来看看机器人遥操作 | §1 |
| 用真实 pico 头显遥操作 | §1.3 |
| 录 VLA 训练数据 | §3 |
| 校验录的数据有没有问题 | §4 / §8 |
| 没有 pico / hub 时验证策略 | §5 |
| 回放旧数据渲染视频 | §6 |
| 测 VLA 部署链路(无 VLA 服务器) | §7 |

---

## 0. 环境准备

```bash
source .venv/bin/activate

# IsaacSim 渲染需要(回放 `replay-jaka` 会自动设置)
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs:$LD_LIBRARY_PATH"
```

---

## 1. 单终端全身 MF 策略遥操作(核心)

仿真 + 全身 MF 策略 + 运动缓冲都在 SIMPLE 一个终端跑。启动后**自动进入 policy 模式**
(`init_base_z=0.596` 站立高度),机器人直接站住并跟踪参考。

```bash
export TASK_NAME=JakaTabletopPickTeleop-v0
python -m simple.cli.teleop_jaka_mf \
  simple/$TASK_NAME \
  --controller keyboard \
  --no-headless
```

- CLI 只保留"每次运行会变"的开关:`env_id`、`--config`、`--target`、`--controller`、
  `--headless/--no-headless`、`--debug-log`。**其余参数全在配置文件
  `data/jaka_mf/teleop_jaka_mf.yaml`**(策略、频率、初始高度、ZMQ 端口、相机频率…),
  日常调参改那个文件。state + 相机**始终发布**。
- `JakaTabletopPickTeleop-v0`:桌上固定一个物体(默认 `objaverse:1128` 箱子),随机
  干扰物已关闭;换物体改 `--target` 或
  `src/simple/tasks/jaka_tabletop_pick_teleop.py` 的 `TargetDRCfg(asset_id=...)`。

**常用参数**

| 参数 | 说明 |
| --- | --- |
| `--target <id>` | 目标物体(默认 `objaverse:1128`) |
| `--controller pico/keyboard` | 输入源(只影响 `space` 暂停等;模式切换已禁用,见 §2) |
| `--no-headless` | 打开 MuJoCo 窗口(无左右面板 + 跟踪相机) |
| `--config <yaml>` | 运行配置,默认 `data/jaka_mf/teleop_jaka_mf.yaml` |
| `--debug-log <path>` | 每步 JSONL 诊断快照(对账参考链路的入口) |

**配置文件 `data/jaka_mf/teleop_jaka_mf.yaml` 常用键**

| 键 | 默认 | 说明 |
| --- | --- | --- | 
| `init_base_z` | `0.596` | 初始站立高度。场景 spawn 在 ~0.793、policy 平衡点在 ~0.596,用平衡高度起步才不下坠 |
| `policy_config` | `data/jaka_mf/...` | 策略 yaml(ONNX 默认取其同名 `.onnx`) |
| `motion_backend` | `zmq` | `zmq`(订阅 `:28701`)/ `raw_npz`(离线 npz)/ `zmq_vla`(VLA 部署,见 §7) |
| `rl_rate` / `camera_hz` | `50` / `30` | 控制频率 / 相机发布频率(state 按 `rl_rate` 发) |
| `enable_elastic_band` | `false` | 弹力带(虚拟吊带),当前默认关闭 |
| `reset_on_record_end` | `true` | 运动流 `toggle_data_collection` 由高变低(录制结束)时重置环境 |
| `state_zmq_bind` / `camera_zmq_bind` | `:28711` / `:28712` | 供 `record_jaka_zmq` 订阅的发布端口 |

### 1.3 live 遥操作:用真实的 sim2real pico hub

`motion_backend=zmq`(默认)的数据源可以是真实的 sim2real pico 重定向 hub
(`/home/xu/code/cxr/sim2real-jaka`,conda `teleop` 环境)。真实 hub 同时发运动流
`:28701` 和手柄按钮 `:5592`,与 SIMPLE 的 wire 协议完全一致。

**终端 A —— 启动真实 hub(在 sim2real-jaka 仓库)**:

```bash
cd /home/xu/code/cxr/sim2real-jaka
# 必须显式 --robot jaka(默认 g1),否则 joint/body 顺序不是 JAKA 序,SIMPLE 校验会失败
# publish_hz 建议 50;录制端 30Hz 只取最新帧,所以 hub >= 30 即可
conda run -n teleop python sim2real/teleop/pico_retarget_pub.py \
  --robot jaka \
  --bind tcp://*:28701 \
  --controller_bind tcp://*:5592 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

**终端 B —— SIMPLE 单终端遥操作(不变,默认 `motion-backend zmq` 即订阅 `:28701`)**:

```bash
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 \
  --target=objaverse:425 --controller pico --no-headless
```

要点:

- **必须 `--robot jaka`**:真实 hub 默认 `g1`,不传会发布 G1 序,SIMPLE 校验失败。
- 两个端口都不用改(`:28701` motion / `:5592` 手柄)正好是 SIMPLE 订阅的地址。
- 真实链路需要硬件(pico 头显 + GMR 重定向模型)。不戴头显时参考回退到中立站姿
  `ZMQ_DEFAULT_QPOS`(机器人原地站住),或用 §5.2 的 fake 播 npz 代替。
- 上线前建议先跑 §5 的 fake 验证,再切真实 hub。

---

## 2. 键盘键位

**MuJoCo 窗口**(弹力带/虚拟吊架):`7` 拉高、`8` 放低、`9` 释放/启用。

**终端键盘**:

- `space` —— 暂停/播放参考。
- `r` + 回车 —— 手动触发 reset(重建场景 + 重随机化 DR + 回到 `init_base_z` 站立并
  重新进入 policy)。与 pico 的 `toggle_data_collection` 下降沿(录制结束)是两个并列
  触发条件;没接 hub 时用 `r` 就能测这套 reset。

> ⚠️ **模式切换已禁用**:`BasePolicy.process_controllers()` 里按手柄/键盘切换
> `init/zero/align/policy` 的调度已注释(`jaka_rl/base_policy.py`)。`i` `o` `[` `]`
> 和 pico A/B 都不再切模式(pico A 键被录制开关占用)。需要恢复时取消那段注释。

---

## 3. 录制 VLA 数据(两终端,当前工作流)

录制由**独立进程** `record_jaka_zmq` 完成,遥操作进程只负责把 state + 相机发到 ZMQ。
两个进程用「只取最新帧」(latest-only)对齐,无需共享控制环。

**终端 A —— 遥操作 + 发布 state/相机**(state 50Hz,相机 30Hz):

```bash
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0  --no-headless
```

**终端 B —— 独立录制(30Hz)**:

```bash
python -m simple.cli.record_jaka_zmq \
  --save-dir data/teleop_jaka_mf_zmq \
  --desc "put the bottle on the mouse pad" \
  --frequency 30 \
  --policy-config data/jaka_mf/latest56k_pico_dr.yaml \
  --trigger pico      # 可省略:默认 both(pico A 键 + 键盘回车都行)
  # --run-name exp01  # 可省略:默认用时间戳当这一层的目录名
```

**三路 ZMQ 数据来源(全部 latest-only)**

| 流 | 端口 | 来源 | 频率 |
| --- | --- | --- | --- |
| `action` | `:28701` | pico hub(`pico_retarget_pub`)**直接订阅,不经 motion buffer** | hub `--publish_hz` |
| `state` | `:28711` | 终端 A 的 teleop | 50Hz |
| `camera` | `:28712` | 终端 A 的 teleop(head_stereo_left) | 30Hz |

录制进程按 `--frequency 30` 采样:每次取三路各自的**最新一帧**。`action` 是**参考运动**
(frozen MF 策略要跟踪的目标),不是策略输出。

保存格式(LeRobot,`state`(30) + `actions`(40) + 头图 224×224):
`<save-dir>/<run-id>/level-<dr-level>/`,PNG 字节内嵌 parquet(读数据只读它),
`videos/*.mp4` 只是浏览副本。**已录数据不会被覆盖/删除**:run-id 撞名自动加 `-2`/`-3`。

> ⚠️ 录制端是**跟随电平**的:若它启动时 pico 已经开始录制(电平已是 True),会立即开录。
> 想测"第 N 帧开启沿",要么先起录制端再起 hub,要么把 `--record-start` 调大。
> 另:`--trigger keyboard`(旧默认)完全不看 pico 开关,会出现"teleop 重置了但录制端没反应"。

---

## 4. 校验录制的 action(可选)

对比「用 action 重建的 anchor 运动」与真值参考 `data/motion/motion_mf.npz`:

```bash
PYTHONPATH=src python scripts/verify_jaka_action_recon_error.py \
  --episode 000000 \
  --fps 30 --horizon 30 \
  --num-samples 300 --seed 0 \
  --plot-stats --plot-yaw
```

- `--fps` 必须与录制 `--frequency` 一致(**30**);`--horizon 30` 才是 1 秒。
  脚本默认 `--fps 50 --horizon 50`,**所以这里必须显式传 30**。
- 输出:`POS / ROT / YAW 1s` 误差、`SAVED yaw vs MOTION`(≈0 说明录制忠实)、
  `YAWVEL integ vs SAVED`(yaw_vel 积分漂移)。
- 可选出图:`--plot-stats` / `--plot-yaw` / `--plot-yaw-integ` / `--plot-yaw-dist`
  / `--plot-window <s>` / `--plot-t0 <frame>`。
- 校验对象默认 `--episode 000000`;episode 在 `--save-dir` 下按编号找
  (默认 `data/teleop_jaka_mf`,与录制端 `--save-dir` 对应)。

---

## 5. 离线验证(无需 pico / hub)

### 5.1 方式 A:offline replayer(npz 直喂策略)

```bash
python scripts/run_jaka_mf_npz.py --motion-backend raw_npz --steps 600 --headless
```

常用旋钮:`--env-id`(默认 `simple/JakaOpenTrashCanTeleop-v0`)、`--target`、
`--policy-config` / `--policy-model`、`--motion-npz`、`--steps`、`--no-headless`、
`--band`(挂弹力带)、`--record`(录 npz)。

### 5.2 方式 B:模拟 pico hub(zmq 流,走真实 json 路径)

```bash
# 终端 A:模拟 pico_retarget_hub,播 data/motion 的 motion
python scripts/fake_pico_motion_pub.py
# 终端 B:策略走真实 zmq 路径
python scripts/run_jaka_mf_npz.py --motion-backend zmq --steps 300 --headless
```

`fake_pico_motion_pub.py` 复刻真实 hub 的状态机,默认**先发 1 帧空闲快照再 live**
(空闲姿态 = `ZMQ_DEFAULT_QPOS` 的 FK,和策略无数据时的回退姿态一致,所以起脚本参考不会跳)。

### 5.3 fake 的确定性相位(不用手按键就能复现同一条时间线)

```bash
python scripts/fake_pico_motion_pub.py --live-at 50          # 先空闲 50 帧再 live
python scripts/fake_pico_motion_pub.py --live-at -1          # 永不 live(只发冻结站姿)
python scripts/fake_pico_motion_pub.py --pause-at 200 --pause-frames 100   # 第 200 个 live 帧暂停 100 帧
python scripts/fake_pico_motion_pub.py --record-start 100 --record-stop 400  # 同按 A 开/关录制电平
python scripts/fake_pico_motion_pub.py --keyboard            # 键盘当手柄(见下表)
```

其他旋钮:`--motion-npz`(默认 `data/motion/motion_mf.npz`)、`--bind`(默认
`tcp://*:28701`)、`--fps`(默认 50)、`--start-frame`、`--controller-bind`(默认
`tcp://*:5592`)、`--key-press-ms`(默认 200)。

> `--keyboard` 下开机是 idle(同真实 hub `paused=True`),按 `x` 才 live;
> `--record-start/stop` 失效。暂停时冻结在刚发出的那个姿态上(连续、不跳)。

### 5.4 fake 键盘按键对应(与 `pico_retarget_pub.py` 的手柄映射一致)

| 键盘 | 手柄按键 | hub 侧作用 | SIMPLE `PicoController` 解出的模式 |
| -- | --- | --- | --- |
| `a` | A = RightController key_one | 翻转 `toggle_data_collection`(录制开关) | `init`(单独按 A) |
| `x` | X = LeftController key_one | 翻转 `paused`(冻结/恢复参考) | —(X 不映射模式) |
| `b` | B = RightController key_two | 无(真实 hub 也只转发) | `zero`(单独按 B) |
| `y` | Y = LeftController key_two | 无 | — |
| `q` | — | 退出 | — |

> `a` + `b` 同时按住 = `A+B` 组合 → `policy`。按键会被压住 `--key-press-ms`(默认
> 200ms),两次同键要间隔超过这个时间才算两次边沿。

> 若不开 hub,`motion_backend=zmq` 的参考会回退到 `ZMQ_DEFAULT_QPOS` 的 FK 站姿
> (机器人能站住但不会跟踪运动)。

---

## 6. 回放渲染(Stage 2,`replay-jaka`)

> 仅适用于**旧 jaka-replay 格式**(含 `observation.body_poses` / `observation.scene_qpos` + mp4)。
> §3 的录制写的是 openhlm 格式(仅头图 + state 30 + action 40),**没有** body_poses/scene_qpos,
> 不能用此命令回放。

```bash
python -m simple.cli.replay_jaka simple/JakaTabletopPickTeleop-v0 \
  --data-dir data/jaka_teleop/jaka_tabletop_pick_teleop/level-0 \
  --save-dir data/replay_jaka --sim-mode mujoco --save-all --direct
```

常用旋钮:`--render-hz`、`--dr-level`(仅输出目录后缀)、`--num_episodes`(默认 -1 全部)、
`--policy-model` / `--policy-config`(闭环回放)、`--direct`(直放不跑策略)。

---

## 7. VLA 部署链路测试(`motion_backend: zmq_vla`,无 VLA 服务器 / 机器人)

测的是 `RealtimeMotionBufferVla`(接收 30Hz VLA 动作流 + 周期性推理停顿)。
前提:把 `data/jaka_mf/teleop_jaka_mf.yaml` 的 `motion_backend` 改成 `zmq_vla`
(改回 `zmq` 即恢复 pico 遥操作);`motion_zmq_connect` 保持 `tcp://127.0.0.1:28701`。

### 7.1 两终端仿真测试(核心跑法)

```bash
# 终端 B —— 被测的接收端(motion_backend 已设为 zmq_vla),先起来
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 \
    --headless --controller keyboard --debug-log /tmp/vla.jsonl
#   等到打印 [zmq] state ... Hz(约 13~15 s)再起终端 A

# 终端 A —— 仿真 openpi-eval 下发(把录制数据按 VLA 下发模式发到 :28701)
python scripts/test_vla_motion_stream.py --episode 0
```

**顺序很重要**:ZMQ PUB 不缓存,订阅者没连上前发的消息直接丢弃,而 teleop 启动到进
控制循环要 ~15s。推荐**先 B 后 A**;想先 A 后 B,A 要加 `--loop`(数据循环回放,
B 随时加入都能收到)。

`--debug-log` 的 JSONL(`ref_joint` / `joint_qpos` / `base_pos` …)就是对账接收端的入口。

### 7.2 test_vla_motion_stream 的数据源与旋钮

- 默认数据源(`--session`):`data/teleop_jaka_mf_real/20261015-135528`
  (也可 `--parquet` 直指 episode 文件,或 `--session` 指别的目录——两种落盘布局
  `<session>/data/chunk-000/episode_*.parquet` 与
  `<session>/<run-id>/level-0/data/chunk-000/episode_*.parquet` 都能直接指)。
- 下发模式(对标 openpi-eval):`--control-hz 30`(控制拍)、
  `--open-loop-horizon 25`(每 25 拍重推理一次)、`--inference-ms 120`(该拍阻塞、
  一帧不发)、`--num-frames-to-send 2`(2 帧滑动窗口 `[i-1, i]`)。
- `--anchor-source recorded`(默认,发录制 anchor 真值,参考 === 录制轨迹)vs
  `integrated`(只用 VLA 真会输出的 `[30:33]` 积分,与真实部署逐位一致,但有
  mean 5.6cm / max 8.0cm 漂移,脚本会打印漂移量)。
- `--hold-s`(数据放完后保持末帧的秒数,默认 0;跑长实验加大)、`--connect-settle-s`
  (开流前等订阅者,默认 1s)、`--seed`、`--max-seconds`、`--bind`(默认 `tcp://*:28701`)。

### 7.3 延时/异常场景扫法

```bash
python scripts/test_vla_motion_stream.py --inference-ms 0            # 无停顿的理想流(基线)
python scripts/test_vla_motion_stream.py --inference-ms 400          # 停顿比 delay(120ms)还长(长停顿语义)
python scripts/test_vla_motion_stream.py --inference-ms 120 --inference-jitter-ms 60   # 推理耗时抖动
python scripts/test_vla_motion_stream.py --drop-every 10             # 每 10 条消息丢 1 条(验滑动窗抗丢帧)
python scripts/test_vla_motion_stream.py --final-stall-s 5           # VLA 服务挂掉:数据流彻底停住
python scripts/test_vla_motion_stream.py --anchor-source integrated  # 复现 openpi-eval 的积分路径 + 漂移报告
```

### 7.4 用真的 OpenHLM env 当发送端

`test_vla_motion_stream.py` 是按 openpi-eval 逻辑**复刻**的发送端。要验证**真代码**,
直接用 `docs/openhlm_vla/jaka_tabletop_env.py`,它的 `step()` 就是那条打包路径:

```python
from jaka_tabletop_env import JakaTabletopEnv
env = JakaTabletopEnv(mock=False, publish=True, control_hz=30)   # PUB :28701
env.step(actions33[i])                                            # 逐帧下发
```

(规范集的 `actions[:, 0:33]` 正好就是它要的策略序 action。2026-09-19 实测:
真 env 与复刻发送端两条路径的参考链路误差 / 闭环相关系数一致,复刻是忠实的。)

### 7.5 关键注意事项

1. **`[send]` 显示 ~27 Hz 而不是 30 Hz 是正常的**,不是丢帧:每 25 帧里有 1 拍要阻塞
   推理时间,`25 / (24×33.3ms + 120ms) ≈ 27.2 Hz`。openpi-eval 真机同样到不了 30Hz,
   这也意味着**参考被放慢约 9%** 播放。
2. **两份数据集的 `actions[0:27]` 关节序不一样**,喂参考时按 `meta/info.json` 的
   `features.actions.names` 按名字判、别按位置猜:
   - `data/simple/JakaTabletopPickTeleop-v0/level-0`(规范集,VLA 训练用)——
     **OpenPI 策略序**(手臂在前),= VLA 的输出空间;
   - `data/teleop_jaka_mf/<run>/level-0`(`record_jaka_zmq` 直出)——
     **JAKA/MuJoCo 序**(腿在前)。
   搞错不会报错,只会让参考关节整体错位、机器人直接摔。脚本启动会打印
   `[data] actions 关节序: ...`,接别的数据源先看这一行。
3. `--anchor-source integrated` 时若 episode 有全零 action 行(pico 未就绪时录到的),
   积分链会塌到零位姿 —— 建议换一集。
4. 观测流对齐:openpi-eval 侧 `main.py` 的 `jaka_state_port` / `jaka_head_image_port`
   要与 `teleop_jaka_mf.yaml` 的 `state_zmq_bind` / `camera_zmq_bind`(28711/28712)同步改。

---

## 8. 辅助自检命令

```bash
# action_trunk_reconstruct 自检:从 action trunk 反推参考轨迹与 155 维 command
# (实测:位置误差 3e-8 m,command 误差 4e-9。⚠️ 自检里 fps 硬编码 50,录制数据是 30,套用要改)
PYTHONPATH=src python -m simple.jaka_rl.action_trunk_reconstruct
```

```bash
# 查看已录 episode 列表 / 删除 episode(delete_episodes.py)
python scripts/delete_episodes.py --help
```

---

## 9. 历史命令(MIGRATION_PLAN.md,已过时)

> 以下来自迁移计划文档,**CLI 参数已收敛**(见 §1),仅当考古/对照设计用:

```bash
# 当时的目标形态(--record/--success-criteria 等参数现由配置文件承担)
export TASK_NAME=JakaOpenTrashCanTeleop-v0
python -m simple.cli.teleop_jaka_mf \
  simple/$TASK_NAME \
  --target=graspnet1b:0 \
  --sim-mode=mujoco \
  --record \
  --no-headless \
  --success-criteria=0.9 \
  --policy-config checkpoints/jaka_mf_v1_dr/latest56k_pico_dr.yaml
```

---

## 10. 端口 / 频率速查

| 端口 | 协议 | 方向 | 内容 |
| --- | --- | --- | --- |
| `:28701` | ZMQ PUB(JSON(pico)或二进制 v1(VLA)) | pico hub / openpi-eval → SIMPLE | 参考运动(30~50Hz) |
| `:28711` | ZMQ PUB(裸 JSON) | SIMPLE → 录制端 / openpi-eval | state(`joint_pos` 27 JAKA 序 / `body_quat_w` …),50Hz |
| `:28712` | ZMQ PUB(裸字节) | SIMPLE → 录制端 / openpi-eval | `struct.pack("iii", w,h,c)` + HWC uint8 RGB 头图,30Hz |
| `:5592` | ZMQ(手柄按钮) | pico hub → SIMPLE | A/B/X/Y 电平(模式切换已禁用,A 只当录制开关) |

| 频率 | 值 | 含义 |
| --- | --- | --- |
| `rl_rate` | 50 Hz | 策略控制循环(get_obs 每 tick 推进播放时钟 20ms) |
| `camera_hz` | 30 Hz | 头相机发布 |
| 录制 `--frequency` | 30 Hz | record_jaka_zmq 采样 |
| VLA 下发 | 30 Hz 名义 / ≈27.2 Hz 实际 | 30Hz 控制拍,每 25 拍 1 次 120ms 推理停顿 |

---

## 11. 相关文件索引

| 文件 | 作用 |
| --- | --- |
| `src/simple/cli/teleop_jaka_mf.py` | 单终端遥操作 CLI(§1) |
| `src/simple/cli/record_jaka_zmq.py` | 独立录制进程(§3) |
| `src/simple/cli/replay_jaka.py` | 回放渲染 CLI(§6) |
| `data/jaka_mf/teleop_jaka_mf.yaml` | 遥操作/部署配置(motion_backend、端口、频率) |
| `data/jaka_mf/latest56k_pico_dr.yaml` | 策略配置(future_steps、anchor_body_name、顺序表) |
| `scripts/fake_pico_motion_pub.py` | 模拟 pico hub(§5.2/5.3) |
| `scripts/run_jaka_mf_npz.py` | 离线策略 replayer(§5.1) |
| `scripts/test_vla_motion_stream.py` | 仿真 openpi-eval 下发(§7) |
| `scripts/verify_jaka_action_recon_error.py` | 录制 action 重建误差校验(§4) |
| `scripts/delete_episodes.py` | 删除已录 episode(§8) |
| `src/simple/jaka_rl/action_trunk_reconstruct.py` | 从 action trunk 重建参考轨迹(§8) |
| `src/simple/jaka_rl/motion_buffer.py` | `RealtimeMotionBuffer`(pico)/ `RealtimeMotionBufferVla`(VLA) |
| `src/simple/interfaces/jaka_zmq_pub.py` | state / 相机 ZMQ 发布端 |
| `src/simple/datasets/jaka_lerobot.py` | openhlm 数据格式(state 30 / action 40) |
| `docs/openhlm_vla/{main.py, jaka_tabletop_env.py}` | openpi-eval 侧真代码(§7.4) |
| `data/motion/motion_mf.npz` | 离线参考运动 |
