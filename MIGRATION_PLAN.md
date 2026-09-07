# sim2real-jaka 策略栈整合进 SIMPLE 框架 — 迁移计划

> **实现状态：已完成**（实现见 `src/simple/jaka_rl/` + `src/simple/agents/jaka_mf_agent.py` +
> `src/simple/cli/teleop_jaka_mf.py` + `src/simple/robots/jaka.py`）。本文件保留为设计文档。
>
> **更新**：已按要求移除 MF **v2**（835 维）观测类与配置文件，只用 v1（620 维，`latest56k_pico_dr`）。
> 本文件下方的 v2 描述仅作历史背景，实际代码中已无 `jaka_mf_v2`。
>
> 运行方式（单终端，仿真 + 策略一体；仅 sim2real 的 `pico_retarget_hub` 需单独终端）：
> ```bash
> python -m simple.cli.teleop_jaka_mf simple/JakaOpenTrashCanTeleop-v0 \
>   --target=graspnet1b:0 --sim-mode=mujoco --no-headless --record \
>   --policy-config /home/xu/code/sim2real-jaka/checkpoints/jaka_mf_v1_dr/latest56k_pico_dr.yaml
> ```
> 关键实现决策（与 sim2real 的两处增强，均不改变离线 npz 输出）：
> - 观测用**按名解析 anchor body 与关节重排**（`_refresh_motion_indices`），使 live `zmq`
>   后端（pico 流为 MuJoCo 关节/body 序）与 offline `raw_npz`（IsaacLab 序）都正确。
> - v2 命令里 `motion_data.joint_pos` 用 `[0][:, reindex]`（而非 `[0, :, reindex]`），
>   规避 NumPy 高级索引把维度前移导致的关节乱序 bug。

## 0. 目标

把 `../sim2real-jaka` 的 Jaka MF 全身 RL 策略栈**融合进 SIMPLE 的 Task / Env / Agent / CLI 框架**，而不是简单把两个进程拼起来。最终形态对标 SIMPLE 现有的 `teleop_decoupled_wbc`：

```bash
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

- 终端 A（sim2real，**不迁移**）：`pico_retarget_hub`（`pico_retarget_pub.py`）→ 发布运动流 `:28701`、手柄按钮 `:5592`。
- 终端 B（SIMPLE，本次迁移）：单进程内，用 SIMPLE 定义的 Jaka task 场景（HSSD 房间 + 桌子 + 物体 + 铰链物体）跑仿真，`JakaMFAgent` 订阅 pico 运动、构建 MF 观测、跑 ONNX、输出全身 q_target，机器人按 sim2real 的 PD 增益闭环跟踪，并可录制 LeRobot VLA 数据。

---

## 1. 现状与对应关系

### 1.1 sim2real 模块 → 迁移动作

| sim2real 模块 | 作用 | 迁移动作 |
|---|---|---|
| `rl_policy/base_policy.py` (`BasePolicy`) | 控制模式状态机 + 观测组装 + ONNX + 发命令 | **迁移**，改造成 in-process（去 ZMQ） |
| `rl_policy/tracking.py` | 注入 zmq motion backend | 并入 `JakaMFAgent` / runner |
| `rl_policy/observations/{base,jaka_mf,jaka_mf_v2}.py` | 观测类 + 注册表（620/835 维） | **迁移**（几乎原样） |
| `rl_policy/utils/motion_buffer.py` (`RealtimeMotionBuffer`) | 订阅 pico `:28701` 运动流并插值缓冲 | **迁移**（原样，保留 ZMQ） |
| `rl_policy/utils/motion.py` | `MotionData` + slerp | **迁移** |
| `rl_policy/utils/npz_motion.py` | 离线 npz 后端 | 可选迁移（离线对齐/回放用） |
| `rl_policy/utils/state_processor.py` | 收 LowState → root_quat/joint_pos；管理 motion backend | **迁移 + 改造**：改 in-process 读 `mjData` |
| `rl_policy/utils/command_sender.py` (`ActionManager`) | 发 LowCmd | **迁移 + 改造**：改 in-process 写 PD 目标 |
| `rl_policy/inference/onnx_module.py` | onnxruntime 封装 | **迁移**（处理 `next_*` 隐状态） |
| `rl_policy/controllers/{base,keyboard,pico}.py` + `control_mode.py` | init/align/zero/policy 来源 | **迁移** |
| `utils/math.py` | 四元数运算 | **迁移**（收敛成一个模块） |
| `config/robots/Jaka.py` (`JAKA_CFG`) | 关节/body 名、限位、kp/kd、default qpos | **收敛**到 `simple/jaka_rl/config.py` |
| `utils/common.py` (`LowState/LowCmd`) | 二进制消息 | **不迁移**（SIMPLE 已有 `interfaces/messages.py`） |
| `sim_env/base_sim.py` / `bridge.py` / `elastic_band.py` | 裸仿真 + ZMQ bridge + 弹力带 | **已被 SIMPLE 覆盖**（见 1.2），不迁移，复用 |

### 1.2 SIMPLE 已有基础（直接复用）

- **Task 场景**：`src/simple/tasks/jaka_*_teleop.py`（已镜像全部 G1 teleop 任务），含 `dr_cfgs`（HSSD 房间/桌子/物体/铰链物体/光照/材质）、`jaka_vla_cameras.py`（head/front 双目相机）、`check_success`。
- **Env 注册**：`src/simple/envs/__init__.py` 已注册 `simple/Jaka*Teleop-v0` → `LocoManipulationEnv`。
- **机器人**：`src/simple/robots/jaka.py`（27-DOF `Jaka`，`setup_control`/`apply_action`/`get_robot_qpos`）。
- **PD bridge（等价 sim2real `SimulationBridge`）**：`src/simple/interfaces/zmq_bridge.py` `ZMQSimBridge`，已含 `_init_mujoco_indices`（关节/actuator/root/IMU 地址）、`apply_pd`（子步 PD + 扭矩限幅）、`create_without_zmq`（无 ZMQ 模式）。
- **录制**：`src/simple/datasets/jaka_lerobot.py`（LeRobot exporter + `build_jaka_frame` + `body_poses` + `scene_qpos`）。
- **离线回放策略**：`src/simple/datasets/jaka_policy.py`（`JakaPolicyReplay`，620/835 维观测已内联实现，可做交叉验证）。
- **CLI 范式**：`src/simple/cli/teleop_decoupled_wbc.py` + `src/simple/agents/pico_decoupled_agent.py`（agent + env + 录制状态机的标准写法）。

---

## 2. 目标架构（SIMPLE 框架内）

```
python -m simple.cli.teleop_jaka_mf simple/JakaOpenTrashCanTeleop-v0 ...

  gym.make(env_id) ──► LocoManipulationEnv (task=JakaOpenTrashCanTeleop)
                         ├─ MujocoSimulator (HSSD room + table + objects + jaka)
                         └─ task.robot = Jaka (扩展后支持 MF 策略闭环)

  JakaMFAgent(robot)      (对标 PicoDecoupledAgent)
     ├─ RealtimeMotionBuffer ──ZMQ SUB──► tcp://127.0.0.1:28701 (pico hub)
     ├─ JakaMFPolicy (迁移的 BasePolicy)
     │     ├─ StateProcessor (in-process 读 mjData)
     │     ├─ Observation: jaka_frame_stack_mf / _v2 (620/835)
     │     ├─ ONNXModule (onnxruntime)
     │     └─ Controller (pico :5592 / keyboard) → control_mode
     └─ get_action() → ActionCmd(target_q 27 维 / elastic_band)

  主循环 (50 Hz，镜像 teleop_decoupled_wbc):
     action = agent.get_action(observation, instruction, privileged_info)
     observation, ... = env.step(action)   # robot.step(command) 内部 500 Hz 子步 PD
     recording 状态机 (record)
```

关键点：**仿真↔策略之间不再有 ZMQ**。策略直接读 `mjData`、直接写 PD 目标；唯一保留的 ZMQ 是「pico 运动流 `:28701`」和「pico 手柄 `:5592`」（数据源在另一台终端）。

---

## 3. 文件落点

新增包 `src/simple/jaka_rl/`（策略栈，纯逻辑，不碰 SIMPLE 核心）：

```
src/simple/jaka_rl/
  __init__.py
  config.py            # JAKA_CFG：JOINT_NAMES / BODY_NAMES / POLICY_JOINT_NAMES / NPZ_*_NAMES /
                       #           KP / KD / DEFAULT_JOINT_POS / DEFAULT_QPOS / 限位 / IMU / anchor
  math.py              # 四元数（quat_mul/conjugate/rotate_inverse/rotate/yaw_quat…）
  motion.py            # MotionData + _normalize_quat_batch + _quat_slerp_batch
  motion_buffer.py     # RealtimeMotionBuffer（订阅 :28701）
  npz_motion.py        # NpzMotionDataset（可选）
  control_mode.py      # ControlMode / PicoButtonState / resolve_*
  controllers/{__init__,base,keyboard,pico}.py
  inference/{__init__,onnx_module}.py
  observations/{__init__,base,jaka_mf,jaka_mf_v2}.py
  state_processor.py   # 改造：in-process 读 mjData
  action_manager.py    # 改造：in-process 写 PD
  base_policy.py       # BasePolicy（状态机 + 观测 + ONNX）
```

改动现有文件：

```
src/simple/robots/jaka.py            # 扩展：elastic_band / prepare_obs / command / step(子步 PD)
src/simple/agents/jaka_mf_agent.py   # 新增 JakaMFAgent(BaseAgent)
src/simple/cli/teleop_jaka_mf.py     # 新增 CLI（镜像 teleop_decoupled_wbc）
src/simple/datasets/jaka_lerobot.py  # 复用（可能小改：action 来源 = 策略 q_target）
src/simple/envs/__init__.py          # 可选：新增 Jaka*Teleop → 专用 Env（若需要）
```

---

## 4. 逐模块迁移要点

### 4.1 `jaka_rl/math.py`、`motion.py`、`npz_motion.py` — 原样迁移

去 sim2real import，改成 `simple.jaka_rl.*`；去 `any4hdmi`/`scipy` 依赖（`math.py` 只保留 numpy 四元数运算）。

### 4.2 `jaka_rl/motion_buffer.py` — `RealtimeMotionBuffer`（核心，保留 ZMQ）

- **原样保留**：ZMQ SUB `motion_zmq_connect=tcp://127.0.0.1:28701`，`recv_string` → `json.loads`，`__append_payload`（时间戳排序/二分插入/去重）、`_fill_sample_frames_locked`（线性 + slerp 插值）、`cleanup`、`get_obs()`（返回 `MotionData`）。
- **改造**：构造函数的「默认站姿 FK」不再用 sim2real 的 `robot_cfg.resolve_mjcf_path()`，改为**传入 SIMPLE 已编译的 `mj_model/mj_data`**（或 `jaka_rl/config` 里的 body/joint 名 + SIMPLE 的模型路径）。`joint_names`/`body_names` 取自 `config`。
- **payload 字段**（来自 pico_retarget_pub，见 §7）：`smplx_t_ns`、`joint_pos`（或 `dof_pos`/`qpos`）、`body_pos_w`、`body_quat_w`。`get_obs()` 里 `body_lin_vel_w`/`body_ang_vel_w` 恒 0（payload 不含速度）。
- `future_steps` 来自 yaml `motion.future_steps`。

### 4.3 `jaka_rl/observations/` — 原样迁移 + 注册表

- `base.py`：`_RegistryMixin` + `Observation` + `ObsGroup`（`__init_subclass__` 自动注册）。
- `jaka_mf.py` / `jaka_mf_v2.py`：**逻辑原样**，只改 import。依赖 `env.policy_config`、`env.default_dof_angles`、`state_processor.*`（§4.6 补齐）。
- `__init__.py`：导出注册表（只 export jaka_mf / jaka_mf_v2，G1 的 track/velocity 不用迁）。

### 4.4 `jaka_rl/config.py` — 收敛 Jaka 配置

把散落在 `robots/jaka.py`（ALL_JOINTS）、`zmq_bridge.py`（DEFAULT_KP/KD/DEFAULT_JOINT_POS）、sim2real `Jaka.py`（限位/effort/armature/default_qpos）合并为单一 `config.py`，字段对齐 `JAKA_CFG`：

- `JOINT_NAMES`（MuJoCo 顺序 27）、`BODY_NAMES`（28，含 `Right_wrist_yaw__Link` 双下划线）、`POLICY_JOINT_NAMES`、`NPZ_JOINT_NAMES`/`NPZ_BODY_NAMES`（IsaacLab 顺序）。
- `JOINT_KP`/`JOINT_KD`（regex）、`DEFAULT_JOINT_POS`、`DEFAULT_QPOS`、限位/effort。
- `ROOT_JOINT_NAMES=("base_joint",)`、`IMU_SITE_NAME="waist_imu"`、`ANCHOR_BODY="waist_yaw_Link"`、`ANCHOR_BODY_INDEX=3`。

> `ZMQSimBridge.DEFAULT_KP/KD/DEFAULT_JOINT_POS` 后续也改为从此 import，消除重复。

### 4.5 `jaka_rl/state_processor.py` — 迁移 + in-process 改造

- **保留**：`joint_names`、`qpos/qvel` 视图（`root_quat_w`/`root_ang_vel_b`/`joint_pos`/`joint_vel`）、`motion_config/motion_backend/motion_future_steps/motion_joint_names/motion_body_names/motion_data`、`_init_motion_backend()`（npz/raw_npz/zmq）、`_update_motion_data()`、`reset()/update()`。
- **改造 `_prepare_low_state()`**：删除 ZMQ LowState 订阅；改为持有一个「状态源」（bridge / robot / `mj_data`），直接读：
  - `root_quat_w` ← IMU framequat `waist_imu`（无则 `qpos[root+3:root+7]`）
  - `root_ang_vel_b` ← IMU gyro（无则 `qvel[root+3:root+6]`）
  - `joint_pos/joint_vel` ← 关节 qpos/qvel 地址（复用 `ZMQSimBridge._init_mujoco_indices` 的解析结果，或把该解析提为独立函数供两者共用）。
- G1 遗留的 `mocap_subscribers`/`register_subscriber` 可裁剪。

### 4.6 `jaka_rl/action_manager.py` — 迁移 + in-process 改造

- **保留**：从 yaml 解析 `joint_kp/joint_kd/default_joint_pos`（regex 匹配，复用 `zmq_bridge._match_param` 或内联）、`controlled_joint_indices`、`InitLowCmd`。
- **改造 `send_command()`**：不建 ZMQ；改为写 `bridge.cmd_q/cmd_dq/cmd_tau/cmd_kp/cmd_kd`，`reset_qpos/reset_qvel` 直接写 `mj_data`。扭矩计算仍复用 `ZMQSimBridge.apply_pd()`。

### 4.7 `jaka_rl/base_policy.py` — 迁移（核心状态机）

原样保留：`__init__`（读 yaml → StateProcessor → ActionManager → 解析 default/action_scale/policy_joint/controlled_joint/limits → setup_policy(ONNX) → setup_observations(注册表)）、`reset/update/prepare_obs_for_rl/get_init_target/get_align_target`、`set_init/align/zero/policy_mode`、`process_controllers`、`step()`、`policy(input_dict)`（ONNX + `next_*` 更新 + `action_clip` + `q_target=default+action*scale`）。

改造：`StateProcessor`/`ActionManager` 构造走 in-process（§4.5/4.6）；删 `get_robot_cfg`，用 `config`。保留 `run()` 的 sched 定时循环（供离线/单进程模式），但 SIMPLE 集成下主循环由 CLI 驱动，`step()` 作为 `JakaMFAgent.get_action()` 的底层。

### 4.8 `jaka_rl/controllers/` + `control_mode.py` — 原样迁移

键盘（`i/o/[/]`）、pico（`:5592` A/B → policy/init/zero）。

### 4.9 `jaka_rl/inference/onnx_module.py` — 迁移

处理 onnx 输入/输出名归一化（`_orig`/`next_`）、动态 batch、JSON meta。MF 模型若带 `next_*` 隐状态必须用这个（裸 `session.run` 会丢状态）。

---

## 5. SIMPLE 框架整合（本次的关键）

### 5.1 扩展 `Jaka` 机器人（对标 `G1Sonic`）

在 `src/simple/robots/jaka.py` 增加（或新建 `JakaMF` 子类/混入，避免污染现有 `Jaka`）：

- `sim_dt = 0.002`、`viewer_dt = reward_dt = image_dt = 0.02`（与 sim2real `sim_dt=0.002` + `decimation=10` 对齐）。
- `elastic_band`（从 `run_jaka_sim_server.py` 的 `ElasticBand` 或 gear_sonic 的 `ElasticBand` 迁移）+ `band_attached_link`（`waist_yaw_Link`）+ `use_floating_root_link`。
- `command`（复用现有 `command=None` 字段）→ 非 None 时 `MujocoSimulator.step()` 走 `robot.step(command)` 分支。
- `prepare_obs()` → 返回 `root_quat_w`（IMU framequat）、`root_ang_vel_b`（gyro）、`joint_pos`/`joint_vel`（MuJoCo 顺序）、`floating_base_pose/vel`。
- `step(command, replay, eval)`：**每 `env.step` 内跑 10 个物理子步**（`sim_dt=0.002` × 10 = 0.02s），每个子步用 `ZMQSimBridge.create_without_zmq` 的 `apply_pd()` 重算扭矩并 `mj_step`——与 `run_jaka_sim_server.py` / `replay_jaka.py` 闭环完全一致。
- `apply_action(action_cmd)`：识别 `position`（q_target）、`elastic_band`（释放/下放）、`reset_qpos`。

> 这样 `LocoManipulationEnv.step(action)` 的语义与 G1 一致：CLI 50 Hz 驱动，机器人内部 500 Hz 子步 PD。无需改 `MujocoSimulator`。

### 5.2 新增 `JakaMFAgent`（`src/simple/agents/jaka_mf_agent.py`）

**策略形态（已确认）**：与 sim2real 完全一致，采用**全身控制 MF 策略**——单个全身 ONNX 一次输出 27 维关节动作，不做 G1 那种「下肢 RL + 上肢 IK」的 decoupled 解耦。部署同样用这个全身策略。因此 `JakaMFAgent` 内部是「读状态 → 构造 620/835 维观测 → 全身 ONNX → 27 维 q_target → 机器人子步 PD 闭环」，比 `PicoDecoupledAgent` 简单，不需要 `TeleopPolicy`/`TeleopRetargetingIK`/手 IK 那一整套。

`BaseAgent` 子类，对标 `PicoDecoupledAgent`：

- `__init__(robot, policy_config, ...)`：建 `RealtimeMotionBuffer`（`config.joint_names/body_names` + zmq `:28701`）、`JakaMFPolicy`（`base_policy` 的 `BasePolicy`，`motion_backend=zmq`）、controller（pico/keyboard）。
- `get_action(observation, instruction, privileged_info)`：
  1. 读 `robot.prepare_obs()` → 喂给 `state_processor`（in-process）。
  2. `policy.step()` → 依据 control_mode（init/align/zero/policy）选 q_target。
  3. pico 按钮 → 弹力带下放/落地/释放、`reset_requested`。
  4. 返回 `ActionCmd("position", target_qpos=dict(zip(joint_names, q_target)))` 或 `ActionCmd("elastic_band", ...)`。
- `reset_policy()`：新 episode 时重置观测历史 / motion buffer / ref_to_robot_quat_init（对应 `BasePolicy.reset` + `set_align_mode`）。
- `publish_low_state`：pass（无需 Unitree bridge）。

### 5.3 新增 CLI `src/simple/cli/teleop_jaka_mf.py`

镜像 `teleop_decoupled_wbc.main`：

- `gym.make(env_id, sim_mode, render_hz, headless, max_episode_steps, target, dr_level, success_criteria)`。
- `agent = JakaMFAgent(robot, policy_config=..., ...)`。
- 主循环：`action = agent.get_action(...)` → `env.step(action)` → `env.update_viewer/update_reward`（若存在）→ 录制状态机（复用 `jaka_lerobot` 的 exporter + `_build_frame`，策略动作即 `q_target`）。
- 参数：`--policy-config`、`--policy-model`、`--motion-zmq-connect tcp://127.0.0.1:28701`、`--controller pico/keyboard`、`--pico-zmq-connect tcp://127.0.0.1:5592`、`--record`、`--success-criteria`。

---

## 6. 实施步骤（建议顺序）

1. **纯函数迁移**：`jaka_rl/math.py`、`motion.py`、`npz_motion.py`、`control_mode.py`、`controllers/*`、`inference/*`、`config.py`。跑 `py_compile`。
2. **运动缓冲**：`motion_buffer.py`，单测喂 JSON payload 验证 `get_obs()` 插值。
3. **观测类**：`observations/*` + 注册表。用 `JakaPolicyReplay`（`datasets/jaka_policy.py`，已知正确）对同一份 npz 交叉验证 `jaka_frame_stack_mf/_v2` 输出一致。
4. **StateProcessor / ActionManager 改造**：in-process 读状态 / 写 PD。
5. **BasePolicy 迁移**：状态机 + 观测 + ONNX 组装；先 `raw_npz` 离线跑通。
6. **机器人扩展**：`Jaka` 增加 elastic_band / prepare_obs / command / step(子步 PD)。
7. **JakaMFAgent + CLI**：`teleop_jaka_mf.py`，先 `raw_npz` 离线验证，再 `zmq` 连 pico hub 实测。
8. **录制打通**：复用 `jaka_lerobot`，把策略 q_target 作为 action 落盘，`--record` 出 VLA 数据。
9. 收敛重复配置（`zmq_bridge.DEFAULT_KP/KD` ↔ `jaka_rl.config`），更新 `JAKA_QUICKSTART.md`。

---

## 7. 风险与注意点

1. **关节顺序**：MuJoCo(27) ≠ npz/policy(27, IsaacLab)。观测用 `mujoco_to_isaaclab_reindex`，动作散射用 `controlled_joint_indices`，**下标必须与 yaml 完全一致**。
2. **IMU anchor 语义**：`jaka_mf.py` 里 `sp.root_quat_w` 直接是 `waist_imu`（装在 `waist_yaw_Link`）framequat，projected_gravity 用它；旧 `jaka.py` 是 root quat 再乘 `waist_yaw_joint`。**不要混淆**。bridge 读 `waist_imu` 的逻辑已实现，保留。
3. **`anchor_body_index=3`** 依赖 npz_body_names 顺序（base=0, L/R hip_pitch=1/2, waist_yaw_Link=3），迁移 body 名表顺序不能变。
4. **`Right_wrist_yaw__Link`（双下划线）**：body 名双下划线、关节名单下划线，按名匹配要小心。
5. **live 运动流 payload**：`joint_pos` 可能 NaN（SMPLX 流只给 body 姿态），可能带 `qpos`（configuration 源，含 base 7 维需剥掉）；无 `body_lin_vel_w/ang_vel_w`（v2 命令 vel 恒 0）。`motion_buffer.__append_payload` 已处理，保留。
6. **ONNX 隐状态**：若模型有 `next_*` 输出（GRU/LSTM），必须用 `ONNXModule` 的 `input_dict.update(next_state_dict)`。MF v1/v2 从维度看是纯堆帧无隐状态，但保险用 ONNXModule。
7. **子步 PD**：策略 50 Hz 只更新目标；`apply_pd()` 必须每个 500 Hz 子步重算（复用 bridge 语义），否则跟踪不稳。Jaka 机器人的 `step()` 要保证 10 子步。
8. **弹力带**：SIMPLE scene 模式 band anchor 按 `mj_data.qpos[:2]` 设（现有行为）；sim2real 固定 `[0,0,2.7]`。整合时保留 SIMPLE 行为。
9. **`need_gravity=True`**：Jaka teleop task 已设 `need_gravity=True`，机器人 `command` 非 None 时 `MujocoSimulator.step` 才走 `robot.step(command)` 分支——因此 `Jaka` 必须在策略激活时把 `command` 置非 None（即使只是占位），否则落到 `mj_step(nstep=1)` 慢动作。
10. **依赖**：`sshkeyboard`、`onnxruntime`、`pyyaml`、`zmq`、`glfw` 需在 `pyproject.toml` 确认（`onnxruntime/pyyaml/zmq` 已有，`sshkeyboard` 需确认/添加）。
11. **录制 action 来源**：录制时 action 应为策略 `q_target`（`bridge.cmd_q` / agent 缓存），与 `run_jaka_sim_server.py` 的 `action = bridge.cmd_q` 一致，保证回放闭环一致。

---

## 8. 验证

1. **离线对齐**（不连 pico）：`motion_backend=raw_npz`，喂 `jaka_data/*/motion*.npz`，对比 `JakaPolicyReplay` 与新 `jaka_frame_stack_mf/_v2` 的观测数值完全一致。
2. **闭环跟踪**（离线 npz）：机器人从默认站姿 `set_align_mode` → `set_policy_mode`，看是否稳定跟踪参考运动不倒地。
3. **live 遥操作**（连 pico hub）：`--motion-backend zmq`，戴头显动捕，机器人实时跟踪；`--record` 出 LeRobot 数据。
4. **回放**：录制的数据用 `replay_jaka --direct` / `--policy-model` 验证渲染与闭环。
