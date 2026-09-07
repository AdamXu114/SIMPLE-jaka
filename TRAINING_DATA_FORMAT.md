# OpenHLM 训练 VLA 所需数据格式

> 本文档是给「准备训练数据的 agent / 工程」用的自包含参考。
> 目标模型：OpenHLM 的 π₀.₅（PaliGemma 2B + Action Expert）。训练代码位于 `src/openpi4OpenHLM`。
>
> **一句话总结**：训练时模型直接消费 **LeRobot 格式** 数据集；原始遥操作/人类数据需先通过 `convert_g1_data_to_lerobot(_multi).py` 转换。最终每个 episode 的每条帧数据 = **3 张 224×224 图像 + 34 维 state + 34 维 action**，外加**每 episode 一句语言指令**。

---

## 1. 数据流总览

```
采集(原始)                                       训练消费(LeRobot)
┌───────────────────────────┐   convert_...py   ┌──────────────────────────┐
│ episode_XXXX/              │ ────────────────▶ │ $LEROBOT_HOME/<repo_id>/  │
│   data.json                │                    │  (meta + videos + parquet)│
│   videos/*.mp4             │                    └───────────┬──────────────┘
└───────────────────────────┘                                │
                                                            ▼
                                              ┌──────────────────────────┐
                                              │ compute_norm_stats.py    │→ 归一化统计
                                              │        +                 │
                                              │ train_pytorch.py         │→ π₀.₅ fine-tune
                                              └──────────────────────────┘
```

训练代码只认 **LeRobot 格式**（`src/openpi4OpenHLM/src/openpi/training/config.py:359` 的 `LeRobotG1DataConfig` / `DataConfig`）。所有原始数据必须先转换。

---

## 2. 原始采集数据格式（转换前的输入）

由 `src/GR00T-WholeBodyControl4OpenHLM/gear_sonic/data_utils/episode_writer.py` 生成。
**每个 episode 一个目录** `episode_XXXX/`：

```
episode_0000/
├── data.json
└── videos/
    ├── rgb_left.mp4          # 头部摄像头（可省略）
    ├── wrist_rgb_left.mp4    # 左手腕
    └── wrist_rgb_right.mp4   # 右手腕
```

### `data.json` schema

```json
{
  "info": {
    "version": "1.0.0",
    "video": { "format": "mp4", "codec": "mp4v", "fps": 30, "cameras": { "rgb_left": {"height":..,"width":..,"channels":3,"fps":30,"path":"videos/rgb_left.mp4"} } }
  },
  "text": {
    "goal": "pick up the red cup on the table",   // ← 作为语言指令（prompt）
    "desc": "...",
    "steps": "..."
  },
  "data": [
    {
      "idx": 0,
      "rgb_left":        {"video_path": "videos/rgb_left.mp4", "frame_index": 0},
      "wrist_rgb_left":  {"video_path": "videos/wrist_rgb_left.mp4", "frame_index": 0},
      "wrist_rgb_right": {"video_path": "videos/wrist_rgb_right.mp4", "frame_index": 0},
      "state_body":        [/* 32 个 float */],
      "state_hand_left":   <scalar float>,
      "state_hand_right":  <scalar float>,
      "action_body":       [/* 32 个 float */],
      "action_hand_left":  <scalar float>,
      "action_hand_right": <scalar float>
    }
  ]
}
```

关键点：

- **视频不存单帧图片**，每帧只记录 `{video_path, frame_index}`，转换时按帧号从 mp4 读回。
- `state_body` / `action_body` = **32 维**；`state_hand_*` / `action_hand_*` = **夹爪标量**。
- `text.goal` 是该 episode 的语言指令。

### `state_body`(32 维) 布局

转换脚本见 `examples/unitree_g1/convert_g1_data_to_lerobot.py:331-345`：

| 索引    | 内容          |
|---------|---------------|
| 0:3     | root(roll,pitch,yaw_vel) |
| 3:9     | leg_left (6)  |
| 9:15    | leg_right (6) |
| 15:18   | waist (3)     |
| 18:25   | arm_left (7)  |
| 25:32   | arm_right (7) |

---

## 3. 转换后的 LeRobot 格式（训练真正消费）

由 `examples/unitree_g1/convert_g1_data_to_lerobot.py`（单目录）或 `convert_g1_data_to_lerobot_multi.py`（多目录/采样/HuMI 混合）生成。

### features 定义（`convert_g1_data_to_lerobot_multi.py:641-667`）

| feature           | dtype   | shape         | 说明 |
|-------------------|---------|---------------|------|
| `head_image_left` | image   | (224, 224, 3) | 头部图像，统一 resize 到 224×224 |
| `left_wrist_image`| image   | (224, 224, 3) | 左手腕 |
| `right_wrist_image`| image  | (224, 224, 3) | 右手腕 |
| `state`           | float32 | (34,)         | 机器人状态 |
| `actions`         | float32 | (34,)         | 动作 |

另有**每个 episode 一个字符串 `task`**（来自 `text.goal`）。训练时通过 `task_index → prompt` 映射生成语言指令（`PromptFromLeRobotTask`，`src/openpi4OpenHLM/src/openpi/transforms.py:328`）。

### 34 维 OpenPI state/action 布局（**核心，顺序必须严格**）

| 维       | 内容          |
|----------|---------------|
| 0-6  (7) | left arm      |
| 7   (1)  | left gripper  |
| 8-14 (7) | right arm     |
| 15  (1)  | right gripper |
| 16-21(6) | left leg      |
| 22-27(6) | right leg     |
| 28-30(3) | waist         |
| 31-33(3) | root(roll,pitch,yaw_vel) |

原始 `state_body`(32)+夹爪标量 → 34 维的重排（`convert_g1_data_to_lerobot.py:350-395`）：

```python
state = np.concatenate([
    state_arm_left,   state_hand_left,    # 7 + 1
    state_arm_right,  state_hand_right,   # 7 + 1
    state_leg_left,   state_leg_right,    # 6 + 6
    state_waist,      state_root,         # 3 + 3
])
# actions 同理由 action_body + action_hand_* 组装，顺序一致。
```

### 图像处理注意

- 用 `resize_with_pad`：等比缩放 + 零填充，**不拉伸**（`convert_g1_data_to_lerobot.py:27-70`）。
- `convert_..._multi.py` 会把 2704×2028 的图像**先中心裁剪成 2028×2028** 再缩放（`:72-76`），该逻辑专门为头相机的鱼眼/立体图设计——若你的相机宽高比不同需自行调整。
- **头部相机可不提供**。`_multi.py` 自动检测 absence，缺头相机的 episode 把 `head_image_left` 存成**全零黑图**（`:333-339`）。模型侧 `G1Inputs` 用 `np.any(img>0)` 判断并生成 `image_mask` 把占位图屏蔽（`src/openpi4OpenHLM/src/openpi/policies/g1_policy.py:117-126`）。

### 存储位置

输出写到 `$LEROBOT_HOME/<repo_id>/`，默认 `~/.cache/huggingface/lerobot/<repo_id>/`。
`repo_id` 形如 `OpenHLM/example` → 数据应在 `$LEROBOT_HOME/OpenHLM/example/`。

### 磁盘存储结构（LeRobot v2 落盘）

> 参考 pin 版本 `huggingface/lerobot@0cf8648` 的 `lerobot/common/datasets/utils.py:54-57`。
> LeRobot v2 **不是每个 episode 一个目录**，而是按「数据类型 + chunk」散开存，靠**命名**与 `meta/` 关联。

**存储方式由 features 里视觉字段的 `dtype` 决定**：

| features 里的 `dtype` | 视觉数据落盘 | 路径模板（`info.json` 中的 `data_path`/`video_path` 用同名常量） |
|---|---|---|
| `"video"` | 每相机一个 `.mp4` | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| `"image"`（**OpenHLM 使用**） | 每帧一个 `.png` | `images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png` |
| 数值/向量字段 | 全部进 parquet | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |

`chunk` 每 **1000 个 episode** 一个目录（`DEFAULT_CHUNK_SIZE=1000`）。
OpenHLM 转换脚本把三个相机都声明为 `dtype:"image"` 且**未传 `use_videos`**（其默认值只决定 `video_path` 字段、不改变 dtype），故 `video_keys` 为空 → **不产生任何 mp4，全部走 `images/` 的 PNG**。

**一个转换好的数据集实际目录树**（`$LEROBOT_HOME/OpenHLM/example/`）：

```
OpenHLM/example/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet      # 第 0 个 episode 的表格
│       ├── episode_000001.parquet
│       └── ...
├── images/                              # dtype="image" 的视觉字段逐帧 PNG
│   ├── head_image_left/
│   │   ├── episode_000000/
│   │   │   ├── frame_000000.png
│   │   │   ├── frame_000001.png
│   │   │   └── ...
│   │   └── episode_000001/
│   ├── left_wrist_image/
│   │   └── episode_000000/frame_000000.png
│   └── right_wrist_image/
│       └── episode_000000/frame_000000.png
└── meta/
    ├── info.json             # features 定义 + data_path/video_path 格式化串 + fps + 统计
    ├── episodes.jsonl        # 每 episode 的 episode_index / tasks / length
    ├── tasks.jsonl           # task 字符串 → 供 prompt_from_task 用
    └── stats.json
```

**每个 episode 的 parquet 列**（`add_frame` + `save_episode` 写入）：

| 列 | 类型 | 含义 |
|---|---|---|
| `index` | int64 | 全局帧号 |
| `episode_index` | int64 | 所属 episode |
| `frame_index` | int64 | episode 内帧号 |
| `timestamp` | float32 | `frame_index / fps`（30Hz） |
| `task_index` | int64 | 指向 `meta/tasks.jsonl` 的任务索引 |
| `state` | float32[34] | 机器人状态（OpenPI 布局） |
| `actions` | float32[34] | 动作 |
| `head_image_left` / `left_wrist_image` / `right_wrist_image` | image struct | **内嵌**图像（`{"path": "...png", "image": <PIL>}`），训练读取直接拿到解码图；PNG 文件是可备份的副本 |

`index / episode_index / task_index` 由 `save_episode` 自动补；`state`/`actions` 由转换脚本塞进 `add_frame`；图像列经 `embed_images` 内嵌进 parquet（`save_episode_table` → `embed_images(ep_dataset)`）。

> 若你想让图像走 **mp4**（省磁盘、下载更快），把 features 里三个相机的 `dtype` 改成 `"video"` 即可——但**转换脚本需同步修改**，且相关 `image_writer` 路径与编码逻辑随之变化。默认 OpenHLM 数据集是 `images/` PNG。

---

## 4. 训练配置要求（`config.py`）

`openhlm_example` 的 TrainConfig（`config.py:871-886`）：

```python
TrainConfig(
    name="openhlm_example",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=34,           # 动作维数（= state 维数）
        action_horizon=50,       # 一次预测 50 帧 → 每样本 actions 形状 (50, 34)
        discrete_state_input=True,
    ),
    data=LeRobotG1DataConfig(
        repo_id="OpenHLM/example",     # → $LEROBOT_HOME/OpenHLM/example
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=False,
    ),
    ...
)
```

### 必须匹配的 4 个数据相关参数

1. **`action_dim=34`** — 与 `features` 里 `state`/`actions` 的维度一致。
2. **`action_horizon=50`** — 数据加载器用 `delta_timestamps` 打包每帧未来 50 帧动作成 `(50, 34)`（`data_loader.py:140-146`）。即你提供的 `actions` 必须能按帧号往后回溯出未来 50 帧。
3. **`discrete_state_input=True`** — 状态含离散接口（夹爪）。
4. **`prompt_from_task=True`** — prompt 由每帧的 `task_index` 从 dataset `tasks` 表取字符串（`data_loader.py:148-149`）。所以 `task` 字段是**训练集必备**的。

### 可选：增量动作

`LeRobotG1DataConfig` 的 `use_delta_joint_actions`（默认 True，示例设 False）决定是否把绝对关节角转为增量。mask 由 `make_bool_mask(7, -1, 7, -1, 15, -3)` 生成：手臂(7)/腿+腰(15)算 delta；夹爪(1)×2 与 root 角速度(3)保持绝对（`config.py:415-428`）。若为 True，`DeltaActions`/`AbsoluteActions` 变换会在训练与推理时执行，**务必保持转换脚本与 config 一致**。

### 归一化统计（训练前置，必须）

训练前必须跑：

```bash
cd src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name openhlm_example --max-frames 500000
```

否则 data loader 抛 `"Normalization stats not found"`（`data_loader.py:176-181`）。
默认 `use_quantile_norm=True`（`config.py:384`），即**分位数归一化**。

### 数据加载时 key 重映射

LeRobot 原始 key 经 `RepackTransform` 映射到模型输入（`config.py:391-403`）：

```
head_image_left      → observation/head_image_left
left_wrist_image     → observation/left_wrist_image
right_wrist_image    → observation/right_wrist_image
state                → observation/state
actions              → actions
task_index 的 prompt → prompt
```

---

## 5. 端到端准备 checklist

为训练准备数据的最小动作序列：

1. **采集原始数据**：`launch_data_collection.sh` / HuMI 流程，产出标准 `episode_*/data.json` + `videos/`。
2. **转换到 LeRobot**：
   ```bash
   cd src/openpi4OpenHLM
   uv run examples/unitree_g1/convert_g1_data_to_lerobot.py \
       --data_dir ./example_demonstration/20260520_1602_example_1 \
       --repo_name OpenHLM/example
   # 或多目录：
   uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
       --parent_dir ./example_demonstration \
       --dataset_folders 20260520_1602_example_1 20260520_1602_example_2 \
       --repo_name OpenHLM/example
   ```
3. **确认结果**：脚本会打印每 episode 帧数、视频尺寸、独特 `task` 数量。
4. **在 config.py 加 TrainConfig**：`repo_id` 指向你的数据集，设 `action_dim=34, action_horizon=50`。
5. **算归一化**：跑 `compute_norm_stats.py --config-name <你的config>`。
6. **训练**：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     scripts/train_pytorch.py <config> \
     --pytorch-weight-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
     --exp-name openhlm_example
   ```

---

## 6. 常见坑（最容易出问题的地方）

| 坑 | 说明 |
|----|------|
| **34 维顺序错** | state/action 必须严格「左臂7→左夹爪1→右臂7→右夹爪1→左腿6→右腿6→腰3→root3」。手把手臂/腿/root 顺序改错会静默伤害性能。 |
| **action_horizon 对齐** | 每样本 actions 需按帧回溯未来 50 帧；`delta_timestamps` 依赖 LeRobot 数据集中 actions 按帧对齐。 |
| **图像尺寸** | 必须 224×224（`features` shape & 模型 `ResizeImages(224,224)`）。未 resize 会报错或隐式拉伸。 |
| **头相机缺失** | 用全零黑图 + `image_mask`；不要真的不写 `head_image_left` feature。 |
| **缺失归一化** | 不跑 `compute_norm_stats.py` 会直接报 `"Normalization stats not found"`。 |
| **`use_delta_joint_actions` 不一致** | 转换侧若做 delta、config 若不做（或反之），训练与推理动作空间不一致。 |
| **数值类型** | `state`/`actions` 必须是 dtype `float32`（features 里声明，转换时 `np.array(...,float32)`）。 |

---

## 7. 参考代码文件清单

- 原始写入：`src/GR00T-WholeBodyControl4OpenHLM/gear_sonic/data_utils/episode_writer.py`
- 单目录转换：`src/openpi4OpenHLM/examples/unitree_g1/convert_g1_data_to_lerobot.py`
- 多目录/HuMI 转换：`src/openpi4OpenHLM/examples/unitree_g1/convert_g1_data_to_lerobot_multi.py`
- 训练数据配置：`src/openpi4OpenHLM/src/openpi/training/config.py`（`LeRobotG1DataConfig`、`openhlm_example`）
- G1 输入输出变换：`src/openpi4OpenHLM/src/openpi/policies/g1_policy.py`
- 数据加载器：`src/openpi4OpenHLM/src/openpi/training/data_loader.py`
- 归一化脚本：`src/openpi4OpenHLM/scripts/compute_norm_stats.py`
