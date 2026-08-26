# SAM3 — 文本/点/框提示图像分割服务

基于 [Meta AI SAM3](https://github.com/facebookresearch/sam3) 的图像分割服务，支持两类提示：

- **文本提示**（`/segment-text`）：接收图片和**一个或多个文本提示**，通过一次前向推理返回每个提示对应的分割掩码、边界框和置信度分数（检出所有匹配目标）。
- **点/框提示**（`/segment-geometry`）：接收批量的正点 `[x, y]` 和/或框 `[x1, y1, x2, y2]` 提示，**每个提示返回其所指那一个目标的 mask**（SAM1 风格单目标分割）。

**环境要求：** NVIDIA GPU + CUDA 12.8, Python 3.12

**默认端口：** 8004

## 安装

```bash
cd services/sam3
bash setup.bash
```

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8004 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8004` | 监听端口 |
| `--checkpoint-path` | `./checkpoints/sam3.pt` | 模型检查点路径 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/segment-text` | 文本提示图像分割（检出所有匹配目标） |
| `POST` | `/segment-geometry` | 点/框提示分割（每个提示对应一个目标的 mask） |

## `/segment-text` 请求格式

`text_prompts` 支持单个字符串或字符串列表，多个提示会在一次前向中同时分割：

```json
{
  "image": "<base64 编码的图片>",
  "text_prompts": "a red car",
  "confidence_threshold": 0.5
}
```

```json
{
  "image": "<base64 编码的图片>",
  "text_prompts": ["a red car", "a person", "a dog"],
  "confidence_threshold": 0.5
}
```

- `image`：base64 编码的图片数据
- `text_prompts`：分割文本提示，可为单个字符串或字符串列表（多提示一次推理完成）
- `confidence_threshold`（可选）：置信度阈值，所有提示共用，默认 0.5

### 成功响应

```json
{
  "status": "success",
  "results": [
    {
      "text_prompt": "a red car",
      "masks": ["<base64 编码的 PNG 掩码>"],
      "boxes": [[x1, y1, x2, y2]],
      "scores": [0.95],
      "num_objects": 1
    },
    {
      "text_prompt": "a person",
      "masks": ["<base64 编码的 PNG 掩码>", "<base64 编码的 PNG 掩码>"],
      "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]],
      "scores": [0.91, 0.82],
      "num_objects": 2
    }
  ],
  "metadata": {
    "text_prompts": ["a red car", "a person"],
    "confidence_threshold": 0.5,
    "num_objects": [1, 2],
    "generation_time": 2.5,
    "image_size": "1024x768"
  }
}
```

`results` 列表按请求中 `text_prompts` 的顺序排列，每个元素包含该提示对应的掩码、边界框、分数与检出数量。

## `/segment-geometry` 请求格式

`points` 与 `boxes` 至少提供一个，均为**原图像素坐标**；响应中结果与输入列表按索引一一对应：

```json
{
  "image": "<base64 编码的图片>",
  "points": [[520, 375], [300, 200]],
  "boxes": [[59, 144, 76, 163]]
}
```

- `image`：base64 编码的图片数据
- `points`（可选）：正点提示列表，每个 `[x, y]`，返回该点所点中目标的 mask
- `boxes`（可选）：框提示列表，每个 `[x1, y1, x2, y2]`（XYXY），返回框内目标的 mask

### 成功响应

```json
{
  "status": "success",
  "results": {
    "points": [
      {"mask": "<base64 编码的 PNG 掩码>", "score": 0.95},
      {"mask": "<base64 编码的 PNG 掩码>", "score": 0.88}
    ],
    "boxes": [
      {"mask": "<base64 编码的 PNG 掩码>", "score": 0.92}
    ]
  },
  "metadata": {
    "num_point_prompts": 2,
    "num_box_prompts": 1,
    "generation_time": 0.9,
    "image_size": "1024x768"
  }
}
```

- `results.points[i]` 对应 `points[i]`，`results.boxes[i]` 对应 `boxes[i]`
- `mask`：base64 编码的 PNG 二值掩码（前景 255），尺寸与原图一致
- `score`：模型预测的 mask 质量（IoU），越高越好
- 点提示内部采用官方推荐的 multimask 策略（三个候选中取分数最高者）；框提示直接输出单一 mask
- 校验失败（`points`/`boxes` 均为空、坐标越界、框坐标逆序等）返回 422

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8004 health

# 文本提示分割
uv run example_client.py --port 8004 segment-text --image photo.jpg --text "a red car"

# 多个文本提示同时分割（一次请求，多个物体；掩码按 prompt 保存为 prompt{p}_mask{m:03d}.png）
uv run example_client.py --port 8004 segment-text --image photo.jpg --text "a red car" "a person" --output-dir masks/

# 点/框提示分割（点、框均可重复给出；掩码保存为 point{i}_mask.png / box{i}_mask.png）
uv run example_client.py --port 8004 segment-geometry --image photo.jpg --point 520,375 --point 300,200 --box 59,144,76,163 --output-dir masks/
```
