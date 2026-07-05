# SAM3 — 文本提示图像分割服务

基于 [Meta AI SAM3](https://github.com/facebookresearch/sam3) 的文本提示图像分割服务。接收图片和**一个或多个文本提示**，通过一次前向推理返回每个提示对应的分割掩码、边界框和置信度分数。

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
| `POST` | `/segment` | 文本提示图像分割 |

## 请求格式

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

## 成功响应

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

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8004 health

# 单个文本提示分割
uv run example_client.py --port 8004 segment --image photo.jpg --text "a red car"

# 多个文本提示同时分割（一次请求，多个物体；掩码按 prompt 保存为 prompt{p}_mask{m:03d}.png）
uv run example_client.py --port 8004 segment --image photo.jpg --text "a red car" "a person" "a dog" --output-dir masks/

# 指定置信度阈值并保存掩码
uv run example_client.py --port 8004 segment --image photo.jpg --text "car" "person" --confidence-threshold 0.8 --output-dir masks/
```
