# FLUX.2-klein-9B — 文本生成图片服务

基于 [Black Forest Labs FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) 的文本生成图片服务，通过 HuggingFace diffusers 提供。

**环境要求：** NVIDIA GPU + CUDA (24GB VRAM 及以上), Python 3.11

**默认端口：** 8003

## 前置条件

该模型为 **Gated 模型**，需要先在 Hugging Face 上登录并同意 [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) 才能下载权重。

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 登录（需要访问 token）
huggingface-cli login
```

## 安装

```bash
cd services/huggingface
bash setup.bash
```

## 启动服务

```bash
cd flux2-klein-9b
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8003 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8003` | 监听端口 |
| `--model` | `black-forest-labs/FLUX.2-klein-9B` | HuggingFace 模型标识 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/text-to-image` | 文本生成图片 |

## 请求格式

```json
{
  "prompt": "A cat holding a sign that says hello world",
  "seed": 42,
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "num_images_per_prompt": 1
}
```

### 参数说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string` | **必填** | 文本提示词 |
| `seed` | `integer` | `null` | 随机种子，设为 null 则随机 |
| `height` | `integer` | `1024` | 图片高度 (1-2048) |
| `width` | `integer` | `1024` | 图片宽度 (1-2048) |
| `num_inference_steps` | `integer` | `4` | 去噪步数。FLUX.2-klein 为 4-step 蒸馏模型，默认 4 步即可 |
| `guidance_scale` | `float` | `1.0` | CFG 引导系数。蒸馏模型推荐 1.0 |
| `num_images_per_prompt` | `integer` | `1` | 生成图片数量 (1-4) |

## 成功响应

```json
{
  "status": "success",
  "images": ["<base64 编码的 PNG 图片>"],
  "metadata": {
    "prompt": "A cat holding a sign that says hello world",
    "seed": 42,
    "generation_time": 3.2,
    "num_images": 1,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0
  }
}
```

## 显存优化

该服务默认启用 **`enable_model_cpu_offload()`**，将模型权重的非活跃部分自动卸载到 CPU 内存，峰值 VRAM 占用控制在 24GB 以内。

```python
# main.py 中的关键配置
pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()  # 关键：CPU offload
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8003 health

# 文本生成图片（默认 4 步蒸馏）
uv run example_client.py --port 8003 generate --prompt "a cat sitting on a sofa"

# 指定更多参数
uv run example_client.py --port 8003 generate \
  --prompt "A futuristic city at sunset, highly detailed, 8k" \
  --seed 42 \
  --steps 4 \
  --guidance 1.0 \
  --output city
```
