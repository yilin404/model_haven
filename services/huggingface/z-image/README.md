# Z-Image — 文本生成图片服务

基于 [Alibaba Tongyi Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 的文本生成图片服务，通过 HuggingFace diffusers 提供。Z-Image 是 6B 参数的图像生成模型，Turbo 版本经蒸馏仅需 8 步推理，擅长写实图像生成、中英双语文本渲染。

**环境要求：** NVIDIA GPU + CUDA (16GB VRAM 及以上), Python 3.11

**默认端口：** 8007

## 模型缓存

`scripts/start-services.sh` 以 `HF_HUB_OFFLINE=1` 启动服务，因此首次使用前必须先以同一系统用户预下载权重到 HuggingFace 缓存：

```bash
cd services/huggingface
uv run -- hf download Tongyi-MAI/Z-Image-Turbo

# 国内网络可走 hf-mirror.com 镜像
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="${no_proxy:+$no_proxy,}hf-mirror.com"
uv run -- hf download Tongyi-MAI/Z-Image-Turbo
```

## 安装

```bash
cd services/huggingface
bash setup.bash
```

## 启动服务

```bash
cd z-image
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8007 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8007` | 监听端口 |
| `--model` | `Tongyi-MAI/Z-Image-Turbo` | HuggingFace 模型标识 |
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
  "prompt": "一只戴着墨镜的柴犬，坐在冲浪板上，海浪背景",
  "seed": 42,
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 8,
  "guidance_scale": 0.0,
  "num_images_per_prompt": 1
}
```

### 参数说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string` | **必填** | 文本提示词，支持中英文 |
| `seed` | `integer` | `null` | 随机种子，设为 null 则随机 |
| `height` | `integer` | `1024` | 图片高度 (1-2048) |
| `width` | `integer` | `1024` | 图片宽度 (1-2048) |
| `num_inference_steps` | `integer` | `8` | 去噪步数。Z-Image-Turbo 为 8-step 蒸馏模型，默认 8 步即可 |
| `guidance_scale` | `float` | `0.0` | CFG 引导系数。蒸馏模型推荐 0.0（不使用 CFG） |
| `negative_prompt` | `string` | `null` | 负面提示词，仅 `guidance_scale > 1` 时生效 |
| `num_images_per_prompt` | `integer` | `1` | 生成图片数量 (1-4) |

## 成功响应

```json
{
  "status": "success",
  "images": ["<base64 编码的 PNG 图片>"],
  "metadata": {
    "prompt": "一只戴着墨镜的柴犬，坐在冲浪板上，海浪背景",
    "negative_prompt": null,
    "seed": 42,
    "generation_time": 5.4,
    "num_images": 1,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 8,
    "guidance_scale": 0.0
  }
}
```

## 显存优化

该服务默认启用 **`enable_model_cpu_offload()`**，将模型权重的非活跃部分自动卸载到 CPU 内存，6B 模型（transformer + 文本编码器）峰值 VRAM 占用可控制在 16GB 左右。

```python
# main.py 中的关键配置
pipeline = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()  # 关键：CPU offload
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8007 health

# 文本生成图片（默认 8 步蒸馏推理）
uv run example_client.py --port 8007 generate --prompt "a cat sitting on a sofa"

# 指定更多参数
uv run example_client.py --port 8007 generate \
  --prompt "A futuristic city at sunset, highly detailed, 8k" \
  --seed 42 \
  --steps 8 \
  --width 1328 --height 1328 \
  --output city
```
