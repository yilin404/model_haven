# SDXL — 文本生成图片服务

基于 [Stability AI SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 的文本生成图片服务，通过 HuggingFace diffusers 提供。

**环境要求：** NVIDIA GPU + CUDA, Python 3.11

**默认端口：** 8002

## 安装

```bash
cd services/huggingface
bash setup.bash
```

## 启动服务

```bash
cd sdxl
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8002 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8002` | 监听端口 |
| `--model` | `stabilityai/stable-diffusion-xl-base-1.0` | HuggingFace 模型标识 |
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
  "prompt": "a photo of an astronaut riding a horse on mars",
  "seed": 42,
  "options": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    "negative_prompt": null,
    "num_images_per_prompt": 1
  }
}
```

## 成功响应

```json
{
  "status": "success",
  "images": ["<base64 编码的 PNG 图片>"],
  "metadata": {
    "seed": 42,
    "generation_time": 8.5,
    "num_images": 1,
    "height": 1024,
    "width": 1024
  }
}
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8002 health

# 文本生成图片
uv run example_client.py --port 8002 generate --prompt "a cat sitting on a sofa"
```
