# Qwen-Image 2.1 — 文本生成图片 & 多参考图编辑服务

基于 [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 的文生图与图像编辑服务，通过 HuggingFace diffusers 提供。Qwen-Image 2.1 是 7B 参数的单流 block-causal DiT 模型，配合 Qwen3-VL 文本编码器统一编码提示词与参考图，同一管线同时支持文生图与多参考图编辑（最多 10 张），原生 2048×2048 分辨率与 RGBA 透明输出。

**环境要求：** NVIDIA GPU (24GB VRAM 及以上，配合 CPU offload), Python 3.11

**默认端口：** 8010

## 模型缓存

`scripts/start-services.sh` 以 `HF_HUB_OFFLINE=1` 启动服务，因此首次使用前必须先以同一系统用户预下载权重到 HuggingFace 缓存（约 35GB+）：

```bash
cd services/huggingface
uv run -- hf download Qwen/Qwen-Image-2.1

# 国内网络可走 hf-mirror.com 镜像
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="${no_proxy:+$no_proxy,}hf-mirror.com"
uv run -- hf download Qwen/Qwen-Image-2.1
```

加速变体：`Qwen/Qwen-Image-2.1-Turbo`（8 步蒸馏、自带 sigma 网格、不支持 CFG），启动时通过 `--model Qwen/Qwen-Image-2.1-Turbo` 切换，`num_inference_steps` 会被其自带 sigma 网格覆盖。

## 安装

```bash
cd services/huggingface
bash setup.bash
```

依赖要求：`diffusers >= 0.38`（`QwenImage21Pipeline` 自 v0.38.0 引入）、`transformers >= 5.17`（Qwen3-VL 编码器），`setup.bash` 会自动安装。注意 huggingface 下各服务（flux2-klein-9b / 本服务）共享同一 `.venv`，升级后建议对现有服务做一次回归验证。

## 启动服务

```bash
cd qwen-image-2.1
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8010 --idle-timeout 600

# 使用 Turbo 8 步蒸馏版本
uv run main.py --model Qwen/Qwen-Image-2.1-Turbo
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8010` | 监听端口 |
| `--model` | `Qwen/Qwen-Image-2.1` | HuggingFace 模型标识 |
| `--idle-timeout` | `1800` | 空闲超时秒数 |
| `--idle-check-interval` | `60` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/unload` | 手动卸载模型释放显存 |
| `POST` | `/text-to-image` | 文本生成图片 |
| `POST` | `/image-editing` | 多参考图编辑（1-10 张） |

## 请求格式

### 文本生成图片 `POST /text-to-image`

```json
{
  "prompt": "一只戴着墨镜的柴犬，坐在冲浪板上，海浪背景",
  "seed": 42,
  "height": 2048,
  "width": 2048,
  "num_inference_steps": 40,
  "true_cfg_scale": 1.0,
  "num_images_per_prompt": 1
}
```

### 多参考图编辑 `POST /image-editing`

```json
{
  "prompt": "把第一张图里的花放进第二张图的场景中",
  "images": ["<base64 参考图 1>", "<base64 参考图 2>"],
  "seed": 42,
  "num_inference_steps": 40
}
```

`height`/`width` 省略时按第一张参考图的宽高比推导输出尺寸。

### 参数说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string` | **必填** | 提示词 / 编辑指令，支持中英文 |
| `images` | `string[]` | 编辑必填 | 1-10 张 base64 参考图（仅编辑端点） |
| `seed` | `integer` | `null` | 随机种子，设为 null 则随机 |
| `height` | `integer` | `2048` | 图片高度 (1-4096)，Qwen-Image 2.1 原生分辨率为 2048 |
| `width` | `integer` | `2048` | 图片宽度 (1-4096) |
| `num_inference_steps` | `integer` | `40` | 去噪步数，官方推荐 40 |
| `true_cfg_scale` | `float` | `1.0` | true CFG 强度。**该管线没有 `guidance_scale`**；官方推荐默认不开 CFG；设为 >1（如 4.0）开启 true CFG（每步计算量翻倍），需配合 `negative_prompt`（可为空串） |
| `negative_prompt` | `string` | `null` | 负面提示词，仅 `true_cfg_scale > 1` 时生效 |
| `num_images_per_prompt` | `integer` | `1` | 生成图片数量 (1-4) |

## 成功响应

```json
{
  "status": "success",
  "images": ["<base64 编码的 PNG 图片，可能为 RGBA>"],
  "metadata": {
    "mode": "text-to-image",
    "prompt": "一只戴着墨镜的柴犬，坐在冲浪板上，海浪背景",
    "negative_prompt": null,
    "seed": 42,
    "generation_time": 180.5,
    "num_images": 1,
    "height": 2048,
    "width": 2048,
    "num_inference_steps": 40,
    "true_cfg_scale": 1.0,
    "num_reference_images": 0
  }
}
```

## 显存优化

该服务默认启用两项优化（已在 RTX 4090 24GB 上实测）：

1. **`enable_model_cpu_offload()`**：约 30.9GB bf16 权重（7B DiT + Qwen3-VL 编码器 + VAE）常驻主机内存，推理时逐组件搬运上卡，每次请求结束后权重归还内存，GPU 稳态占用仅 0.01GB。
2. **VAE tiling**：`vae.enable_tiling()` 分块解码最终图像。未开启时 2048×2048 解码需要约 4.5GB 激活尖峰，会在 24GB 卡上 OOM；开启后峰值被打平。

实测峰值显存：1024² 与 2048²（含 tiling）均约 **17.2GB**（含 ~0.7GB 桌面基线与 CUDA context），24GB 卡剩余约 6.5GB 余量；**18GB 以下显存的显卡需改用 `enable_sequential_cpu_offload()`**（逐层搬运，速度大幅下降）。

```python
# main.py 中的关键配置
pipeline = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()
```

耗时参考（RTX 4090）：1024²/20 步约 23s；2048²/40 步（默认配置）约 125s。显存不足请求会触发 OOM，服务会自动卸载模型释放显存并以结构化错误返回（`common.py` 内置自愈），下次请求自动重新加载。

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8010 health

# 文本生成图片（默认 2048x2048、40 步）
uv run example_client.py --port 8010 generate \
  --prompt "a cat sitting on a sofa" --seed 42 --no-show

# 低分辨率快速出图
uv run example_client.py --port 8010 generate \
  --prompt "A futuristic city at sunset" \
  --height 1024 --width 1024 --steps 20 --output city --no-show

# 多参考图编辑（--image 可重复传入，最多 10 张）
uv run example_client.py --port 8010 edit \
  --prompt "Move it to a snowy mountain top" \
  --image input.png --output edited --no-show
```
