# TRELLIS — 文本/图像 → 3D 生成服务

基于 [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) 的 3D 模型生成服务，支持文本和图像两种输入，返回 GLB 格式的 3D 模型。

**环境要求：** NVIDIA GPU + CUDA 12.8, Python 3.11

**默认端口：** 8000

## 安装

```bash
cd services/trellis
bash setup.bash
```

> `setup.bash` 会自动安装 PyTorch (cu128)、TRELLIS 及其全部扩展依赖（nvdiffrast、diffoctreerast、spconv 等），编译耗时较长，请耐心等待。

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8000 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8000` | 监听端口 |
| `--text-model` | `microsoft/TRELLIS-text-xlarge` | 文本→3D 模型 |
| `--image-model` | `microsoft/TRELLIS-image-large` | 图像→3D 模型 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/text-to-3d` | 文本生成 3D |
| `POST` | `/image-to-3d` | 图像生成 3D |

## 文本生成 3D 请求格式

```json
{
  "text": "A modern chair with wooden legs",
  "seed": 42,
  "options": {
    "simplify": 0.95,
    "texture_size": 1024
  }
}
```

## 图像生成 3D 请求格式

```json
{
  "image": "<base64 编码的图片数据>",
  "seed": 42,
  "options": {
    "simplify": 0.95,
    "texture_size": 1024,
    "preprocess_image": true
  }
}
```

## 成功响应

```json
{
  "status": "success",
  "glb_data": "<base64 编码的 GLB 文件>",
  "metadata": {
    "seed": 42,
    "generation_time": 12.34,
    "file_size": 524288,
    "simplify": 0.95,
    "texture_size": 1024
  }
}
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8000 health

# 文本生成 3D
uv run example_client.py --port 8000 text-to-3d --text "A modern chair"

# 图像生成 3D
uv run example_client.py --port 8000 image-to-3d photo.png
```
