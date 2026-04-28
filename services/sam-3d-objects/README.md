# SAM 3D Objects — 单图像 3D 重建服务

基于 [Meta SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) 的单图像 3D 物体重建服务。接收图片和对应掩码，返回 Gaussian Splat PLY 格式的 3D 模型。

**环境要求：** NVIDIA GPU + CUDA 12.1, Python 3.11

**默认端口：** 8003

## 安装

```bash
cd services/sam-3d-objects
bash setup.bash
```

> `setup.bash` 会安装 PyTorch (cu121)、pytorch3d、kaolin 等依赖，并执行 hydra 补丁。编译耗时较长，请耐心等待。

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8003 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8003` | 监听端口 |
| `--config-path` | `checkpoints/pipeline.yaml` | pipeline 配置 YAML |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/generate` | 图像 + 掩码 → 3D 重建 |

## 请求格式

```json
{
  "image": "<base64 编码的 RGBA 图片>",
  "mask": "<base64 编码的灰度掩码图片>",
  "seed": 42
}
```

- `image`：base64 编码的 RGBA 图片
- `mask`：base64 编码的灰度掩码图片
- `seed`（可选）：随机种子，默认 42

> 注意：`image` 和 `mask` 尺寸必须一致。

## 成功响应

```json
{
  "status": "success",
  "ply_data": "<base64 编码的 PLY 文件>",
  "metadata": {
    "generation_time": 15.0,
    "rotation": [...],
    "translation": [...],
    "scale": [...],
    "file_size": 1048576
  }
}
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8003 health

# 从图像和掩码生成 3D
uv run example_client.py --port 8003 generate --image object.png --mask mask.png

# 指定种子和输出路径
uv run example_client.py --port 8003 generate --image object.png --mask mask.png --seed 123 --output result.ply
```
