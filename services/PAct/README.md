# PAct — 零件铰接生成服务

基于 [PAct](https://github.com/user/PAct) 的单视图零件铰接 3D 生成服务。接收 RGB 图片（可选零件分割掩码），返回 URDF 格式的铰接 3D 模型。

**环境要求：** NVIDIA GPU + CUDA, Python 3.11

**默认端口：** 8005

## 安装

```bash
cd services/PAct
bash setup.bash
```

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8005 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8005` | 监听端口 |
| `--sam-ckpt` | `../../deps/PAct/ckpt/sam_vit_h_4b8939.pth` | SAM 检查点路径 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/generate` | 零件铰接生成 |

## 请求格式

```json
{
  "image": "<base64 编码的 RGB 图片>",
  "mask": "<base64 编码的灰度掩码（可选）>",
  "seed": 42,
  "cfg_strength": 7.5
}
```

- `image`：base64 编码的 RGB 图片
- `mask`（可选）：base64 编码的灰度零件分割掩码。若不提供，将自动使用 SAM 分割
- `seed`（可选）：随机种子，默认 42
- `cfg_strength`（可选）：Classifier-free guidance 强度，默认 7.5

## 成功响应

```json
{
  "status": "success",
  "urdf_zip": "<base64 编码的 URDF 压缩包>",
  "metadata": {
    "seed": 42,
    "num_parts": 3,
    "generation_time": 30.0
  }
}
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8005 health

# 从图片生成铰接模型（自动分割零件）
uv run example_client.py --port 8005 generate photo.png

# 使用指定掩码
uv run example_client.py --port 8005 generate photo.png --mask parts.png

# 指定种子和输出目录
uv run example_client.py --port 8005 generate photo.png --seed 123 --output-dir ./output
```
