# TRELLIS.2 — 图像 → 3D 生成服务

基于 [Microsoft TRELLIS.2](https://github.com/microsoft/trellis.2)（`microsoft/TRELLIS.2-4B`）的高保真图像生成 3D 服务。输入单张图片，输出带 PBR 材质（base color / metallic / roughness）烘焙纹理的 GLB 网格，最高支持 1536³ 分辨率。

**环境要求：** NVIDIA GPU (**显存 ≥ 24GB**) + CUDA 12.4, Python 3.10

**默认端口：** 8009

## 安装

```bash
cd services/trellis.2
bash setup.bash
```

> `setup.bash` 会安装 PyTorch (cu124)、TRELLIS.2 全部 CUDA 扩展（flash-attn、nvdiffrast、nvdiffrec、CuMesh、FlexGEMM、o-voxel），并预热 `microsoft/TRELLIS.2-4B`、DINO 条件模型与 BiRefNet 抠图权重到 HuggingFace 缓存。编译耗时较长，请耐心等待。

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8009 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8009` | 监听端口 |
| `--model` | `microsoft/TRELLIS.2-4B` | 图像→3D 模型 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/generate` | 图像生成 3D（GLB） |
| `POST` | `/unload` | 卸载模型释放显存 |

## 请求格式

```json
{
  "image": "<base64 编码的图片数据>",
  "seed": 42,
  "pipeline_type": "1024_cascade",
  "preprocess_image": true,
  "decimation_target": 1000000,
  "texture_size": 2048,
  "remesh": true
}
```

- `image`：base64 编码的图片（无透明通道时服务端自动用 BiRefNet 去背景）
- `seed`（可选）：随机种子，默认随机
- `pipeline_type`（可选）：分辨率档位，`512` / `1024` / `1024_cascade`（默认）/ `1536_cascade`；显存不足或需要更快生成时用 `512`，最高质量用 `1536_cascade`
- `preprocess_image`（可选）：是否服务端去背景，默认 `true`；输入已有干净 alpha 通道时可设 `false`
- `decimation_target`（可选）：导出网格目标顶点数，默认 `1000000`
- `texture_size`（可选）：烘焙纹理分辨率，默认 `2048`，可到 `4096`
- `remesh`（可选）：UV 展开前是否重网格化，默认 `true`

## 成功响应

```json
{
  "status": "success",
  "glb_data": "<base64 编码的 GLB 文件>",
  "metadata": {
    "seed": 42,
    "generation_time": 17.5,
    "file_size": 5242880,
    "pipeline_type": "1024_cascade",
    "decimation_target": 1000000,
    "texture_size": 2048,
    "remesh": true,
    "preprocess_image": true
  }
}
```

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8009 health

# 从图像生成 3D
uv run example_client.py --port 8009 generate --image photo.png

# 指定种子、分辨率档位与输出路径（生成 result.glb）
uv run example_client.py --port 8009 generate --image photo.png --seed 123 \
  --pipeline-type 1536_cascade --texture-size 4096 --output result
```

## 注意事项

- 输出 GLB 纹理的 alpha 通道默认未连接到材质 opacity，需要透明效果时请在 3D 软件中手动连接
- 显存紧张时可降低 `pipeline_type`（`512`）或 `texture_size`（`2048`）
- 服务仅导出 GLB；PBR 预览视频需要 HDRI 环境贴图，可参考上游 `example.py` 离线渲染
