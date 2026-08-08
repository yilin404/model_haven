# Depth Anything V3 — 深度与相机估计服务

基于官方 [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
源码，提供单视图/多视图相对深度、相机位姿估计和 pose conditioning。

**默认模型：** `depth-anything/DA3-LARGE-1.1`

**默认端口：** `8006`

默认模型是 any-view 相对深度模型，不输出真实米制深度。官方仓库将
`DA3-LARGE-1.1` 权重标为 CC BY-NC 4.0，而 Hugging Face 模型页当前显示
Apache-2.0；商业使用前应向上游确认有效的权重许可证。

## 安装

先在仓库根目录初始化 submodule：

```bash
git submodule update --init --recursive
```

然后安装独立环境并预热默认模型的 Hugging Face cache：

```bash
cd services/depth-anything-v3
bash setup.bash
```

预下载另一个模型：

```bash
DA3_MODEL_NAME=depth-anything/DA3-BASE bash setup.bash
```

模型由 Hugging Face cache 管理，不保存在服务目录的 `checkpoints/` 下。
`scripts/start-services.sh` 会设置 `HF_HUB_OFFLINE=1`，因此启动前必须已经使用
同一系统用户和相同 `HF_HOME` 缓存目标模型。

## 启动

默认模型：

```bash
uv run -- python main.py
```

通过命令行切换已缓存的模型：

```bash
uv run -- python main.py \
  --model-name depth-anything/DA3-BASE \
  --port 8006
```

通过仓库启动器运行默认模型：

```bash
bash scripts/start-services.sh --depth-anything-v3:8006
```

通用启动器当前只向服务传递 `--port`，因此非默认模型应直接通过
`main.py --model-name ...` 启动。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8006` | 监听端口 |
| `--model-name` | `depth-anything/DA3-LARGE-1.1` | Hugging Face repo id 或本地模型目录 |
| `--idle-timeout` | `300` | 空闲多久后卸载模型 |
| `--idle-check-interval` | `30` | 空闲检查间隔 |
| `--log-level` | `INFO` | 日志级别 |

## Interface

### 健康检查

```text
GET /health
```

健康检查只表示进程和 GPU 可见性；模型采用懒加载，首次真实请求才会从 cache
加载权重。

### 深度估计

```text
POST /estimate-depth
```

请求示例：

```json
{
  "images": ["<base64 PNG/JPEG>"],
  "extrinsics": null,
  "intrinsics": null,
  "process_res": 504,
  "process_res_method": "upper_bound_resize"
}
```

相机参数必须成对提供：

- `extrinsics`：`(N, 4, 4)` 浮点数组；
- `intrinsics`：`(N, 3, 3)` 浮点数组；
- `N` 必须等于图片数量。

NumPy 数组统一使用以下无损 JSON 表示：

```json
{
  "data": "<base64 encoded C-order bytes>",
  "shape": [1, 4, 4],
  "dtype": "float32"
}
```

成功响应包含：

```json
{
  "status": "success",
  "depth": {"data": "...", "shape": [1, 504, 672], "dtype": "float32"},
  "confidence": {"data": "...", "shape": [1, 504, 672], "dtype": "float32"},
  "extrinsics": {"data": "...", "shape": [1, 3, 4], "dtype": "float32"},
  "intrinsics": {"data": "...", "shape": [1, 3, 3], "dtype": "float32"},
  "metadata": {
    "model_name": "depth-anything/DA3-LARGE-1.1",
    "num_images": 1,
    "is_metric": false,
    "generation_time": 1.23
  }
}
```

第一版不开放服务器文件导出、GLB/PLY 或 Gaussian Splatting。

## 使用官方样例做集成测试

官方仓库没有单独的 pytest 测试套件，但 README 的 Basic Usage 使用
`assets/examples/SOH/000.png` 和 `010.png` 验证多视图推理。服务启动后可复用
同一组输入，依次检查两次单图请求、一次双图请求和两个错误请求：

```bash
uv run -- python tests/validate_official_examples.py \
  --base-url http://127.0.0.1:8006

uv run -- python tests/visualize_official_examples.py
```

原始数组、指标报告、深度/置信度对比图会写入
`output/official-soh/`。深度图使用上游 `visualize_depth()` 的 Spectral 配色；
颜色仅用于显示经过 2%–98% 分位裁剪的相对深度，不代表米制距离。

## 示例客户端

```bash
# 健康检查
uv run -- python example_client.py health

# 单图；结果保存为客户端本地 NPZ
uv run -- python example_client.py estimate \
  --image image.jpg \
  --output prediction.npz

# 多视图
uv run -- python example_client.py estimate \
  --image view0.jpg \
  --image view1.jpg \
  --output prediction.npz

# 带相机参数，NPZ 内需包含 extrinsics 和 intrinsics
uv run -- python example_client.py estimate \
  --image view0.jpg \
  --image view1.jpg \
  --camera-npz cameras.npz
```
