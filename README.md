# Model Haven

AI 模型服务聚合仓库，通过 FastAPI 提供 RESTful 模型推理服务接口。

## 项目结构

```
model_haven/
├── deps/           # Git submodule 依赖库
│   ├── graspgen/   # NVlabs/GraspGen (6-DOF 抓取生成)
│   └── trellis/    # microsoft/TRELLIS (文本/图像 → 3D)
└── services/       # FastAPI 模型服务
    ├── graspgen/       # 6-DOF 抓取生成服务
    ├── trellis/        # 文本/图像 → 3D 生成服务
    └── huggingface/    # HuggingFace 模型服务 (SDXL 等)
```

## 依赖管理 (deps)

`deps/` 目录存放 git submodule 库。使用以下命令添加新的依赖：

```bash
git submodule add <repository_url> deps/<name>
```

示例：
```bash
git submodule add https://github.com/microsoft/TRELLIS.git deps/trellis
```

初始化/更新 submodule：
```bash
git submodule update --init --recursive
```

## 服务项目结构 (services)

每个服务目录包含以下文件：

```
services/<project_name>/
├── main.py            # FastAPI Server 主程序
├── setup.bash         # uv 环境安装脚本
├── pyproject.toml     # uv 项目配置
├── uv.lock            # uv 锁文件
├── .python-version    # Python 版本 (如 3.11)
└── example_client.py  # (可选) 客户端示例代码
```

每个服务使用 **uv** 独立管理 Python 环境，互不干扰。

### 启动服务

```bash
# 1. 安装环境
cd services/<project_name>
bash setup.bash

# 2. 启动服务 (uv run 自动激活虚拟环境)
uv run main.py --host 0.0.0.0 --port <port>
```

### 通用特性

所有服务共享以下特性：

- **自动 GPU 选择** — 选择空闲显存最多的 GPU
- **懒加载** — 首次请求时才加载模型，减少启动时间
- **空闲超时卸载** — 长时间无请求自动卸载模型，释放 GPU 显存
- **线程安全推理** — 通过 asyncio Lock 序列化推理请求
- **健康检查** — `GET /health` 端点查看模型状态和 GPU 信息

---

## FastAPI Server 类书写规范

所有 `main.py` 中的 Server 类应遵循以下接口规范：

### 必需接口

```python
class XxxServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        idle_timeout: int = 300,
        idle_check_interval: int = 30,
    ):
        """Initialize the server.

        Args:
            host: Host to bind to
            port: Port to listen on
            idle_timeout: Seconds before idle model unload (must be > 0)
            idle_check_interval: Seconds between idle checks (must be > 0)
        """
        # Model state management
        self._model_state: ModelState = ModelState.NOT_LOADED
        self._state_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._last_activity: float = time.monotonic()
        self._gpu_id: Optional[int] = None

        # FastAPI app with lifespan for idle monitor
        self._app = FastAPI(title="...", lifespan=lifespan)
        self._register_routes()

    def _register_routes(self) -> None:
        """Register FastAPI routes (health, inference endpoints)."""
        ...

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")


def main():
    """Main entry point for the FastAPI server."""
    parser = argparse.ArgumentParser(description="Xxx FastAPI Server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--idle-timeout", type=int, default=300)
    parser.add_argument("--idle-check-interval", type=int, default=30)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    server = XxxServer(host=args.host, port=args.port, ...)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 开发新服务

1. 创建目录: `mkdir services/<new_service>`
2. 添加依赖: `git submodule add <url> deps/<dep_name>`
3. 创建文件结构（参考上方规范）
4. 编写 `setup.bash` 安装脚本
5. 实现 Server 类（继承通用模式：懒加载、空闲卸载、健康检查）
6. 编写 `example_client.py` 示例客户端

## 服务列表

### GraspGen — 6-DOF 抓取生成服务

基于 [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) 的 6 自由度抓取姿态生成服务。接收点云数据（base64 编码的 numpy 数组），返回抓取姿态和置信度。

**环境要求：** NVIDIA GPU + CUDA 12.1, Python 3.10

**默认端口：** 8001

#### 安装

```bash
cd services/graspgen
bash setup.bash
```

#### 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8001 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8001` | 监听端口 |
| `--gripper-config` | `graspgen_robotiq_2f_140.yml` | 夹爪配置文件 |
| `--idle-timeout` | `300` | 空闲超时秒数 |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/generate` | 抓取生成 |

#### 请求格式

```json
{
  "point_cloud": {
    "data": "<base64 编码的 float32 字节>",
    "shape": [2000, 3],
    "dtype": "float32"
  },
  "num_grasps": 200,
  "topk_num_grasps": -1,
  "grasp_threshold": -1.0,
  "min_grasps": 40,
  "max_tries": 6,
  "remove_outliers": true
}
```

#### 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8001 health

# 从点云文件生成抓取
uv run example_client.py --port 8001 generate --pcd-file input.npy

# 从网格文件生成抓取
uv run example_client.py --port 8001 generate --mesh-file model.obj --visualize
```

---

### TRELLIS — 文本/图像 → 3D 生成服务

基于 [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) 的 3D 模型生成服务，支持文本和图像两种输入，返回 GLB 格式的 3D 模型。

**环境要求：** NVIDIA GPU + CUDA 12.8, Python 3.11

**默认端口：** 8000

#### 安装

```bash
cd services/trellis
bash setup.bash
```

> `setup.bash` 会自动安装 PyTorch (cu128)、TRELLIS 及其全部扩展依赖（nvdiffrast、diffoctreerast、spconv 等），编译耗时较长，请耐心等待。

#### 启动服务

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

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/text-to-3d` | 文本生成 3D |
| `POST` | `/image-to-3d` | 图像生成 3D |

#### 文本生成 3D 请求格式

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

#### 图像生成 3D 请求格式

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

#### 成功响应

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

#### 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8000 health

# 文本生成 3D
uv run example_client.py --port 8000 text-to-3d --text "A modern chair"

# 图像生成 3D
uv run example_client.py --port 8000 image-to-3d photo.png
```

---

### SDXL — 文本生成图片服务

基于 [Stability AI SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 的文本生成图片服务，通过 HuggingFace diffusers 提供。

**环境要求：** NVIDIA GPU + CUDA, Python 3.11

**默认端口：** 8002

#### 安装

```bash
cd services/huggingface
bash setup.bash
```

#### 启动服务

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

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/text-to-image` | 文本生成图片 |

#### 请求格式

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

#### 成功响应

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

#### 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8002 health

# 文本生成图片
uv run example_client.py --port 8002 generate --prompt "a cat sitting on a sofa"
```
