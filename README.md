# Model Haven

AI 模型服务聚合仓库，通过 FastAPI 提供 RESTful 模型推理服务接口。

## 使用介绍

### 快速开始

```bash
# 1. 初始化子模块
git submodule update --init --recursive

# 2. 安装并启动某个服务（以 trellis 为例）
cd services/trellis
bash setup.bash
uv run main.py --host 0.0.0.0 --port 8000
```

### 通用特性

- **自动 GPU 选择** — 选择空闲显存最多的 GPU
- **懒加载** — 首次请求时才加载模型，减少启动时间
- **空闲超时卸载** — 长时间无请求自动卸载模型，释放 GPU 显存
- **线程安全推理** — 通过 asyncio Lock 序列化推理请求
- **健康检查** — `GET /health` 端点查看模型状态和 GPU 信息

---

## 项目介绍

### 项目结构

```
model_haven/
├── deps/                # Git submodule 依赖库
│   ├── GraspGen/        # NVlabs/GraspGen (6-DOF 抓取生成)
│   ├── trellis/         # microsoft/TRELLIS (文本/图像 → 3D)
│   ├── sam3/            # facebookresearch/sam3 (文本提示图像分割)
│   ├── sam-3d-objects/  # facebookresearch/sam-3d-objects (单图像 3D 重建)
│   ├── vggt/            # facebookresearch/vggt
│   └── hamer/           # geopavlakos/hamer
└── services/            # FastAPI 模型服务
    ├── graspgen/            # 6-DOF 抓取生成服务
    ├── trellis/             # 文本/图像 → 3D 生成服务
    ├── sam3/                # SAM3 文本提示图像分割服务
    ├── sam-3d-objects/      # SAM 3D 物体重建服务
    └── huggingface/sdxl/    # HuggingFace 模型服务 (SDXL 等)
```

### 依赖管理 (deps)

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

### 服务项目结构 (services)

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

#### 启动服务

```bash
# 1. 安装环境
cd services/<project_name>
bash setup.bash

# 2. 启动服务 (uv run 自动激活虚拟环境)
uv run main.py --host 0.0.0.0 --port <port>
```

### FastAPI Server 类书写规范

所有 `main.py` 中的 Server 类应遵循以下接口规范：

#### 必需接口

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

### 开发新服务

1. 创建目录: `mkdir services/<new_service>`
2. 添加依赖: `git submodule add <url> deps/<dep_name>`
3. 创建文件结构（参考上方规范）
4. 编写 `setup.bash` 安装脚本
5. 实现 Server 类（继承通用模式：懒加载、空闲卸载、健康检查）
6. 编写 `example_client.py` 示例客户端

---

## 服务列表

详细文档请参阅各服务目录下的 README.md：

| 服务 | 路径 | 说明 |
|------|------|------|
| [GraspGen](services/graspgen/README.md) | `services/graspgen/` | 6-DOF 抓取生成 |
| [TRELLIS](services/trellis/README.md) | `services/trellis/` | 文本/图像 → 3D |
| [SDXL](services/huggingface/sdxl/README.md) | `services/huggingface/sdxl/` | 文本生成图片 |
| [SAM3](services/sam3/README.md) | `services/sam3/` | 文本提示图像分割 |
| [SAM 3D Objects](services/sam-3d-objects/README.md) | `services/sam-3d-objects/` | 单图像 3D 重建 |
