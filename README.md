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

# 3. 或使用脚本快速启动（自动管理多个服务）
bash scripts/start-services.sh --all
```

### 通用特性

- **自动 GPU 选择** — 选择空闲显存最多的 GPU
- **懒加载** — 首次请求时才加载模型，减少启动时间
- **空闲超时卸载** — 长时间无请求自动卸载模型，释放 GPU 显存
- **线程安全推理** — 通过 asyncio Lock 序列化推理请求
- **健康检查** — `GET /health` 端点查看模型状态和 GPU 信息

### 服务管理脚本

`scripts/start-services.sh` 通过 tmux 管理多个服务，自动发现 `services/` 下所有包含 `main.py` 的目录。

```bash
# 查看可用服务
bash scripts/start-services.sh --help

# 启动所有服务
bash scripts/start-services.sh --all

# 启动指定服务并指定端口
bash scripts/start-services.sh --sam3:8014 --trellis:8010
```

启动后服务运行在 tmux session `model-haven` 中，关闭终端不影响运行。

```bash
# 查看服务日志
tmux attach -t model-haven

# 列出所有 window（每个服务一个 window）
tmux list-windows -t model-haven

# 停止所有服务
tmux kill-session -t model-haven
```

---

## 项目介绍

### 项目结构

```
model_haven/
├── deps/                # Git submodule 依赖库
│   ├── GraspGen/        # NVlabs/GraspGen (6-DOF 抓取生成)
│   ├── trellis/         # microsoft/TRELLIS (文本/图像 → 3D)
│   ├── sam3/            # facebookresearch/sam3 (文本提示图像分割)
│   └── sam-3d-objects/  # facebookresearch/sam-3d-objects (单图像 3D 重建)
└── services/            # FastAPI 模型服务
    ├── __init__.py          # 包初始化
    ├── common.py            # ModelEngine + BaseFastAPIServer 基类
    ├── GraspGen/            # 6-DOF 抓取生成服务
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

### FastAPI Server 基类架构

所有服务基于 `services/common.py` 中的两个基类：

- **`ModelEngine`** — 模型生命周期基类（load/unload/inference）
- **`BaseFastAPIServer`** — FastAPI 服务基类（health、idle monitor、lifespan）

#### 实现 ModelEngine

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

class XxxEngine(ModelEngine):
    def __init__(self, model_path: str):
        super().__init__("model")  # engine name, used in health response
        self.model_path = model_path
        self.model = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        self.model = load_model(self.model_path).to(f"cuda:{self.gpu_id}")

    def _unload_impl(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def _run_inference_impl(self, input_data, **kwargs) -> dict:
        result = self.model(input_data)
        return {"status": "success", "data": result}
```

#### 实现 BaseFastAPIServer

```python
class XxxServer(BaseFastAPIServer):
    def __init__(self, model_path: str, **kwargs):
        self._engine = XxxEngine(model_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/predict")
        async def predict(request: PredictRequest):
            result = await self._engine.run_inference(request.input_data)
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result
```

#### 启动入口

```python
def main():
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

#### 健康检查响应

所有服务共享统一的健康检查格式：

```json
{
  "status": "ok",
  "model_state": {"model": "not_loaded"},
  "gpu": "NVIDIA RTX 4090",
  "gpu_memory_allocated_gb": 0.0,
  "gpu_memory_reserved_gb": 0.0,
  "idle_timeout": 300
}
```

`model_state` 为 `Dict[str, str]`，key 为引擎名称，value 为状态。多引擎服务（如 TRELLIS）会有多个 key。

---

### 开发新服务

1. 创建目录: `mkdir services/<new_service>`
2. 添加依赖: `git submodule add <url> deps/<dep_name>`
3. 创建文件结构（参考上方规范）
4. 编写 `setup.bash` 安装脚本
5. 实现 `ModelEngine` 子类（`_load_impl`、`_unload_impl`、`_run_inference_impl`）
6. 实现 `BaseFastAPIServer` 子类（`_register_routes`）
7. 编写 `example_client.py` 示例客户端

---

## 服务列表

详细文档请参阅各服务目录下的 README.md：

| 服务 | 路径 | 说明 |
|------|------|------|
| [TRELLIS](services/trellis/README.md) | `services/trellis/` | 文本/图像 → 3D |
| [GraspGen](services/GraspGen/README.md) | `services/GraspGen/` | 6-DOF 抓取生成 |
| [SDXL](services/huggingface/sdxl/README.md) | `services/huggingface/sdxl/` | 文本生成图片 |
| [SAM3](services/sam3/README.md) | `services/sam3/` | 文本提示图像分割 |
| [SAM 3D Objects](services/sam-3d-objects/README.md) | `services/sam-3d-objects/` | 单图像 3D 重建 |
