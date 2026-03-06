# Model Haven

> 模型服务港湾 - 为 MobRT 项目提供远程模型推理服务

**核心设计理念：每个模型服务独立管理，拥有独立的虚拟环境和依赖**

---

## 📁 目录结构

```
model_haven/
├── deps/                      # 📦 第三方源码（git submodules）
│   ├── trellis/              # TRELLIS 源码
│   ├── grounding_dino/       # GroundingDINO 源码
│   └── sam2/                 # SAM2 源码
│
├── services/                  # 🚀 模型服务（每个服务独立环境）
│   ├── trellis/
│   │   ├── .venv/            # 🔒 TRELLIS 专用虚拟环境
│   │   ├── server.py         # 服务脚本
│   │   ├── pyproject.toml    # uv 配置文件
│   │   └── README.md         # 服务说明
│   ├── grounding_dino/
│   │   ├── .venv/            # 🔒 GroundingDINO 专用虚拟环境
│   │   ├── server.py
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── sam2/
│       ├── .venv/            # 🔒 SAM2 专用虚拟环境
│       ├── server.py
│       ├── pyproject.toml
│       └── README.md
│
├── shared/                    # 🔧 共享基础设施（被所有服务引用）
│   ├── __init__.py
│   ├── base_server.py        # 服务基类
│   ├── gpu_manager.py        # GPU 管理器
│   ├── protocol.py           # ZMQ 通信协议
│   └── utils.py              # 通用工具
│
├── clients/                   # 💻 客户端封装（本地使用）
│   ├── __init__.py
│   ├── trellis_client.py
│   ├── grounding_dino_client.py
│   └── sam2_client.py
│
├── scripts/                   # 🛠️ 管理脚本
│   ├── start_all.sh          # 一键启动所有服务
│   ├── stop_all.sh           # 一键停止所有服务
│   ├── setup.sh              # 初始化脚本
│   └── status.sh             # 查看服务状态
│
└── README.md                  # 本文件
```

---

## 🎯 设计优势

### ✅ 独立环境隔离

每个模型服务拥有独立的虚拟环境：
- **不同的 Python 版本**：TRELLIS 可能需要 3.10，其他模型可能需要 3.8
- **不同的依赖版本**：避免依赖冲突
- **独立的依赖管理**：通过 `pyproject.toml` 精确控制

### ✅ 便于部署和维护

- **模块化**：每个服务独立运行，互不影响
- **易于扩展**：添加新模型只需创建新目录
- **便于调试**：问题隔离到具体服务

### ✅ 共享基础设施

通过 `shared/` 目录共享通用代码：
- **避免重复**：base_server、protocol 等只写一次
- **统一接口**：所有服务遵循相同的通信协议

---

## 🚀 快速开始

### 1. 初始化项目结构

```bash
# 在 MobRT 项目根目录
cd /path/to/MobRT

# 创建目录结构
mkdir -p model_haven/{deps,services,shared,clients,scripts}
mkdir -p model_haven/services/{trellis,grounding_dino,sam2}

# 创建标记文件
touch model_haven/deps/.gitkeep
```

### 2. 添加第三方依赖（git submodules）

```bash
cd model_haven

# 添加 TRELLIS
git submodule add https://github.com/microsoft/TRELLIS.git deps/trellis

# 添加 GroundingDINO
git submodule add https://github.com/IDEA-Research/GroundingDINO.git deps/grounding_dino

# 添加 SAM2
git submodule add https://github.com/facebookresearch/segment-anything-2.git deps/sam2

# 初始化并更新
git submodule update --init --recursive
```

### 3. 为每个服务创建独立虚拟环境

#### TRELLIS 服务

```bash
cd model_haven/services/trellis

# 创建独立虚拟环境（Python 3.10）
uv venv --python 3.10

# 激活环境
source .venv/bin/activate

# 安装依赖（从 pyproject.toml）
uv pip install -e .

# 或者手动安装
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install flash-attn spconv-cu118 diffoctreerast mip-gaussian nvdiffrast kaolin
```

#### GroundingDINO 服务

```bash
cd model_haven/services/grounding_dino

# 创建独立虚拟环境（可能需要不同 Python 版本）
uv venv --python 3.9

source .venv/bin/activate
uv pip install -e .
```

#### SAM2 服务

```bash
cd model_haven/services/sam2

uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### 4. 配置共享代码引用

**方法 1：Python 路径（推荐）**

在每个服务的 `server.py` 中添加：

```python
import sys
import os

# 添加 shared 目录到 Python 路径
shared_path = os.path.abspath("../../shared")
sys.path.insert(0, shared_path)

from base_server import BaseModelServer
```

**方法 2：符号链接**

```bash
cd model_haven/services/trellis
ln -s ../../shared/base_server.py .
ln -s ../../shared/protocol.py .
```

### 5. 启动服务

#### 启动单个服务

```bash
cd model_haven/services/trellis
source .venv/bin/activate
python server.py --gpu 0 --port 5555
```

#### 启动所有服务

```bash
cd model_haven
bash scripts/start_all.sh
```

---

## 📦 已集成的模型

| 模型 | 服务目录 | 虚拟环境 | Python | 服务端口 | GPU 预估 |
|------|---------|---------|--------|---------|---------|
| **TRELLIS** | `services/trellis/` | ✅ 独立 | 3.10 | 5555 | ~16GB |
| **GroundingDINO** | `services/grounding_dino/` | ✅ 独立 | 3.9 | 5556 | ~4GB |
| **SAM2** | `services/sam2/` | ✅ 独立 | 3.10 | 5557 | ~8GB |

---

## 🔧 服务开发示例

### TRELLIS 服务的文件结构

```
services/trellis/
├── .venv/                 # 虚拟环境（不提交到 git）
├── server.py              # 服务实现
├── pyproject.toml         # uv 依赖配置
└── README.md              # 服务说明
```

### pyproject.toml 示例

```toml
[project]
name = "trellis-service"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "pillow",
    "imageio",
    "huggingface-hub",
    "pyzmq",
]

[project.optional-dependencies]
# TRELLIS 特定依赖
trellis = [
    "flash-attn>=2.0.0",
    "spconv-cu118",
    "diffoctreerast",
    "mip-gaussian",
    "nvdiffrast",
    "kaolin>=0.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### server.py 示例

```python
#!/usr/bin/env python3
"""
TRELLIS 模型服务
"""
import sys
import os

# 添加共享代码到路径
shared_path = os.path.abspath("../../shared")
sys.path.insert(0, shared_path)

# 添加 TRELLIS 源码到路径
deps_path = os.path.abspath("../../deps/trellis")
sys.path.insert(0, deps_path)

from base_server import BaseModelServer
from trellis.pipelines import TrellisImageTo3DPipeline

class TrellisServer(BaseModelServer):
    def _load_model(self):
        """加载 TRELLIS 模型"""
        self.logger.info("Loading TRELLIS model from HuggingFace...")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        self.pipeline.cuda()
        self.logger.info("✅ TRELLIS model loaded")

    def _process_request(self, request: dict) -> dict:
        """处理生成请求"""
        if request["type"] == "generate":
            from PIL import Image

            # 调用 TRELLIS 生成
            image = Image.open(request["image_path"])
            outputs = self.pipeline.run(
                image,
                seed=request.get("seed", 42),
            )

            # 返回结果路径
            return {
                "status": "success",
                "mesh_path": "/path/to/output.glb",
            }
        else:
            raise ValueError(f"Unknown request type: {request['type']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    server = TrellisServer(
        host=args.host,
        port=args.port,
        gpu_id=args.gpu
    )
    server.start()
```

---

## 🛠️ 管理脚本

### start_all.sh

```bash
#!/bin/bash
# model_haven/scripts/start_all.sh

MODEL_HAVEN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVICES=(
    "trellis:5555:0"
    "grounding_dino:5556:1"
    "sam2:5557:2"
)

echo "🚀 Starting Model Haven services..."

for service_config in "${SERVICES[@]}"; do
    IFS=':' read -r name port gpu <<< "$service_config"

    echo "Starting $name on port $port, GPU $gpu..."

    cd "$MODEL_HAVEN/services/$name"

    # 激活该服务的独立环境
    source .venv/bin/activate

    # 启动服务
    python server.py --gpu "$gpu" --port "$port" &

    echo "✅ $name started (PID: $!)"
done

echo "✨ All services started!"
```

---

## 🔍 故障排查

### 查看服务状态

```bash
cd model_haven
bash scripts/status.sh
```

### 停止所有服务

```bash
cd model_haven
bash scripts/stop_all.sh
```

### 单独重启某个服务

```bash
cd model_haven/services/trellis
source .venv/bin/activate
python server.py --gpu 0 --port 5555
```

---

## 📖 依赖管理对比

| 方案 | 优点 | 缺点 | 本项目选择 |
|------|------|------|-----------|
| **统一环境** | 安装简单 | 依赖冲突风险高 | ❌ |
| **独立环境** | 完全隔离，无冲突 | 需分别安装 | ✅ **采用** |
| **Docker** | 环境一致性好 | 学习曲线陡，资源重 | 🚧 未来可选 |

---

## 📝 TODO

- [ ] 创建 shared/ 目录的共享代码
- [ ] 实现 TRELLIS 服务
- [ ] 实现 GroundingDINO 服务
- [ ] 实现 SAM2 服务
- [ ] 创建管理脚本（start_all.sh, stop_all.sh, status.sh）
- [ ] 添加服务监控和自动重启
- [ ] 编写单元测试

---

## 📄 许可证

MIT License

---

**最后更新：** 2026-02-26
**维护者：** MobRT Team
