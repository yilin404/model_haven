# Model Haven

AI 模型服务聚合仓库，通过 ZMQ 协议提供统一的模型推理服务接口。

## 项目结构

```
model_haven/
├── deps/           # Git submodule 依赖库
│   └── ...         # 模型依赖
└── services/       # ZMQ 服务项目
    └── ...         # 可扩展模型服务
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

每个 server project 目录包含以下文件：

```
services/<project_name>/
├── main.py            # ZMQ Server 主程序
├── setup.bash         # uv 环境安装脚本
├── pyproject.toml     # uv 项目配置
├── uv.lock            # uv 锁文件
├── .python-version    # Python 版本 (如 3.11)
├── README.md          # 服务说明文档
└── example_client.py  # (可选) 客户端示例代码
```

每个服务使用 **uv** 独立管理 Python 环境，互不干扰。

### 启动服务

```bash
# 1. 安装环境
cd services/<project_name>
bash setup.bash

# 2. 启动服务 (uv run 自动激活虚拟环境)
uv run main.py --port <port> --gpu <gpu_id>
```

---

## ZMQServer 类书写规范

所有 `main.py` 中的 ZMQ Server 类应遵循以下接口规范：

### 必需接口

```python
class XxxZMQServer:
    def __init__(
        self,
        host: str = "*",
        port: int = 5555,
        gpu_id: int = 0,
        recv_timeout: int = 5000,
    ):
        """Initialize the server.

        Args:
            host: Host to bind to ("*" for all interfaces)
            port: ZMQ port to listen on
            gpu_id: GPU device ID (-1 for CPU)
            recv_timeout: ZMQ receive timeout in milliseconds
        """
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.recv_timeout = recv_timeout

        # ZMQ context and socket
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

    def handle_request(self, request_json: str) -> str:
        """Process a single JSON request.

        Args:
            request_json: JSON string containing the request

        Returns:
            JSON string containing the response
        """
        # 实现请求处理逻辑...
        pass

    def start(self) -> None:
        """Start the ZMQ server and begin processing requests.

        This method blocks until the server is stopped.
        """
        # 创建 ZMQ context 和 REP socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout)
        self.socket.bind(f"tcp://{self.host}:{self.port}")

        # 请求处理循环
        try:
            while True:
                try:
                    request_json = self.socket.recv_string()
                    response_json = self.handle_request(request_json)
                    self.socket.send_string(response_json)
                except zmq.Again:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the ZMQ server and cleanup resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


def main():
    """Main entry point for the ZMQ server."""
    parser = argparse.ArgumentParser(description="Xxx ZMQ Server")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    server = XxxZMQServer(host=args.host, port=args.port, gpu_id=args.gpu)
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
5. 实现 `XxxZMQServer` 类
6. 编写 `README.md` 说明文档
