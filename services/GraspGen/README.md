# GraspGen — 6-DOF 抓取生成服务

基于 [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) 的 6 自由度抓取姿态生成服务。接收点云数据（base64 编码的 numpy 数组），返回抓取姿态和置信度。

**环境要求：** NVIDIA GPU + CUDA 12.1, Python 3.10

**默认端口：** 8001

## 安装

```bash
cd services/graspgen
bash setup.bash
```

## 启动服务

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

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/generate` | 抓取生成 |

## 请求格式

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

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8001 health

# 从点云文件生成抓取
uv run example_client.py --port 8001 generate --pcd-file input.npy

# 从网格文件生成抓取
uv run example_client.py --port 8001 generate --mesh-file model.obj --visualize
```
