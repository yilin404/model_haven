# P3-SAM — 点云级 3D 部件分割服务

基于 [Tencent Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) 中 P3-SAM 点提示分割模型的服务。接收**调用方自行采样的点云 + 法线**，返回**逐点部件标签**。服务端复刻官方 `auto_mask.py::mesh_sam` 的点级流程（Sonata 特征提取 → FPS 提示点 → 批量 mask 推理 → NMS 聚类 / bbox 合并 / 遗漏修补），刻意排除官方管线中一切 mesh 相关步骤（内部采样、face 投票、face 修复）——调用方持有 mesh，由调用方完成"点标签 → 面标签"的多数投票。

**环境要求：** NVIDIA GPU + CUDA 12.8, Python 3.10

**默认端口：** 8008

## 模型缓存

`start-services.sh` 以 `HF_HUB_OFFLINE=1` 启动服务，**必须先运行 `setup.bash` 预热权重**：

- P3-SAM checkpoint（`tencent/Hunyuan3D-Part` 的 `p3sam/p3sam.safetensors`）→ 共享 HF 缓存（`~/.cache/huggingface`）
- Sonata 骨干（`facebook/sonata` 的 `sonata.pth`）→ `~/.cache/sonata/ckpt`（`sonata.load` 使用 local_dir 约定）

## 安装

```bash
cd services/p3-sam
bash setup.bash
```

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8008 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8008` | 监听端口 |
| `--hf-repo` | `tencent/Hunyuan3D-Part` | P3-SAM checkpoint 所在 HF 仓库 |
| `--ckpt-filename` | `p3sam/p3sam.safetensors` | checkpoint 在仓库内的文件名 |
| `--idle-timeout` | `300` | 空闲超时秒数（超时自动卸载模型） |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（不触发模型加载） |
| `POST` | `/segment` | 点云级部件分割 |

## 请求格式

`points` / `normals` 使用 `NDArrayData` 编码（`{data: base64, shape, dtype}`，与 GraspGen 服务一致）：

```json
{
  "points":  {"data": "<base64>", "shape": [100000, 3], "dtype": "float32"},
  "normals": {"data": "<base64>", "shape": [100000, 3], "dtype": "float32"},
  "prompt_num": 400,
  "prompt_bs": 32,
  "seed": 42
}
```

- `points`：`(N, 3)` 点坐标。模型内部按 bbox 归一化到 `[-1, 1]`，尺度不变，坐标系只需自洽；建议 N = 100000（与官方一致），下限为 `prompt_num`，上限 2,000,000
- `normals`：`(N, 3)` 点法线（调用方采样时保留的 face 法线），shape 必须与 `points` 一致
- `prompt_num`（可选）：FPS 提示点数，默认 400（官方默认）
- `prompt_bs`（可选）：提示点前向批大小，默认 8（EmbodiedGen 同款；官方 demo 的 32 在 100k 点下需 ~40GB 显存，24GB 卡请用 8 或更小）
- `seed`（可选）：随机种子，默认 42，保证可复现

## 成功响应

```json
{
  "status": "success",
  "labels": {"data": "<base64>", "shape": [100000], "dtype": "int32"},
  "num_parts": 3,
  "metadata": {
    "generation_time": 18.4,
    "num_points": 100000,
    "num_parts": 3,
    "num_unassigned": 12,
    "part_sizes": [52130, 40215, 7643],
    "prompt_num": 400,
    "prompt_bs": 32,
    "seed": 42
  }
}
```

- `labels`：`(N,)` int32，取值 `0..P-1`（按部件点数**降序**编号，0 号最大）或 `-1`（未分配）。官方管线的原始输出是簇索引、由调用方压缩编号，本服务直接完成压缩以简化客户端
- `num_parts`：部件数
- `metadata.part_sizes`：与标签 0..P-1 对应的每部件点数

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8008 health

# 内置合成双球冒烟测试（两个分离球面点云，预期恰好分出 2 个部件，PASS 时退出码 0）
uv run example_client.py --port 8008 segment
```

## 实现说明

- **monkey patch**：官方 `P3-SAM/model.py::build_P3SAM` 把 Sonata 权重路径硬编码为 `/root/sonata`；本服务按 EmbodiedGen 的做法在导入 `auto_mask` 前替换该函数——改用 `~/.cache/sonata/ckpt` 的预热权重，并通过 `custom_config={"enable_flash": False}` 关闭 flash attention（PointTransformerV3 回退普通注意力后端，**无需安装 flash-attn**）
- **上游函数直接复用**：`get_feat` / `get_mask` / `normalize_pc` / `set_seed` / `cal_iou` 系列全部从 `deps/hunyuan3d-part/P3-SAM/demo/auto_mask.py` 导入，服务内只实现点级编排（`segment_points`），保证与官方行为一致
- 模型惰性加载（首个请求触发）、空闲自动卸载、CUDA OOM 自愈由共享基类 `services/common.py` 提供
