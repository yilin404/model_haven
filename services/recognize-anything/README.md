# RAM++ — 开放词汇图像标注服务

基于 [Recognize Anything Plus (RAM++)](https://github.com/xinyu1205/recognize-anything) 的开放词汇图像标注服务。接收一张图片,返回图中物体的开放词汇标签(英文 + 中文,4585 类词表)。标签可作为 SAM3 等 segmentation 模型的 text prompt,用于下游场景图构建的"图像 → 物体(label, mask)"流水线。

**环境要求:** NVIDIA GPU + CUDA 12.1, Python 3.10

**默认端口:** 8002

## 安装

```bash
cd services/recognize-anything
bash setup.bash
```

## 启动服务

```bash
uv run main.py

# 自定义配置
uv run main.py --host 0.0.0.0 --port 8002 --idle-timeout 600
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务绑定地址 |
| `--port` | `8002` | 监听端口 |
| `--checkpoint-path` | `./checkpoints/ram_plus_swin_large_14m.pth` | RAM++ 权重路径 |
| `--idle-timeout` | `300` | 空闲超时秒数(超时自动卸载模型) |
| `--idle-check-interval` | `30` | 空闲检查间隔秒数 |
| `--log-level` | `INFO` | 日志级别 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查(GPU/显存/模型状态) |
| `POST` | `/tag` | 开放词汇图像标注 |

## 请求格式

```json
{
  "image": "<base64 编码的图片>",
  "threshold": 0.5
}
```

- `image`:base64 编码的图片数据(支持 `data:image/...;base64,` 前缀,自动剥离)
- `threshold`(可选):统一阈值覆盖,范围 `[0,1]`。**不传则使用 RAM++ 自带的逐类最佳阈值**(`ram/data/ram_tag_list_threshold.txt`),通常效果最好

## 成功响应

```json
{
  "status": "success",
  "tags": ["cat", "laptop", "desk"],
  "tags_chinese": ["猫", "笔记本电脑", "桌子"],
  "metadata": {
    "num_tags": 3,
    "threshold": null,
    "generation_time": 0.45,
    "image_size": "640x425"
  }
}
```

- `tags`:英文标签列表
- `tags_chinese`:对应中文标签列表
- `metadata.num_tags`:标签数量
- `metadata.threshold`:本次请求生效的阈值(`null` 表示用默认逐类阈值)

## 示例客户端

```bash
# 健康检查
uv run example_client.py --host localhost --port 8002 health

# 标注(默认逐类阈值)
uv run example_client.py --port 8002 tag --image photo.jpg

# 指定统一阈值
uv run example_client.py --port 8002 tag --image photo.jpg --threshold 0.5

# 保存结果为 JSON
uv run example_client.py --port 8002 tag --image photo.jpg --output tags.json
```

## 与 SAM3 串联(场景图节点提取)

将本服务返回的 `tags` 作为 `text_prompts` 调用 SAM3 `/segment` 接口(默认端口 8004),即可一次性得到每个标签对应的物体掩码,完成"图像 → 物体(label, mask)":

```
image --RAM++ /tag--> tags --SAM3 /segment (text_prompts=tags)--> per-tag masks
```
