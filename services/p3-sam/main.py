#!/usr/bin/env python3
"""
P3-SAM FastAPI Server - Point-Level 3D Part Segmentation Service

A FastAPI-based server that wraps Tencent's P3-SAM (Hunyuan3D-Part) point-promptable
segmentation model. It accepts an externally sampled point cloud with normals and
returns per-point part labels, replicating the official ``mesh_sam`` point-level
pipeline (feature extraction -> FPS prompts -> batched mask inference -> NMS
clustering / bbox merge / repair). All mesh handling of the official pipeline
(internal sampling, face voting, face repair) is intentionally excluded: the
caller owns the mesh and performs the face-level majority vote itself.
"""

import argparse
import importlib
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu
from serialization import NDArrayData

logger = logging.getLogger(__name__)


# ===========================================================================
# Upstream (deps/hunyuan3d-part) integration
# ===========================================================================
HUNYUAN_PART_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../deps/hunyuan3d-part")
)
P3SAM_ROOT = os.path.join(HUNYUAN_PART_ROOT, "P3-SAM")
P3SAM_DEMO_ROOT = os.path.join(P3SAM_ROOT, "demo")
PARTGEN_ROOT = os.path.join(HUNYUAN_PART_ROOT, "XPart", "partgen")

DEFAULT_HF_REPO = "tencent/Hunyuan3D-Part"
DEFAULT_CKPT_FILENAME = "p3sam/p3sam.safetensors"
DEFAULT_SONATA_CKPT = "~/.cache/sonata/ckpt/sonata.pth"


def _patch_p3sam_sonata_loader() -> None:
    """Patch upstream ``model.build_P3SAM`` before ``auto_mask`` is imported.

    Differences from upstream ``P3-SAM/model.py::build_P3SAM``:
    - upstream hardcodes ``download_root='/root/sonata'``; we resolve the
      preheated local sonata checkpoint (setup.bash downloads it to
      ``~/.cache/sonata/ckpt``) and fall back to the hub download with the
      loader's default cache;
    - flash attention is disabled via ``custom_config`` so the service does
      not need to compile flash-attn (PointTransformerV3 falls back to the
      plain attention backend).

    Note:
        The patch must run before ``from auto_mask import ...`` because
        ``auto_mask`` binds ``build_P3SAM`` into its own namespace at import
        time. Path insertion order (partgen before p3sam) mirrors EmbodiedGen
        so that the generic top-level names ``utils``/``models`` resolve to
        the XPart/partgen packages that ``model.py`` expects.
    """
    for path in [P3SAM_ROOT, PARTGEN_ROOT]:
        if path not in sys.path:
            sys.path.insert(0, path)

    from models import sonata

    p3sam_model = importlib.import_module("model")

    sonata_name: Any = "sonata"
    local_sonata = os.path.expanduser(DEFAULT_SONATA_CKPT)
    if os.path.isfile(local_sonata):
        sonata_name = local_sonata

    def build_P3SAM(self) -> None:  # noqa: N802 - keep upstream name
        self.sonata = sonata.load(
            sonata_name,
            repo_id="facebook/sonata",
            custom_config={"enable_flash": False},
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1232, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 512),
        )
        self.transform = sonata.transform.default()

        def seg_mlp() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(512 + 3 + 3, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 1),
            )

        def seg_s2_mlp(width: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(512 + 3 + 3 + 3 + 256, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, 1),
            )

        self.seg_mlp_1 = seg_mlp()
        self.seg_mlp_2 = seg_mlp()
        self.seg_mlp_3 = seg_mlp()

        self.seg_s2_mlp_g = torch.nn.Sequential(
            torch.nn.Linear(512 + 3 + 3 + 3, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
        )
        self.seg_s2_mlp_1 = seg_s2_mlp(256)
        self.seg_s2_mlp_2 = seg_s2_mlp(256)
        self.seg_s2_mlp_3 = seg_s2_mlp(256)

        self.iou_mlp = torch.nn.Sequential(
            torch.nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
        )
        self.iou_mlp_out = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 3),
        )
        self.iou_criterion = torch.nn.MSELoss()

    p3sam_model.build_P3SAM = build_P3SAM


_patch_p3sam_sonata_loader()

sys.path.insert(0, P3SAM_DEMO_ROOT)
from auto_mask import (  # noqa: E402 - must import after the patch above
    P3SAM,
    cal_bbox_iou,
    cal_iou,
    cal_single_iou,
    get_feat,
    get_mask,
    normalize_pc,
    set_seed,
)

import fpsample  # noqa: E402 - upstream demo dependency


# ===========================================================================
# Point-level segmentation pipeline (official mesh_sam, points-only port)
# ===========================================================================
def segment_points(
    model: P3SAM,
    points_raw: np.ndarray,
    normals: np.ndarray,
    prompt_num: int,
    prompt_bs: int,
    seed: int,
) -> Dict[str, Any]:
    """Segment a point cloud into parts, mirroring the official ``mesh_sam``.

    Steps 1-3 follow ``auto_mask.py::mesh_sam`` verbatim on the point level:
    normalize to [-1, 1] (bbox center / longest half extent), extract Sonata
    features, pick FPS prompts, run batched mask inference keeping the best-IoU
    head per prompt. Steps 4-6 are the official clustering cascade: NMS with
    IoU > 0.9, drop clusters with <= 2 member masks, bbox-IoU > 0.5 re-merge,
    and repair of uncovered points with single-IoU > 0.7 masks. Labels are
    finally compacted to ``0..P-1`` ordered by part area (descending), with -1
    for unassigned points - the official pipeline emits raw cluster indices and
    leaves the compaction to the caller (EmbodiedGen's ``remap_to_palette``).

    Args:
        model: P3SAM module already on GPU.
        points_raw: (N, 3) float32 point positions in the caller's frame; the
            model is scale-invariant, the frame only has to be consistent.
        normals: (N, 3) float32 point normals (caller-side face normals).
        prompt_num: number of FPS prompt points (official default 400).
        prompt_bs: prompts per forward batch (official default 32).
        seed: RNG seed applied before inference for reproducibility.

    Returns:
        Dict with ``labels`` (N,) int32 in {0..P-1, -1}, ``num_parts``,
        ``part_sizes`` (per-label point counts, area-descending) and
        ``num_unassigned``.
    """
    num_points = points_raw.shape[0]

    # 阶段 1：归一化 + 特征 + FPS 提示点（官方 mesh_sam 的采样段替换为外部点云）
    set_seed(seed)
    points = normalize_pc(points_raw)
    feats = get_feat(model, points, normals)  # [N, 512] cuda tensor

    fps_idx = fpsample.fps_sampling(points, prompt_num)
    point_prompts = points[fps_idx]

    # 阶段 2：分批推理，每个 prompt 取 IoU 最优 head 的 mask（官方逐批循环）
    mask_list: list[np.ndarray] = []
    iou_list: list[float] = []
    for start in range(0, prompt_num, prompt_bs):
        batch_prompts = point_prompts[start : start + prompt_bs]
        if len(batch_prompts) == 0:
            break
        pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
            model, feats, points, batch_prompts
        )
        pred_mask = np.stack([pred_mask_1, pred_mask_2, pred_mask_3], axis=-1)
        best_head = np.argmax(pred_iou, axis=-1)  # [K]
        for j in range(best_head.shape[0]):
            mask_list.append(pred_mask[:, j, best_head[j]])
            iou_list.append(float(pred_iou[j, best_head[j]]))

    # 阶段 3：按 IoU 降序排序后做 NMS 聚类（IoU > 0.9 并入首个命中簇）
    order = np.argsort(np.array(iou_list))[::-1]
    mask_sorted = [mask_list[i] for i in order]
    iou_sorted = [iou_list[i] for i in order]

    clusters: Dict[int, list[int]] = defaultdict(list)
    for i in range(prompt_num):
        mask_i = mask_sorted[i]
        for j in clusters.keys():
            if cal_iou(mask_i, mask_sorted[j]) > 0.9:
                clusters[j].append(i)
                break
        else:
            clusters[i].append(i)

    # 阶段 4：过滤成员数 <= 2 的簇，再按 bbox IoU > 0.5 二次合并（官方两步）
    filtered = [i for i in clusters if len(clusters[i]) > 2]
    merged: Dict[int, list[int]] = {}
    consumed = [False] * len(filtered)
    for pos, i in enumerate(filtered):
        if consumed[pos]:
            continue
        merged[i] = [i]
        for pos_j in range(pos + 1, len(filtered)):
            j = filtered[pos_j]
            if consumed[pos_j]:
                continue
            if cal_bbox_iou(points, mask_sorted[j], mask_sorted[i]) > 0.5:
                merged[i].append(j)
                consumed[pos_j] = True

    # 阶段 5：修补无 mask 覆盖的点（single IoU > 0.7 的落选 mask 补位）
    no_mask = np.ones(num_points, dtype=bool)
    for i in merged:
        no_mask[mask_sorted[i]] = False
    for i in range(prompt_num):
        if i in merged:
            continue
        part_mask = mask_sorted[i]
        if cal_single_iou(part_mask, no_mask) > 0.7:
            merged[i] = [i]
            no_mask[part_mask] = False

    # 阶段 6：按面积降序赋标签并压缩为 0..P-1（-1 = 未分配）
    final_indices = sorted(merged.keys(), key=lambda i: int(mask_sorted[i].sum()), reverse=True)
    labels = np.full(num_points, -1, dtype=np.int32)
    for new_id, i in enumerate(final_indices):
        labels[mask_sorted[i]] = new_id

    part_sizes = [int((labels == k).sum()) for k in range(len(final_indices))]
    return {
        "labels": labels,
        "num_parts": len(final_indices),
        "part_sizes": part_sizes,
        "num_unassigned": int((labels == -1).sum()),
    }


# ===========================================================================
# ModelEngine
# ===========================================================================
class P3SamEngine(ModelEngine):
    def __init__(self, hf_repo: str, ckpt_filename: str):
        super().__init__("model")
        self.hf_repo = hf_repo
        self.ckpt_filename = ckpt_filename
        self.model: Optional[P3SAM] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)

        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id=self.hf_repo, filename=self.ckpt_filename
        )
        logger.info(f"P3-SAM checkpoint resolved at: {ckpt_path}")

        # build_P3SAM 已被打补丁：sonata 权重走本地预热路径、flash attention 关闭
        model = P3SAM()
        model.load_state_dict(ckpt_path=ckpt_path)
        model.eval()
        model.cuda()
        self.model = model

    def _unload_impl(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def _run_inference_impl(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        prompt_num: int,
        prompt_bs: int,
        seed: int,
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)

        logger.info(
            f"Segmenting {points.shape[0]} points "
            f"(prompt_num={prompt_num}, prompt_bs={prompt_bs}, seed={seed})"
        )
        start_time = time.time()

        try:
            result = segment_points(
                self.model, points, normals, prompt_num, prompt_bs, seed
            )
            generation_time = time.time() - start_time

            logger.info(
                f"Segmented into {result['num_parts']} parts in "
                f"{generation_time:.2f}s (unassigned: {result['num_unassigned']})"
            )

            return {
                "status": "success",
                "labels": NDArrayData.from_array(result["labels"]),
                "num_parts": result["num_parts"],
                "metadata": {
                    "generation_time": round(generation_time, 2),
                    "num_points": int(points.shape[0]),
                    "num_parts": result["num_parts"],
                    "num_unassigned": result["num_unassigned"],
                    "part_sizes": result["part_sizes"],
                    "prompt_num": prompt_num,
                    "prompt_bs": prompt_bs,
                    "seed": seed,
                },
            }

        except Exception as e:  # noqa: BLE001 - engine contract: never raise
            logger.error(f"Point segmentation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class SegmentRequest(BaseModel):
    points: NDArrayData = Field(
        ...,
        description="Point positions (N, 3) as {data: base64, shape, dtype}",
    )
    normals: NDArrayData = Field(
        ...,
        description="Point normals (N, 3), e.g. face normals at sampling time",
    )
    prompt_num: int = Field(
        default=400, gt=0, description="Number of FPS prompt points"
    )
    prompt_bs: int = Field(
        default=8,
        gt=0,
        description="Prompts per forward batch; official demo default 32 needs "
        "~40GB VRAM at 100k points, 8 fits 24GB cards (EmbodiedGen default)",
    )
    seed: int = Field(default=42, description="RNG seed for reproducibility")


class SegmentResponse(BaseModel):
    status: str
    labels: Optional[NDArrayData] = None
    num_parts: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


MAX_POINTS = 2_000_000


class P3SamServer(BaseFastAPIServer):
    def __init__(self, hf_repo: str, ckpt_filename: str, **kwargs):
        self._engine = P3SamEngine(hf_repo, ckpt_filename)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/segment", response_model=SegmentResponse)
        async def segment(request: SegmentRequest):
            points = request.points.to_array().astype(np.float32, copy=False)
            normals = request.normals.to_array().astype(np.float32, copy=False)

            if points.ndim != 2 or points.shape[1] != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"points must be (N, 3), got shape {points.shape}",
                )
            if normals.shape != points.shape:
                raise HTTPException(
                    status_code=400,
                    detail=f"normals must match points shape {points.shape}, "
                    f"got {normals.shape}",
                )
            if points.shape[0] < request.prompt_num:
                raise HTTPException(
                    status_code=400,
                    detail=f"points must contain at least prompt_num="
                    f"{request.prompt_num} points, got {points.shape[0]}",
                )
            if points.shape[0] > MAX_POINTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"points must contain at most {MAX_POINTS} points, "
                    f"got {points.shape[0]}",
                )
            if not (np.all(np.isfinite(points)) and np.all(np.isfinite(normals))):
                raise HTTPException(
                    status_code=400,
                    detail="points/normals contain non-finite values (NaN or Inf)",
                )

            result = await self._engine.run_inference(
                points=points,
                normals=normals,
                prompt_num=request.prompt_num,
                prompt_bs=request.prompt_bs,
                seed=request.seed,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8008
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description="P3-SAM FastAPI Server - Point-Level 3D Part Segmentation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="HuggingFace repo holding the P3-SAM checkpoint",
    )
    parser.add_argument(
        "--ckpt-filename",
        type=str,
        default=DEFAULT_CKPT_FILENAME,
        help="Checkpoint filename inside the repo",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before unloading model",
    )
    parser.add_argument(
        "--idle-check-interval",
        type=int,
        default=DEFAULT_IDLE_CHECK_INTERVAL,
        help="Seconds between idle checks",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(os.path.dirname(__file__), "p3sam_server.log")
            ),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("P3-SAM Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"HF Repo: {args.hf_repo}")
    log.info(f"Checkpoint: {args.ckpt_filename}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = P3SamServer(
        hf_repo=args.hf_repo,
        ckpt_filename=args.ckpt_filename,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        log.info("Server interrupted by user")
    except Exception as e:
        log.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
