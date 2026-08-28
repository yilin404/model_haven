#!/usr/bin/env python3
"""
SAM3 FastAPI Server - Text / Geometry Prompted Image Segmentation Service

A FastAPI-based server that wraps Meta AI's SAM3 (Segment Anything Model 3).

- /segment-text: one or more text prompts per request, segmented together in
  a single batched forward pass (detector pipeline).
- /segment-geometry: point [x, y] and box [x1, y1, x2, y2] prompts, one mask
  per prompt via the SAM1-style interactive predictor (predict_inst).
"""

import argparse
import base64
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_CHECKPOINT_PATH = "./checkpoints/sam3.pt"
# Inference resolution and default confidence threshold match the official
# batched-inference demo (deps/sam3/examples/sam3_image_batched_inference.ipynb).
DEFAULT_INFERENCE_RESOLUTION = 1008
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class Sam3Engine(ModelEngine):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        super().__init__("model")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None  # Sam3Processor for geometry prompts, built in _load_impl
        self.transform = None  # built once in _load_impl and reused

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        # Build on CPU, load checkpoint, then move to GPU (matches prior behaviour).
        # enable_inst_interactivity adds the SAM1-task predictor (point/box
        # prompts) from the same checkpoint; the batched detector forward is
        # unchanged.
        self.model = build_sam3_image_model(
            device="cpu",
            load_from_HF=False,
            checkpoint_path=self.checkpoint_path,
            enable_inst_interactivity=True,
        )
        self.model = self.model.to(f"cuda:{self.gpu_id}")
        # Its default device is cuda:0, so pass the selected GPU explicitly.
        self.processor = Sam3Processor(
            self.model,
            resolution=DEFAULT_INFERENCE_RESOLUTION,
            device=f"cuda:{self.gpu_id}",
        )
        # Image preprocessing pipeline (resize -> tensor -> normalize to [-1, 1]).
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=DEFAULT_INFERENCE_RESOLUTION,
                    max_size=DEFAULT_INFERENCE_RESOLUTION,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _unload_impl(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.processor = None
        self.transform = None

        # SAM3 keeps a long-lived autocast context whose cached weight casts
        # otherwise survive after the model references above are released.
        torch.clear_autocast_cache()

    @staticmethod
    def _encode_mask_as_png(mask_bool: np.ndarray) -> str:
        mask_uint8 = (mask_bool.squeeze().astype(np.uint8)) * 255
        mask_pil = PILImage.fromarray(mask_uint8, mode="L")
        buffer = BytesIO()
        mask_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _run_inference_impl(self, task: str, **kwargs) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)
        try:
            if task == "text":
                return self._segment_text_impl(**kwargs)
            if task == "geometry":
                return self._segment_geometry_impl(**kwargs)
            raise ValueError(f"Unknown inference task: {task}")
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _segment_text_impl(
        self,
        image: PILImage.Image,
        text_prompts: list[str],
        confidence_threshold: Optional[float],
    ) -> Dict[str, Any]:
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else DEFAULT_CONFIDENCE_THRESHOLD
        )

        logger.info(
            f"Segmenting image with {len(text_prompts)} prompt(s): {text_prompts}"
        )
        start_time = time.time()

        w, h = image.size
        # 1) Build a Datapoint holding a single image plus N text prompts.
        dp = Datapoint(
            find_queries=[],
            images=[SAMImage(data=image, objects=[], size=[h, w])],
        )
        for i, txt in enumerate(text_prompts):
            dp.find_queries.append(
                FindQueryLoaded(
                    query_text=txt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=i + 1,
                        original_image_id=i + 1,
                        original_category_id=1,
                        original_size=[h, w],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )

        # 2) transform -> collate -> move to device.
        dp = self.transform(dp)
        batch = collate_fn_api([dp], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(
            batch, torch.device(f"cuda:{self.gpu_id}"), non_blocking=True
        )

        # 3) Single batched forward (inference_mode + bf16 autocast).
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = self.model(batch)

        # 4) Postprocess; PostProcessImage is stateless, so build it per
        #    request to apply this request's confidence threshold directly.
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=threshold,
            to_cpu=False,
        )
        results = postprocessor.process_results(output, batch.find_metadatas)

        # 5) Group results by prompt and encode each mask to base64 PNG.
        grouped = []
        for i, txt in enumerate(text_prompts):
            r = results.get(i + 1)
            if r is None or len(r["scores"]) == 0:
                grouped.append(
                    {
                        "text_prompt": txt,
                        "masks": [],
                        "boxes": [],
                        "scores": [],
                        "num_objects": 0,
                    }
                )
                continue
            # Cast to fp32 before numpy: outputs are bf16 under autocast,
            # and numpy does not support bfloat16.
            scores_np = r["scores"].float().cpu().numpy()
            boxes_np = r["boxes"].float().cpu().numpy()
            masks_t = r["masks"]  # shape [N, 1, H, W], bool
            masks_b64 = [
                self._encode_mask_as_png(masks_t[j].squeeze(0).cpu().numpy())
                for j in range(len(scores_np))
            ]
            grouped.append(
                {
                    "text_prompt": txt,
                    "masks": masks_b64,
                    "boxes": boxes_np.tolist(),
                    "scores": scores_np.tolist(),
                    "num_objects": len(masks_b64),
                }
            )

        generation_time = time.time() - start_time
        total = sum(g["num_objects"] for g in grouped)
        metadata = {
            "text_prompts": text_prompts,
            "confidence_threshold": threshold,
            "num_objects": [g["num_objects"] for g in grouped],
            "generation_time": round(generation_time, 2),
            "image_size": f"{w}x{h}",
        }
        logger.info(
            f"Segmentation complete in {generation_time:.2f}s, "
            f"{total} object(s) across {len(text_prompts)} prompt(s)"
        )
        return {
            "status": "success",
            "results": grouped,
            "metadata": metadata,
        }

    def _segment_geometry_impl(
        self,
        image: PILImage.Image,
        points: Optional[list[list[float]]],
        boxes: Optional[list[list[float]]],
    ) -> Dict[str, Any]:
        points = points or []
        boxes = boxes or []

        logger.info(
            f"Segmenting image with {len(points)} point prompt(s) and "
            f"{len(boxes)} box prompt(s)"
        )
        start_time = time.time()

        # fp32 without autocast, matching the official interactive-predictor
        # example (examples/sam3_for_sam1_task_example.ipynb).
        with torch.inference_mode():
            # Backbone features are computed once and shared by all prompts;
            # each prompt then runs one SAM1-style predict_inst call.
            state = self.processor.set_image(image)

            point_results = []
            for pt in points:
                # A single positive click is ambiguous, so request three mask
                # candidates and keep the best one by predicted IoU.
                masks, scores, _ = self.model.predict_inst(
                    state,
                    point_coords=np.array([pt], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                point_results.append(
                    {
                        "mask": self._encode_mask_as_png(masks[best]),
                        "score": float(scores[best]),
                    }
                )

            box_results = []
            for bx in boxes:
                # Boxes are unambiguous prompts: a single mask is more accurate.
                masks, scores, _ = self.model.predict_inst(
                    state,
                    box=np.array(bx, dtype=np.float32),
                    multimask_output=False,
                )
                box_results.append(
                    {
                        "mask": self._encode_mask_as_png(masks[0]),
                        "score": float(scores[0]),
                    }
                )

        generation_time = time.time() - start_time
        w, h = image.size
        metadata = {
            "num_point_prompts": len(point_results),
            "num_box_prompts": len(box_results),
            "generation_time": round(generation_time, 2),
            "image_size": f"{w}x{h}",
        }
        logger.info(
            f"Segmentation complete in {generation_time:.2f}s, "
            f"{len(point_results)} point mask(s) and "
            f"{len(box_results)} box mask(s)"
        )
        return {
            "status": "success",
            "results": {"points": point_results, "boxes": box_results},
            "metadata": metadata,
        }


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class BaseSegmentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded image string and will be decoded to PILImage.Image",
    )

    @field_validator("image", mode="before", json_schema_input_type=str)
    @classmethod
    def parse_image(cls, v: str) -> PILImage.Image:
        if not isinstance(v, str):
            raise ValueError("Image must be a base64-encoded string")
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Image must be valid base64-encoded data: {e}")
        try:
            return PILImage.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Image data is not a valid image format: {e}")


class SegmentRequest(BaseSegmentRequest):
    text_prompts: Union[str, list[str]] = Field(
        ...,
        description=(
            "Text prompt(s) for segmentation: a single string or a list of "
            "strings. Multiple prompts are segmented together in one forward pass."
        ),
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence threshold shared by all prompts (default: 0.5)",
    )

    @field_validator("text_prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, v):
        # Accept a single string and normalize it to a one-element list.
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(
                "text_prompts must be a non-empty string or list of strings"
            )
        for x in v:
            if not isinstance(x, str) or not x.strip():
                raise ValueError("each text prompt must be a non-empty string")
        return v


class GeometrySegmentRequest(BaseSegmentRequest):
    points: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Point prompts, each [x, y] in pixels on the original image. "
            "Each point returns the mask of the object it clicks on."
        ),
    )
    boxes: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Box prompts, each [x1, y1, x2, y2] (XYXY) in pixels on the "
            "original image. Each box returns the mask of the object inside it."
        ),
    )

    @model_validator(mode="after")
    def validate_prompts(self):
        points = self.points or []
        boxes = self.boxes or []
        if not points and not boxes:
            raise ValueError("at least one of points / boxes must be non-empty")
        w, h = self.image.size
        for i, pt in enumerate(points):
            if len(pt) != 2 or not all(np.isfinite(pt)):
                raise ValueError(f"points[{i}] must be [x, y] with finite values")
            if not (0 <= pt[0] <= w - 1 and 0 <= pt[1] <= h - 1):
                raise ValueError(
                    f"points[{i}] = {pt} is outside the image ({w}x{h})"
                )
        for i, bx in enumerate(boxes):
            if len(bx) != 4 or not all(np.isfinite(bx)):
                raise ValueError(
                    f"boxes[{i}] must be [x1, y1, x2, y2] with finite values"
                )
            if bx[2] <= bx[0] or bx[3] <= bx[1]:
                raise ValueError(
                    f"boxes[{i}] must be XYXY with x2 > x1 and y2 > y1"
                )
            if bx[0] < 0 or bx[1] < 0 or bx[2] > w - 1 or bx[3] > h - 1:
                raise ValueError(
                    f"boxes[{i}] extends outside the image ({w}x{h})"
                )
        return self


class PromptResult(BaseModel):
    text_prompt: str
    masks: list[str]
    boxes: list[list[float]]
    scores: list[float]
    num_objects: int


class SegmentResponse(BaseModel):
    status: str
    results: Optional[list[PromptResult]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class GeometryMaskResult(BaseModel):
    mask: str
    score: float


class GeometrySegmentResponse(BaseModel):
    status: str
    results: Optional[Dict[str, list[GeometryMaskResult]]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class Sam3Server(BaseFastAPIServer):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, **kwargs):
        self._engine = Sam3Engine(checkpoint_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/segment-text", response_model=SegmentResponse)
        async def segment_text(request: SegmentRequest):
            result = await self._engine.run_inference(
                "text",
                image=request.image,
                text_prompts=request.text_prompts,
                confidence_threshold=request.confidence_threshold,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result

        @self._app.post(
            "/segment-geometry", response_model=GeometrySegmentResponse
        )
        async def segment_geometry(request: GeometrySegmentRequest):
            result = await self._engine.run_inference(
                "geometry",
                image=request.image,
                points=request.points,
                boxes=request.boxes,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8004
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SAM3 FastAPI Server - Text / Geometry Prompted "
            "Image Segmentation Service"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to SAM3 model checkpoint (PyTorch .pt file)",
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
            logging.FileHandler(os.path.join(os.path.dirname(__file__), "sam3_server.log")),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("SAM3 Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Checkpoint: {args.checkpoint_path}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = Sam3Server(
        checkpoint_path=args.checkpoint_path,
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
