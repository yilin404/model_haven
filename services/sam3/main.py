#!/usr/bin/env python3
"""
SAM3 FastAPI Server - Text-Prompted Image Segmentation Service

A FastAPI-based server that wraps Meta AI's SAM3 (Segment Anything Model 3)
for text-prompted image segmentation.
"""

import argparse
import base64
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_CHECKPOINT_PATH = "./checkpoints/sam3.pt"


class Sam3Engine(ModelEngine):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        super().__init__("model")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor: Optional[Sam3Processor] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        self.model = build_sam3_image_model(
            device="cpu",
            load_from_HF=False,
            checkpoint_path=self.checkpoint_path,
        )
        self.model = self.model.to(f"cuda:{self.gpu_id}")
        self.processor = Sam3Processor(self.model, device=f"cuda:{self.gpu_id}")

    def _unload_impl(self) -> None:
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.model is not None:
            del self.model
            self.model = None

    @staticmethod
    def _encode_mask_as_png(mask_bool: np.ndarray) -> str:
        mask_uint8 = (mask_bool.squeeze().astype(np.uint8)) * 255
        mask_pil = PILImage.fromarray(mask_uint8, mode="L")
        buffer = BytesIO()
        mask_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        text_prompt: str,
        confidence_threshold: Optional[float],
    ) -> Dict[str, Any]:
        old_threshold = self.processor.confidence_threshold
        if confidence_threshold is not None:
            self.processor.set_confidence_threshold(confidence_threshold)

        logger.info(f"Segmenting image with prompt: '{text_prompt[:80]}'")
        start_time = time.time()

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(text_prompt, state)

            masks = state["masks"].cpu().to(torch.uint8).numpy()
            boxes = state["boxes"].cpu().float().numpy()
            scores = state["scores"].cpu().float().numpy()

            masks_b64 = [self._encode_mask_as_png(m) for m in masks]

            generation_time = time.time() - start_time
            metadata = {
                "text_prompt": text_prompt,
                "confidence_threshold": self.processor.confidence_threshold,
                "num_objects": len(masks_b64),
                "generation_time": round(generation_time, 2),
                "image_size": f"{image.width}x{image.height}",
            }

            logger.info(
                f"Segmentation complete in {generation_time:.2f}s, "
                f"{len(masks_b64)} object(s) found"
            )

            return {
                "status": "success",
                "masks": masks_b64,
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}
        finally:
            self.processor.set_confidence_threshold(old_threshold)


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class SegmentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded image string and will be decoded to PILImage.Image",
    )
    text_prompt: str = Field(
        ..., min_length=1, description="Text prompt for segmentation"
    )
    confidence_threshold: Optional[float] = Field(
        default=None, ge=0, le=1, description="Confidence threshold (default: 0.5)"
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


class SegmentResponse(BaseModel):
    status: str
    masks: Optional[list[str]] = None
    boxes: Optional[list[list[float]]] = None
    scores: Optional[list[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class Sam3Server(BaseFastAPIServer):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, **kwargs):
        self._engine = Sam3Engine(checkpoint_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/segment", response_model=SegmentResponse)
        async def segment(request: SegmentRequest):
            result = await self._engine.run_inference(
                request.image, request.text_prompt, request.confidence_threshold
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
        description="SAM3 FastAPI Server - Text-Prompted Image Segmentation Service",
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
            logging.FileHandler("sam3_server.log"),
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
