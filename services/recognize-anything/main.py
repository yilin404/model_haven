#!/usr/bin/env python3
"""
RAM++ (Recognize Anything Plus) FastAPI Server - Open-Vocabulary Image Tagging

A FastAPI-based server that wraps the RAM++ model to produce open-vocabulary
tags for an input image. Tags can be fed downstream to a segmenter (e.g. SAM3)
to obtain object masks for scene-graph construction.
"""

import argparse
import base64
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_CHECKPOINT_PATH = "./checkpoints/ram_plus_swin_large_14m.pth"
DEFAULT_IMAGE_SIZE = 384
# When threshold is None, the model uses its built-in per-class thresholds
# (ram/data/ram_tag_list_threshold.txt), which are the tuned defaults.


class RamEngine(ModelEngine):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        super().__init__("model")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.transform = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        # Build on CPU, load checkpoint, then move to GPU (same pattern as sam3).
        self.model = ram_plus(
            pretrained=self.checkpoint_path,
            image_size=DEFAULT_IMAGE_SIZE,
            vit="swin_l",
        )
        self.model.eval()
        self.model = self.model.to(f"cuda:{self.gpu_id}")
        self.transform = get_transform(image_size=DEFAULT_IMAGE_SIZE)

    def _unload_impl(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.transform = None

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)
        # Optional per-request threshold override on the stateful model; restore in finally.
        old_threshold = None
        if threshold is not None:
            old_threshold = self.model.class_threshold.clone()
            self.model.class_threshold.fill_(threshold)

        logger.info("Tagging image")
        start_time = time.time()

        try:
            x = self.transform(image).unsqueeze(0).to(f"cuda:{self.gpu_id}")
            with torch.inference_mode():
                en_str, zh_str = inference(x, self.model)
            tags_en = [t for t in en_str.split(" | ") if t]
            tags_zh = [t for t in zh_str.split(" | ") if t] if zh_str else []

            generation_time = time.time() - start_time
            metadata = {
                "num_tags": len(tags_en),
                "threshold": threshold,
                "generation_time": round(generation_time, 2),
                "image_size": f"{image.width}x{image.height}",
            }
            logger.info(
                f"Tagging complete in {generation_time:.2f}s, {len(tags_en)} tag(s)"
            )
            return {
                "status": "success",
                "tags": tags_en,
                "tags_chinese": tags_zh,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Tagging failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
        finally:
            if old_threshold is not None:
                self.model.class_threshold.copy_(old_threshold)


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class TagRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded image string and will be decoded to PILImage.Image",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional unified threshold override (default: RAM++ per-class thresholds)",
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


class TagResponse(BaseModel):
    status: str
    tags: Optional[List[str]] = None
    tags_chinese: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class RamServer(BaseFastAPIServer):
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, **kwargs):
        self._engine = RamEngine(checkpoint_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/tag", response_model=TagResponse)
        async def tag(request: TagRequest):
            result = await self._engine.run_inference(
                request.image, request.threshold
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8002
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description="RAM++ FastAPI Server - Open-Vocabulary Image Tagging",
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
        help="Path to RAM++ checkpoint (ram_plus_swin_large_14m.pth)",
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
            logging.FileHandler("ram_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("RAM++ Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Checkpoint: {args.checkpoint_path}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = RamServer(
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
