#!/usr/bin/env python3
"""
TRELLIS Server - Text/Image-to-3D Generation Service

A FastAPI-based server that wraps Microsoft's TRELLIS pipelines for
generating 3D models from text prompts or images.
"""

import argparse
import base64
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Optional

import torch
from PIL import Image as PILImage
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/trellis"))

from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils

os.environ["SPCONV_ALGO"] = "native"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngines
# ===========================================================================
DEFAULT_TEXT_MODEL = "microsoft/TRELLIS-text-xlarge"
DEFAULT_IMAGE_MODEL = "microsoft/TRELLIS-image-large"


def _export_glb(outputs, simplify: float, texture_size: int) -> bytes:
    gaussian = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    glb_mesh = postprocessing_utils.to_glb(
        gaussian,
        mesh,
        simplify=simplify,
        texture_size=texture_size,
        fill_holes=True,
        verbose=False,
    )
    buffer = BytesIO()
    glb_mesh.export(file_obj=buffer, file_type="glb")
    return buffer.getvalue()


class TrellisTextEngine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL):
        super().__init__("text_pipeline")
        self.model_name = model_name
        self.pipeline: Optional[TrellisTextTo3DPipeline] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_name)
        self.pipeline.to(f"cuda:{self.gpu_id}")

    def _unload_impl(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def _run_inference_impl(
        self,
        text: str,
        seed: Optional[int],
        simplify: float,
        texture_size: int,
    ) -> Dict[str, Any]:
        seed = seed if seed is not None else torch.randint(0, 2**32, (1,)).item()

        logger.info(
            f"Generating 3D from text: '{text}' (seed={seed}, "
            f"simplify={simplify}, texture_size={texture_size})"
        )
        start_time = time.time()

        try:
            outputs = self.pipeline.run(
                text,
                seed=seed,
                formats=["mesh", "gaussian"],
            )

            glb_bytes = _export_glb(outputs, simplify, texture_size)
            generation_time = time.time() - start_time

            logger.info(
                f"Generation complete in {generation_time:.2f}s, "
                f"file size: {len(glb_bytes)} bytes"
            )

            return {
                "status": "success",
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8"),
                "metadata": {
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "file_size": len(glb_bytes),
                    "simplify": simplify,
                    "texture_size": texture_size,
                },
            }

        except Exception as e:
            logger.error(f"Text-to-3D generation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


class TrellisImageEngine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL):
        super().__init__("image_pipeline")
        self.model_name = model_name
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_name)
        self.pipeline.to(f"cuda:{self.gpu_id}")

    def _unload_impl(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        seed: Optional[int],
        simplify: float,
        texture_size: int,
        preprocess_image: bool,
    ) -> Dict[str, Any]:
        seed = seed if seed is not None else torch.randint(0, 2**32, (1,)).item()

        logger.info(
            f"Generating 3D from image (seed={seed}, "
            f"simplify={simplify}, texture_size={texture_size})"
        )
        start_time = time.time()

        try:
            outputs = self.pipeline.run(
                image,
                seed=seed,
                formats=["mesh", "gaussian"],
                preprocess_image=preprocess_image,
            )

            glb_bytes = _export_glb(outputs, simplify, texture_size)
            generation_time = time.time() - start_time

            logger.info(
                f"Image-to-3D complete in {generation_time:.2f}s, "
                f"file size: {len(glb_bytes)} bytes"
            )

            return {
                "status": "success",
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8"),
                "metadata": {
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "file_size": len(glb_bytes),
                    "simplify": simplify,
                    "texture_size": texture_size,
                    "preprocess_image": preprocess_image,
                },
            }

        except Exception as e:
            logger.error(f"Image-to-3D generation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class TextTo3DRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text prompt for 3D generation")
    seed: Optional[int] = Field(default=None, description="Random seed")
    simplify: float = Field(
        default=0.95, ge=0, le=1, description="Mesh simplification ratio (0-1)"
    )
    texture_size: int = Field(default=1024, gt=0, description="Texture resolution")


class ImageTo3DRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded image string and will be decoded to PILImage.Image",
    )
    seed: Optional[int] = Field(default=None, description="Random seed")
    simplify: float = Field(
        default=0.95, ge=0, le=1, description="Mesh simplification ratio (0-1)"
    )
    texture_size: int = Field(default=1024, gt=0, description="Texture resolution")
    preprocess_image: bool = Field(
        default=True, description="Whether to preprocess (remove background)"
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
            return PILImage.open(BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Image data is not a valid image format: {e}")


class GenerationResponse(BaseModel):
    status: str
    glb_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class TrellisServer(BaseFastAPIServer):
    def __init__(
        self,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        image_model_name: str = DEFAULT_IMAGE_MODEL,
        **kwargs,
    ):
        self._text_engine = TrellisTextEngine(text_model_name)
        self._image_engine = TrellisImageEngine(image_model_name)
        super().__init__(engines=[self._text_engine, self._image_engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/text-to-3d", response_model=GenerationResponse)
        async def text_to_3d(request: TextTo3DRequest):
            result = await self._text_engine.run_inference(
                text=request.text, seed=request.seed,
                simplify=request.simplify, texture_size=request.texture_size,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result

        @self._app.post("/image-to-3d", response_model=GenerationResponse)
        async def image_to_3d(request: ImageTo3DRequest):
            result = await self._image_engine.run_inference(
                image=request.image, seed=request.seed,
                simplify=request.simplify, texture_size=request.texture_size,
                preprocess_image=request.preprocess_image,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS FastAPI Server - Text/Image-to-3D Generation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=DEFAULT_TEXT_MODEL,
        help="HuggingFace model identifier for text-to-3D",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=DEFAULT_IMAGE_MODEL,
        help="HuggingFace model identifier for image-to-3D",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before unloading models",
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
            logging.FileHandler("trellis_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("TRELLIS Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Text Model: {args.text_model}")
    log.info(f"Image Model: {args.image_model}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = TrellisServer(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
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
