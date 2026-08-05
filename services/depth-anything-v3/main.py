#!/usr/bin/env python3
"""Depth Anything 3 FastAPI server for depth and camera estimation."""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from binascii import Error as BinasciiError
from io import BytesIO
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu
from serialization import NDArrayData

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "depth-anything/DA3-LARGE-1.1"
DEFAULT_PROCESS_RES = 504
MAX_IMAGES = 32
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000


def _decode_image(encoded: str) -> PILImage.Image:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("each image must be a non-empty base64 string")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]

    max_encoded_length = ((MAX_IMAGE_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded_length:
        raise ValueError(
            f"encoded image exceeds the {MAX_IMAGE_BYTES}-byte decoded limit"
        )

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (BinasciiError, ValueError) as exc:
        raise ValueError("image must be valid base64") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"decoded image exceeds the {MAX_IMAGE_BYTES}-byte limit"
        )

    try:
        image = PILImage.open(BytesIO(raw))
    except Exception as exc:
        raise ValueError("decoded data is not a supported image") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image exceeds the {MAX_IMAGE_PIXELS}-pixel limit"
        )
    try:
        image.load()
    except Exception as exc:
        raise ValueError("decoded data is not a supported image") from exc
    return image.convert("RGB")


def _decode_images(encoded_images: list[str]) -> list[PILImage.Image]:
    return [_decode_image(encoded) for encoded in encoded_images]


class DepthRequest(BaseModel):
    images: list[str] = Field(
        min_length=1,
        max_length=MAX_IMAGES,
        description="Base64-encoded PNG/JPEG images",
    )
    extrinsics: Optional[NDArrayData] = Field(
        default=None,
        description="Optional float array with shape (N, 4, 4)",
    )
    intrinsics: Optional[NDArrayData] = Field(
        default=None,
        description="Optional float array with shape (N, 3, 3)",
    )
    process_res: int = Field(default=DEFAULT_PROCESS_RES, ge=128, le=2048)
    process_res_method: Literal[
        "upper_bound_resize", "lower_bound_resize"
    ] = "upper_bound_resize"

    @model_validator(mode="after")
    def validate_camera_payloads(self) -> "DepthRequest":
        if (self.extrinsics is None) != (self.intrinsics is None):
            raise ValueError(
                "extrinsics and intrinsics must be provided together"
            )
        if self.extrinsics is None:
            return self

        num_images = len(self.images)
        if self.extrinsics.shape != [num_images, 4, 4]:
            raise ValueError(
                "extrinsics must have shape "
                f"({num_images}, 4, 4), got {tuple(self.extrinsics.shape)}"
            )
        if self.intrinsics.shape != [num_images, 3, 3]:
            raise ValueError(
                "intrinsics must have shape "
                f"({num_images}, 3, 3), got {tuple(self.intrinsics.shape)}"
            )

        for name, payload in (
            ("extrinsics", self.extrinsics),
            ("intrinsics", self.intrinsics),
        ):
            if not np.issubdtype(np.dtype(payload.dtype), np.floating):
                raise ValueError(f"{name} must use a floating-point dtype")
        return self


class DepthResponse(BaseModel):
    status: str
    depth: Optional[NDArrayData] = None
    confidence: Optional[NDArrayData] = None
    extrinsics: Optional[NDArrayData] = None
    intrinsics: Optional[NDArrayData] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class DepthAnythingV3Engine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        super().__init__("model")
        self.model_name = model_name
        self.model: Optional[DepthAnything3] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        logger.info(
            "Loading %s on cuda:%s", self.model_name, self.gpu_id
        )
        self.model = DepthAnything3.from_pretrained(self.model_name)
        self.model = self.model.to(device=f"cuda:{self.gpu_id}")
        self.model.eval()

    def _unload_impl(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def _run_inference_impl(
        self,
        images: list[PILImage.Image],
        extrinsics: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        process_res: int,
        process_res_method: str,
    ) -> Dict[str, Any]:
        if self.model is None or self.gpu_id is None:
            raise RuntimeError("model is not loaded")

        torch.cuda.set_device(self.gpu_id)
        start_time = time.time()
        prediction = self.model.inference(
            image=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        generation_time = time.time() - start_time

        depth = np.asarray(prediction.depth, dtype=np.float32)
        confidence = (
            np.asarray(prediction.conf, dtype=np.float32)
            if prediction.conf is not None
            else None
        )
        predicted_extrinsics = (
            np.asarray(prediction.extrinsics, dtype=np.float32)
            if prediction.extrinsics is not None
            else None
        )
        predicted_intrinsics = (
            np.asarray(prediction.intrinsics, dtype=np.float32)
            if prediction.intrinsics is not None
            else None
        )

        logger.info(
            "Depth estimation complete in %.2fs for %d image(s)",
            generation_time,
            len(images),
        )
        return {
            "status": "success",
            "depth": NDArrayData.from_array(depth),
            "confidence": NDArrayData.from_array(confidence)
            if confidence is not None
            else None,
            "extrinsics": NDArrayData.from_array(predicted_extrinsics)
            if predicted_extrinsics is not None
            else None,
            "intrinsics": NDArrayData.from_array(predicted_intrinsics)
            if predicted_intrinsics is not None
            else None,
            "metadata": {
                "model_name": self.model_name,
                "num_images": len(images),
                "process_res": process_res,
                "process_res_method": process_res_method,
                "is_metric": bool(prediction.is_metric),
                "scale_factor": float(prediction.scale_factor)
                if prediction.scale_factor is not None
                else None,
                "generation_time": round(generation_time, 2),
            },
        }


class DepthAnythingV3Server(BaseFastAPIServer):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, **kwargs):
        self._engine = DepthAnythingV3Engine(model_name)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/estimate-depth", response_model=DepthResponse)
        async def estimate_depth(request: DepthRequest):
            try:
                images = _decode_images(request.images)
                extrinsics = (
                    request.extrinsics.to_array().astype(np.float32, copy=False)
                    if request.extrinsics is not None
                    else None
                )
                intrinsics = (
                    request.intrinsics.to_array().astype(np.float32, copy=False)
                    if request.intrinsics is not None
                    else None
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if extrinsics is not None and not np.all(np.isfinite(extrinsics)):
                raise HTTPException(
                    status_code=400,
                    detail="extrinsics contains NaN or Inf",
                )
            if intrinsics is not None and not np.all(np.isfinite(intrinsics)):
                raise HTTPException(
                    status_code=400,
                    detail="intrinsics contains NaN or Inf",
                )

            result = await self._engine.run_inference(
                images=images,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                process_res=request.process_res,
                process_res_method=request.process_res_method,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500),
                    detail=result,
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8006
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 FastAPI Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model repository or local model directory",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before unloading the model",
    )
    parser.add_argument(
        "--idle-check-interval",
        type=int,
        default=DEFAULT_IDLE_CHECK_INTERVAL,
        help="Seconds between idle checks",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("depth_anything_v3_server.log"),
        ],
    )

    logger.info("Model: %s", args.model_name)
    logger.info("Host: %s", args.host)
    logger.info("Port: %s", args.port)
    logger.info("Idle timeout: %ss", args.idle_timeout)

    server = DepthAnythingV3Server(
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
    )
    server.start()


if __name__ == "__main__":
    main()
