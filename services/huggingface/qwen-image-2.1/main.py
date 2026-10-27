#!/usr/bin/env python3
"""
Qwen-Image-2.1 FastAPI Server - Text-to-Image & Image Editing Service

A FastAPI-based server that wraps Alibaba Qwen's Qwen-Image-2.1 model via
HuggingFace diffusers for text-to-image generation and multi-reference
image editing (a single pipeline handles both).

Key features:
- 7B single-stream block-causal DiT + Qwen3-VL text encoder that jointly
  encodes the prompt and the condition images
- CPU offload via enable_model_cpu_offload() to keep peak VRAM low
- Native 2048x2048 generation and RGBA transparency output
- Multi-reference image editing with up to 10 condition images
- True CFG opt-in via true_cfg_scale > 1 + negative_prompt (default off)
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
from diffusers import QwenImage21Pipeline
from fastapi import HTTPException
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import (
    BaseFastAPIServer,
    DEFAULT_IDLE_CHECK_INTERVAL,
    DEFAULT_IDLE_TIMEOUT,
    ModelEngine,
    select_free_gpu,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_MODEL = "Qwen/Qwen-Image-2.1"
# Qwen-Image-2.1 accepts up to 10 condition images in one editing request
MAX_REFERENCE_IMAGES = 10


def _decode_base64_image(encoded: str) -> Image.Image:
    """Decode a base64 PNG/JPEG payload into an RGB PIL image.

    Args:
        encoded: Base64 string of the image file bytes. Accepts the
            optional ``data:...;base64,`` data-URL prefix.

    Returns:
        Decoded image converted to RGB mode.

    Raises:
        ValueError: If the payload is not valid base64 or not a
            decodable image.
    """
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        # binascii.Error subclasses ValueError, so this also covers it.
        raise ValueError("image payload is not valid base64") from error
    try:
        with Image.open(BytesIO(raw)) as loaded:
            return loaded.convert("RGB")
    except Exception as error:
        raise ValueError("image payload is not a decodable image") from error


class QwenImage21Engine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__("model")
        self.model_name = model_name
        self.pipeline: Optional[QwenImage21Pipeline] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        logger.info(
            f"Loading Qwen-Image-2.1 from {self.model_name} on cuda:{self.gpu_id} ..."
        )
        self.pipeline = QwenImage21Pipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        # CPU offload: the full weights (~35GB bf16: 7B DiT + Qwen3-VL
        # encoder + VAE) stay in host memory; one component at a time moves
        # to the GPU, so peak VRAM ~= largest single component (~16GB).
        self.pipeline.enable_model_cpu_offload(gpu_id=self.gpu_id)
        # Tiled VAE decode: without it, decoding a 2048x2048 image needs a
        # ~4.5GB activation spike on top of the resident DiT and OOMs on a
        # 24GB GPU. Tiling only affects the final decode step.
        self.pipeline.vae.enable_tiling()
        logger.info("Pipeline loaded with model_cpu_offload + vae tiling enabled")

    def _unload_impl(self) -> None:
        if self.pipeline is None:
            return
        del self.pipeline
        self.pipeline = None

    def _run_inference_impl(
        self,
        prompt: str,
        seed: Optional[int],
        height: Optional[int],
        width: Optional[int],
        num_inference_steps: int,
        true_cfg_scale: float,
        negative_prompt: Optional[str],
        num_images_per_prompt: int,
        images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        mode = "image-editing" if images else "text-to-image"
        logger.info(
            f"Generating image ({mode}): '{prompt[:80]}' (seed={seed}, "
            f"steps={num_inference_steps}, true_cfg={true_cfg_scale}, "
            f"size={width}x{height}, refs={len(images) if images else 0})"
        )
        start_time = time.monotonic()

        try:
            generator = (
                torch.Generator(f"cuda:{self.gpu_id}").manual_seed(seed)
                if seed is not None
                else None
            )

            # Qwen-Image-2.1 has no guidance_scale; true CFG is enabled by
            # true_cfg_scale > 1 together with a negative prompt string
            # (an empty string is sufficient per the official model card).
            effective_negative_prompt = negative_prompt
            if true_cfg_scale > 1 and negative_prompt is None:
                effective_negative_prompt = ""

            output = self.pipeline(
                prompt=prompt,
                image=images,  # None for t2i; list of PIL images for editing
                negative_prompt=effective_negative_prompt,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

            images_b64 = []
            for img in output.images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")  # keeps RGBA when present
                images_b64.append(
                    base64.b64encode(buffer.getvalue()).decode("utf-8")
                )

            generation_time = time.monotonic() - start_time
            logger.info(
                f"Generation complete in {generation_time:.2f}s, "
                f"{len(images_b64)} image(s) generated"
            )

            return {
                "status": "success",
                "images": images_b64,
                "metadata": {
                    "mode": mode,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "num_images": len(images_b64),
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "true_cfg_scale": true_cfg_scale,
                    "num_reference_images": len(images) if images else 0,
                },
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class TextToImageRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, description="Text prompt for image generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    height: int = Field(
        default=2048,
        gt=0,
        le=4096,
        description="Image height in pixels (native resolution is 2048)",
    )
    width: int = Field(
        default=2048,
        gt=0,
        le=4096,
        description="Image width in pixels (native resolution is 2048)",
    )
    num_inference_steps: int = Field(
        default=40,
        gt=0,
        le=100,
        description="Number of denoising steps (Qwen recommends 40)",
    )
    true_cfg_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description=(
            "True CFG scale; >1 enables classifier-free guidance (doubles "
            "per-step compute) and requires a negative prompt"
        ),
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Negative prompt, only used when true_cfg_scale > 1 "
            "(an empty string is sent if omitted)"
        ),
    )
    num_images_per_prompt: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate"
    )


class ImageEditingRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Editing instruction, encoded together with the reference images",
    )
    images: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_REFERENCE_IMAGES,
        description="Base64-encoded reference images (PNG/JPEG), 1-10 images",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    height: Optional[int] = Field(
        default=None,
        gt=0,
        le=4096,
        description=(
            "Output height; derived from the first reference image's "
            "aspect ratio when omitted"
        ),
    )
    width: Optional[int] = Field(
        default=None,
        gt=0,
        le=4096,
        description=(
            "Output width; derived from the first reference image's "
            "aspect ratio when omitted"
        ),
    )
    num_inference_steps: int = Field(
        default=40, gt=0, le=100, description="Number of denoising steps"
    )
    true_cfg_scale: float = Field(
        default=1.0, ge=1.0, le=20.0, description="True CFG scale (>1 enables CFG)"
    )
    negative_prompt: Optional[str] = Field(
        default=None, description="Negative prompt (used when true_cfg_scale > 1)"
    )
    num_images_per_prompt: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate"
    )


class ImageResponse(BaseModel):
    status: str
    images: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class QwenImage21Server(BaseFastAPIServer):
    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self._engine = QwenImage21Engine(model_name)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/text-to-image", response_model=ImageResponse)
        async def text_to_image(request: TextToImageRequest):
            result = await self._engine.run_inference(
                prompt=request.prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                true_cfg_scale=request.true_cfg_scale,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result

        @self._app.post("/image-editing", response_model=ImageResponse)
        async def image_editing(request: ImageEditingRequest):
            try:
                images = [_decode_base64_image(img) for img in request.images]
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error))

            result = await self._engine.run_inference(
                prompt=request.prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                true_cfg_scale=request.true_cfg_scale,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
                images=images,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8010


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen-Image-2.1 FastAPI Server - Text-to-Image & Image Editing Service"
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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "HuggingFace model identifier "
            "(e.g. Qwen/Qwen-Image-2.1-Turbo for the 8-step distilled variant)"
        ),
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
                os.path.join(os.path.dirname(__file__), "qwen_image_server.log")
            ),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("Qwen-Image-2.1 Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Model: {args.model}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = QwenImage21Server(
        model_name=args.model,
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
