#!/usr/bin/env python3
"""
SDXL FastAPI Server - Text-to-Image Generation Service

A FastAPI-based server that wraps Stability AI's SDXL model via
HuggingFace diffusers for text-to-image generation.
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
from diffusers import StableDiffusionXLPipeline
from fastapi import HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class SDXLEngine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__("model")
        self.model_name = model_name
        self.pipeline: Optional[StableDiffusionXLPipeline] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        logger.info(
            f"Loading SDXL from {self.model_name} on cuda:{self.gpu_id} ..."
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipeline.enable_model_cpu_offload(gpu_id=self.gpu_id)

    def _unload_impl(self) -> None:
        if self.pipeline is None:
            return
        del self.pipeline
        self.pipeline = None

    def _run_inference_impl(
        self,
        prompt: str,
        seed: Optional[int],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        num_images_per_prompt: int,
    ) -> Dict[str, Any]:
        logger.info(
            f"Generating image: '{prompt[:80]}' (seed={seed}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}, "
            f"size={width}x{height})"
        )
        start_time = time.monotonic()

        try:
            generator = (
                torch.Generator(f"cuda:{self.gpu_id}").manual_seed(seed)
                if seed is not None
                else None
            )

            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

            images_b64 = []
            for img in output.images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                images_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            generation_time = time.monotonic() - start_time
            logger.info(
                f"Generation complete in {generation_time:.2f}s, "
                f"{len(images_b64)} image(s) generated"
            )

            return {
                "status": "success",
                "images": images_b64,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "num_images": len(images_b64),
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                },
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


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
        default=1024, gt=0, le=2048, description="Image height in pixels"
    )
    width: int = Field(default=1024, gt=0, le=2048, description="Image width in pixels")
    num_inference_steps: int = Field(
        default=40, gt=0, le=100, description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=5.0, ge=0, le=20, description="Guidance scale (CFG)"
    )
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt")
    num_images_per_prompt: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate"
    )


class ImageResponse(BaseModel):
    status: str
    images: Optional[list[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class SDXLServer(BaseFastAPIServer):
    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self._engine = SDXLEngine(model_name)
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
                guidance_scale=request.guidance_scale,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
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
        description="SDXL FastAPI Server - Text-to-Image Generation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model identifier"
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
            logging.FileHandler("sdxl_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("SDXL Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Model: {args.model}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = SDXLServer(
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
