#!/usr/bin/env python3
"""
SDXL FastAPI Server - Text-to-Image Generation Service

A FastAPI-based server that wraps Stability AI's SDXL model via
HuggingFace diffusers for text-to-image generation.

Features:
- Automatic GPU selection (chooses GPU with most free memory)
- Lazy model loading (loads on first request)
- Idle timeout with automatic model unloading to free GPU memory
- Thread-safe inference serialization
"""

import argparse
import asyncio
import base64
import enum
import logging
import sys
import time
from contextlib import asynccontextmanager, suppress
from io import BytesIO
from typing import Any, Dict, Optional

import torch
import uvicorn
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sdxl_server.log"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models
# ===========================================================================
class TextToImageOptions(BaseModel):
    """Options for image generation."""

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


class TextToImageRequest(BaseModel):
    """Request body for text-to-image generation."""

    prompt: str = Field(
        ..., min_length=1, description="Text prompt for image generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    options: Optional[TextToImageOptions] = None


class ImageResponse(BaseModel):
    """Response for image generation."""

    status: str

    images: Optional[list[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    error: Optional[str] = None
    error_type: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str

    model_state: str

    gpu: str
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None

    idle_timeout: Optional[int] = None


# ===========================================================================
# Server Implementation
# ===========================================================================
# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8002
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes
DEFAULT_IDLE_CHECK_INTERVAL = 30  # seconds


# Enum for model states
class ModelState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class SDXLServer:
    """
    FastAPI-based server for SDXL text-to-image generation.

    Features automatic GPU selection, lazy model loading, idle-timeout
    unloading, and thread-safe serialized inference.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model_name: str = DEFAULT_MODEL,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        idle_check_interval: int = DEFAULT_IDLE_CHECK_INTERVAL,
    ):
        # SDXL model configuration
        self.model_name = model_name

        # Model state management
        self.pipeline: Optional[StableDiffusionXLPipeline] = None

        self._model_state: ModelState = ModelState.NOT_LOADED
        self._state_lock = asyncio.Lock()  # protects load/unload state transitions
        self._inference_lock = asyncio.Lock()  # serializes inference
        self._last_activity: float = time.monotonic()
        self._gpu_id: Optional[int] = None

        # Server configuration
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.idle_check_interval = idle_check_interval
        self._idle_monitor_task: Optional[asyncio.Task] = None
        assert self.idle_timeout >= 0, "Idle timeout must be non-negative"
        assert self.idle_check_interval > 0, "Idle check interval must be positive"

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Start idle monitor on startup, cleanup on shutdown."""
            # Auto-load pipeline on startup
            try:
                await self._ensure_model_loaded()
            except Exception as e:
                logger.warning(f"Startup model loading failed (will retry on first request): {e}")

            # Start idle monitor (no-op if idle_timeout == 0)
            if self.idle_timeout > 0:
                self._idle_monitor_task = asyncio.create_task(self._idle_monitor_loop())
            logger.info("Startup complete")

            try:
                yield
            finally:
                if self._idle_monitor_task:
                    self._idle_monitor_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._idle_monitor_task

                await asyncio.to_thread(self._unload_pipeline)
                self._model_state = ModelState.NOT_LOADED

                logger.info("Shutdown complete, pipeline unloaded")

        self._app = FastAPI(
            title="SDXL Image Generation Service",
            description="Generate images from text prompts using Stability AI SDXL model.",
            version="1.0.0",
            lifespan=lifespan,
        )
        self._register_routes()

    def _register_routes(self) -> None:
        """Register FastAPI routes."""

        @self._app.get("/health", response_model=HealthResponse)
        async def health():
            """Check server health and pipeline status."""
            if torch.cuda.is_available():
                gpu_names = ", ".join(
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                )
                total_alloc = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                ) / (1024**3)
                total_reserved = sum(
                    torch.cuda.memory_reserved(i)
                    for i in range(torch.cuda.device_count())
                ) / (1024**3)
            else:
                gpu_names = "N/A"
                total_alloc = None
                total_reserved = None

            return {
                "status": "ok"
                if self._model_state == ModelState.LOADED
                else "degraded",
                "model_state": self._model_state.value,
                "gpu": gpu_names,
                "gpu_memory_allocated_gb": round(total_alloc, 2)
                if total_alloc is not None
                else None,
                "gpu_memory_reserved_gb": round(total_reserved, 2)
                if total_reserved is not None
                else None,
                "idle_timeout": self.idle_timeout,
            }

        @self._app.post("/text-to-image", response_model=ImageResponse)
        async def text_to_image(request: TextToImageRequest):
            """Generate image(s) from a text prompt."""
            self._last_activity = time.monotonic()  # update activity timestamp immediately to prevent idle unload during generation
            await self._ensure_model_loaded()

            async with self._inference_lock:
                result = await asyncio.to_thread(
                    self._generate_image,
                    prompt=request.prompt,
                    seed=request.seed,
                    options=request.options,
                )

            self._last_activity = (
                time.monotonic()
            )  # update activity timestamp after generation completes

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        logger.info(f"Starting SDXL server on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    @staticmethod
    def _select_free_gpu() -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SDXL requires GPU acceleration.")
        best_gpu = 0
        best_free = 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_gpu = i
        logger.info(
            f"Selected GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)} "
            f"(free: {best_free / (1024**3):.1f}GB)"
        )
        return best_gpu

    # ----------------------------------------------------------------
    # Model Load / Unload
    # ----------------------------------------------------------------
    def _load_pipeline(self) -> None:
        """Load the SDXL pipeline onto the selected GPU."""
        logger.info(f"Loading SDXL pipeline: {self.model_name}")
        start_time = time.time()

        try:
            self._gpu_id = self._select_free_gpu()

            logger.info(
                f"Initializing StableDiffusionXLPipeline (model={self.model_name})"
            )
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(f"cuda:{self._gpu_id}")

            load_time = time.time() - start_time
            logger.info(f"SDXL pipeline loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load SDXL pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}") from e

    def _unload_pipeline(self) -> None:
        """Unload the pipeline and release GPU memory."""
        if self.pipeline is None:
            return

        logger.info("[model] Unloading...")
        gpu_id = self._gpu_id
        self._gpu_id = None

        if gpu_id is not None:
            mem_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        del self.pipeline
        self.pipeline = None
        torch.cuda.empty_cache()

        if gpu_id is not None:
            mem_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(
                f"[model] Unloaded from GPU {gpu_id} "
                f"(freed {mem_before - mem_after:.2f}GB, GPU now: {mem_after:.2f}GB)"
            )

    # ----------------------------------------------------------------
    # Model State Management & Idle Monitor
    # ----------------------------------------------------------------
    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded, loading on demand if needed (double-check locking)."""
        async with self._state_lock:
            if self._model_state == ModelState.LOADED:
                return

            logger.info(f"[model] {self._model_state.value} -> LOADING")
            self._model_state = ModelState.LOADING
            try:
                await asyncio.to_thread(self._load_pipeline)
                self._model_state = ModelState.LOADED
                logger.info("[model] LOADING -> LOADED")
            except Exception as e:
                self._model_state = ModelState.NOT_LOADED
                logger.error(f"[model] Failed to load: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load SDXL pipeline: {e}",
                )

    async def _idle_monitor_loop(self) -> None:
        """Background task that unloads the model after idle timeout."""
        logger.info(
            f"Idle monitor started (timeout={self.idle_timeout}s, "
            f"check_interval={self.idle_check_interval}s)"
        )
        while True:
            try:
                await asyncio.sleep(self.idle_check_interval)
                now = time.monotonic()

                if self._model_state == ModelState.LOADED:
                    elapsed = now - self._last_activity
                    if elapsed > self.idle_timeout:
                        async with self._state_lock:
                            can_unload = (
                                self._model_state == ModelState.LOADED
                            ) and not self._inference_lock.locked()

                            if can_unload:
                                logger.info(
                                    f"[model] Idle timeout ({elapsed:.0f}s), unloading"
                                )
                                self._model_state = ModelState.UNLOADING
                                await asyncio.to_thread(self._unload_pipeline)
                                self._model_state = ModelState.NOT_LOADED

            except asyncio.CancelledError:
                logger.info("Idle monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Idle monitor error: {e}")

    # ----------------------------------------------------------------
    # Image Generation (Inference)
    # ----------------------------------------------------------------
    def _generate_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        options: Optional[TextToImageOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate images from text prompt using SDXL.

        Returns a dict matching ImageResponse fields.
        """
        options = options or TextToImageOptions()

        logger.info(
            f"Generating image: '{prompt[:80]}...' (seed={seed}, "
            f"steps={options.num_inference_steps}, guidance={options.guidance_scale}, "
            f"size={options.width}x{options.height})"
        )

        start_time = time.time()

        try:
            generator = (
                torch.Generator(f"cuda:{self._gpu_id}").manual_seed(seed) if seed is not None else None
            )

            output = self.pipeline(
                prompt=prompt,
                negative_prompt=options.negative_prompt,
                height=options.height,
                width=options.width,
                num_inference_steps=options.num_inference_steps,
                guidance_scale=options.guidance_scale,
                num_images_per_prompt=options.num_images_per_prompt,
                generator=generator,
            )

            # Encode images to base64 PNG
            images_b64 = []
            for img in output.images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                images_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            generation_time = time.time() - start_time
            metadata = {
                "prompt": prompt,
                "negative_prompt": options.negative_prompt,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "num_images": len(images_b64),
                "height": options.height,
                "width": options.width,
                "num_inference_steps": options.num_inference_steps,
                "guidance_scale": options.guidance_scale,
            }

            logger.info(
                f"Generation complete in {generation_time:.2f}s, "
                f"{len(images_b64)} image(s) generated"
            )

            return {
                "status": "success",
                "images": images_b64,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


def main():
    """
    Main entry point for the SDXL FastAPI server.
    """
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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model identifier (e.g. stabilityai/stable-diffusion-xl-base-1.0)",
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
    logger.setLevel(getattr(logging, args.log_level))

    # Print configuration
    logger.info("=" * 60)
    logger.info("SDXL FastAPI Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Idle Timeout: {args.idle_timeout}s")
    logger.info(f"Idle Check Interval: {args.idle_check_interval}s")
    logger.info("=" * 60)

    server = SDXLServer(
        host=args.host,
        port=args.port,
        model_name=args.model,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
