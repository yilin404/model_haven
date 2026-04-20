#!/usr/bin/env python3
"""
TRELLIS Server - Text/Image-to-3D Generation Service

A FastAPI-based server that wraps Microsoft's TRELLIS pipelines for
generating 3D models from text prompts or images.
"""

import argparse
import asyncio
import base64
import enum
import logging
import os
import sys
import time
from io import BytesIO
from contextlib import asynccontextmanager, suppress

import torch

from PIL import Image as PILImage

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from typing import Any, Dict, Optional

# Add TRELLIS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/trellis"))

from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils

# Set required environment variables for TRELLIS
os.environ["SPCONV_ALGO"] = "native"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trellis_server.log"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models
# ===========================================================================
class TextTo3DOptions(BaseModel):
    """Options for text-to-3D generation."""

    simplify: float = Field(
        default=0.95, ge=0, le=1, description="Mesh simplification ratio (0-1)"
    )
    texture_size: int = Field(default=1024, gt=0, description="Texture resolution")


class ImageTo3DOptions(BaseModel):
    """Options for image-to-3D generation."""

    simplify: float = Field(
        default=0.95, ge=0, le=1, description="Mesh simplification ratio (0-1)"
    )
    texture_size: int = Field(default=1024, gt=0, description="Texture resolution")
    preprocess_image: bool = Field(
        default=True, description="Whether to preprocess (remove background)"
    )


class TextTo3DRequest(BaseModel):
    """Request body for text-to-3D generation."""

    text: str = Field(..., min_length=1, description="Text prompt for 3D generation")

    seed: Optional[int] = Field(default=None, description="Random seed")
    options: Optional[TextTo3DOptions] = None


class ImageTo3DRequest(BaseModel):
    """Request body for image-to-3D generation."""

    image: str = Field(
        ..., description="Base64-encoded image string"
    )

    seed: Optional[int] = Field(default=None, description="Random seed")
    options: Optional[ImageTo3DOptions] = None

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, v):
        if not isinstance(v, str):
            raise ValueError("Image must be a base64-encoded string")
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Image must be valid base64-encoded data: {e}")

        try:
            return PILImage.open(BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Image data is not a valid image format: {e}")


class GenerationResponse(BaseModel):
    """Response for 3D generation endpoints."""

    status: str

    glb_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    error: Optional[str] = None
    error_type: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str

    text_pipeline_state: str
    image_pipeline_state: str

    gpu: str
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None

    idle_timeout: Optional[int] = None


# ===========================================================================
# Server Implementation
# ===========================================================================
# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_TEXT_MODEL = "microsoft/TRELLIS-text-xlarge"
DEFAULT_IMAGE_MODEL = "microsoft/TRELLIS-image-large"
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes
DEFAULT_IDLE_CHECK_INTERVAL = 30  # seconds


# Enum for pipeline states
class PipelineState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class TrellisServer:
    """
    FastAPI-based server for TRELLIS text/image-to-3D generation.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        image_model_name: str = DEFAULT_IMAGE_MODEL,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        idle_check_interval: int = DEFAULT_IDLE_CHECK_INTERVAL,
    ):
        # Text Pipeline (and its state management)
        self.text_model_name = text_model_name
        self.text_pipeline: Optional[TrellisTextTo3DPipeline] = None

        self._text_pipeline_state: PipelineState = PipelineState.NOT_LOADED
        self._text_state_lock = asyncio.Lock()  # protects load/unload state transitions
        self._text_inference_lock = asyncio.Lock()  # serializes inference
        self._text_last_activity: float = time.monotonic()
        self._text_gpu_id: Optional[int] = None

        # Image Pipeline (and its state management)
        self.image_model_name = image_model_name
        self.image_pipeline: Optional[TrellisImageTo3DPipeline] = None

        self._image_pipeline_state: PipelineState = PipelineState.NOT_LOADED
        self._image_state_lock = (
            asyncio.Lock()
        )  # protects load/unload state transitions
        self._image_inference_lock = asyncio.Lock()  # serializes inference
        self._image_last_activity: float = time.monotonic()
        self._image_gpu_id: Optional[int] = None

        # Server configuration
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.idle_check_interval = idle_check_interval
        self._idle_monitor_task: Optional[asyncio.Task] = None
        assert self.idle_timeout > 0, "Idle timeout must be positive"
        assert self.idle_check_interval > 0, "Idle check interval must be positive"

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Load pipelines on startup and start idle monitor."""
            # Start idle monitor
            self._idle_monitor_task = asyncio.create_task(self._idle_monitor_loop())
            logger.info("Startup complete (lazy model loading enabled)")

            try:
                yield  # App is running...
            finally:
                # Shutdown cleanup
                if self._idle_monitor_task:
                    self._idle_monitor_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._idle_monitor_task

                await asyncio.to_thread(self._unload_text_pipeline)
                self._text_pipeline_state = PipelineState.NOT_LOADED
                await asyncio.to_thread(self._unload_image_pipeline)
                self._image_pipeline_state = PipelineState.NOT_LOADED

                logger.info("Shutdown complete, pipelines unloaded")

        self._app = FastAPI(
            title="TRELLIS 3D Generation Service",
            description="Generate 3D models from text prompts or images using Microsoft TRELLIS.",
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

            any_loaded = (
                self._text_pipeline_state == PipelineState.LOADED
                or self._image_pipeline_state == PipelineState.LOADED
            )
            return {
                "status": "ok" if any_loaded else "degraded",
                "text_pipeline_state": self._text_pipeline_state.value,
                "image_pipeline_state": self._image_pipeline_state.value,
                "gpu": gpu_names,
                "gpu_memory_allocated_gb": round(total_alloc, 2)
                if total_alloc is not None
                else None,
                "gpu_memory_reserved_gb": round(total_reserved, 2)
                if total_reserved is not None
                else None,
                "idle_timeout": self.idle_timeout,
            }

        @self._app.post("/text-to-3d", response_model=GenerationResponse)
        async def text_to_3d(request: TextTo3DRequest):
            """Generate a 3D model from a text prompt."""
            self._text_last_activity = time.monotonic()
            await self._ensure_text_pipeline_loaded()

            async with self._text_inference_lock:
                result = await asyncio.to_thread(
                    self._generate_text_to_3d,
                    text=request.text,
                    seed=request.seed,
                    options=request.options,
                )

            self._text_last_activity = time.monotonic()

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

        @self._app.post("/image-to-3d", response_model=GenerationResponse)
        async def image_to_3d(request: ImageTo3DRequest):
            """Generate a 3D model from an image."""
            self._image_last_activity = time.monotonic()
            await self._ensure_image_pipeline_loaded()

            # image is already decoded to PIL.Image by field_validator
            image = request.image

            async with self._image_inference_lock:
                result = await asyncio.to_thread(
                    self._generate_image_to_3d,
                    image=image,
                    seed=request.seed,
                    options=request.options,
                )

            self._image_last_activity = time.monotonic()

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        logger.info(f"Starting TRELLIS server on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    @staticmethod
    def _select_free_gpu() -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. TRELLIS requires GPU acceleration."
            )
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
    # Pipeline Load / Unload
    # ----------------------------------------------------------------
    def _load_text_pipeline(self) -> None:
        """Load the TRELLIS text-to-3D pipeline."""
        logger.info(f"Loading text-to-3D pipeline: {self.text_model_name}")
        start_time = time.time()

        try:
            self._text_gpu_id = self._select_free_gpu()
            self.text_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                self.text_model_name
            )
            self.text_pipeline.to(f"cuda:{self._text_gpu_id}")

            logger.info(f"Text-to-3D pipeline loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            self._text_gpu_id = None
            logger.error(f"Failed to load text pipeline: {e}")
            raise RuntimeError(f"Text pipeline loading failed: {e}") from e

    def _load_image_pipeline(self) -> None:
        """Load the TRELLIS image-to-3D pipeline."""
        logger.info(f"Loading image-to-3D pipeline: {self.image_model_name}")
        start_time = time.time()

        try:
            self._image_gpu_id = self._select_free_gpu()
            self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                self.image_model_name
            )
            self.image_pipeline.to(f"cuda:{self._image_gpu_id}")

            logger.info(f"Image-to-3D pipeline loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            self._image_gpu_id = None
            logger.error(f"Failed to load image pipeline: {e}")
            raise RuntimeError(f"Image pipeline loading failed: {e}") from e

    def _unload_text_pipeline(self) -> None:
        """Unload the text pipeline and release GPU memory."""
        if self.text_pipeline is None:
            return

        logger.info("[text-pipeline] Unloading...")
        gpu_id = self._text_gpu_id
        self._text_gpu_id = None

        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            mem_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        del self.text_pipeline
        self.text_pipeline = None
        torch.cuda.empty_cache()

        if gpu_id is not None:
            mem_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(
                f"[text-pipeline] Unloaded from GPU {gpu_id} "
                f"(freed {mem_before - mem_after:.2f}GB, GPU now: {mem_after:.2f}GB)"
            )

    def _unload_image_pipeline(self) -> None:
        """Unload the image pipeline and release GPU memory."""
        if self.image_pipeline is None:
            return

        logger.info("[image-pipeline] Unloading...")
        gpu_id = self._image_gpu_id
        self._image_gpu_id = None

        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            mem_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        del self.image_pipeline
        self.image_pipeline = None
        torch.cuda.empty_cache()

        if gpu_id is not None:
            mem_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logger.info(
                f"[image-pipeline] Unloaded from GPU {gpu_id} "
                f"(freed {mem_before - mem_after:.2f}GB, GPU now: {mem_after:.2f}GB)"
            )

    # ----------------------------------------------------------------
    # Pipeline State Management & Idle Monitor
    # ----------------------------------------------------------------
    async def _ensure_text_pipeline_loaded(self) -> None:
        """Ensure the text pipeline is loaded, loading on demand if needed."""
        if self._text_pipeline_state == PipelineState.LOADED:
            return

        async with self._text_state_lock:
            if self._text_pipeline_state == PipelineState.LOADED:
                return
            logger.info(f"[text-pipeline] {self._text_pipeline_state.value} -> LOADING")
            self._text_pipeline_state = PipelineState.LOADING
            try:
                await asyncio.to_thread(self._load_text_pipeline)
                self._text_pipeline_state = PipelineState.LOADED
                logger.info("[text-pipeline] LOADING -> LOADED")
            except Exception as e:
                self._text_pipeline_state = PipelineState.NOT_LOADED
                logger.error(f"[text-pipeline] Failed to load: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load text pipeline: {e}",
                )

    async def _ensure_image_pipeline_loaded(self) -> None:
        """Ensure the image pipeline is loaded, loading on demand if needed."""
        if self._image_pipeline_state == PipelineState.LOADED:
            return

        async with self._image_state_lock:
            if self._image_pipeline_state == PipelineState.LOADED:
                return
            logger.info(
                f"[image-pipeline] {self._image_pipeline_state.value} -> LOADING"
            )
            self._image_pipeline_state = PipelineState.LOADING
            try:
                await asyncio.to_thread(self._load_image_pipeline)
                self._image_pipeline_state = PipelineState.LOADED
                logger.info("[image-pipeline] LOADING -> LOADED")
            except Exception as e:
                self._image_pipeline_state = PipelineState.NOT_LOADED
                logger.error(f"[image-pipeline] Failed to load: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load image pipeline: {e}",
                )

    async def _idle_monitor_loop(self) -> None:
        """Background task that unloads pipelines independently after idle timeout."""
        logger.info(
            f"Idle monitor started (timeout={self.idle_timeout}s, "
            f"check_interval={self.idle_check_interval}s)"
        )
        while True:
            try:
                await asyncio.sleep(self.idle_check_interval)
                now = time.monotonic()

                if self._text_pipeline_state == PipelineState.LOADED:
                    elapsed = now - self._text_last_activity
                    if elapsed > self.idle_timeout:
                        async with self._text_state_lock:
                            if self._text_pipeline_state == PipelineState.LOADED:
                                if self._text_inference_lock.locked():
                                    continue

                                logger.info(
                                    f"[text-pipeline] Idle timeout ({elapsed:.0f}s), unloading"
                                )
                                self._text_pipeline_state = PipelineState.UNLOADING
                                await asyncio.to_thread(self._unload_text_pipeline)
                                self._text_pipeline_state = PipelineState.NOT_LOADED

                if self._image_pipeline_state == PipelineState.LOADED:
                    elapsed = now - self._image_last_activity
                    if elapsed > self.idle_timeout:
                        async with self._image_state_lock:
                            if self._image_pipeline_state == PipelineState.LOADED:
                                if self._image_inference_lock.locked():
                                    continue

                                logger.info(
                                    f"[image-pipeline] Idle timeout ({elapsed:.0f}s), unloading"
                                )
                                self._image_pipeline_state = PipelineState.UNLOADING
                                await asyncio.to_thread(self._unload_image_pipeline)
                                self._image_pipeline_state = PipelineState.NOT_LOADED

            except asyncio.CancelledError:
                logger.info("Idle monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Idle monitor error: {e}")

    # ----------------------------------------------------------------
    # 3D Generation (Inference)
    # ----------------------------------------------------------------
    def _generate_text_to_3d(
        self,
        text: str,
        seed: Optional[int] = None,
        options: Optional[TextTo3DOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate a 3D model from text prompt using TRELLIS.

        Returns a dict matching GenerationResponse fields.
        """
        seed = seed if seed is not None else torch.randint(0, 2**32, (1,)).item()
        options = options or TextTo3DOptions()

        logger.info(
            f"Generating 3D from text: '{text}' (seed={seed}, "
            f"simplify={options.simplify}, texture_size={options.texture_size})"
        )

        start_time = time.time()

        try:
            outputs = self.text_pipeline.run(
                text,
                seed=seed,
                formats=["mesh", "gaussian"],
            )

            gaussian = outputs["gaussian"][0]
            mesh = outputs["mesh"][0]

            glb_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=options.simplify,
                texture_size=options.texture_size,
                fill_holes=True,
                verbose=False,
            )

            buffer = BytesIO()
            glb_mesh.export(file_obj=buffer, file_type="glb")
            glb_bytes = buffer.getvalue()

            generation_time = time.time() - start_time
            metadata = {
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "file_size": len(glb_bytes),
                "simplify": options.simplify,
                "texture_size": options.texture_size,
            }

            logger.info(
                f"Generation complete in {generation_time:.2f}s, "
                f"file size: {len(glb_bytes)} bytes"
            )

            return {
                "status": "success",
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8"),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Text-to-3D generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _generate_image_to_3d(
        self,
        image: PILImage.Image,
        seed: Optional[int] = None,
        options: Optional[ImageTo3DOptions] = None,
    ) -> Dict[str, Any]:
        """
        Generate a 3D model from an image using TRELLIS.

        Returns a dict matching GenerationResponse fields.
        """
        seed = seed if seed is not None else torch.randint(0, 2**32, (1,)).item()
        options = options or ImageTo3DOptions()

        logger.info(
            f"Generating 3D from image (seed={seed}, "
            f"simplify={options.simplify}, texture_size={options.texture_size})"
        )

        start_time = time.time()

        try:
            outputs = self.image_pipeline.run(
                image,
                seed=seed,
                formats=["mesh", "gaussian"],
                preprocess_image=options.preprocess_image,
            )

            gaussian = outputs["gaussian"][0]
            mesh = outputs["mesh"][0]

            glb_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=options.simplify,
                texture_size=options.texture_size,
                fill_holes=True,
                verbose=False,
            )

            buffer = BytesIO()
            glb_mesh.export(file_obj=buffer, file_type="glb")
            glb_bytes = buffer.getvalue()

            generation_time = time.time() - start_time
            metadata = {
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "file_size": len(glb_bytes),
                "simplify": options.simplify,
                "texture_size": options.texture_size,
                "preprocess_image": options.preprocess_image,
            }

            logger.info(
                f"Image-to-3D complete in {generation_time:.2f}s, "
                f"file size: {len(glb_bytes)} bytes"
            )

            return {
                "status": "success",
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8"),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Image-to-3D generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


def main():
    """
    Main entry point for the TRELLIS FastAPI server.
    """
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
    logger.setLevel(getattr(logging, args.log_level))

    # Print configuration
    logger.info("=" * 60)
    logger.info("TRELLIS Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Text Model: {args.text_model}")
    logger.info(f"Image Model: {args.image_model}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Idle Timeout: {args.idle_timeout}s")
    logger.info(f"Idle Check Interval: {args.idle_check_interval}s")
    logger.info("=" * 60)

    server = TrellisServer(
        host=args.host,
        port=args.port,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
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
