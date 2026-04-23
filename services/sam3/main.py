#!/usr/bin/env python3
"""
SAM3 FastAPI Server - Text-Prompted Image Segmentation Service

A FastAPI-based server that wraps Meta AI's SAM3 (Segment Anything Model 3)
for text-prompted image segmentation.

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

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sam3_server.log"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models
# ===========================================================================
class SegmentRequest(BaseModel):
    """Request body for text-prompted image segmentation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any = Field(
        ...,
        description="Base64-encoded image (decoded to PIL Image by validator)",
    )
    text_prompt: str = Field(
        ..., min_length=1, description="Text prompt for segmentation"
    )

    confidence_threshold: Optional[float] = Field(
        default=None, ge=0, le=1, description="Confidence threshold (default: 0.5)"
    )

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, v):
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
    """Response for image segmentation."""

    status: str

    masks: Optional[list[str]] = None
    boxes: Optional[list[list[float]]] = None
    scores: Optional[list[float]] = None
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
DEFAULT_PORT = 8004
DEFAULT_CHECKPOINT_PATH = "./checkpoints/sam3.pt"
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes
DEFAULT_IDLE_CHECK_INTERVAL = 30  # seconds


# Enum for model states
class ModelState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class Sam3Server:
    """
    FastAPI-based server for SAM3 text-prompted image segmentation.

    Features automatic GPU selection, lazy model loading, idle-timeout
    unloading, and thread-safe serialized inference.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        idle_check_interval: int = DEFAULT_IDLE_CHECK_INTERVAL,
    ):
        # Model state management
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

        self._model_state: ModelState = ModelState.NOT_LOADED
        self._state_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
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
            # Models are loaded lazily on first request

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

                await asyncio.to_thread(self._unload_model)
                self._model_state = ModelState.NOT_LOADED

                logger.info("Shutdown complete, model unloaded")

        self._app = FastAPI(
            title="SAM3 Image Segmentation Service",
            description="Segment objects in images using text prompts with Meta AI SAM3.",
            version="1.0.0",
            lifespan=lifespan,
        )
        self._register_routes()

    def _register_routes(self) -> None:
        """Register FastAPI routes."""

        @self._app.get("/health", response_model=HealthResponse)
        async def health():
            """Check server health and model status."""
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

        @self._app.post("/segment", response_model=SegmentResponse)
        async def segment(request: SegmentRequest):
            """Segment objects in an image using a text prompt."""
            self._last_activity = time.monotonic()  # update activity timestamp immediately to prevent idle unload during generation
            await self._ensure_model_loaded()

            async with self._inference_lock:
                result = await asyncio.to_thread(
                    self._run_segmentation,
                    image=request.image,
                    text_prompt=request.text_prompt,
                    confidence_threshold=request.confidence_threshold,
                )

            self._last_activity = (
                time.monotonic()
            )  # update activity timestamp after generation completes

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        logger.info(f"Starting SAM3 server on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    @staticmethod
    def _select_free_gpu() -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SAM3 requires GPU acceleration.")
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
    def _load_model(self) -> None:
        """Load the SAM3 model onto the selected GPU."""
        logger.info("Loading SAM3 image model")
        start_time = time.time()

        try:
            self._gpu_id = self._select_free_gpu()

            # Build on CPU first to avoid torch.cuda.set_device side effect
            self.model = build_sam3_image_model(
                device="cpu",
                load_from_HF=False,
                checkpoint_path=self.checkpoint_path,
            )
            self.model = self.model.to(f"cuda:{self._gpu_id}")
            self.processor = Sam3Processor(self.model, device=f"cuda:{self._gpu_id}")

            load_time = time.time() - start_time
            logger.info(f"SAM3 model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            # Clean up partially loaded resources on failure
            self._gpu_id = None
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            torch.cuda.empty_cache()
            logger.error(f"Failed to load SAM3 model: {e}")
            raise RuntimeError(f"SAM3 model loading failed: {e}") from e

    def _unload_model(self) -> None:
        """Unload the model and release GPU memory."""
        if self.model is None:
            return

        logger.info("[model] Unloading...")
        gpu_id = self._gpu_id
        self._gpu_id = None

        if gpu_id is not None:
            mem_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        del self.processor
        self.processor = None
        del self.model
        self.model = None
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
                await asyncio.to_thread(self._load_model)
                self._model_state = ModelState.LOADED
                logger.info("[model] LOADING -> LOADED")
            except Exception as e:
                self._model_state = ModelState.NOT_LOADED
                logger.error(f"[model] Failed to load: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load SAM3 model: {e}",
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
                                await asyncio.to_thread(self._unload_model)
                                self._model_state = ModelState.NOT_LOADED

            except asyncio.CancelledError:
                logger.info("Idle monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Idle monitor error: {e}")

    # ----------------------------------------------------------------
    # Segmentation (Inference)
    # ----------------------------------------------------------------
    @staticmethod
    def _encode_mask_as_png(mask_bool: np.ndarray) -> str:
        """Encode a single boolean mask as a base64 PNG string."""
        mask_uint8 = (mask_bool.squeeze().astype(np.uint8)) * 255
        mask_pil = PILImage.fromarray(mask_uint8, mode="L")
        buffer = BytesIO()
        mask_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _run_segmentation(
        self,
        image: PILImage.Image,
        text_prompt: str,
        confidence_threshold: Optional[float],
    ) -> Dict[str, Any]:
        """
        Run SAM3 text-prompted segmentation on an image.

        Returns a dict matching SegmentResponse fields.
        """
        # Request-scoped confidence threshold (safe because _inference_lock serializes all callers)
        old_threshold = self.processor.confidence_threshold
        if confidence_threshold is not None:
            self.processor.set_confidence_threshold(confidence_threshold)

        logger.info(f"Segmenting image with prompt: '{text_prompt[:80]}'")
        start_time = time.time()

        try:
            # Run inference under bf16 autocast (SAM3 is designed for this)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(text_prompt, state)

            # Extract results (cast to float32 for numpy compatibility)
            masks = state["masks"].cpu().to(torch.uint8).numpy()  # [N, 1, H, W] bool
            boxes = state["boxes"].cpu().float().numpy()  # [N, 4] float
            scores = state["scores"].cpu().float().numpy()  # [N] float

            # Encode masks as base64 PNGs
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
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
        finally:
            # Restore original threshold to avoid affecting subsequent requests
            self.processor.set_confidence_threshold(old_threshold)


def main():
    """
    Main entry point for the SAM3 FastAPI server.
    """
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
    logger.setLevel(getattr(logging, args.log_level))

    # Print configuration
    logger.info("=" * 60)
    logger.info("SAM3 FastAPI Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Idle Timeout: {args.idle_timeout}s")
    logger.info(f"Idle Check Interval: {args.idle_check_interval}s")
    logger.info("=" * 60)

    server = Sam3Server(
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint_path,
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
