#!/usr/bin/env python3
"""
SAM 3D Objects FastAPI Server - Single Image to 3D Reconstruction Service

A FastAPI-based server that wraps Meta's SAM 3D Objects model for single-image
3D object reconstruction, producing Gaussian splat PLY files.

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
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from PIL import Image

# Must be set before importing sam3d_objects
os.environ["LIDRA_SKIP_INIT"] = "true"

from omegaconf import OmegaConf
from hydra.utils import instantiate

import sam3d_objects  # noqa: F401 — registers modules needed by hydra instantiate
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sam3d_objects_server.log"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models
# ===========================================================================
class GenerateRequest(BaseModel):
    """Request body for 3D reconstruction."""

    image: Any = Field(..., description="Base64-encoded image (PNG/JPEG)")
    mask: Any = Field(..., description="Base64-encoded mask image (PNG, grayscale)")

    seed: Optional[int] = Field(default=42, description="Random seed")

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, v):
        if isinstance(v, str):
            return Image.open(io.BytesIO(base64.b64decode(v)))
        raise ValueError("image must be a base64 string")

    @field_validator("mask", mode="before")
    @classmethod
    def parse_mask(cls, v):
        if isinstance(v, str):
            return Image.open(io.BytesIO(base64.b64decode(v)))
        raise ValueError("mask must be a base64 string")


class GenerateResponse(BaseModel):
    """Response for 3D reconstruction."""

    status: str

    ply_data: Optional[str] = None
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
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8003
DEFAULT_CONFIG_PATH = "checkpoints/pipeline.yaml"
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


class ModelState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class SAM3DObjectsServer:
    """
    FastAPI-based server for SAM 3D Objects single-image 3D reconstruction.

    Features automatic GPU selection, lazy model loading, idle-timeout
    unloading, and thread-safe serialized inference.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        config_path: str = DEFAULT_CONFIG_PATH,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        idle_check_interval: int = DEFAULT_IDLE_CHECK_INTERVAL,
    ):
        # SAM 3D Objects configuration
        self._config_path = config_path
        self.pipeline: Optional[InferencePipelinePointMap] = None

        # Model state management
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
            try:
                await self._ensure_model_loaded()
            except Exception as e:
                logger.warning(
                    f"Startup model loading failed (will retry on first request): {e}"
                )

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
                logger.info("Shutdown complete, pipeline unloaded")

        self._app = FastAPI(
            title="SAM 3D Objects Reconstruction Service",
            description="Reconstruct 3D objects from single images using Meta SAM 3D Objects.",
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

        @self._app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Reconstruct 3D object from image and mask."""
            image = request.image
            mask = request.mask

            if image.size != mask.size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size {image.size} does not match mask size {mask.size}",
                )

            self._last_activity = time.monotonic()
            await self._ensure_model_loaded()

            async with self._inference_lock:
                result = await asyncio.to_thread(
                    self._generate_3d,
                    image,
                    mask,
                    request.seed,
                )

            self._last_activity = time.monotonic()

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        logger.info(f"Starting SAM 3D Objects server on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    @staticmethod
    def _select_free_gpu() -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. SAM 3D Objects requires GPU acceleration."
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
    # Model Load / Unload
    # ----------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the SAM 3D Objects pipeline onto the selected GPU."""
        logger.info("Loading SAM 3D Objects pipeline")
        start_time = time.time()

        try:
            self._gpu_id = self._select_free_gpu()
            torch.cuda.set_device(self._gpu_id)

            abs_config = os.path.abspath(
                os.path.join(os.path.dirname(__file__), self._config_path)
            )
            config = OmegaConf.load(abs_config)
            config.compile_model = False
            config.rendering_engine = "pytorch3d"
            config.workspace_dir = os.path.dirname(abs_config)

            self.pipeline = instantiate(config)

            load_time = time.time() - start_time
            logger.info(
                f"SAM 3D Objects pipeline loaded successfully in {load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load SAM 3D Objects pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}") from e

    def _unload_model(self) -> None:
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
                await asyncio.to_thread(self._load_model)
                self._model_state = ModelState.LOADED
                logger.info("[model] LOADING -> LOADED")
            except Exception as e:
                self._model_state = ModelState.NOT_LOADED
                logger.error(f"[model] Failed to load: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load SAM 3D Objects pipeline: {e}",
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
    # Inference
    # ----------------------------------------------------------------
    def _generate_3d(
        self,
        image: Image.Image,
        mask: Image.Image,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D object from image and mask.

        Returns a dict matching GenerateResponse fields.
        """
        if self._gpu_id is not None:
            torch.cuda.set_device(self._gpu_id)

        logger.info(
            f"Starting 3D reconstruction (image size={image.size}, seed={seed})"
        )

        start_time = time.time()

        try:
            output = self.pipeline.run(
                image,
                mask,
                seed,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=True,
            )
            generation_time = time.time() - start_time

            # Export Gaussian splat to PLY bytes
            ply_buffer = io.BytesIO()
            output["gs"].save_ply(ply_buffer)
            ply_bytes = ply_buffer.getvalue()

            # Extract pose information
            rotation = output["rotation"].cpu().numpy().tolist()
            translation = output["translation"].cpu().numpy().tolist()
            scale = output["scale"].cpu().numpy().tolist()

            logger.info(
                f"3D reconstruction complete in {generation_time:.2f}s "
                f"(PLY size: {len(ply_bytes)} bytes)"
            )

            return {
                "status": "success",
                "ply_data": base64.b64encode(ply_bytes).decode("utf-8"),
                "metadata": {
                    "generation_time": round(generation_time, 2),
                    "rotation": rotation,
                    "translation": translation,
                    "scale": scale,
                    "file_size": len(ply_bytes),
                },
            }

        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


def main():
    """Main entry point for the SAM 3D Objects FastAPI server."""
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects FastAPI Server - Single Image 3D Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline config YAML",
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
    logger.info("SAM 3D Objects FastAPI Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Config Path: {args.config_path}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Idle Timeout: {args.idle_timeout}s")
    logger.info(f"Idle Check Interval: {args.idle_check_interval}s")
    logger.info("=" * 60)

    server = SAM3DObjectsServer(
        host=args.host,
        port=args.port,
        config_path=args.config_path,
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
