#!/usr/bin/env python3
"""
GraspGen FastAPI Server - 6-DOF Grasp Generation Service

A FastAPI-based server that wraps NVIDIA's GraspGen diffusion model for 6-DOF
grasp generation, enabling remote clients to generate grasp poses from
point clouds or meshes.

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
import os
import sys
import time
from contextlib import asynccontextmanager, suppress

import numpy as np
import torch

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from typing import Any, Dict, Optional

# Add GraspGen to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/GraspGen"))

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("graspgen_server.log"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models
# ===========================================================================
class PointCloudData(BaseModel):
    """Serialized numpy array for point cloud transport."""

    data: str
    shape: list
    dtype: str


class GraspGenRequest(BaseModel):
    """Request body for JSON-based grasp generation."""

    point_cloud: Any = Field(
        ..., description="Point cloud as {data: base64, shape: list, dtype: str}"
    )

    num_grasps: int = Field(default=200, gt=0, description="Number of grasps to sample")
    topk_num_grasps: int = Field(
        default=-1, description="Return only top-k grasps (-1 = use threshold)"
    )
    grasp_threshold: float = Field(
        default=-1.0, description="Minimum confidence threshold (-1 = use topk)"
    )
    min_grasps: int = Field(
        default=40, gt=0, description="Minimum grasps required before stopping retries"
    )
    max_tries: int = Field(
        default=6, gt=0, description="Maximum inference retry attempts"
    )
    remove_outliers: bool = Field(
        default=True, description="Remove point cloud outliers before inference"
    )

    @field_validator("point_cloud", mode="before")
    @classmethod
    def parse_point_cloud(cls, v):
        if isinstance(v, dict):
            # Use PointCloudData for schema validation, then decode to numpy
            pc = PointCloudData(**v)

            raw = base64.b64decode(pc.data)
            return np.frombuffer(raw, dtype=pc.dtype).reshape(pc.shape)

        raise ValueError("point_cloud must be a dict with {data, shape, dtype}")


class GraspResponse(BaseModel):
    """Response for grasp generation."""

    status: str

    grasps: Optional[Dict[str, Any]] = None
    confidences: Optional[Dict[str, Any]] = None
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
DEFAULT_PORT = 8001
DEFAULT_GRIPPER_CONFIG = "graspgen_robotiq_2f_140.yml"
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes
DEFAULT_IDLE_CHECK_INTERVAL = 30  # seconds


# Enum for model states
class ModelState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class GraspGenServer:
    """
    FastAPI-based server for GraspGen 6-DOF grasp generation.

    Features automatic GPU selection, lazy model loading, idle-timeout
    unloading, and thread-safe serialized inference.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        gripper_config: str = DEFAULT_GRIPPER_CONFIG,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        idle_check_interval: int = DEFAULT_IDLE_CHECK_INTERVAL,
    ):
        # GraspGen model configuration
        config_path = os.path.join("GraspGenModels", "checkpoints", gripper_config)
        logger.info(f"Loading gripper config from: {config_path}")
        self.cfg = load_grasp_cfg(config_path)

        self.gripper_name = self.cfg.data.gripper_name
        self.model_name = self.cfg.eval.model_name

        # Model state management
        self.sampler: Optional[GraspGenSampler] = None

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
            # Models are loaded lazily on first request

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

                await asyncio.to_thread(self._unload_model)
                self._model_state = ModelState.NOT_LOADED

                logger.info("Shutdown complete, pipelines unloaded")

        self._app = FastAPI(
            title="GraspGen 6-DOF Grasp Generation Service",
            description="Generate 6-DOF grasp poses from point clouds using NVIDIA GraspGen.",
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

        @self._app.post("/generate", response_model=GraspResponse)
        async def generate(request: GraspGenRequest):
            """Generate grasps from a JSON request with base64-encoded point cloud."""
            point_cloud = np.asarray(request.point_cloud, dtype=np.float32)

            # Validate point cloud shape and values
            if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point cloud must be (N, 3), got shape {point_cloud.shape}",
                )
            if point_cloud.shape[0] < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point cloud must have at least 10 points, got {point_cloud.shape[0]}",
                )
            if not np.all(np.isfinite(point_cloud)):
                raise HTTPException(
                    status_code=400,
                    detail="Point cloud contains non-finite values (NaN or Inf)",
                )

            self._last_activity = time.monotonic()  # update activity timestamp immediately to prevent idle unload during generation
            await self._ensure_model_loaded()

            async with self._inference_lock:
                result = await asyncio.to_thread(
                    self._generate_grasps,
                    point_cloud,
                    num_grasps=request.num_grasps,
                    topk_num_grasps=request.topk_num_grasps,
                    grasp_threshold=request.grasp_threshold,
                    min_grasps=request.min_grasps,
                    max_tries=request.max_tries,
                    remove_outliers=request.remove_outliers,
                )

            self._last_activity = (
                time.monotonic()
            )  # update activity timestamp after generation completes

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result)

            return result

    def start(self) -> None:
        """Start the FastAPI server using uvicorn."""
        logger.info(f"Starting GraspGen server on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    @staticmethod
    def _select_free_gpu() -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. GraspGen requires GPU acceleration."
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
        """Load the GraspGen model onto the selected GPU."""
        logger.info(f"Loading GraspGen model: {self.gripper_name}")
        start_time = time.time()

        try:
            self._gpu_id = self._select_free_gpu()
            torch.cuda.set_device(self._gpu_id)

            logger.info(
                f"Initializing GraspGenSampler (model={self.model_name}, gripper={self.gripper_name})"
            )
            self.sampler = GraspGenSampler(self.cfg)

            load_time = time.time() - start_time
            logger.info(f"GraspGen model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load GraspGen model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _unload_model(self) -> None:
        """Unload the model and release GPU memory."""
        if self.sampler is None:
            return

        logger.info("[model] Unloading...")
        gpu_id = self._gpu_id
        self._gpu_id = None

        if gpu_id is not None:
            mem_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        del self.sampler
        self.sampler = None
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
                    detail=f"Failed to load GraspGen model: {e}",
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
    # Serialization Helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _encode_numpy_array(array: np.ndarray) -> Dict[str, Any]:
        """Encode numpy array to base64 format."""
        return {
            "data": base64.b64encode(array.tobytes()).decode("utf-8"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def _generate_grasps(
        self,
        point_cloud: np.ndarray,
        num_grasps: int,
        topk_num_grasps: int,
        grasp_threshold: float,
        min_grasps: int,
        max_tries: int,
        remove_outliers: bool,
    ) -> Dict[str, Any]:
        """Generate grasp poses from point cloud using GraspGen.

        Returns a dict matching GraspResponse fields.
        """
        # Ensure correct CUDA device before inference
        if self._gpu_id is not None:
            torch.cuda.set_device(self._gpu_id)

        logger.info(
            f"Generating grasps for {len(point_cloud)} points "
            f"(num_grasps={num_grasps}, topk={topk_num_grasps}, "
            f"threshold={grasp_threshold})"
        )

        start_time = time.time()

        try:
            grasps, grasp_conf = GraspGenSampler.run_inference(
                point_cloud,
                self.sampler,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                grasp_threshold=grasp_threshold,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )

            generation_time = time.time() - start_time

            # Convert to numpy
            if len(grasps) > 0:
                grasps_np = grasps.cpu().numpy().astype(np.float32)
                conf_np = grasp_conf.cpu().numpy().astype(np.float32)
            else:
                grasps_np = np.empty((0, 4, 4), dtype=np.float32)
                conf_np = np.empty((0,), dtype=np.float32)

            logger.info(
                f"Generated {len(grasps_np)} grasps in {generation_time:.2f}s "
                f"(conf range: {conf_np.min():.3f} - {conf_np.max():.3f})"
                if len(conf_np) > 0
                else f"Generated 0 grasps in {generation_time:.2f}s"
            )

            return {
                "status": "success",
                "grasps": self._encode_numpy_array(grasps_np),
                "confidences": self._encode_numpy_array(conf_np),
                "metadata": {
                    "num_grasps": len(grasps_np),
                    "generation_time": round(generation_time, 2),
                    "num_points": len(point_cloud),
                    "gripper_name": self.gripper_name,
                    "model_name": self.model_name,
                },
            }

        except Exception as e:
            logger.error(f"Grasp generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }


def main():
    """
    Main entry point for the GraspGen FastAPI server.
    """
    parser = argparse.ArgumentParser(
        description="GraspGen FastAPI Server - 6-DOF Grasp Generation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--gripper-config",
        type=str,
        default=DEFAULT_GRIPPER_CONFIG,
        help="Gripper configuration file",
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
    logger.info("GraspGen FastAPI Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Gripper Config: {args.gripper_config}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Idle Timeout: {args.idle_timeout}s")
    logger.info(f"Idle Check Interval: {args.idle_check_interval}s")
    logger.info("=" * 60)

    server = GraspGenServer(
        host=args.host,
        port=args.port,
        gripper_config=args.gripper_config,
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
