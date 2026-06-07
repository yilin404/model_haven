"""
Shared infrastructure for GPU-backed FastAPI model servers.

- ModelEngine: abstract base for a single model's lifecycle (load/unload/inference)
- BaseFastAPIServer: abstract base that composes ModelEngines into a FastAPI service
- select_free_gpu(): standalone GPU selection utility
"""

import abc
import asyncio
import enum
import gc
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ===========================================================================
# Shared types
# ===========================================================================
class ModelState(str, enum.Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class HealthResponse(BaseModel):
    status: str
    model_state: Dict[str, str]
    gpu: str
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
    idle_timeout: Optional[int] = None


# ===========================================================================
# GPU utility
# ===========================================================================
def select_free_gpu() -> int:
    """Select the GPU with the most free memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A GPU is required.")
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


# ===========================================================================
# ModelEngine — model lifecycle base class
# ===========================================================================
class ModelEngine(abc.ABC):
    """
    Abstract base for a single model/pipeline's lifecycle.

    Subclass must implement:
        _load_impl()           — load model onto GPU
        _unload_impl()         — delete model objects, set refs to None
        _run_inference_impl()  — run inference, return result dict
    """

    def __init__(self, name: str):
        self.name = name
        self.state: ModelState = ModelState.NOT_LOADED
        self.state_lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()
        self.last_activity: float = time.monotonic()
        self.gpu_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Abstract: subclass must implement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _load_impl(self) -> None:
        """Load the model onto a GPU. Set self.gpu_id and model refs."""

    @abc.abstractmethod
    def _unload_impl(self) -> None:
        """Delete model objects and set refs to None. Base calls empty_cache."""

    @abc.abstractmethod
    def _run_inference_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """Run inference. Return {"status": "success", ...} or {"status": "error", ...}."""

    # ------------------------------------------------------------------
    # Fully implemented
    # ------------------------------------------------------------------
    @staticmethod
    def _is_cuda_oom(error: Any) -> bool:
        text = str(error).lower()
        return "cuda" in text and ("out of memory" in text or "cuda oom" in text)

    def _clear_cuda_cache(self, gpu_id: Optional[int] = None) -> None:
        gc.collect()
        if not torch.cuda.is_available():
            return

        device_ids = [gpu_id] if gpu_id is not None else range(torch.cuda.device_count())
        for device_id in device_ids:
            if device_id is None:
                continue
            with torch.cuda.device(device_id):
                with suppress(Exception):
                    torch.cuda.synchronize(device_id)
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    with suppress(Exception):
                        torch.cuda.ipc_collect()

    async def ensure_loaded(self) -> None:
        """Double-check locking lazy load."""
        async with self.state_lock:
            if self.state == ModelState.LOADED:
                return
            logger.info(f"[{self.name}] {self.state.value} -> LOADING")
            self.state = ModelState.LOADING
            try:
                await asyncio.to_thread(self._load_impl)
                self.state = ModelState.LOADED
                logger.info(f"[{self.name}] LOADING -> LOADED")
            except Exception as e:
                gpu_id = self.gpu_id
                with suppress(Exception):
                    self._unload_impl()
                self.gpu_id = None
                self._clear_cuda_cache(gpu_id)

                self.state = ModelState.NOT_LOADED
                logger.error(f"[{self.name}] Failed to load: {e}")
                raise RuntimeError(f"Failed to load {self.name}: {e}") from e

    async def ensure_unloaded(self) -> None:
        """Unload if loaded and no inference is running. Safe to call from any state."""
        async with self.state_lock:
            if self.state != ModelState.LOADED or self.inference_lock.locked():
                return

            elapsed = time.monotonic() - self.last_activity
            logger.info(f"[{self.name}] Unloading (idle {elapsed:.0f}s)")
            self.state = ModelState.UNLOADING

            gpu_id = self.gpu_id
            self.gpu_id = None
            mem_before = (
                torch.cuda.memory_allocated(gpu_id) / (1024**3)
                if gpu_id is not None
                else None
            )

            try:
                self._unload_impl()
            except Exception as e:
                logger.error(f"[{self.name}] Unload failed: {e}")
            finally:
                self._clear_cuda_cache(gpu_id)
                self.state = ModelState.NOT_LOADED

            if gpu_id is not None and mem_before is not None:
                mem_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                freed = mem_before - mem_after
                logger.info(
                    f"[{self.name}] Unloaded from GPU {gpu_id} "
                    f"(freed {freed:.2f}GB, now {mem_after:.2f}GB)"
                )

    async def run_inference(self, *args, **kwargs) -> Dict[str, Any]:
        """Standard inference pattern: touch -> ensure -> lock -> thread -> touch."""
        self.last_activity = time.monotonic()
        try:
            await self.ensure_loaded()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "http_status": 503,
            }

        oom_detected = False
        try:
            async with self.inference_lock:
                result = await asyncio.to_thread(
                    self._run_inference_impl, *args, **kwargs
                )
        except Exception as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            oom_detected = self._is_cuda_oom(e)
            result = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }
        finally:
            self.last_activity = time.monotonic()
            self._clear_cuda_cache(self.gpu_id)

        if result.get("status") == "error" and self._is_cuda_oom(
            result.get("error", "")
        ):
            oom_detected = True

        if oom_detected:
            logger.warning(
                f"[{self.name}] CUDA OOM detected; unloading model to recover memory"
            )
            await self.ensure_unloaded()

        return result

    def should_idle_unload(self, idle_timeout: int) -> bool:
        if self.state != ModelState.LOADED:
            return False
        return (time.monotonic() - self.last_activity) > idle_timeout


# ===========================================================================
# BaseFastAPIServer — service layer base class
# ===========================================================================
class BaseFastAPIServer(abc.ABC):
    """
    Abstract base that composes ModelEngines into a FastAPI service.

    Subclass must implement:
        _register_routes()  — register inference endpoints
    """

    def __init__(
        self,
        engines: list[ModelEngine],
        host: str = "0.0.0.0",
        port: int = 8000,
        idle_timeout: int = 300,
        idle_check_interval: int = 30,
    ):
        self._engines = engines
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.idle_check_interval = idle_check_interval
        self._idle_monitor_task: Optional[asyncio.Task] = None

        assert self.idle_timeout >= 0
        assert self.idle_check_interval > 0

        names = [e.name for e in engines]
        assert len(names) == len(set(names)), (
            f"Engine names must be unique, got: {names}"
        )

        self._app = self._create_app()
        self._register_routes()

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _register_routes(self) -> None:
        """Register service-specific FastAPI routes (inference endpoints)."""

    # ------------------------------------------------------------------
    # App creation
    # ------------------------------------------------------------------
    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title=f"{self.__class__.__name__}",
            description=f"{self.__class__.__name__} with engines: {', '.join(e.name for e in self._engines)}",
            version="1.0.0",
            lifespan=self._make_lifespan(),
        )

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return self._build_health_dict()

        return app

    # ------------------------------------------------------------------
    # Lifespan
    # ------------------------------------------------------------------
    def _make_lifespan(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
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

                for engine in self._engines:
                    try:
                        await engine.ensure_unloaded()
                    except Exception as e:
                        logger.error(f"Shutdown unload failed for {engine.name}: {e}")

                logger.info("Shutdown complete")

        return lifespan

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def _build_health_dict(self) -> dict:
        if torch.cuda.is_available():
            gpu_names = ", ".join(
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            )
            total_alloc = sum(
                torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
            ) / (1024**3)
            total_reserved = sum(
                torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())
            ) / (1024**3)
        else:
            gpu_names = "N/A"
            total_alloc = None
            total_reserved = None

        return {
            "status": "ok" if torch.cuda.is_available() else "degraded",
            "model_state": {
                engine.name: engine.state.value for engine in self._engines
            },
            "gpu": gpu_names,
            "gpu_memory_allocated_gb": round(total_alloc, 2)
            if total_alloc is not None
            else None,
            "gpu_memory_reserved_gb": round(total_reserved, 2)
            if total_reserved is not None
            else None,
            "idle_timeout": self.idle_timeout,
        }

    # ------------------------------------------------------------------
    # Idle monitor
    # ------------------------------------------------------------------
    async def _idle_monitor_loop(self) -> None:
        logger.info(
            f"Idle monitor started (timeout={self.idle_timeout}s, "
            f"check_interval={self.idle_check_interval}s)"
        )
        while True:
            try:
                await asyncio.sleep(self.idle_check_interval)
                for engine in self._engines:
                    if engine.should_idle_unload(self.idle_timeout):
                        await engine.ensure_unloaded()
            except asyncio.CancelledError:
                logger.info("Idle monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Idle monitor error: {e}")

    # ------------------------------------------------------------------
    # Server start
    # ------------------------------------------------------------------
    def start(self) -> None:
        logger.info(f"Starting {self.__class__.__name__} on {self.host}:{self.port}")
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")
