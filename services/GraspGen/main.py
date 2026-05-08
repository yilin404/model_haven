#!/usr/bin/env python3
"""
GraspGen FastAPI Server - 6-DOF Grasp Generation Service

A FastAPI-based server that wraps NVIDIA's GraspGen diffusion model for 6-DOF
grasp generation, enabling remote clients to generate grasp poses from
point clouds or meshes.
"""

import argparse
import base64
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/GraspGen"))
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_GRIPPER_CONFIG = "graspgen_robotiq_2f_140.yml"


class GraspGenEngine(ModelEngine):
    def __init__(self, gripper_config: str = DEFAULT_GRIPPER_CONFIG):
        super().__init__("model")
        config_path = os.path.join("GraspGenModels", "checkpoints", gripper_config)
        logger.info(f"Loading gripper config from: {config_path}")
        self.cfg = load_grasp_cfg(config_path)
        self.gripper_name = self.cfg.data.gripper_name
        self.model_name = self.cfg.eval.model_name
        self.sampler: Optional[GraspGenSampler] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        logger.info(
            f"Initializing GraspGenSampler (model={self.model_name}, gripper={self.gripper_name})"
        )
        self.sampler = GraspGenSampler(self.cfg)

    def _unload_impl(self) -> None:
        if self.sampler is not None:
            del self.sampler
            self.sampler = None

    @staticmethod
    def _encode_numpy_array(array: np.ndarray) -> Dict[str, Any]:
        return {
            "data": base64.b64encode(array.tobytes()).decode("utf-8"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def _run_inference_impl(
        self,
        point_cloud: np.ndarray,
        num_grasps: int,
        topk_num_grasps: int,
        grasp_threshold: float,
        min_grasps: int,
        max_tries: int,
        remove_outliers: bool,
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)

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
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class PointCloudData(BaseModel):
    data: str
    shape: list[int]
    dtype: str


class GraspGenRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    point_cloud: np.ndarray = Field(
        ...,
        description="Point cloud as {data: base64, shape: list, dtype: str} and will be decoded to (N, 3) float32 numpy array",
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

    @field_validator("point_cloud", mode="before", json_schema_input_type=dict)
    @classmethod
    def parse_point_cloud(cls, v: dict) -> np.ndarray:
        if isinstance(v, dict):
            pc = PointCloudData(**v)
            raw = base64.b64decode(pc.data, validate=True)
            arr = np.frombuffer(raw, dtype=pc.dtype).reshape(pc.shape)
            assert arr.ndim == 2 and arr.shape[1] == 3, (
                "point_cloud must have shape (N, 3)"
            )
            return arr.astype(np.float32, copy=False)
        raise ValueError("point_cloud must be a dict with {data, shape, dtype}")


class GraspResponse(BaseModel):
    status: str
    grasps: Optional[Dict[str, Any]] = None
    confidences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class GraspGenServer(BaseFastAPIServer):
    def __init__(self, gripper_config: str = DEFAULT_GRIPPER_CONFIG, **kwargs):
        self._engine = GraspGenEngine(gripper_config)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/generate", response_model=GraspResponse)
        async def generate(request: GraspGenRequest):
            point_cloud = np.asarray(request.point_cloud, dtype=np.float32)

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

            result = await self._engine.run_inference(
                point_cloud=point_cloud,
                num_grasps=request.num_grasps,
                topk_num_grasps=request.topk_num_grasps,
                grasp_threshold=request.grasp_threshold,
                min_grasps=request.min_grasps,
                max_tries=request.max_tries,
                remove_outliers=request.remove_outliers,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("graspgen_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("GraspGen Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Gripper Config: {args.gripper_config}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = GraspGenServer(
        gripper_config=args.gripper_config,
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
