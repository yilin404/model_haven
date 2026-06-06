#!/usr/bin/env python3
"""
SAM 3D Objects FastAPI Server - Single Image to 3D Reconstruction Service

A FastAPI-based server that wraps Meta's SAM 3D Objects model for single-image
3D object reconstruction, producing Gaussian splat PLY files.
"""

import argparse
import base64
import io
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

os.environ["LIDRA_SKIP_INIT"] = "true"

from omegaconf import OmegaConf
from hydra.utils import instantiate

import sam3d_objects  # noqa: F401 — registers modules needed by hydra instantiate
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_CONFIG_PATH = "checkpoints/pipeline.yaml"


class SAM3DObjectsEngine(ModelEngine):
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        super().__init__("model")
        self._config_path = config_path
        self.pipeline: Optional[InferencePipelinePointMap] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)

        abs_config = os.path.abspath(
            os.path.join(os.path.dirname(__file__), self._config_path)
        )
        config = OmegaConf.load(abs_config)
        config.compile_model = False
        config.rendering_engine = "pytorch3d"
        config.workspace_dir = os.path.dirname(abs_config)

        self.pipeline = instantiate(config)

    def _unload_impl(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        mask: PILImage.Image,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)

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

            ply_buffer = io.BytesIO()
            output["gs"].save_ply(ply_buffer)
            ply_bytes = ply_buffer.getvalue()

            glb_mesh = output.get("glb")
            glb_bytes = None
            if glb_mesh is not None:
                glb_buffer = io.BytesIO()
                glb_mesh.export(glb_buffer, file_type="glb")
                glb_bytes = glb_buffer.getvalue()

            rotation = output["rotation"].cpu().numpy().tolist()
            translation = output["translation"].cpu().numpy().tolist()
            scale = output["scale"].cpu().numpy().tolist()

            logger.info(
                f"3D reconstruction complete in {generation_time:.2f}s "
                f"(PLY: {len(ply_bytes)} bytes, GLB: {len(glb_bytes) if glb_bytes else 0} bytes)"
            )

            return {
                "status": "success",
                "ply_data": base64.b64encode(ply_bytes).decode("utf-8"),
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8") if glb_bytes else None,
                "metadata": {
                    "generation_time": round(generation_time, 2),
                    "rotation": rotation,
                    "translation": translation,
                    "scale": scale,
                    "ply_file_size": len(ply_bytes),
                    "glb_file_size": len(glb_bytes) if glb_bytes else 0,
                },
            }

        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class GenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded image string and will be decoded to PILImage.Image",
    )
    mask: PILImage.Image = Field(
        ...,
        description="Base64-encoded mask image string and will be decoded to PILImage.Image",
    )
    seed: Optional[int] = Field(default=42, description="Random seed")

    @field_validator("image", mode="before", json_schema_input_type=str)
    @classmethod
    def parse_image(cls, v: str) -> PILImage.Image:
        if not isinstance(v, str):
            raise ValueError("Image must be a base64-encoded string")
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Image must be valid base64-encoded data: {e}")
        try:
            return PILImage.open(io.BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Image data is not a valid image format: {e}")

    @field_validator("mask", mode="before", json_schema_input_type=str)
    @classmethod
    def parse_mask(cls, v: str) -> PILImage.Image:
        if not isinstance(v, str):
            raise ValueError("Mask must be a base64-encoded string")
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Mask must be valid base64-encoded data: {e}")
        try:
            return PILImage.open(io.BytesIO(raw)).convert("L")
        except Exception as e:
            raise ValueError(f"Mask data is not a valid image format: {e}")


class GenerateResponse(BaseModel):
    status: str
    ply_data: Optional[str] = None
    glb_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class SAM3DObjectsServer(BaseFastAPIServer):
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, **kwargs):
        self._engine = SAM3DObjectsEngine(config_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            image = request.image
            mask = request.mask

            if image.size != mask.size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size {image.size} does not match mask size {mask.size}",
                )

            result = await self._engine.run_inference(
                image=image, mask=mask, seed=request.seed
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8005
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("sam3d_objects_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("SAM 3D Objects Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Config Path: {args.config_path}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = SAM3DObjectsServer(
        config_path=args.config_path,
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
