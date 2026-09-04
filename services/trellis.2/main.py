#!/usr/bin/env python3
"""
TRELLIS.2 Server - Image-to-3D Generation Service

A FastAPI-based server that wraps Microsoft's TRELLIS.2 (microsoft/TRELLIS.2-4B)
for high-fidelity image-to-3D generation, returning GLB assets with baked
PBR textures (base color / metallic / roughness).
"""

import argparse
import base64
import logging
import os
import sys
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, Literal, Optional

import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/trellis.2"))

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")


def _export_glb(mesh, decimation_target: int, texture_size: int, remesh: bool) -> bytes:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    buffer = BytesIO()
    # file_type is required when exporting to a buffer (path-based exports infer it)
    glb.export(buffer, file_type="glb", extension_webp=True)
    return buffer.getvalue()


class Trellis2Engine(ModelEngine):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__("trellis2")
        self.model_name = model_name
        self.pipeline: Optional[Trellis2ImageTo3DPipeline] = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        torch.cuda.set_device(self.gpu_id)
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_name)
        # low_vram pipelines keep models on CPU and move stages to GPU on demand;
        # .to() only records the target device
        self.pipeline.to(f"cuda:{self.gpu_id}")

    def _unload_impl(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        seed: Optional[int],
        pipeline_type: Optional[str],
        preprocess_image: bool,
        decimation_target: int,
        texture_size: int,
        remesh: bool,
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)
        seed = seed if seed is not None else torch.randint(0, 2**32, (1,)).item()

        logger.info(
            f"Generating 3D from image (seed={seed}, pipeline_type={pipeline_type}, "
            f"decimation_target={decimation_target}, texture_size={texture_size})"
        )
        start_time = time.time()

        try:
            mesh = self.pipeline.run(
                image,
                seed=seed,
                pipeline_type=pipeline_type,
                preprocess_image=preprocess_image,
            )[0]
            mesh.simplify(16777216)  # nvdiffrast face limit

            glb_bytes = _export_glb(mesh, decimation_target, texture_size, remesh)
            generation_time = time.time() - start_time

            logger.info(
                f"Image-to-3D complete in {generation_time:.2f}s, "
                f"file size: {len(glb_bytes)} bytes"
            )

            return {
                "status": "success",
                "glb_data": base64.b64encode(glb_bytes).decode("utf-8"),
                "metadata": {
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "file_size": len(glb_bytes),
                    "pipeline_type": pipeline_type or "default",
                    "decimation_target": decimation_target,
                    "texture_size": texture_size,
                    "remesh": remesh,
                    "preprocess_image": preprocess_image,
                },
            }

        except Exception as e:
            logger.error(f"Image-to-3D generation failed: {e}")
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
    seed: Optional[int] = Field(default=None, description="Random seed")
    pipeline_type: Optional[Literal["512", "1024", "1024_cascade", "1536_cascade"]] = Field(
        default=None,
        description="Resolution pipeline; None uses the model default (1024_cascade)",
    )
    preprocess_image: bool = Field(
        default=True, description="Whether to remove the background (BiRefNet)"
    )
    decimation_target: int = Field(
        default=1000000, gt=0, description="Target number of vertices for mesh decimation"
    )
    texture_size: int = Field(
        default=2048, gt=0, description="Baked texture resolution"
    )
    remesh: bool = Field(default=True, description="Whether to remesh before UV unwrapping")

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
            return PILImage.open(BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Image data is not a valid image format: {e}")


class GenerateResponse(BaseModel):
    status: str
    glb_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class Trellis2Server(BaseFastAPIServer):
    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self._engine = Trellis2Engine(model_name)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            result = await self._engine.run_inference(
                image=request.image,
                seed=request.seed,
                pipeline_type=request.pipeline_type,
                preprocess_image=request.preprocess_image,
                decimation_target=request.decimation_target,
                texture_size=request.texture_size,
                remesh=request.remesh,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8009
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 FastAPI Server - Image-to-3D Generation Service",
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
        help="HuggingFace model identifier for image-to-3D",
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
            logging.FileHandler(os.path.join(os.path.dirname(__file__), "trellis2_server.log")),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("TRELLIS.2 Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Model: {args.model}")
    log.info(f"Log Level: {args.log_level}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = Trellis2Server(
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
