#!/usr/bin/env python3
"""
PAct Server - Single-View Part Articulation Generation Service

A FastAPI-based server that wraps PAct for generating part-aware articulated
3D objects from a single RGB image (+ optional part segmentation mask).
"""

import argparse
import base64
import logging
import os
import sys
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from transformers import AutoModelForImageSegmentation
from segment_anything import SamAutomaticMaskGenerator, build_sam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/PAct"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/PAct/scripts"))

from modules.pact.pipelines import PActPipeline

from utils import preprocess, run_pact_pipeline, prepare_urdf_package

os.environ["SPCONV_ALGO"] = "native"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)


# ===========================================================================
# ModelEngine
# ===========================================================================
DEFAULT_SAM_CKPT = os.path.join(
    os.path.dirname(__file__), "checkpoints/sam_vit_h_4b8939.pth"
)


class PActEngine(ModelEngine):
    def __init__(self, sam_ckpt_path: str = DEFAULT_SAM_CKPT):
        super().__init__("model")
        self._sam_ckpt_path = sam_ckpt_path
        self.pact_pipeline: Optional[PActPipeline] = None
        self.rmbg_model = None
        self.sam_mask_generator = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        device = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(self.gpu_id)

        logger.info("Loading BriaRMBG 2.0...")
        self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        self.rmbg_model.to(device)
        self.rmbg_model.eval()

        logger.info(f"Loading SAM from {self._sam_ckpt_path}...")
        if not os.path.exists(self._sam_ckpt_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {self._sam_ckpt_path}")
        sam_model = build_sam(checkpoint=self._sam_ckpt_path).to(device)
        self.sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

        logger.info("Loading PAct pipeline...")
        self.pact_pipeline = PActPipeline.from_pretrained(
            "PAct000/PAct", revision="main"
        )
        self.pact_pipeline.to(device)
        self.pact_pipeline.verbose = False

    def _unload_impl(self) -> None:
        if self.pact_pipeline is not None:
            del self.pact_pipeline
            self.pact_pipeline = None
        if self.rmbg_model is not None:
            del self.rmbg_model
            self.rmbg_model = None
        if self.sam_mask_generator is not None:
            del self.sam_mask_generator
            self.sam_mask_generator = None

    def _preprocess(
        self,
        image: PILImage.Image,
        mask: Optional[PILImage.Image],
    ) -> tuple[dict, int]:
        return preprocess(image, self.rmbg_model, self.sam_mask_generator, mask)

    def _run_pact_pipeline(
        self,
        batch: dict,
        seed: int,
        cfg_strength: float,
        device: torch.device,
        outdir: str,
    ) -> None:
        run_pact_pipeline(self.pact_pipeline, batch, seed, cfg_strength, device, outdir)

    def _run_inference_impl(
        self,
        image: PILImage.Image,
        mask: Optional[PILImage.Image],
        seed: int,
        cfg_strength: float,
    ) -> Dict[str, Any]:
        device = torch.device(f"cuda:{self.gpu_id}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        start_time = time.time()

        try:
            batch, num_parts = self._preprocess(image, mask)
            with tempfile.TemporaryDirectory(prefix="pact_req_") as tmpdir:
                self._run_pact_pipeline(batch, seed, cfg_strength, device, tmpdir)

                export_root = os.path.join(tmpdir, "exported_arti_objects")
                urdf_zip_path = prepare_urdf_package(export_root, tmpdir)
                if not urdf_zip_path or not os.path.exists(urdf_zip_path):
                    raise RuntimeError("URDF package generation failed")

                with open(urdf_zip_path, "rb") as f:
                    urdf_b64 = base64.b64encode(f.read()).decode("utf-8")

                return {
                    "status": "success",
                    "urdf_zip": urdf_b64,
                    "metadata": {
                        "seed": seed,
                        "cfg_strength": cfg_strength,
                        "num_parts": num_parts,
                        "generation_time": round(time.time() - start_time, 2),
                    },
                }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# FastAPI Server and Pydantic models
# ===========================================================================
class GenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image = Field(
        ...,
        description="Base64-encoded RGB image",
    )
    mask: Optional[PILImage.Image] = Field(
        default=None,
        description="Base64-encoded part segmentation mask (optional)",
    )
    seed: Optional[int] = Field(default=42, description="Random seed")
    cfg_strength: float = Field(
        default=7.5, ge=0.0, le=20.0, description="Classifier-free guidance strength"
    )

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

    @field_validator("mask", mode="before", json_schema_input_type=str)
    @classmethod
    def parse_mask(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Mask must be a base64-encoded string")
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Mask must be valid base64-encoded data: {e}")
        try:
            return PILImage.open(BytesIO(raw)).convert("L")
        except Exception as e:
            raise ValueError(f"Mask data is not a valid image format: {e}")


class GenerateResponse(BaseModel):
    status: str
    urdf_zip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class PActServer(BaseFastAPIServer):
    def __init__(self, sam_ckpt_path: str = DEFAULT_SAM_CKPT, **kwargs):
        self._engine = PActEngine(sam_ckpt_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            result = await self._engine.run_inference(
                image=request.image,
                mask=request.mask,
                seed=request.seed,
                cfg_strength=request.cfg_strength,
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
        description="PAct FastAPI Server - Part Articulation Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--sam-ckpt",
        type=str,
        default=DEFAULT_SAM_CKPT,
        help="Path to SAM checkpoint",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pact_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("PAct Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"SAM checkpoint: {args.sam_ckpt}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = PActServer(
        sam_ckpt_path=args.sam_ckpt,
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
