"""
Particulate Server - 3D Object Articulation Inference Service

A FastAPI-based server that wraps Particulate for feed-forward 3D object
articulation inference from a single static 3D mesh.
"""

import argparse
import base64
import logging
import os
import sys
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# Must be set before any torch checkpoint loading
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
import trimesh
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Add particulate deps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/particulate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/particulate/PartField"))

from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from particulate.articulation_utils import plucker_to_axis_point
from particulate.data_utils import load_obj_raw_preserve, sample_points
from particulate.export_utils import export_animated_glb_file, export_mjcf, export_urdf
from particulate.models import PAT_B, PAT_L, PAT_S, PAT_XL
from particulate.postprocessing_utils import find_part_ids_for_faces
from particulate.visualization_utils import (
    ARROW_COLOR_PRISMATIC,
    ARROW_COLOR_REVOLUTE,
    create_arrow,
    create_ring,
    create_textured_mesh_parts,
    get_3D_arrow_on_points,
)
from partfield_utils import get_partfield_model, obtain_partfield_feats

from yacs.config import CfgNode

torch.serialization.add_safe_globals([CfgNode])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BaseFastAPIServer, ModelEngine, select_free_gpu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../../deps/particulate/configs/particulate-B.yaml"
)
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints/model.pt")

DATA_CONFIG = {
    "sharp_point_ratio": 0.5,
    "normalize_points": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zip_directory_to_b64(dir_path: str) -> str:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, dir_path)
                zf.write(filepath, arcname)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _transform_up_dir(mesh: trimesh.Trimesh, up_dir: str) -> trimesh.Trimesh:
    mesh_t = mesh.copy()
    rot = {
        "X":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        "-X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
        "Y":  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        "-Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
        "Z":  None,
        "-Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    }
    R = rot[up_dir]
    if R is not None:
        mesh_t.vertices = mesh_t.vertices @ R.T
    return mesh_t


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh.vertices -= center
    scale = (bbox_max - bbox_min).max()
    mesh.vertices /= scale
    return mesh


def _save_articulated_meshes(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    outputs: list,
    output_dir: str,
    strict: bool = True,
    animation_frames: int = 50,
) -> Tuple[
    List[trimesh.Trimesh],
    np.ndarray,
    np.ndarray,
    list,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    hyp_idx = 0
    part_ids = outputs[hyp_idx]["part_ids"]
    motion_hierarchy = outputs[hyp_idx]["motion_hierarchy"]
    is_part_revolute = outputs[hyp_idx]["is_part_revolute"]
    is_part_prismatic = outputs[hyp_idx]["is_part_prismatic"]
    revolute_plucker = outputs[hyp_idx]["revolute_plucker"]
    revolute_range = outputs[hyp_idx]["revolute_range"]
    prismatic_axis = outputs[hyp_idx]["prismatic_axis"]
    prismatic_range = outputs[hyp_idx]["prismatic_range"]

    face_part_ids = find_part_ids_for_faces(mesh, part_ids, face_indices, strict=strict)
    unique_part_ids = np.unique(face_part_ids)
    num_parts = len(unique_part_ids)

    mesh_parts_original = [
        mesh.submesh([face_part_ids == pid], append=True) for pid in unique_part_ids
    ]
    mesh_parts_segmented = create_textured_mesh_parts([mp.copy() for mp in mesh_parts_original])

    axes = []
    for i, mesh_part in enumerate(mesh_parts_segmented):
        pid = unique_part_ids[i]
        if is_part_revolute[pid]:
            axis, point = plucker_to_axis_point(revolute_plucker[pid])
            arrow_start, arrow_end = get_3D_arrow_on_points(
                axis, mesh_part.vertices, fixed_point=point, extension=0.2
            )
            axes.append(create_arrow(arrow_start, arrow_end, color=ARROW_COLOR_REVOLUTE, radius=0.01, radius_tip=0.018))
            arrow_dir = arrow_end - arrow_start
            axes.append(create_ring(arrow_start, arrow_dir, major_radius=0.03, minor_radius=0.006, color=ARROW_COLOR_REVOLUTE))
            axes.append(create_ring(arrow_end, arrow_dir, major_radius=0.03, minor_radius=0.006, color=ARROW_COLOR_REVOLUTE))
        elif is_part_prismatic[pid]:
            axis = prismatic_axis[pid]
            arrow_start, arrow_end = get_3D_arrow_on_points(
                axis, mesh_part.vertices, extension=0.2
            )
            axes.append(create_arrow(arrow_start, arrow_end, color=ARROW_COLOR_PRISMATIC, radius=0.01, radius_tip=0.018))

    segmented_path = os.path.join(output_dir, "mesh_parts_with_axes.glb")
    trimesh.Scene(mesh_parts_segmented + axes).export(segmented_path)

    animated_path = os.path.join(output_dir, "animated_textured.glb")
    export_animated_glb_file(
        mesh_parts_original,
        unique_part_ids,
        motion_hierarchy,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range,
        animation_frames,
        animated_path,
        include_axes=False,
        axes_meshes=None,
    )

    return (
        mesh_parts_original,
        face_part_ids,
        unique_part_ids,
        motion_hierarchy,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range,
    )


# ===========================================================================
# ModelEngine
# ===========================================================================
class ParticulateEngine(ModelEngine):
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        ckpt_path: str = DEFAULT_CKPT_PATH,
    ):
        super().__init__("particulate")
        self._config_path = config_path
        self._ckpt_path = ckpt_path
        self.pat_model = None
        self.partfield_model = None

    def _load_impl(self) -> None:
        self.gpu_id = select_free_gpu()
        device = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(self.gpu_id)

        logger.info("Loading PAT model config...")
        cfg = OmegaConf.load(self._config_path)
        model_size = cfg.pop("model_size", "B")
        pat_cls = {"S": PAT_S, "B": PAT_B, "L": PAT_L, "XL": PAT_XL}
        self.pat_model = pat_cls[model_size](**cfg)
        self.pat_model.eval()

        logger.info("Loading PAT checkpoint...")
        ckpt = self._ckpt_path
        if not os.path.exists(ckpt):
            logger.info("Checkpoint not found locally, downloading from HuggingFace...")
            ckpt = hf_hub_download("rayli/Particulate", "model.pt")
        self.pat_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.pat_model.to(device)

        # Ensure PartField checkpoint exists (download if missing)
        partfield_ckpt = os.path.join(
            os.path.dirname(__file__), "../../deps/particulate/PartField/model/model_objaverse.ckpt"
        )
        if not os.path.exists(partfield_ckpt):
            logger.info("PartField checkpoint not found, downloading...")
            os.makedirs(os.path.dirname(partfield_ckpt), exist_ok=True)
            hf_hub_download(
                repo_id="mikaelaangel/partfield-ckpt",
                filename="model_objaverse.ckpt",
                local_dir=os.path.dirname(partfield_ckpt),
            )

        logger.info("Loading PartField model...")
        self.partfield_model = get_partfield_model(device=f"cuda:{self.gpu_id}")
        logger.info("All models loaded successfully.")

    def _unload_impl(self) -> None:
        if self.pat_model is not None:
            del self.pat_model
            self.pat_model = None
        if self.partfield_model is not None:
            del self.partfield_model
            self.partfield_model = None

    def _prepare_inputs(
        self,
        mesh: trimesh.Trimesh,
        num_points_global: int = 40000,
        num_points_decode: int = 2048,
    ) -> Tuple[dict, np.ndarray, np.ndarray]:
        device = f"cuda:{self.gpu_id}"
        sharp_point_ratio = DATA_CONFIG["sharp_point_ratio"]

        all_points, _, _, _ = sample_points(mesh, num_points_global, sharp_point_ratio)
        points, normals, sharp_flag, face_indices = sample_points(
            mesh, num_points_decode, sharp_point_ratio
        )

        if DATA_CONFIG["normalize_points"]:
            bbmin = np.concatenate([all_points, points], axis=0).min(0)
            bbmax = np.concatenate([all_points, points], axis=0).max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 1.0 / (bbmax - bbmin).max()
            all_points = (all_points - center) * scale
            points = (points - center) * scale

        all_points_t = torch.from_numpy(all_points).to(device).float().unsqueeze(0)
        points_t = torch.from_numpy(points).to(device).float().unsqueeze(0)
        normals_t = torch.from_numpy(normals).to(device).float().unsqueeze(0)

        feats = obtain_partfield_feats(self.partfield_model, all_points_t, points_t)

        return dict(xyz=points_t, normals=normals_t, feats=feats), sharp_flag, face_indices

    def _run_inference_impl(
        self,
        mesh_file_bytes: bytes,
        mesh_filename: str,
        up_dir: str = "-Z",
        num_points: int = 102400,
        min_part_confidence: float = 0.0,
        strict: bool = True,
        animation_frames: int = 50,
        export_urdf_flag: bool = False,
        export_mjcf_flag: bool = False,
    ) -> Dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)
        start_time = time.time()

        try:
            with tempfile.TemporaryDirectory(prefix="particulate_") as tmpdir:
                # Write mesh bytes to temp file
                mesh_ext = Path(mesh_filename).suffix.lower()
                tmp_mesh_path = os.path.join(tmpdir, f"input{mesh_ext}")
                with open(tmp_mesh_path, "wb") as f:
                    f.write(mesh_file_bytes)

                # Load mesh
                if mesh_ext == ".obj":
                    verts, faces = load_obj_raw_preserve(Path(tmp_mesh_path))
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                else:
                    mesh = trimesh.load(tmp_mesh_path, process=False)
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(mesh.geometry.values())

                if mesh_ext == ".ply":
                    if not (
                        isinstance(mesh, trimesh.Trimesh)
                        and mesh.vertices is not None
                        and mesh.faces is not None
                        and len(mesh.vertices) > 0
                        and len(mesh.faces) > 0
                    ):
                        raise ValueError("Invalid PLY mesh: empty vertices or faces")

                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"Failed to load mesh as Trimesh from {mesh_ext} file")
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    raise ValueError("Mesh has empty vertices or faces")

                # Transform and normalize
                mesh_transformed = _transform_up_dir(mesh, up_dir)
                mesh_transformed = _normalize_mesh(mesh_transformed)

                # Prepare inputs (uses cached PartField model)
                inputs, _sharp_flag, face_indices = self._prepare_inputs(
                    mesh_transformed,
                    num_points_global=40000,
                    num_points_decode=num_points,
                )

                # Run model
                with torch.no_grad():
                    outputs = self.pat_model.infer(
                        xyz=inputs["xyz"],
                        feats=inputs["feats"],
                        normals=inputs["normals"],
                        output_all_hyps=True,
                        min_part_confidence=min_part_confidence,
                    )

                # Post-process: save GLB files to tmpdir
                (
                    mesh_parts_original,
                    _face_part_ids,
                    unique_part_ids,
                    motion_hierarchy,
                    is_part_revolute,
                    is_part_prismatic,
                    revolute_plucker,
                    revolute_range,
                    prismatic_axis,
                    prismatic_range,
                ) = _save_articulated_meshes(
                    mesh_transformed,
                    face_indices,
                    outputs,
                    tmpdir,
                    strict=strict,
                    animation_frames=animation_frames,
                )

                num_parts = len(unique_part_ids)

                # Read output files (MUST read before tempdir exits)
                segmented_path = os.path.join(tmpdir, "mesh_parts_with_axes.glb")
                animated_path = os.path.join(tmpdir, "animated_textured.glb")

                with open(segmented_path, "rb") as f:
                    segmented_b64 = base64.b64encode(f.read()).decode("utf-8")
                with open(animated_path, "rb") as f:
                    animated_b64 = base64.b64encode(f.read()).decode("utf-8")

                result = {
                    "status": "success",
                    "segmented_glb": segmented_b64,
                    "animated_glb": animated_b64,
                    "metadata": {
                        "num_parts": int(num_parts),
                        "generation_time": round(time.time() - start_time, 2),
                        "up_dir": up_dir,
                        "num_points": num_points,
                    },
                }

                # Optionally export URDF
                if export_urdf_flag:
                    urdf_dir = os.path.join(tmpdir, "urdf")
                    urdf_output_path = os.path.join(urdf_dir, "model.urdf")
                    export_urdf(
                        mesh_parts_original,
                        unique_part_ids,
                        motion_hierarchy,
                        is_part_revolute,
                        is_part_prismatic,
                        revolute_plucker,
                        revolute_range,
                        prismatic_axis,
                        prismatic_range,
                        output_path=urdf_output_path,
                        name="model",
                    )
                    result["urdf_zip"] = _zip_directory_to_b64(urdf_dir)

                # Optionally export MJCF
                if export_mjcf_flag:
                    mjcf_dir = os.path.join(tmpdir, "mjcf")
                    mjcf_output_path = os.path.join(mjcf_dir, "model.xml")
                    export_mjcf(
                        mesh_parts_original,
                        unique_part_ids,
                        motion_hierarchy,
                        is_part_revolute,
                        is_part_prismatic,
                        revolute_plucker,
                        revolute_range,
                        prismatic_axis,
                        prismatic_range,
                        output_path=mjcf_output_path,
                        name="model",
                    )
                    result["mjcf_zip"] = _zip_directory_to_b64(mjcf_dir)

                return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# ===========================================================================
# Pydantic models
# ===========================================================================
class InferRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mesh_file: str = Field(..., description="Base64-encoded mesh file")
    filename: str = Field(
        default="input.obj",
        description="Original filename (determines mesh format from suffix)",
    )
    up_dir: Literal["X", "Y", "Z", "-X", "-Y", "-Z"] = Field(
        default="-Z", description="Up direction of the input mesh",
    )
    num_points: int = Field(
        default=102400, ge=1024, le=500000,
        description="Number of points to sample",
    )
    min_part_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum part confidence threshold",
    )
    strict: bool = Field(
        default=True,
        description="Use strict connected component refinement",
    )
    animation_frames: int = Field(
        default=50, ge=10, le=200,
        description="Number of animation frames for animated GLB",
    )
    export_urdf: bool = Field(default=False, description="Export URDF package")
    export_mjcf: bool = Field(default=False, description="Export MJCF package")

    @field_validator("mesh_file", mode="before", json_schema_input_type=str)
    @classmethod
    def validate_mesh_file(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("mesh_file must be a base64-encoded string")
        if "," in v and v.startswith("data:"):
            v = v.split(",", 1)[1]
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"mesh_file must be valid base64: {e}")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        suffix = Path(v).suffix.lower()
        if suffix not in {".obj", ".ply", ".glb"}:
            raise ValueError("filename must end with .obj, .ply, or .glb")
        return v


class InferResponse(BaseModel):
    status: str
    segmented_glb: Optional[str] = None
    animated_glb: Optional[str] = None
    urdf_zip: Optional[str] = None
    mjcf_zip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


# ===========================================================================
# FastAPI Server
# ===========================================================================
class ParticulateServer(BaseFastAPIServer):
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        ckpt_path: str = DEFAULT_CKPT_PATH,
        **kwargs,
    ):
        self._engine = ParticulateEngine(config_path, ckpt_path)
        super().__init__(engines=[self._engine], **kwargs)

    def _register_routes(self) -> None:
        @self._app.post("/infer", response_model=InferResponse)
        async def infer(request: InferRequest):
            mesh_bytes = base64.b64decode(request.mesh_file)
            result = await self._engine.run_inference(
                mesh_file_bytes=mesh_bytes,
                mesh_filename=request.filename,
                up_dir=request.up_dir,
                num_points=request.num_points,
                min_part_confidence=request.min_part_confidence,
                strict=request.strict,
                animation_frames=request.animation_frames,
                export_urdf_flag=request.export_urdf,
                export_mjcf_flag=request.export_mjcf,
            )
            if result["status"] == "error":
                raise HTTPException(
                    status_code=result.get("http_status", 500), detail=result
                )
            return result


# ===========================================================================
# Main
# ===========================================================================
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8006
DEFAULT_IDLE_TIMEOUT = 300
DEFAULT_IDLE_CHECK_INTERVAL = 30


def main():
    parser = argparse.ArgumentParser(
        description="Particulate FastAPI Server - 3D Articulation Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to model config")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH, help="Path to model checkpoint")
    parser.add_argument("--idle-timeout", type=int, default=DEFAULT_IDLE_TIMEOUT, help="Seconds of inactivity before unloading models")
    parser.add_argument("--idle-check-interval", type=int, default=DEFAULT_IDLE_CHECK_INTERVAL, help="Seconds between idle checks")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("particulate_server.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.setLevel(getattr(logging, args.log_level))

    log.info("=" * 60)
    log.info("Particulate Server Configuration")
    log.info("=" * 60)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Config: {args.config}")
    log.info(f"Checkpoint: {args.ckpt}")
    log.info(f"Idle Timeout: {args.idle_timeout}s")
    log.info(f"Idle Check Interval: {args.idle_check_interval}s")
    log.info("=" * 60)

    server = ParticulateServer(
        config_path=args.config,
        ckpt_path=args.ckpt,
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
