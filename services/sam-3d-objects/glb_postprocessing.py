"""Build textured GLBs from frozen SAM3D mesh and Gaussian outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from PIL import Image as PILImage
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


def _linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Encode normalized linear RGB values as sRGB.

    Args:
        linear_rgb: Float RGB values with shape ``(..., 3)`` in ``[0, 1]``.

    Returns:
        sRGB values with the same shape and float32 dtype in ``[0, 1]``.
    """
    clipped_rgb = np.clip(linear_rgb.astype(np.float32), 0.0, 1.0)
    return np.where(
        clipped_rgb <= 0.0031308,
        12.92 * clipped_rgb,
        1.055 * np.power(clipped_rgb, 1.0 / 2.4) - 0.055,
    )


def _encode_base_color_texture_as_srgb(
    mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Encode a baked linear base-color texture for glTF storage.

    Args:
        mesh: Textured mesh returned by SAM3D ``postprocessing_utils.to_glb``.

    Returns:
        The same mesh with an sRGB-encoded base-color image. Vertices, faces,
        UVs, texture alpha, and material factors remain unchanged.

    Raises:
        RuntimeError: If the baked mesh has no base-color texture.

    Note:
        This function mutates only ``mesh.visual.material.baseColorTexture``.
    """
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        raise RuntimeError("SAM3D texture baking returned no base-color texture")

    texture_rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8).copy()
    linear_rgb = texture_rgba[..., :3].astype(np.float32) / 255.0
    texture_rgba[..., :3] = np.rint(_linear_to_srgb(linear_rgb) * 255.0).astype(
        np.uint8
    )
    material.baseColorTexture = PILImage.fromarray(texture_rgba)
    return mesh


def build_textured_glb(output: dict[str, Any]) -> trimesh.Trimesh:
    """Build a RoboSnap-style textured GLB from one raw SAM3D output.

    Args:
        output: Raw SAM3D result containing ``mesh`` and ``gaussian`` lists.
            Their first elements must come from the same inference run and use
            the same internal Z-up coordinate frame.

    Returns:
        Y-up textured triangle mesh ready to export as GLB.

    Raises:
        KeyError: If the raw mesh or Gaussian output is absent.
        RuntimeError: If SAM3D fails to produce a base-color texture.
    """
    textured_mesh = postprocessing_utils.to_glb(
        app_rep=output["gaussian"][0],
        mesh=output["mesh"][0],
        simplify=0.95,
        fill_holes=False,
        texture_size=1024,
        verbose=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        rendering_engine="nvdiffrast",
    )
    return _encode_base_color_texture_as_srgb(textured_mesh)
