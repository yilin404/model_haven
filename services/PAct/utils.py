"""PAct server utility functions."""

import json
import logging
import os
import zipfile
from glob import glob
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from PIL import Image as PILImage

from modules.inference_utils import load_img_mask
from modules.label_2d_mask.label_parts import (
    prepare_image,
    resize_and_pad_to_square,
    split_disconnected_parts,
)
from modules.pact.modules.sparse.basic import SparseTensor, sparse_cat
from modules.pact.process_utils import (
    make_slat_coords_from_voxel_coords,
    merge_multi_parts,
)
from modules.pact.utils import postprocessing_utils
from scripts.json_to_urdf import (
    ConverterConfig,
    build_robot_element,
    prettify,
)

logger = logging.getLogger(__name__)


# ================================================
# PreProcess
# ================================================

def segment_parts(rgba_image, mask_generator, size_threshold=2000):
    """SAM segmentation returning int32 label map (background=0, parts=1..N)."""
    # White-background composite
    white_bg = PILImage.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    image_np = np.array(
        PILImage.alpha_composite(white_bg, rgba_image.convert("RGBA")).convert("RGB")
    )

    # SAM generation + background filtering
    masks = sorted(
        mask_generator.generate(image_np), key=lambda x: x["area"], reverse=True
    )
    rgba_arr = np.array(rgba_image)
    bg_alpha = rgba_arr[:, :, 3] <= 10

    group_ids = np.full(image_np.shape[:2], -1, dtype=int)
    group_counter = 0
    for m in masks:
        if m["area"] < size_threshold:
            continue
        seg = m["segmentation"]
        if np.sum(seg) == 0:
            continue
        if np.sum(seg & bg_alpha) / np.sum(seg) > 0.1:
            continue
        group_ids[seg] = group_counter
        group_counter += 1

    # Split disconnected parts
    group_ids = split_disconnected_parts(group_ids, size_threshold)
    if np.max(group_ids) >= 0:
        group_counter = np.max(group_ids) + 1

    # Detect undetected regions via alpha channel
    alpha_fg = rgba_arr[:, :, 3] > 0
    existing = group_ids != -1
    dilated_existing = cv2.dilate(existing.astype(np.uint8), np.ones((4, 4), np.uint8))
    undetected = alpha_fg & (~dilated_existing.astype(bool))

    if np.sum(undetected) > size_threshold:
        num_labels, labels = cv2.connectedComponents(
            undetected.astype(np.uint8), connectivity=8
        )

        # Union-Find
        parent = list(range(num_labels))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        areas = np.bincount(labels.flatten())
        valid = np.where(areas[1:] >= size_threshold / 5)[0] + 1

        kernel = np.ones((3, 3), np.uint8)
        dilated_map = {
            i: cv2.dilate((labels == i).astype(np.uint8), kernel, iterations=2)
            for i in valid
        }

        barrier_mask = existing
        for idx_a, i in enumerate(valid[:-1]):
            for j in valid[idx_a + 1 :]:
                overlap = dilated_map[i] & dilated_map[j]
                overlap_size = np.sum(overlap)
                if overlap_size > 40 and not np.any(overlap & barrier_mask):
                    overlap_ratio_i = overlap_size / areas[i - 1]
                    overlap_ratio_j = overlap_size / areas[j - 1]
                    if max(overlap_ratio_i, overlap_ratio_j) > 0.03:
                        union(i, j)

        merged_labels = np.zeros_like(labels)
        for label in range(1, num_labels):
            merged_labels[labels == label] = find(label)
        unique_merged = np.unique(merged_labels[merged_labels > 0])

        for label_val in unique_merged:
            region_mask = merged_labels == label_val
            if np.sum(region_mask) > size_threshold:
                group_ids[region_mask] = group_counter
                group_counter += 1

    return group_ids.astype(np.int32) + 1


def preprocess(image, rmbg_model, sam_mask_generator, mask=None):
    """背景去除 + SAM 分割 + 构造 batch dict。返回 (batch, num_parts)。"""
    logger.info("Removing background...")
    processed = prepare_image(image.convert("RGB"), rmbg_net=rmbg_model)
    processed_sq = resize_and_pad_to_square(processed)

    if mask is not None:
        logger.info("Using provided mask")
        labels = np.array(
            mask.resize(processed_sq.size, resample=PILImage.Resampling.NEAREST)
        ).astype(np.int32)
    else:
        logger.info("Auto-segmenting with SAM...")
        labels = segment_parts(processed_sq, sam_mask_generator)

    num_parts = len(np.unique(labels[labels > 0]))
    if num_parts == 0:
        raise ValueError("No parts detected in mask")

    rgba = np.array(processed_sq)
    if rgba.shape[2] == 4 and rgba[:, :, 3].sum() == 0:
        raise ValueError("No foreground after background removal")

    logger.info(f"Detected {num_parts} parts")

    labels_3d = einops.repeat(labels, "h w -> h w c", c=1)
    _, img_black, ordered_mask, _ = load_img_mask(
        None,
        None,
        size=(518, 518),
        img=processed_sq,
        mask_part=labels_3d,
        is_sort=True,
    )

    n = int(ordered_mask.max().item())
    if n == 0:
        raise ValueError("No parts survived mask downsampling to 37x37")

    batch = {
        "id": ["request"] * n,
        "cond": img_black.unsqueeze(0).repeat(n, 1, 1, 1),
        "masks": ordered_mask.unsqueeze(0).repeat(n, 1, 1),
        "num_parts": torch.tensor([n], dtype=torch.int32),
        "part_idx": torch.tensor(list(range(n)), dtype=torch.int32),
    }
    return batch, n


# ================================================
# Run PAct Pipeline
# ================================================

def run_pact_pipeline(pipeline, batch, seed, cfg_strength, device, outdir):
    """执行 PAct 推理管线：voxel 生成 → SLat → articulation → 导出。"""
    num_parts_list = batch["num_parts"].cpu().numpy().tolist()

    pipeline._setup_rng(seed)
    pipeline.is_decode_coords = True
    ss_params = {
        "steps": 25,
        "cfg_strength": cfg_strength,
        "arti_out_mode": "mean_feature_regression_steps",
        "part_idx": batch["part_idx"].to(device),
    }
    with torch.inference_mode():
        raw_output = pipeline.get_part_coords(
            batch["cond"].to(device),
            batch["masks"].to(device),
            num_parts=batch["num_parts"].to(device),
            seed=seed,
            return_raw=True,
            sparse_structure_sampler_params=ss_params,
        )
        occ = raw_output["occ"]

    x_0 = occ.to(device)
    x_0_coords = [
        torch.nonzero(x_0[i, 0] > 0, as_tuple=False) for i in range(len(x_0))
    ]
    merge_multi_parts(x_0_coords, num_parts_list)
    _, _, slat_data_list = make_slat_coords_from_voxel_coords(
        x_0_coords, num_parts_list
    )
    coords = sparse_cat(
        [
            SparseTensor(
                coords=slat_data_list[i]["coords"],
                feats=torch.zeros(
                    slat_data_list[i]["coords"].shape[0], 1, device=device
                ),
            )
            for i in range(len(slat_data_list))
        ]
    ).coords
    layouts = [
        slat_data_list[i]["part_layouts"] for i in range(len(slat_data_list))
    ]

    conds = batch["cond"].to(device)
    masks = batch["masks"].to(device)
    if conds.shape[0] == sum(num_parts_list):
        idx = torch.cumsum(torch.tensor([0] + num_parts_list), dim=0)[
            : len(num_parts_list)
        ]
        conds = conds[idx]
        masks = masks[idx]

    slat_params = {
        "steps": 25,
        "cfg_strength": cfg_strength,
        "arti_out_mode": "mean_feature_regression_steps",
    }
    with torch.inference_mode():
        pact_output = pipeline.get_slat_arti(
            conds,
            coords,
            layouts,
            masks,
            seed=seed,
            slat_sampler_params=slat_params,
            formats=["mesh", "gaussian"],
            preprocess_image=False,
        )

    x_0_arti_list = pact_output["articulation"]["info"]
    gs_list = pact_output["gaussian"]
    mesh_list = pact_output["mesh"]

    num_parts_slat = torch.tensor(num_parts_list) + 1
    idxs = torch.cumsum(torch.tensor([0] + num_parts_slat.tolist()), dim=0)

    textured_meshes = [
        postprocessing_utils.to_glb(
            gs_list[i],
            mesh_list[i],
            simplify=0.95,
            texture_size=1024,
            textured=True,
            transform_zup_to_yup=False,
        )
        for i in range(len(gs_list))
    ]
    rep_parts_mesh = [
        mesh_list[idxs[i] : idxs[i + 1]] for i in range(len(num_parts_slat))
    ]
    tex_mesh_per_obj = [
        textured_meshes[idxs[i] + 1 : idxs[i + 1]]
        for i in range(len(num_parts_slat))
    ]

    pipeline.export_arti_objects(
        [rep_parts_mesh[0][1:]],
        x_0_arti_list[0:1],
        [
            {
                "img_id": "01",
                "class_id": "pact",
                "obj_id": "request",
                "scale": 1.0,
                "offset": [0.0, 0.0, 0.0],
                "out_dir": outdir,
                "tag": "exported_arti_objects",
            }
        ],
        texture_mesh_list=[tex_mesh_per_obj[0]],
    )
    logger.info(
        f"Inference complete ({num_parts_list[0]} parts, exported to {outdir})"
    )


# ================================================
# PostProcess
# ================================================

def prepare_urdf_package(export_root, outdir):
    """调用原项目 json_to_urdf 生成 URDF，然后打包为 zip。"""
    json_candidates = glob(
        os.path.join(export_root, "**", "object.json"), recursive=True
    )
    if not json_candidates:
        logger.warning("No object.json found for URDF export.")
        return None

    target_json = Path(json_candidates[0])
    export_dir = target_json.parent
    urdf_path = export_dir / "object.urdf"

    try:
        with target_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        config = ConverterConfig(
            json_path=target_json,
            output_path=urdf_path,
            asset_root=export_dir,
            mesh_priority="obj",
            geometry_format="glb",
            absolute_mesh_paths=False,
            robot_name=None,
            min_mass=1.0,
            limit_effort=50.0,
            limit_velocity=1.0,
        )
        robot_element, _ = build_robot_element(data, config)
        pretty_xml = prettify(robot_element)
        urdf_path.write_text(pretty_xml, encoding="utf-8")

    except Exception as exc:
        logger.error(f"Failed to build URDF: {exc}")
        return None

    zip_path = Path(outdir) / "pact_urdf_package.zip"
    try:
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for root, _, files in os.walk(export_dir):
                for filename in files:
                    abs_path = Path(root) / filename
                    rel_path = abs_path.relative_to(export_dir)
                    archive.write(abs_path, arcname=rel_path.as_posix())
    except Exception as exc:
        logger.error(f"Failed to zip URDF package: {exc}")
        return None

    return zip_path.as_posix()
