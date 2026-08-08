#!/usr/bin/env python3
"""Example client for the Depth Anything 3 service."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from typing import Any

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serialization import NDArrayData


def check_health(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def _encode_image(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def estimate_depth(
    base_url: str,
    image_paths: list[str],
    process_res: int,
    process_res_method: str,
    camera_npz: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images": [_encode_image(path) for path in image_paths],
        "process_res": process_res,
        "process_res_method": process_res_method,
    }

    if camera_npz is not None:
        with np.load(camera_npz) as camera_data:
            if "extrinsics" not in camera_data or "intrinsics" not in camera_data:
                raise ValueError(
                    "camera NPZ must contain extrinsics and intrinsics"
                )
            payload["extrinsics"] = NDArrayData.from_array(
                camera_data["extrinsics"]
            ).model_dump()
            payload["intrinsics"] = NDArrayData.from_array(
                camera_data["intrinsics"]
            ).model_dump()

    start_time = time.time()
    response = requests.post(
        f"{base_url}/estimate-depth",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    print(f"Response received in {time.time() - start_time:.2f}s")
    return response.json()


def _decode_optional_array(response: dict[str, Any], key: str):
    payload = response.get(key)
    return NDArrayData.model_validate(payload).to_array() if payload else None


def save_prediction(response: dict[str, Any], output_path: str) -> None:
    if response.get("status") != "success":
        raise RuntimeError(response.get("error", "Depth estimation failed"))

    arrays = {
        key: value
        for key in ("depth", "confidence", "extrinsics", "intrinsics")
        if (value := _decode_optional_array(response, key)) is not None
    }
    np.savez_compressed(output_path, **arrays)
    print(f"Saved prediction to {output_path}")

    for name, array in arrays.items():
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")

    metadata = response.get("metadata", {})
    print(f"  model: {metadata.get('model_name')}")
    print(f"  metric depth: {metadata.get('is_metric')}")
    print(f"  generation time: {metadata.get('generation_time')}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 service client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8006)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Check service health")

    estimate = subparsers.add_parser("estimate", help="Estimate depth")
    estimate.add_argument(
        "--image",
        action="append",
        required=True,
        dest="images",
        help="Input image path; repeat for multi-view inference",
    )
    estimate.add_argument("--process-res", type=int, default=504)
    estimate.add_argument(
        "--process-res-method",
        choices=["upper_bound_resize", "lower_bound_resize"],
        default="upper_bound_resize",
    )
    estimate.add_argument(
        "--camera-npz",
        help="Optional NPZ containing extrinsics and intrinsics",
    )
    estimate.add_argument("--output", default="depth_prediction.npz")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    base_url = f"http://{args.host}:{args.port}"
    if args.command == "health":
        print(check_health(base_url))
        return

    response = estimate_depth(
        base_url=base_url,
        image_paths=args.images,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        camera_npz=args.camera_npz,
    )
    save_prediction(response, args.output)


if __name__ == "__main__":
    main()

