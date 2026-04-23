#!/usr/bin/env python3
"""
Example client for SAM 3D Objects FastAPI Server

Demonstrates how to call the SAM 3D Objects FastAPI server for single-image
3D object reconstruction using the requests library.
"""

import argparse
import base64
import sys
import time

import requests


def check_health(base_url: str) -> dict:
    """Check server health status."""
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def generate_3d(
    base_url: str,
    image_path: str,
    mask_path: str,
    seed: int | None = 42,
) -> dict:
    """
    Reconstruct 3D object from image and mask.

    Args:
        base_url: Server base URL
        image_path: Path to input image file
        mask_path: Path to mask image file
        seed: Random seed

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "mask": mask_b64,
        "seed": seed,
    }

    print(f"Sending 3D reconstruction request")
    print(f"  Image: {image_path}")
    print(f"  Mask:  {mask_path}")
    start_time = time.time()

    resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_ply(response: dict, output_path: str) -> None:
    """Save PLY data from response to file."""
    ply_bytes = base64.b64decode(response["ply_data"])
    with open(output_path, "wb") as f:
        f.write(ply_bytes)
    print(f"Saved: {output_path} ({len(ply_bytes)} bytes)")


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8003, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # generate 3D reconstruction
    gen_parser = subparsers.add_parser("generate", help="Reconstruct 3D from image and mask")
    gen_parser.add_argument("--image", type=str, required=True, help="Input image path")
    gen_parser.add_argument("--mask", type=str, required=True, help="Mask image path")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--output", type=str, default="output.ply", help="Output PLY file path")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "health":
        result = check_health(base_url)
        print(f"Status: {result['status']}")
        print(f"Model state: {result.get('model_state', 'N/A')}")
        print(f"GPU: {result['gpu']}")
        if result.get("gpu_memory_allocated_gb") is not None:
            print(f"GPU memory allocated: {result['gpu_memory_allocated_gb']:.2f} GB")
        if result.get("gpu_memory_reserved_gb") is not None:
            print(f"GPU memory reserved: {result['gpu_memory_reserved_gb']:.2f} GB")
        if result.get("idle_timeout") is not None:
            print(f"Idle timeout: {result['idle_timeout']}s")

    elif args.command == "generate":
        print("=" * 60)
        print("SAM 3D Objects FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Image:  {args.image}")
        print(f"Mask:   {args.mask}")
        print(f"Seed:   {args.seed}")
        print(f"Output: {args.output}")
        print("=" * 60)

        response = generate_3d(
            base_url=base_url,
            image_path=args.image,
            mask_path=args.mask,
            seed=args.seed,
        )

        if response["status"] == "success":
            print("3D reconstruction successful!")
            save_ply(response, args.output)
            metadata = response.get("metadata", {})
            print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
            print(f"Rotation: {metadata.get('rotation', 'N/A')}")
            print(f"Translation: {metadata.get('translation', 'N/A')}")
            print(f"Scale: {metadata.get('scale', 'N/A')}")
        else:
            print(f"3D reconstruction failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
