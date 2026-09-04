#!/usr/bin/env python3
"""
Example client for TRELLIS.2 FastAPI Server

Demonstrates how to call the TRELLIS.2 FastAPI server for image-to-3D
generation using the requests library.
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
    seed: int | None = None,
    pipeline_type: str | None = None,
    preprocess_image: bool = True,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = True,
) -> dict:
    """
    Generate a 3D asset from an image.

    Args:
        base_url: Server base URL
        image_path: Path to input image file
        seed: Random seed (None for random)
        pipeline_type: Resolution pipeline ('512', '1024', '1024_cascade',
            '1536_cascade' or None for model default)
        preprocess_image: Whether to remove the background server-side
        decimation_target: Target vertex count of the exported mesh
        texture_size: Baked texture resolution
        remesh: Whether to remesh before UV unwrapping

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "seed": seed,
        "preprocess_image": preprocess_image,
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "remesh": remesh,
    }
    if pipeline_type is not None:
        payload["pipeline_type"] = pipeline_type

    print("Sending image-to-3D generation request")
    print(f"  Image: {image_path}")
    start_time = time.time()

    resp = requests.post(f"{base_url}/generate", json=payload, timeout=1800)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_glb(response: dict, output_path: str) -> None:
    """Save GLB data from response to file."""
    glb_bytes = base64.b64decode(response["glb_data"])
    with open(output_path, "wb") as f:
        f.write(glb_bytes)
    print(f"Saved: {output_path} ({len(glb_bytes)} bytes)")


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8009, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # generate 3D from image
    gen_parser = subparsers.add_parser("generate", help="Generate 3D from image")
    gen_parser.add_argument("--image", type=str, required=True, help="Input image path")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument(
        "--pipeline-type",
        type=str,
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default=None,
        help="Resolution pipeline (None = model default 1024_cascade)",
    )
    gen_parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip background removal (input already has clean alpha)",
    )
    gen_parser.add_argument(
        "--decimation-target", type=int, default=1000000, help="Target vertex count"
    )
    gen_parser.add_argument(
        "--texture-size", type=int, default=2048, help="Baked texture resolution"
    )
    gen_parser.add_argument(
        "--no-remesh", action="store_true", help="Skip remeshing before UV unwrapping"
    )
    gen_parser.add_argument(
        "--output", type=str, default="output", help="Output file path (without extension)"
    )

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "health":
        result = check_health(base_url)
        print(f"Status: {result['status']}")
        model_states = result.get("model_state", {})
        for name, state in model_states.items():
            print(f"Engine [{name}] state: {state}")
        print(f"GPU: {result['gpu']}")
        if result.get("gpu_memory_allocated_gb") is not None:
            print(f"GPU memory allocated: {result['gpu_memory_allocated_gb']:.2f} GB")
        if result.get("gpu_memory_reserved_gb") is not None:
            print(f"GPU memory reserved: {result['gpu_memory_reserved_gb']:.2f} GB")
        if result.get("idle_timeout") is not None:
            print(f"Idle timeout: {result['idle_timeout']}s")

    elif args.command == "generate":
        print("=" * 60)
        print("TRELLIS.2 FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Image:  {args.image}")
        print(f"Seed:   {args.seed}")
        print(f"Pipeline type: {args.pipeline_type or 'default'}")
        print(f"Output: {args.output}")
        print("=" * 60)

        response = generate_3d(
            base_url=base_url,
            image_path=args.image,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            preprocess_image=not args.no_preprocess,
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=not args.no_remesh,
        )

        if response["status"] == "success":
            print("Image-to-3D generation successful!")
            save_glb(response, f"{args.output}.glb")
            metadata = response.get("metadata", {})
            print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
            print(f"Seed: {metadata.get('seed', 'N/A')}")
            print(f"File size: {metadata.get('file_size', 'N/A')} bytes")
        else:
            print(f"Image-to-3D generation failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
