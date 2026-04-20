#!/usr/bin/env python3
"""
Example client for TRELLIS FastAPI Server

Demonstrates how to call the TRELLIS FastAPI server for text-to-3D and
image-to-3D generation using the requests library.
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


def text_to_3d(
    base_url: str,
    text: str,
    seed: int | None = None,
    simplify: float = 0.95,
    texture_size: int = 1024,
) -> dict:
    """
    Generate a 3D model from a text prompt.

    Args:
        base_url: Server base URL
        text: Text prompt for 3D generation
        seed: Random seed
        simplify: Mesh simplification ratio (0-1)
        texture_size: Texture resolution

    Returns:
        Response dictionary from server
    """
    payload = {
        "text": text,
        "seed": seed,
        "options": {
            "simplify": simplify,
            "texture_size": texture_size,
        },
    }

    print(f"Sending text-to-3D request: '{text}'")
    start_time = time.time()

    resp = requests.post(f"{base_url}/text-to-3d", json=payload, timeout=300)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def image_to_3d(
    base_url: str,
    image_path: str,
    seed: int | None = None,
    simplify: float = 0.95,
    texture_size: int = 1024,
    preprocess_image: bool = True,
) -> dict:
    """
    Generate a 3D model from an image file.

    Args:
        base_url: Server base URL
        image_path: Path to the input image file
        seed: Random seed
        simplify: Mesh simplification ratio (0-1)
        texture_size: Texture resolution
        preprocess_image: Whether to preprocess (remove background)

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "seed": seed,
        "options": {
            "simplify": simplify,
            "texture_size": texture_size,
            "preprocess_image": preprocess_image,
        },
    }

    print(f"Sending image-to-3D request: {image_path}")
    start_time = time.time()

    resp = requests.post(f"{base_url}/image-to-3d", json=payload, timeout=300)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_glb(response: dict, output_file: str, show: bool = True) -> None:
    """
    Save GLB data from response to file.

    Args:
        response: Response dictionary from server
        output_file: Output file path
        show: Whether to visualize the mesh using trimesh
    """
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    glb_data = base64.b64decode(response["glb_data"])

    with open(output_file, "wb") as f:
        f.write(glb_data)

    metadata = response.get("metadata", {})
    print(f"Saved GLB file: {output_file}")
    print(f"File size: {len(glb_data)} bytes")
    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
    print(f"Seed: {metadata.get('seed', 'N/A')}")

    if show:
        try:
            import trimesh

            mesh = trimesh.load(output_file, file_type="glb")
            mesh.show()
        except ImportError:
            print("trimesh not installed, skipping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # text-to-3d
    text_parser = subparsers.add_parser("text-to-3d", help="Generate 3D from text")
    text_parser.add_argument("--text", type=str, default="A modern chair with wooden legs", help="Text prompt")
    text_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    text_parser.add_argument("--simplify", type=float, default=0.95, help="Mesh simplification ratio")
    text_parser.add_argument("--texture-size", type=int, default=1024, help="Texture resolution")
    text_parser.add_argument("--output", type=str, default="output_text.glb", help="Output GLB file")

    # image-to-3d
    image_parser = subparsers.add_parser("image-to-3d", help="Generate 3D from image")
    image_parser.add_argument("image", type=str, help="Path to input image file")
    image_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    image_parser.add_argument("--simplify", type=float, default=0.95, help="Mesh simplification ratio")
    image_parser.add_argument("--texture-size", type=int, default=1024, help="Texture resolution")
    image_parser.add_argument("--no-preprocess", action="store_true", help="Skip image preprocessing")
    image_parser.add_argument("--output", type=str, default="output_image.glb", help="Output GLB file")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "health":
        result = check_health(base_url)
        print(f"Status: {result['status']}")
        print(f"Text pipeline: {result['text_pipeline_state']}")
        print(f"Image pipeline: {result['image_pipeline_state']}")
        print(f"GPU: {result['gpu']}")
        if result.get("gpu_memory_allocated_gb") is not None:
            print(f"GPU memory allocated: {result['gpu_memory_allocated_gb']:.2f} GB")
        if result.get("gpu_memory_reserved_gb") is not None:
            print(f"GPU memory reserved: {result['gpu_memory_reserved_gb']:.2f} GB")
        if result.get("idle_timeout") is not None:
            print(f"Idle timeout: {result['idle_timeout']}s")

    elif args.command == "text-to-3d":
        response = text_to_3d(
            base_url=base_url,
            text=args.text,
            seed=args.seed,
            simplify=args.simplify,
            texture_size=args.texture_size,
        )
        if response["status"] == "success":
            print("Generation successful!")
            save_glb(response, args.output)
        else:
            print(f"Generation failed: {response.get('error')}")
            sys.exit(1)

    elif args.command == "image-to-3d":
        response = image_to_3d(
            base_url=base_url,
            image_path=args.image,
            seed=args.seed,
            simplify=args.simplify,
            texture_size=args.texture_size,
            preprocess_image=not args.no_preprocess,
        )
        if response["status"] == "success":
            print("Generation successful!")
            save_glb(response, args.output)
        else:
            print(f"Generation failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
