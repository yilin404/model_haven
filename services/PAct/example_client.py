#!/usr/bin/env python3
"""
Example client for PAct FastAPI Server

Demonstrates how to call the PAct server for part articulation generation
from an RGB image (+ optional part segmentation mask).
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


def generate(
    base_url: str,
    image_path: str,
    mask_path: str | None = None,
    seed: int = 42,
    cfg_strength: float = 7.5,
) -> dict:
    """Generate articulation from an RGB image (+ optional mask)."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "seed": seed,
        "cfg_strength": cfg_strength,
    }

    if mask_path:
        with open(mask_path, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload["mask"] = mask_b64

    print(f"Sending generate request: {image_path}")
    if mask_path:
        print(f"  with mask: {mask_path}")
    else:
        print("  auto-segmentation enabled (no mask provided)")

    start_time = time.time()
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    resp.raise_for_status()
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_results(response: dict, output_dir: str) -> None:
    """Save generation results to files."""
    import os

    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    os.makedirs(output_dir, exist_ok=True)
    metadata = response.get("metadata", {})

    if response.get("articulation_video"):
        path = os.path.join(output_dir, "articulation.mp4")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["articulation_video"]))
        print(f"Saved: {path}")

    if response.get("exploded_video"):
        path = os.path.join(output_dir, "exploded.mp4")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["exploded_video"]))
        print(f"Saved: {path}")

    if response.get("urdf_zip"):
        path = os.path.join(output_dir, "pact_urdf_package.zip")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["urdf_zip"]))
        print(f"Saved: {path}")

    print(f"Seed: {metadata.get('seed', 'N/A')}")
    print(f"Parts: {metadata.get('num_parts', 'N/A')}")
    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")


def main():
    parser = argparse.ArgumentParser(
        description="PAct FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8005, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # generate
    gen_parser = subparsers.add_parser(
        "generate", help="Generate articulation from image"
    )
    gen_parser.add_argument("image", type=str, help="Path to input RGB image")
    gen_parser.add_argument(
        "--mask", type=str, default=None, help="Path to part segmentation mask"
    )
    gen_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    gen_parser.add_argument(
        "--cfg-strength", type=float, default=7.5, help="CFG strength"
    )
    gen_parser.add_argument(
        "--output-dir", type=str, default="./pact_output", help="Output directory"
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
        response = generate(
            base_url=base_url,
            image_path=args.image,
            mask_path=args.mask,
            seed=args.seed,
            cfg_strength=args.cfg_strength,
        )
        if response["status"] == "success":
            print("Generation successful!")
            save_results(response, args.output_dir)
        else:
            print(f"Generation failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
