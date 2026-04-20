#!/usr/bin/env python3
"""
Example client for SDXL FastAPI Server

Demonstrates how to call the SDXL FastAPI server for text-to-image
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


def text_to_image(
    base_url: str,
    prompt: str,
    seed: int | None = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 40,
    guidance_scale: float = 5.0,
    negative_prompt: str | None = None,
    num_images: int = 1,
) -> dict:
    """
    Generate image(s) from a text prompt.

    Args:
        base_url: Server base URL
        prompt: Text prompt for image generation
        seed: Random seed
        height: Image height in pixels
        width: Image width in pixels
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale (CFG)
        negative_prompt: Negative prompt
        num_images: Number of images to generate

    Returns:
        Response dictionary from server
    """
    payload = {
        "prompt": prompt,
        "seed": seed,
        "options": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "num_images_per_prompt": num_images,
        },
    }

    print(f"Sending text-to-image request: '{prompt}'")
    start_time = time.time()

    resp = requests.post(f"{base_url}/text-to-image", json=payload, timeout=300)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_images(response: dict, output_prefix: str = "output", show: bool = True) -> None:
    """
    Save images from response to files.

    Args:
        response: Response dictionary from server
        output_prefix: Output file prefix (e.g. "output" -> output_0.png, output_1.png)
        show: Whether to display images
    """
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    metadata = response.get("metadata", {})
    images_b64 = response.get("images", [])

    for i, img_b64 in enumerate(images_b64):
        img_bytes = base64.b64decode(img_b64)
        filename = f"{output_prefix}_{i}.png" if len(images_b64) > 1 else f"{output_prefix}.png"

        with open(filename, "wb") as f:
            f.write(img_bytes)

        print(f"Saved: {filename} ({len(img_bytes)} bytes)")

    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
    print(f"Seed: {metadata.get('seed', 'N/A')}")
    print(f"Steps: {metadata.get('num_inference_steps', 'N/A')}")
    print(f"Guidance: {metadata.get('guidance_scale', 'N/A')}")

    if show and images_b64:
        try:
            from PIL import Image
            from io import BytesIO

            for i, img_b64 in enumerate(images_b64):
                img = Image.open(BytesIO(base64.b64decode(img_b64)))
                img.show()
        except ImportError:
            print("Pillow not installed, skipping image display")


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="SDXL FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # text-to-image
    gen_parser = subparsers.add_parser("generate", help="Generate image from text")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument("--height", type=int, default=1024, help="Image height")
    gen_parser.add_argument("--width", type=int, default=1024, help="Image width")
    gen_parser.add_argument("--steps", type=int, default=40, help="Number of inference steps")
    gen_parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale")
    gen_parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt")
    gen_parser.add_argument("--num-images", type=int, default=1, help="Number of images")
    gen_parser.add_argument("--output", type=str, default="output", help="Output file prefix")

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
        print("SDXL FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Prompt: {args.prompt}")
        print(f"Size: {args.width}x{args.height}")
        print(f"Steps: {args.steps}")
        print("=" * 60)

        response = text_to_image(
            base_url=base_url,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
        )
        if response["status"] == "success":
            print("Generation successful!")
            save_images(response, args.output)
        else:
            print(f"Generation failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
