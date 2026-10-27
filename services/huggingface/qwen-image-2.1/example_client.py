#!/usr/bin/env python3
"""
Example client for Qwen-Image-2.1 FastAPI Server

Demonstrates how to call the Qwen-Image-2.1 FastAPI server for
text-to-image generation and multi-reference image editing.
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
    height: int = 2048,
    width: int = 2048,
    num_inference_steps: int = 40,
    true_cfg_scale: float = 1.0,
    negative_prompt: str | None = None,
    num_images: int = 1,
) -> dict:
    """Generate image(s) from a text prompt."""
    payload = {
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": num_images,
    }

    print(f"Sending text-to-image request: '{prompt}'")
    start_time = time.time()

    resp = requests.post(f"{base_url}/text-to-image", json=payload, timeout=1800)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")
    return resp.json()


def image_editing(
    base_url: str,
    prompt: str,
    image_paths: list[str],
    seed: int | None = None,
    num_inference_steps: int = 40,
    true_cfg_scale: float = 1.0,
    negative_prompt: str | None = None,
    num_images: int = 1,
) -> dict:
    """Edit reference image(s) with a text instruction (1-10 images)."""
    images_b64 = []
    for path in image_paths:
        with open(path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    payload = {
        "prompt": prompt,
        "images": images_b64,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": num_images,
    }

    print(f"Sending image-editing request: '{prompt}' ({len(images_b64)} refs)")
    start_time = time.time()

    resp = requests.post(f"{base_url}/image-editing", json=payload, timeout=1800)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")
    return resp.json()


def save_images(response: dict, output_prefix: str = "output", show: bool = True) -> None:
    """Save images from response to files."""
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    metadata = response.get("metadata", {})
    images_b64 = response.get("images", [])

    for i, img_b64 in enumerate(images_b64):
        img_bytes = base64.b64decode(img_b64)
        filename = (
            f"{output_prefix}_{i}.png" if len(images_b64) > 1 else f"{output_prefix}.png"
        )

        with open(filename, "wb") as f:
            f.write(img_bytes)

        print(f"Saved: {filename} ({len(img_bytes)} bytes)")

    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
    print(f"Seed: {metadata.get('seed', 'N/A')}")
    print(f"Steps: {metadata.get('num_inference_steps', 'N/A')}")
    print(f"True CFG: {metadata.get('true_cfg_scale', 'N/A')}")

    if show and images_b64:
        try:
            from PIL import Image
            from io import BytesIO

            for img_b64 in images_b64:
                Image.open(BytesIO(base64.b64decode(img_b64))).show()
        except ImportError:
            print("Pillow not installed, skipping image display")


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="Qwen-Image-2.1 FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8010, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # text-to-image
    gen_parser = subparsers.add_parser("generate", help="Generate image from text")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument("--height", type=int, default=2048, help="Image height")
    gen_parser.add_argument("--width", type=int, default=2048, help="Image width")
    gen_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    gen_parser.add_argument(
        "--true-cfg",
        type=float,
        default=1.0,
        help="True CFG scale; >1 enables CFG and needs a negative prompt",
    )
    gen_parser.add_argument("--negative-prompt", type=str, default=None)
    gen_parser.add_argument("--num-images", type=int, default=1)
    gen_parser.add_argument("--output", type=str, default="output", help="Output prefix")
    gen_parser.add_argument("--no-show", action="store_true", help="Skip display")

    # image editing
    edit_parser = subparsers.add_parser("edit", help="Edit reference image(s)")
    edit_parser.add_argument(
        "--prompt", type=str, required=True, help="Editing instruction"
    )
    edit_parser.add_argument(
        "--image",
        type=str,
        action="append",
        required=True,
        help="Reference image file (repeat --image for up to 10 refs)",
    )
    edit_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    edit_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    edit_parser.add_argument("--true-cfg", type=float, default=1.0)
    edit_parser.add_argument("--negative-prompt", type=str, default=None)
    edit_parser.add_argument("--num-images", type=int, default=1)
    edit_parser.add_argument("--output", type=str, default="edited", help="Output prefix")
    edit_parser.add_argument("--no-show", action="store_true", help="Skip display")

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
        print("Qwen-Image-2.1 FastAPI Client")
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
            true_cfg_scale=args.true_cfg,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
        )
        if response["status"] == "success":
            print("Generation successful!")
            save_images(response, args.output, show=not args.no_show)
        else:
            print(f"Generation failed: {response.get('error')}")
            sys.exit(1)

    elif args.command == "edit":
        print("=" * 60)
        print("Qwen-Image-2.1 FastAPI Client (image editing)")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Prompt: {args.prompt}")
        print(f"References: {args.image}")
        print("=" * 60)

        response = image_editing(
            base_url=base_url,
            prompt=args.prompt,
            image_paths=args.image,
            seed=args.seed,
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
        )
        if response["status"] == "success":
            print("Editing successful!")
            save_images(response, args.output, show=not args.no_show)
        else:
            print(f"Editing failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
