#!/usr/bin/env python3
"""
Example client for SAM3 FastAPI Server

Demonstrates how to call the SAM3 FastAPI server for text-prompted
image segmentation using the requests library. Supports one or more
text prompts per request.
"""

import argparse
import base64
import os
import sys
import time

import requests


def check_health(base_url: str) -> dict:
    """Check server health status."""
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def segment(
    base_url: str,
    image_path: str,
    text_prompts: list[str],
    confidence_threshold: float | None = None,
) -> dict:
    """
    Segment objects in an image using one or more text prompts.

    Args:
        base_url: Server base URL
        image_path: Path to input image file
        text_prompts: One or more text prompts for segmentation
        confidence_threshold: Confidence threshold override

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "text_prompts": text_prompts,
    }
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold

    print(
        f"Sending segmentation request with {len(text_prompts)} prompt(s): "
        f"{text_prompts}"
    )
    start_time = time.time()

    resp = requests.post(f"{base_url}/segment", json=payload, timeout=120)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_masks(response: dict, output_dir: str) -> None:
    """
    Save segmentation masks from response to files.

    The response is grouped per prompt; masks for prompt p are saved as
    output_dir/prompt{p}_mask{m:03d}.png.

    Args:
        response: Response dictionary from server
        output_dir: Directory to save mask PNGs
    """
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    os.makedirs(output_dir, exist_ok=True)

    results = response.get("results", [])
    metadata = response.get("metadata", {})
    total_masks = 0

    for p, item in enumerate(results):
        prompt = item.get("text_prompt", f"prompt{p}")
        masks_b64 = item.get("masks", [])
        boxes = item.get("boxes", [])
        scores = item.get("scores", [])
        num = item.get("num_objects", len(masks_b64))

        print(f"\n[prompt {p}] '{prompt}': {num} object(s)")
        for i, mask_b64 in enumerate(masks_b64):
            mask_bytes = base64.b64decode(mask_b64)
            path = os.path.join(output_dir, f"prompt{p}_mask{i:03d}.png")
            with open(path, "wb") as f:
                f.write(mask_bytes)
            score_str = f"{scores[i]:.4f}" if i < len(scores) else "N/A"
            box = boxes[i] if i < len(boxes) else None
            print(f"  prompt{p}_mask{i:03d}.png (score: {score_str}, box: {box})")
        total_masks += len(masks_b64)

    print(f"\nSaved {total_masks} mask(s) to {output_dir}/")
    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")
    print(f"Image size: {metadata.get('image_size', 'N/A')}")
    print(f"Confidence threshold: {metadata.get('confidence_threshold', 'N/A')}")


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="SAM3 FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8004, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # segment
    seg_parser = subparsers.add_parser(
        "segment", help="Segment objects in an image"
    )
    seg_parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    seg_parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        required=True,
        help="One or more text prompts for segmentation (e.g. --text cat dog car)",
    )
    seg_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold (0-1)",
    )
    seg_parser.add_argument(
        "--output-dir",
        type=str,
        default="output_masks",
        help="Directory to save mask PNGs",
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

    elif args.command == "segment":
        print("=" * 60)
        print("SAM3 FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Image: {args.image}")
        print(f"Prompts ({len(args.text)}): {args.text}")
        print(f"Output: {args.output_dir}/")
        print("=" * 60)

        response = segment(
            base_url=base_url,
            image_path=args.image,
            text_prompts=args.text,
            confidence_threshold=args.confidence_threshold,
        )
        if response["status"] == "success":
            print("Segmentation successful!")
            save_masks(response, args.output_dir)
        else:
            print(f"Segmentation failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
