#!/usr/bin/env python3
"""
Example client for SAM3 FastAPI Server

Demonstrates how to call the SAM3 FastAPI server for text-prompted
image segmentation using the requests library.
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
    text_prompt: str,
    confidence_threshold: float | None = None,
) -> dict:
    """
    Segment objects in an image using a text prompt.

    Args:
        base_url: Server base URL
        image_path: Path to input image file
        text_prompt: Text prompt for segmentation
        confidence_threshold: Confidence threshold override

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "text_prompt": text_prompt,
    }
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold

    print(f"Sending segmentation request: '{text_prompt}'")
    start_time = time.time()

    resp = requests.post(f"{base_url}/segment", json=payload, timeout=120)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_masks(response: dict, output_dir: str) -> None:
    """
    Save segmentation masks from response to files.

    Args:
        response: Response dictionary from server
        output_dir: Directory to save mask PNGs
    """
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    os.makedirs(output_dir, exist_ok=True)

    metadata = response.get("metadata", {})
    masks_b64 = response.get("masks", [])
    boxes = response.get("boxes", [])
    scores = response.get("scores", [])

    for i, mask_b64 in enumerate(masks_b64):
        mask_bytes = base64.b64decode(mask_b64)
        path = os.path.join(output_dir, f"mask_{i:03d}.png")
        with open(path, "wb") as f:
            f.write(mask_bytes)
        score = scores[i] if i < len(scores) else "N/A"
        box = boxes[i] if i < len(boxes) else "N/A"
        print(f"  mask_{i:03d}.png (score: {score:.4f}, box: {box})")

    print(f"\nSaved {len(masks_b64)} masks to {output_dir}/")
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
        "--text", type=str, required=True, help="Text prompt for segmentation"
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
        print(f"Model state: {result.get('model_state', 'N/A')}")
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
        print(f"Prompt: {args.text}")
        print(f"Output: {args.output_dir}/")
        print("=" * 60)

        response = segment(
            base_url=base_url,
            image_path=args.image,
            text_prompt=args.text,
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
