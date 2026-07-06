#!/usr/bin/env python3
"""
Example client for RAM++ FastAPI Server

Demonstrates how to call the RAM++ tagging server using the requests library.
"""

import argparse
import base64
import json
import sys
import time

import requests


def check_health(base_url: str) -> dict:
    """Check server health status."""
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def tag(
    base_url: str,
    image_path: str,
    threshold: float | None = None,
) -> dict:
    """
    Tag objects in an image using RAM++.

    Args:
        base_url: Server base URL
        image_path: Path to input image file
        threshold: Optional unified threshold override

    Returns:
        Response dictionary from server
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {"image": image_b64}
    if threshold is not None:
        payload["threshold"] = threshold

    print(f"Sending tagging request: {image_path}")
    start_time = time.time()

    resp = requests.post(f"{base_url}/tag", json=payload, timeout=120)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="RAM++ FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # tag
    tag_parser = subparsers.add_parser("tag", help="Tag objects in an image")
    tag_parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    tag_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional unified threshold override (0-1)",
    )
    tag_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the result as JSON",
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

    elif args.command == "tag":
        print("=" * 60)
        print("RAM++ FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Image: {args.image}")
        print(f"Threshold: {args.threshold}")
        print("=" * 60)

        response = tag(
            base_url=base_url,
            image_path=args.image,
            threshold=args.threshold,
        )
        if response["status"] == "success":
            print("Tagging successful!")
            tags = response.get("tags", [])
            tags_zh = response.get("tags_chinese", [])
            print(f"\nTags ({len(tags)}): {tags}")
            if tags_zh:
                print(f"中文标签 ({len(tags_zh)}): {tags_zh}")
            metadata = response.get("metadata", {})
            print(f"\nGeneration time: {metadata.get('generation_time', 'N/A')}s")
            print(f"Image size: {metadata.get('image_size', 'N/A')}")
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)
                print(f"\nSaved result to {args.output}")
        else:
            print(f"Tagging failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
