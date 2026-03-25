#!/usr/bin/env python3
"""
Example client for TRELLIS ZMQ Server

Demonstrates how to connect to the TRELLIS ZMQ server and request 3D model
generation from text prompts.
"""

import argparse
import base64
import sys
import time
import zmq

import trimesh


def send_request(
    host: str = "localhost",
    port: int = 5555,
    text: str = "A modern chair with wooden legs",
    seed: int = 42,
    options: dict = None,
) -> dict:
    """
    Send a generation request to the TRELLIS ZMQ server.

    Args:
        host: Server host (default: "localhost")
        port: Server port (default: 5555)
        text: Text prompt for 3D generation
        seed: Random seed for generation
        options: Optional generation parameters

    Returns:
        Response dictionary from server
    """
    if options is None:
        options = {}

    # Create request
    request = {"text": text, "seed": seed, "options": options}

    print(f"Connecting to tcp://{host}:{port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 300000)  # 5 minute timeout
    socket.connect(f"tcp://{host}:{port}")

    try:
        print(f"Sending request: {text}")
        start_time = time.time()

        socket.send_json(request)
        print("Request sent, waiting for response...")

        response = socket.recv_json()

        elapsed = time.time() - start_time
        print(f"Response received in {elapsed:.2f}s")

        return response

    finally:
        socket.close()
        context.term()


def save_glb(response: dict, output_file: str, show: bool = True) -> None:
    """
    Save GLB data from response to file.

    Args:
        response: Response dictionary from server
        output_file: Output file path
        show: Whether to visualize the mesh using trimesh (default: True)
    """
    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    # Decode base64
    glb_data = base64.b64decode(response["glb_data"])

    # Save to file
    with open(output_file, "wb") as f:
        f.write(glb_data)

    print(f"Saved GLB file: {output_file}")
    print(f"File size: {len(glb_data)} bytes")
    print(f"Metadata: {response.get('metadata', {})}")

    # Visualize using trimesh
    if show:
        mesh = trimesh.load(output_file, file_type="glb")
        mesh.show()


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="TRELLIS ZMQ Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")

    parser.add_argument("--port", type=int, default=5555, help="Server port")

    parser.add_argument(
        "--text",
        type=str,
        default="A modern chair with wooden legs",
        help="Text prompt for 3D generation",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--output", type=str, default="output.glb", help="Output GLB file"
    )

    parser.add_argument(
        "--simplify", type=float, default=0.95, help="Mesh simplification ratio (0-1)"
    )

    parser.add_argument(
        "--texture-size", type=int, default=1024, help="Texture resolution"
    )

    args = parser.parse_args()

    # Build options
    options = {"simplify": args.simplify, "texture_size": args.texture_size}

    print("=" * 60)
    print("TRELLIS ZMQ Client")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Prompt: {args.text}")
    print(f"Seed: {args.seed}")
    print(f"Options: {options}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Send request
    response = send_request(
        host=args.host, port=args.port, text=args.text, seed=args.seed, options=options
    )

    # Handle response
    if response["status"] == "success":
        print("\n✓ Generation successful!")
        save_glb(response, args.output)
    else:
        print(f"\n✗ Generation failed: {response.get('error')}")
        print(f"Error type: {response.get('error_type')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
