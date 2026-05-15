#!/usr/bin/env python3
"""
Example client for Particulate FastAPI Server

Demonstrates how to call the Particulate server for 3D object articulation
inference from a mesh file (.obj, .ply, or .glb).
"""

import argparse
import base64
import os
import sys
import time

import requests


def check_health(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def infer(
    base_url: str,
    mesh_path: str,
    up_dir: str = "-Z",
    num_points: int = 102400,
    min_part_confidence: float = 0.0,
    strict: bool = True,
    animation_frames: int = 50,
    export_urdf: bool = False,
    export_mjcf: bool = False,
) -> dict:
    with open(mesh_path, "rb") as f:
        mesh_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "mesh_file": mesh_b64,
        "filename": os.path.basename(mesh_path),
        "up_dir": up_dir,
        "num_points": num_points,
        "min_part_confidence": min_part_confidence,
        "strict": strict,
        "animation_frames": animation_frames,
        "export_urdf": export_urdf,
        "export_mjcf": export_mjcf,
    }

    print(f"Sending inference request: {mesh_path}")
    print(f"  up_dir={up_dir}, num_points={num_points}")

    start_time = time.time()
    resp = requests.post(f"{base_url}/infer", json=payload, timeout=600)
    resp.raise_for_status()
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def save_results(response: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if response["status"] != "success":
        print(f"Error: {response.get('error', 'Unknown error')}")
        return

    metadata = response.get("metadata", {})

    if response.get("segmented_glb"):
        path = os.path.join(output_dir, "segmented_parts.glb")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["segmented_glb"]))
        print(f"Saved: {path}")

    if response.get("animated_glb"):
        path = os.path.join(output_dir, "animated.glb")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["animated_glb"]))
        print(f"Saved: {path}")

    if response.get("urdf_zip"):
        path = os.path.join(output_dir, "particulate_urdf.zip")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["urdf_zip"]))
        print(f"Saved: {path}")

    if response.get("mjcf_zip"):
        path = os.path.join(output_dir, "particulate_mjcf.zip")
        with open(path, "wb") as f:
            f.write(base64.b64decode(response["mjcf_zip"]))
        print(f"Saved: {path}")

    print(f"Parts: {metadata.get('num_parts', 'N/A')}")
    print(f"Generation time: {metadata.get('generation_time', 'N/A')}s")


def main():
    parser = argparse.ArgumentParser(
        description="Particulate FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8006, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("health", help="Check server health")

    infer_parser = subparsers.add_parser("infer", help="Run articulation inference")
    infer_parser.add_argument("mesh", type=str, help="Path to input mesh (.obj, .ply, .glb)")
    infer_parser.add_argument(
        "--up-dir", type=str, default="-Z",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        help="Up direction of the input mesh",
    )
    infer_parser.add_argument("--num-points", type=int, default=102400, help="Number of points to sample")
    infer_parser.add_argument("--min-part-confidence", type=float, default=0.0, help="Minimum part confidence")
    infer_parser.add_argument("--no-strict", action="store_true", help="Disable strict refinement")
    infer_parser.add_argument("--animation-frames", type=int, default=50, help="Number of animation frames")
    infer_parser.add_argument("--export-urdf", action="store_true", help="Export URDF package")
    infer_parser.add_argument("--export-mjcf", action="store_true", help="Export MJCF package")
    infer_parser.add_argument("--output-dir", type=str, default="./particulate_output", help="Output directory")

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

    elif args.command == "infer":
        response = infer(
            base_url=base_url,
            mesh_path=args.mesh,
            up_dir=args.up_dir,
            num_points=args.num_points,
            min_part_confidence=args.min_part_confidence,
            strict=not args.no_strict,
            animation_frames=args.animation_frames,
            export_urdf=args.export_urdf,
            export_mjcf=args.export_mjcf,
        )
        if response["status"] == "success":
            print("Inference successful!")
            save_results(response, args.output_dir)
        else:
            print(f"Inference failed: {response.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
