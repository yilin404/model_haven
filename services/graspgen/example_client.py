#!/usr/bin/env python3
"""
Example client for GraspGen FastAPI Server

Demonstrates how to call the GraspGen FastAPI server for 6-DOF grasp
generation from point clouds or meshes using the requests library.
"""

import argparse
import base64
import os
import sys
import time

import numpy as np
import requests


def load_point_cloud_from_mesh(mesh_file: str, scale: float = 1.0, num_points: int = 2000) -> np.ndarray:
    """Load a mesh, scale it, sample surface points, and center them."""
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required for mesh loading. Install with: pip install trimesh")

    mesh = trimesh.load(mesh_file)

    # Handle Scene objects (e.g., from GLB/GLTF files)
    if isinstance(mesh, trimesh.Scene):
        # Merge all meshes in the scene
        meshes = []
        for geometry in mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        if not meshes:
            raise ValueError(f"No valid meshes found in scene: {mesh_file}")
        mesh = trimesh.util.concatenate(meshes)

    mesh.apply_scale(scale)
    xyz, _ = trimesh.sample.sample_surface(mesh, num_points)
    xyz = np.array(xyz, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    return xyz


def load_point_cloud_from_file(pcd_file: str) -> np.ndarray:
    """Load a point cloud from .pcd, .ply, .xyz, or .npy file and center it."""
    ext = pcd_file.rsplit(".", 1)[-1].lower()

    if ext == "npy":
        xyz = np.load(pcd_file).astype(np.float32)
    elif ext == "xyz":
        xyz = np.loadtxt(pcd_file, dtype=np.float32)
    elif ext == "pcd":
        xyz = _read_pcd_ascii(pcd_file)
    elif ext == "ply":
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for PLY loading. Install with: pip install trimesh")
        cloud = trimesh.load(pcd_file)
        xyz = np.array(cloud.vertices, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: .{ext}")

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Expected (N, 3+) array, got shape {xyz.shape}")
    xyz = xyz[:, :3]
    xyz -= xyz.mean(axis=0)
    return xyz


def _read_pcd_ascii(path: str) -> np.ndarray:
    """Minimal ASCII PCD reader (FIELDS x y z)."""
    points = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            if in_data:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            elif line.strip().startswith("DATA"):
                in_data = True
    return np.array(points, dtype=np.float32)


def encode_numpy_array(array: np.ndarray) -> dict:
    """Encode numpy array to dict format for JSON serialization."""
    return {
        "data": base64.b64encode(array.tobytes()).decode("utf-8"),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def grasp_from_numpy(
    base_url: str,
    point_cloud: np.ndarray,
    num_grasps: int = 200,
    topk_num_grasps: int = -1,
    grasp_threshold: float = -1.0,
    min_grasps: int = 40,
    max_tries: int = 6,
    remove_outliers: bool = True,
) -> dict:
    """
    Generate grasps from a numpy point cloud array via JSON request.

    Args:
        base_url: Server base URL
        point_cloud: (N, 3) float32 point cloud
        num_grasps: Number of grasps to sample
        topk_num_grasps: Return only top-k grasps (-1 = use threshold)
        grasp_threshold: Minimum confidence threshold (-1 = use topk)
        min_grasps: Minimum grasps required
        max_tries: Maximum retry attempts
        remove_outliers: Remove outliers before inference

    Returns:
        Response dictionary from server
    """
    payload = {
        "point_cloud": encode_numpy_array(point_cloud),
        "num_grasps": num_grasps,
        "topk_num_grasps": topk_num_grasps,
        "grasp_threshold": grasp_threshold,
        "min_grasps": min_grasps,
        "max_tries": max_tries,
        "remove_outliers": remove_outliers,
    }

    print(f"Sending grasp generation request ({len(point_cloud)} points)")
    start_time = time.time()

    resp = requests.post(f"{base_url}/generate", json=payload, timeout=300)
    resp.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")

    return resp.json()


def check_health(base_url: str) -> dict:
    """Check server health status."""
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def grasp_from_mesh(
    base_url: str,
    mesh_file: str,
    mesh_scale: float = 1.0,
    num_sample_points: int = 2000,
    num_grasps: int = 200,
    topk_num_grasps: int = -1,
    grasp_threshold: float = -1.0,
    min_grasps: int = 40,
    max_tries: int = 6,
    remove_outliers: bool = True,
) -> dict:
    """
    Generate grasps from a mesh file.

    Loads the mesh, samples surface points, and sends to the server.

    Args:
        base_url: Server base URL
        mesh_file: Path to mesh file (.obj/.stl/.ply)
        mesh_scale: Scale factor for mesh
        num_sample_points: Number of points to sample from mesh surface
        num_grasps: Number of grasps to sample
        topk_num_grasps: Return only top-k grasps
        grasp_threshold: Minimum confidence threshold
        min_grasps: Minimum grasps required
        max_tries: Maximum retry attempts
        remove_outliers: Remove outliers before inference

    Returns:
        Response dictionary from server
    """
    point_cloud = load_point_cloud_from_mesh(mesh_file, mesh_scale, num_sample_points)
    print(f"Sampled {len(point_cloud)} points from mesh surface")

    return grasp_from_numpy(
        base_url,
        point_cloud,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        grasp_threshold=grasp_threshold,
        min_grasps=min_grasps,
        max_tries=max_tries,
        remove_outliers=remove_outliers,
    )


def decode_response(response: dict) -> tuple:
    """
    Decode the response from the server.

    Args:
        response: Response dictionary from server

    Returns:
        Tuple of (grasps, confidences, metadata) or (None, None, error_msg)
    """
    if response.get("status") != "success":
        error = response.get("error", "Unknown error")
        error_type = response.get("error_type", "Unknown")
        return None, None, f"{error_type}: {error}"

    # Decode grasps
    grasps_data = response.get("grasps")
    if grasps_data is None:
        return None, None, "No grasp data in response"
    grasps_bytes = base64.b64decode(grasps_data["data"])
    grasps = np.frombuffer(grasps_bytes, dtype=np.float32).reshape(grasps_data["shape"])

    # Decode confidences
    conf_data = response.get("confidences", {})
    conf_bytes = base64.b64decode(conf_data["data"])
    confidences = np.frombuffer(conf_bytes, dtype=np.float32).reshape(conf_data["shape"])

    metadata = response.get("metadata", {})

    return grasps, confidences, metadata


def save_results(grasps: np.ndarray, confidences: np.ndarray, output_file: str) -> None:
    """
    Save grasp results to a file.

    Args:
        grasps: (M, 4, 4) grasp poses
        confidences: (M,) confidence scores
        output_file: Output file path (.npy or .npz)
    """
    if output_file.endswith(".npz"):
        np.savez(output_file, grasps=grasps, confidences=confidences)
    else:
        np.save(output_file, {"grasps": grasps, "confidences": confidences})
    print(f"Saved results to: {output_file}")


def visualize_results(
    point_cloud: np.ndarray,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str = "robotiq_2f_140",
    viser_port: int = 8080,
):
    """Visualize the point cloud and grasps using viser."""
    try:
        from grasp_gen.utils.viser_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_grasp,
            visualize_pointcloud,
        )
    except ImportError:
        print("Warning: viser visualization requires grasp_gen package.")
        print("Install with: pip install -e ../../deps/graspgen")
        return

    vis = create_visualizer(port=viser_port)

    pc_color = np.ones((len(point_cloud), 3), dtype=np.uint8) * 200
    visualize_pointcloud(vis, "point_cloud", point_cloud, pc_color, size=0.003)

    scores = get_color_from_score(confidences, use_255_scale=True)
    for i, grasp in enumerate(grasps):
        grasp = grasp.copy()
        grasp[3, 3] = 1.0
        visualize_grasp(
            vis,
            f"grasps/{i:03d}",
            grasp,
            color=scores[i],
            gripper_name=gripper_name,
            linewidth=0.6,
        )

    print(f"\nViser visualization running at http://localhost:{viser_port}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    """Main entry point for example client."""
    parser = argparse.ArgumentParser(
        description="GraspGen FastAPI Server Example Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # health check
    subparsers.add_parser("health", help="Check server health")

    # generate grasps
    gen_parser = subparsers.add_parser("generate", help="Generate grasps from point cloud or mesh")

    input_group = gen_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--mesh-file", type=str, help="Path to a mesh file (.obj / .stl / .ply)"
    )
    input_group.add_argument(
        "--pcd-file", type=str, help="Path to a point cloud file (.npy / .npz / .ply / .pcd / .xyz)"
    )

    gen_parser.add_argument(
        "--mesh-scale", type=float, default=1.0, help="Scale factor for mesh"
    )
    gen_parser.add_argument(
        "--num-sample-points",
        type=int,
        default=2000,
        help="Number of points to sample from mesh",
    )
    gen_parser.add_argument(
        "--num-grasps", type=int, default=200, help="Number of grasps to generate"
    )
    gen_parser.add_argument(
        "--topk-num-grasps",
        type=int,
        default=-1,
        help="Return only top-k grasps (-1 for all)",
    )
    gen_parser.add_argument(
        "--grasp-threshold",
        type=float,
        default=-1.0,
        help="Minimum confidence threshold (-1 to use topk)",
    )
    gen_parser.add_argument(
        "--min-grasps",
        type=int,
        default=40,
        help="Minimum grasps required before stopping retries",
    )
    gen_parser.add_argument(
        "--max-tries",
        type=int,
        default=6,
        help="Maximum inference retry attempts",
    )
    gen_parser.add_argument(
        "--no-remove-outliers",
        action="store_true",
        help="Disable outlier removal",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (.npz)",
    )
    gen_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results with viser (requires grasp_gen package)",
    )
    gen_parser.add_argument(
        "--viser-port",
        type=int,
        default=8080,
        help="Port for viser visualization server",
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

    elif args.command == "generate":
        print("=" * 60)
        print("GraspGen FastAPI Client")
        print("=" * 60)
        print(f"Server: {base_url}")
        print(f"Num grasps: {args.num_grasps}")
        print(f"Top-k: {args.topk_num_grasps}")
        print(f"Threshold: {args.grasp_threshold}")
        print("=" * 60)

        if args.mesh_file:
            input_source = args.mesh_file
            print(f"Loading mesh: {args.mesh_file} (scale={args.mesh_scale})")
            response = grasp_from_mesh(
                base_url=base_url,
                mesh_file=args.mesh_file,
                mesh_scale=args.mesh_scale,
                num_sample_points=args.num_sample_points,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.topk_num_grasps,
                grasp_threshold=args.grasp_threshold,
                min_grasps=args.min_grasps,
                max_tries=args.max_tries,
                remove_outliers=not args.no_remove_outliers,
            )
        else:
            input_source = args.pcd_file
            print(f"Loading point cloud: {args.pcd_file}")
            point_cloud = load_point_cloud_from_file(args.pcd_file)
            print(f"Loaded {len(point_cloud)} points from file")

            response = grasp_from_numpy(
                base_url=base_url,
                point_cloud=point_cloud,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.topk_num_grasps,
                grasp_threshold=args.grasp_threshold,
                min_grasps=args.min_grasps,
                max_tries=args.max_tries,
                remove_outliers=not args.no_remove_outliers,
            )

        # Decode response
        grasps, confidences, result = decode_response(response)

        if grasps is None:
            print(f"\nError: {result}")
            sys.exit(1)

        # Print results
        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        print(f"Input: {input_source}")
        print(f"Grasps returned: {len(grasps)}")
        if len(confidences) > 0:
            print(f"Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
            print(f"Best grasp confidence: {confidences[0]:.4f}")
        if isinstance(result, dict):
            print(f"Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"Gripper: {result.get('gripper_name', 'N/A')}")
        print(f"{'=' * 60}")

        # Save results if requested
        if args.output:
            save_results(grasps, confidences, args.output)

        # Visualize with viser if requested
        if args.visualize and len(grasps) > 0:
            if isinstance(result, dict):
                gripper_name = result.get("gripper_name", "robotiq_2f_140")
            # Reload point cloud for visualization (mesh needs resampling)
            if args.mesh_file:
                pc = load_point_cloud_from_mesh(
                    args.mesh_file, args.mesh_scale, args.num_sample_points
                )
            else:
                pc = point_cloud
            visualize_results(
                pc, grasps, confidences, gripper_name, args.viser_port
            )


if __name__ == "__main__":
    main()
