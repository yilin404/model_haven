#!/usr/bin/env python3
"""Example client for the P3-SAM point segmentation service.

Subcommands:
    health                       - probe GET /health
    segment                      - run the built-in two-sphere smoke fixture
                                   (expects exactly 2 parts, exit code 0 on pass)
"""

import argparse
import sys

import numpy as np
import requests


def build_two_sphere_fixture(points_per_sphere: int = 50_000, radius: float = 1.0,
                             separation: float = 4.0, noise_std: float = 0.01,
                             seed: int = 0):
    """Create two well-separated noisy spheres as the simplest segmentation case.

    Returns:
        Tuple of points (2*N, 3) and outward unit normals (2*N, 3); sphere A is
        centered at x=-separation/2, sphere B at x=+separation/2, and a boolean
        mask marking which rows belong to sphere B.
    """
    rng = np.random.default_rng(seed)

    def sphere(center_x: float) -> tuple[np.ndarray, np.ndarray]:
        dirs = rng.normal(size=(points_per_sphere, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = center_x + dirs * (radius + rng.normal(0.0, noise_std, (points_per_sphere, 1)))
        return pts.astype(np.float32), dirs.astype(np.float32)

    pts_a, nrm_a = sphere(-separation / 2)
    pts_b, nrm_b = sphere(+separation / 2)
    points = np.concatenate([pts_a, pts_b], axis=0)
    normals = np.concatenate([nrm_a, nrm_b], axis=0)
    is_b = np.zeros(len(points), dtype=bool)
    is_b[points_per_sphere:] = True
    return points, normals, is_b


def encode_array(array: np.ndarray) -> dict:
    """Match the service NDArrayData wire format ({data, shape, dtype})."""
    import base64

    contiguous = np.ascontiguousarray(array)
    return {
        "data": base64.b64encode(contiguous.tobytes(order="C")).decode("utf-8"),
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
    }


def cmd_health(base_url: str) -> int:
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        print(response.json())
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"health check failed: {exc}")
        return 1


def cmd_segment(base_url: str, prompt_num: int, prompt_bs: int, seed: int) -> int:
    points, normals, is_b = build_two_sphere_fixture()
    payload = {
        "points": encode_array(points),
        "normals": encode_array(normals),
        "prompt_num": prompt_num,
        "prompt_bs": prompt_bs,
        "seed": seed,
    }
    print(f"POST {base_url}/segment with {len(points)} points ...")
    response = requests.post(f"{base_url}/segment", json=payload, timeout=600)
    response.raise_for_status()
    body = response.json()
    print(f"status={body['status']} num_parts={body['num_parts']}")
    print(f"metadata={body['metadata']}")

    labels = np.frombuffer(
        __import__("base64").b64decode(body["labels"]["data"]),
        dtype=np.dtype(body["labels"]["dtype"]),
    )
    label_a = labels[~is_b]
    label_b = labels[is_b]

    # 判定：两个球各自内部标签一致、且两球标签不同、无未分配点
    ok = (
        body["num_parts"] == 2
        and len(set(label_a.tolist())) == 1
        and len(set(label_b.tolist())) == 1
        and label_a[0] != label_b[0]
        and (labels == -1).sum() == 0
    )
    print(f"labels sphere A={label_a[0]}, sphere B={label_b[0]} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    seg = sub.add_parser("segment")
    seg.add_argument("--prompt-num", type=int, default=400)
    seg.add_argument("--prompt-bs", type=int, default=8)
    seg.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    if args.command == "health":
        sys.exit(cmd_health(base_url))
    sys.exit(cmd_segment(base_url, args.prompt_num, args.prompt_bs, args.seed))


if __name__ == "__main__":
    main()
