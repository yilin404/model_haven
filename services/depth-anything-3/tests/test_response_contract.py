"""Contract tests for the Depth Anything 3 processed RGB response."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_service_module() -> ModuleType:
    """Load the hyphenated service module directly from its file path.

    Returns:
        Imported Depth Anything 3 service module.

    Raises:
        RuntimeError: If Python cannot create an import specification.
    """
    service_dir = Path(__file__).resolve().parents[1]
    services_dir = service_dir.parent
    sys.path.insert(0, str(services_dir))
    sys.path.insert(0, str(service_dir))

    module_path = service_dir / "main.py"
    spec = importlib.util.spec_from_file_location("depth_anything_3_service", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load service module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def service_module() -> ModuleType:
    """Provide the imported service module to contract tests.

    Returns:
        Imported Depth Anything 3 service module.
    """
    return _load_service_module()


class _FakeModel:
    """Return one preset upstream prediction from ``inference``."""

    def __init__(self, prediction: SimpleNamespace) -> None:
        """Store the prediction returned by the fake model.

        Args:
            prediction: Upstream-like prediction namespace.
        """
        self._prediction = prediction

    def inference(self, **_: Any) -> SimpleNamespace:
        """Return the preset prediction without running a model.

        Args:
            **_: Ignored inference keyword arguments.

        Returns:
            The preset upstream-like prediction.
        """
        return self._prediction


def _make_prediction(
    *,
    processed_images: np.ndarray | None,
    depth_shape: tuple[int, int, int] = (1, 4, 5),
) -> SimpleNamespace:
    """Build a minimal upstream-like DA3 prediction.

    Args:
        processed_images: Processed RGB batch or None.
        depth_shape: Batched depth shape ``(N, H, W)``.

    Returns:
        Namespace exposing every field consumed by the service engine.
    """
    return SimpleNamespace(
        processed_images=processed_images,
        depth=np.ones(depth_shape, dtype=np.float32),
        conf=np.ones(depth_shape, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32)[:3].reshape(1, 3, 4),
        intrinsics=np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        is_metric=False,
        scale_factor=None,
    )


def _run_engine(service_module: ModuleType, prediction: SimpleNamespace) -> dict[str, Any]:
    """Run the engine implementation with CUDA selection and inference mocked.

    Args:
        service_module: Imported service module.
        prediction: Upstream-like prediction returned by the fake model.

    Returns:
        Serialized successful engine result.
    """
    engine = service_module.DepthAnythingV3Engine()
    engine.model = _FakeModel(prediction)
    engine.gpu_id = 0
    original_set_device = service_module.torch.cuda.set_device
    service_module.torch.cuda.set_device = lambda _: None
    try:
        return engine._run_inference_impl(
            images=[],
            extrinsics=None,
            intrinsics=None,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
    finally:
        service_module.torch.cuda.set_device = original_set_device


def test_engine_returns_lossless_processed_rgb(service_module: ModuleType) -> None:
    """The service returns upstream processed RGB through NDArrayData losslessly."""
    image_rgb = np.arange(1 * 4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3)

    result = _run_engine(
        service_module,
        _make_prediction(processed_images=image_rgb),
    )

    decoded = result["image_rgb"].to_array()
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, image_rgb)


def test_depth_response_keeps_only_success_contract_fields(
    service_module: ModuleType,
) -> None:
    """The response schema exposes image geometry fields without error payloads."""
    response_fields = set(service_module.DepthResponse.model_fields)

    assert "image_rgb" in response_fields
    assert "error" not in response_fields
    assert "error_type" not in response_fields


@pytest.mark.parametrize(
    "processed_images, match",
    [
        (None, "did not contain processed_images"),
        (np.zeros((1, 4, 5, 3), dtype=np.float32), "must use uint8"),
        (np.zeros((1, 3, 5, 3), dtype=np.uint8), "shapes are inconsistent"),
    ],
)
def test_engine_rejects_invalid_processed_rgb(
    service_module: ModuleType,
    processed_images: np.ndarray | None,
    match: str,
) -> None:
    """Missing, non-uint8, or spatially misaligned processed RGB is rejected."""
    with pytest.raises(RuntimeError, match=match):
        _run_engine(
            service_module,
            _make_prediction(processed_images=processed_images),
        )
