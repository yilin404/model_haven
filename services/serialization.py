"""Shared wire representations for binary model data."""

from __future__ import annotations

import base64
import math
from binascii import Error as BinasciiError
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator


class NDArrayData(BaseModel):
    """JSON-safe, lossless representation of a C-contiguous NumPy array."""

    data: str = Field(description="Base64-encoded C-order array bytes")
    shape: list[int] = Field(min_length=1, description="Array shape")
    dtype: str = Field(description="NumPy dtype name")

    @model_validator(mode="after")
    def validate_payload(self) -> "NDArrayData":
        if any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            for size in self.shape
        ):
            raise ValueError("shape entries must be non-negative integers")

        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported NumPy dtype: {self.dtype}") from exc

        if dtype.hasobject:
            raise ValueError("object arrays are not supported")

        try:
            raw = base64.b64decode(self.data, validate=True)
        except (BinasciiError, ValueError) as exc:
            raise ValueError("data must be valid base64") from exc

        expected_nbytes = math.prod(self.shape) * dtype.itemsize
        if len(raw) != expected_nbytes:
            raise ValueError(
                "array byte size mismatch: "
                f"expected {expected_nbytes}, got {len(raw)}"
            )
        return self

    @classmethod
    def from_array(cls, array: np.ndarray | Any) -> "NDArrayData":
        """Encode an array without losing its dtype or shape."""
        contiguous = np.ascontiguousarray(np.asarray(array))
        if contiguous.dtype.hasobject:
            raise ValueError("object arrays are not supported")
        return cls(
            data=base64.b64encode(contiguous.tobytes(order="C")).decode("utf-8"),
            shape=list(contiguous.shape),
            dtype=str(contiguous.dtype),
        )

    def to_array(self, *, copy: bool = False) -> np.ndarray:
        """Decode the payload to a NumPy array."""
        raw = base64.b64decode(self.data, validate=True)
        array = np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(self.shape)
        return array.copy() if copy else array
