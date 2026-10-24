"""Type aliases shared across paramnormal."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, TypeAlias

import numpy
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[numpy.float64]

# Anything numpy can coerce into an ndarray: scalars, (nested) sequences,
# and ndarrays (including masked arrays and other ndarray subclasses).
NDArrayLike: TypeAlias = (
    float
    | int
    | numpy.number
    | NDArray[numpy.generic]
    | Sequence[float | int | numpy.number]
    | Sequence["NDArrayLike"]
)


class ParamsLike(Protocol):
    """The namedtuples returned by the ``fit`` methods."""

    @property
    def _fields(self) -> tuple[str, ...]: ...

    def __iter__(self) -> Iterator[float]: ...


__all__ = ["FloatArray", "NDArrayLike", "ParamsLike"]
