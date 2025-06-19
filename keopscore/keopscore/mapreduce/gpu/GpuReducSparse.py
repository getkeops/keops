"""GPU Sparse reduction – provisional implementation.

At the moment we simply *reuse* the dense 1-D scheme (`GpuReduc1D`).
Because the Python front-end densifies the (I,J) mask into a K×1 view
before invoking the map-reduce kernel, running the dense kernel on that
compact representation already gives an **O(K)** algorithm.

This stub therefore just subclasses `GpuReduc1D` so that KeOps can
compile and execute without hitting the previous *NotImplemented*
error. When a native sparse kernel is ready we can replace this file
with the real implementation without touching the high-level API.
"""

from keopscore.mapreduce.gpu.GpuReduc1D import GpuReduc1D


class GpuReducSparse(GpuReduc1D):
    """Temporary alias to the dense 1-D reduction.

    Parameters and constructor signature are identical to `GpuReduc1D`.
    """

    pass
