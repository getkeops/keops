"""CPU Sparse reduction – provisional implementation.

We fallback to the existing *ranges* dense scheme on a K*1 densified
view, which is already O(K). This class just aliases `CpuReduc_ranges`.
"""

from keopscore.mapreduce.cpu.CpuReduc_ranges import CpuReduc_ranges


class CpuReducSparse(CpuReduc_ranges):
    """Temporary alias to the dense ranges reduction."""

    pass
