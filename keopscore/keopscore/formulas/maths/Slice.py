from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_array, VectCopy, c_zero_float
from keopscore.utils.misc_utils import KeOps_Error


class Slice(Operation):
    """A *view-like* slicer that works at compile-time.

    Parameters
    ----------
    f : Operation
        Input KeOps expression.
    start : int
        Starting index of the window on the selected axis.
    length : int
        Number of consecutive indices kept.
    step : int
        Stride between kept indices. Only ``step == 1`` is supported for now.
    axis : int
        0 → slice on the *i*-indexed dimension,
        1 → slice on the *j*-indexed dimension,
        2 → slice on the *feature* dimension (same as existing ``Extract`` op).

    Notes
    -----
    • For ``axis == 2`` this is equivalent to :class:`keopscore.formulas.maths.Extract`.
    • Support for ``axis`` in {0,1} requires shifting the loop index in the
      generated kernel. This first version focuses on the feature-axis use-case
      and raises for the others – they will be implemented in the next
      iteration.
    """

    string_id = "Slice"
    linearity_type = "all"

    def __init__(self, f, start=None, length=None, step=1, axis=2, params=None):
        # init through params for compatibility with the base class
        if start is None:
            # params expected as a 4-uple (start, length, step, axis)
            start, length, step, axis = params
        if step != 1:
            KeOps_Error("Slice currently supports only step == 1.")
        if axis not in (0, 1, 2):
            KeOps_Error("Slice axis must be 0, 1 or 2.")
        if axis == 2 and (start < 0 or length < 1 or start + length > f.dim):
            KeOps_Error("Slice indices out of bounds for feature axis.")
        super().__init__(f, params=(start, length, step, axis))
        self.start = start
        self.length = length
        self.step = step
        self.axis = axis
        # Output dimension identical to the child:
        self.dim = f.dim

    # ---------------------------------------------------------------------
    # Low-level code generation helpers
    # ---------------------------------------------------------------------

    def Op(self, out, table, arg0):
        """Generates C++ code evaluating the slice.

        Current implementation only handles the *feature-axis* case by a simple
        pointer offset, using the same strategy as the existing *Extract* op.
        """
        if self.axis == 2:
            # Feature-axis: nothing fancy – we simply offset the pointer that
            # holds the child vector.
            v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
            return VectCopy(out, v)
        else:
            KeOps_Error("Slice along axis 0 or 1 not implemented yet in this version.")

    # ------------------------------------------------------------------
    # Differential – reuse the same logic as Extract / ExtractT.
    # ------------------------------------------------------------------
    def DiffT(self, v, gradin):
        """Backward rule – for the feature axis only (axis == 2)."""
        if self.axis != 2:
            KeOps_Error("Backward of Slice along axis 0/1 not implemented yet.")
        # For axis == 2 we can rely on ExtractT to scatter the gradient.
        from keopscore.formulas.maths.ExtractT import ExtractT

        f = self.children[0]
        return f.DiffT(v, ExtractT(gradin, self.start, self.length))

    enable_test = True
    nargs = 1
    test_argdims = [10]
    test_params = [3, 5, 1, 2]

    # Torch equivalent (only axis == 2 for now)
    @staticmethod
    def torch_op():
        import torch

        def _torch_op(x, s, l, stp, ax):
            if ax != 2:
                raise NotImplementedError("Torch op only for feature axis.")
            return x[..., s : (s + l)]

        return _torch_op
