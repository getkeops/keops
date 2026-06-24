import re
import unittest
import numpy as np
import pykeops.config

from pykeops.test import assert_np_allclose, assert_torch_allclose


class ShapeNumpyTestCase(unittest.TestCase):

    def setUp(self):
        self.param = np.array([0.4], dtype=np.float32)
        self.param_scalar = np.array(0.4, dtype=np.float32)

        self.x = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)  # shape (3,1)
        self.x_vect = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # shape (3,)

    def test_numpy_pm1_shape_regression(self):
        from pykeops.numpy import Genred

        out = Genred("param", ["param=Pm(1)"], axis=1)(self.param, backend="auto")
        ref = np.full_like(out, self.param[0])
        assert_np_allclose(out, ref, atol=1e-7)

    def test_numpy_pm1_scalar_rejected(self):
        from pykeops.numpy import Genred

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "[pyKeOps] Error: Pm argument #0 requires at least 1 dimension (got a scalar)."
            ),
        ):
            Genred("param", ["param=Pm(1)"], axis=1)(self.param_scalar, backend="auto")

    def test_numpy_vj_single_dim_formula_x(self):
        from pykeops.numpy import Genred

        out = Genred("x", ["x=Vj(1)"], axis=1)(self.x, backend="auto")
        ref = np.sum(self.x, axis=0, keepdims=True)
        assert_np_allclose(out, ref, atol=1e-7)

    def test_numpy_vj_single_dim_formula_x_1d_rejected_python(self):
        from pykeops.numpy import Genred

        # The error is raised in the Python code (parse_type.py), before even calling the C++ code
        with self.assertRaisesRegex(IndexError, "tuple index out of range"):
            Genred("y", ["y=Vj(1)"], axis=1)(self.x_vect, backend="auto")

    def test_numpy_vj_single_dim_formula_x_1d_rejected_cpp(self):
        from pykeops.numpy import Genred

        # The error is raised in the C++ code (LoadKeOps_cpp.py or pykeops_nvrtc.py), after the Python code has accepted the input shape but before launching the kernel
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "[pyKeOps] Error: Vj argument #2 requires at least 2 dimensions, got 1."
            ),
        ):
            Genred("y + z + l", ["y=Vj(1)", "z=Vi(1)", "l=Vj(1)"], axis=1)(
                self.x, self.x, self.x_vect, backend="auto"
            )


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class ShapeTorchTestCase(unittest.TestCase):

    def setUp(self):
        use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        self.param = torch.tensor([0.4], dtype=torch.float32, device=device)
        self.param_scalar = torch.tensor(0.4, dtype=torch.float32, device=device)
        self.x = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32, device=device)
        self.x_vect = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=device)

    def test_torch_pm1_shape_regression(self):
        from pykeops.torch import Genred

        out = Genred("param", ["param=Pm(1)"], axis=1)(self.param, backend="auto")
        ref = torch.full_like(out, self.param[0])
        assert_torch_allclose(out, ref, atol=1e-7)

    def test_torch_pm1_scalar_rejected(self):
        from pykeops.torch import Genred

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "[pyKeOps] Error: Pm argument #0 requires at least 1 dimension (got a scalar)."
            ),
        ):
            Genred("param", ["param=Pm(1)"], axis=1)(self.param_scalar, backend="auto")

    def test_torch_vj_single_dim_formula_x(self):
        from pykeops.torch import Genred

        out = Genred("x", ["x=Vj(1)"], axis=1)(self.x, backend="auto")
        ref = torch.sum(self.x, dim=0, keepdim=True)
        assert_torch_allclose(out, ref, atol=1e-7)

    def test_torch_vj_single_dim_formula_x_1d_rejected_python(self):
        from pykeops.torch import Genred

        with self.assertRaisesRegex(IndexError, "tuple index out of range"):
            Genred("y", ["y=Vj(1)"], axis=1)(self.x_vect, backend="auto")

    def test_torch_vj_single_dim_formula_x_1d_rejected_cpp(self):
        from pykeops.torch import Genred

        # The error is raised in the C++ code (LoadKeOps_cpp.py or pykeops_nvrtc.py), after the Python code has accepted the input shape but before launching the kernel
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "[pyKeOps] Error: Vj argument #2 requires at least 2 dimensions, got 1."
            ),
        ):
            Genred("y + z + l", ["y=Vj(1)", "z=Vi(1)", "l=Vj(1)"], axis=1)(
                self.x, self.x, self.x_vect, backend="auto"
            )


if __name__ == "__main__":
    unittest.main()
