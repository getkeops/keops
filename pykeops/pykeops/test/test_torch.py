import os.path
import sys
from contextlib import redirect_stdout
import io
from math import prod

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.sep.join([os.pardir] * 2)
    )
)
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.sep.join([os.pardir] * 3),
        "keopscore",
    )
)

import unittest

import pykeops
import pykeops.config
from pykeops.test import assert_torch_allclose

HAS_TORCH = False
use_cuda = False

try:
    import torch

    HAS_TORCH = True

    torch.manual_seed(42)

    use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False

except ImportError:
    pass


@unittest.skipUnless(HAS_TORCH, "torch not available")
class PytorchUnitTestCase(unittest.TestCase):

    def setUp(self):
        self.A = int(5)  # Batchdim 1
        self.B = int(3)  # Batchdim 2
        self.M = int(10)
        self.N = int(6)
        self.D = int(3)
        self.E = int(3)
        self.nbatchdims = int(2)

        self.x64 = torch.rand(
            (self.M, self.D), dtype=torch.float64, device=device, requires_grad=True
        )
        self.a64 = torch.rand(
            (self.M, self.E), dtype=torch.float64, device=device, requires_grad=False
        )
        self.e64 = torch.rand(
            (self.M, self.E), dtype=torch.float64, device=device, requires_grad=False
        )
        self.f64 = torch.rand(
            (self.M, 1), dtype=torch.float64, device=device, requires_grad=True
        )
        self.y64 = torch.rand(
            (self.N, self.D), dtype=torch.float64, device=device, requires_grad=False
        )
        self.b64 = torch.rand(
            (self.N, self.E), dtype=torch.float64, device=device, requires_grad=False
        )
        self.g64 = torch.rand(
            (self.N, 1), dtype=torch.float64, device=device, requires_grad=True
        )
        self.p64 = torch.rand(
            2, dtype=torch.float64, device=device, requires_grad=False
        )

        self.sigma64 = torch.tensor([0.4], dtype=torch.float64, device=device)
        self.alpha64 = torch.tensor([0.1], dtype=torch.float64, device=device)

        self.X64 = torch.rand(
            (self.A, self.B, self.M, self.D),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        self.L64 = torch.rand(
            (self.A, 1, self.M, 1),
            dtype=torch.float64,
            device=device,
            requires_grad=False,
        )
        self.Y64 = torch.rand(
            (1, self.B, self.N, self.D),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        self.S64 = 1 + torch.rand(
            (self.A, self.B, 1), dtype=torch.float64, device=device, requires_grad=True
        )

        self.x32 = self.x64.to(torch.float32).clone().requires_grad_(True)
        self.a32 = self.a64.to(torch.float32).clone().requires_grad_(False)
        self.e32 = self.e64.to(torch.float32).clone().requires_grad_(False)
        self.f32 = self.f64.to(torch.float32).clone().requires_grad_(True)
        self.y32 = self.y64.to(torch.float32).clone().requires_grad_(False)
        self.b32 = self.b64.to(torch.float32).clone().requires_grad_(False)
        self.g32 = self.g64.to(torch.float32).clone().requires_grad_(True)
        self.p32 = self.p64.to(torch.float32).clone().requires_grad_(False)

        self.sigma32 = self.sigma64.to(torch.float32).clone().requires_grad_(False)
        self.alpha32 = self.alpha64.to(torch.float32).clone().requires_grad_(False)

        self.X32 = self.X64.to(torch.float32).clone().requires_grad_(True)
        self.L32 = self.L64.to(torch.float32).clone().requires_grad_(False)
        self.Y32 = self.Y64.to(torch.float32).clone().requires_grad_(True)
        self.S32 = self.S64.to(torch.float32).clone().requires_grad_(True)

    ############################################################
    def test_torchtools_function_binding(self):
        ############################################################
        from pykeops.torch.utils import torchtools
        import torch

        tools = torchtools()
        x = self.x32.detach()

        self.assertTrue(torch.equal(tools.copy(x), x))
        assert_torch_allclose(tools.exp(x), torch.exp(x))
        assert_torch_allclose(tools.log(x + 1), torch.log(x + 1))
        assert_torch_allclose(tools.norm(x), torch.norm(x))

    ############################################################
    def test_generic_syntax_float(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = ["p=Pm(1)", "a=Vj(1)", "x=Vi(3)", "y=Vj(3)"]
        formula = "Square(p-a)*Exp(x+y)"
        if pykeops.config.cuda.is_available():
            backend_to_test = ["auto", "GPU_1D", "GPU_2D", "GPU"]
        else:
            backend_to_test = ["auto"]

        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                gamma_keops = Genred(formula, aliases, axis=1)(
                    self.sigma32, self.g32, self.x32, self.y32, backend=b
                )
                # Torch reference in float64 for stable comparisons.
                gamma_ref = torch.sum(
                    (self.sigma64 - self.g64) ** 2
                    * torch.exp(self.y64.T[:, :, None] + self.x64.T[:, None, :]),
                    dim=1,
                ).T
                # compare output
                assert_torch_allclose(
                    gamma_keops.to(torch.float64), gamma_ref, atol=1e-6
                )

    ############################################################
    def test_generic_syntax_double(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = ["p=Pm(1)", "a=Vj(1)", "x=Vi(3)", "y=Vj(3)"]
        formula = "Square(p-a)*Exp(x+y)"
        if pykeops.config.cuda.is_available():
            backend_to_test = ["auto", "GPU_1D", "GPU_2D", "GPU"]
        else:
            backend_to_test = ["auto"]

        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                gamma_keops = Genred(formula, aliases, axis=1)(
                    self.sigma64, self.g64, self.x64, self.y64, backend=b
                )
                # Torch reference in float64.
                gamma_ref = torch.sum(
                    (self.sigma64 - self.g64) ** 2
                    * torch.exp(self.y64.T[:, :, None] + self.x64.T[:, None, :]),
                    dim=1,
                ).T
                # compare output
                assert_torch_allclose(gamma_keops, gamma_ref, atol=1e-6)

    ############################################################
    def test_generic_syntax_softmax(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = ["p=Pm(1)", "a=Vj(1)", "x=Vi(3)", "y=Vj(3)"]
        formula = "Square(p-a)*Exp(-SqNorm2(x-y))"
        formula_weights = "y"
        if pykeops.config.cuda.is_available():
            backend_to_test = ["auto", "GPU_1D", "GPU_2D", "GPU"]
        else:
            backend_to_test = ["auto"]

        for b in backend_to_test:
            with self.subTest(b=b):
                # Call cuda kernel
                myop = Genred(
                    formula,
                    aliases,
                    reduction_op="SumSoftMaxWeight",
                    axis=1,
                    formula2=formula_weights,
                )
                gamma_keops = myop(
                    self.sigma64, self.g64, self.x64, self.y64, backend=b
                )

                # Torch reference
                sqdist = torch.sum(
                    (self.x64[:, None, :] - self.y64[None, :, :]) ** 2, dim=2
                )
                scores = (self.sigma64 - self.g64.T) ** 2 * torch.exp(-sqdist)
                scores = scores - torch.max(scores, dim=1, keepdim=True).values
                gamma_ref = (
                    torch.exp(scores)
                    @ self.y64
                    / torch.sum(torch.exp(scores), dim=1, keepdim=True)
                )

                # compare output
                assert_torch_allclose(gamma_keops, gamma_ref, atol=1e-6)

    ############################################################
    def test_generic_syntax_simple(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = [
            "P = Pm(2)",  # 1st argument,  a parameter, dim 2.
            "X = Vi("
            + str(self.x64.shape[1])
            + ") ",  # 2nd argument, indexed by i, dim D.
            "Y = Vj(" + str(self.y64.shape[1]) + ") ",
        ]  # 3rd argument, indexed by j, dim D.

        formula = "Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))"

        if pykeops.config.cuda.is_available():
            backend_to_test = ["auto", "GPU_1D", "GPU_2D", "GPU"]
        else:
            backend_to_test = ["auto"]

        for b in backend_to_test:
            with self.subTest(b=b):
                my_routine = Genred(formula, aliases, reduction_op="Sum", axis=1)
                gamma_keops = my_routine(self.p64, self.x64, self.y64, backend=b)

                # Torch reference
                scals = (self.x64 @ self.y64.T) ** 2  # Memory-intensive computation!
                gamma_ref = self.p64[0] * scals.sum(1).reshape(
                    -1, 1
                ) * self.x64 + self.p64[1] * (scals @ self.y64)

                # compare output
                assert_torch_allclose(gamma_keops, gamma_ref, atol=1e-6)

    ############################################################
    def test_logSumExp_kernels_feature(self):
        ############################################################
        from pykeops.torch import Vi, Vj, Pm

        kernels = {
            "gaussian": lambda xc, yc, sigmac: (
                -Pm(1 / sigmac**2) * Vi(xc).sqdist(Vj(yc))
            ),
            "laplacian": lambda xc, yc, sigmac: (
                -(Pm(1 / sigmac**2) * Vi(xc).sqdist(Vj(yc))).sqrt()
            ),
            "cauchy": lambda xc, yc, sigmac: (
                1 + Pm(1 / sigmac**2) * Vi(xc).sqdist(Vj(yc))
            )
            .power(-1)
            .log(),
            "inverse_multiquadric": lambda xc, yc, sigmac: (
                1 + Pm(1 / sigmac**2) * Vi(xc).sqdist(Vj(yc))
            )
            .sqrt()
            .power(-1)
            .log(),
        }

        for k in ["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]:
            with self.subTest(k=k):
                # Call cuda kernel
                gamma_lazy = kernels[k](self.x64, self.y64, self.sigma64)
                gamma_lazy = gamma_lazy.logsumexp(dim=1, weight=Vj(self.g64.exp()))

                # Torch reference
                sqdist = torch.sum(
                    (self.x64[:, None, :] - self.y64[None, :, :]) ** 2, dim=2
                )
                inv_s2 = 1 / (self.sigma64**2)
                if k == "gaussian":
                    log_k = -(inv_s2 * sqdist)
                elif k == "laplacian":
                    log_k = -torch.sqrt(inv_s2 * sqdist)
                elif k == "cauchy":
                    log_k = -(1 + inv_s2 * sqdist).log()
                else:
                    log_k = -0.5 * (1 + inv_s2 * sqdist).log()
                gamma_ref = torch.logsumexp(log_k + self.g64.T, dim=1)

                # compare output
                assert_torch_allclose(
                    gamma_lazy.reshape(-1), gamma_ref.reshape(-1), atol=1e-6
                )

    ############################################################
    def test_logSumExp_gradient_kernels_feature(self):
        ############################################################
        import torch
        from pykeops.torch import Genred

        aliases = [
            "P = Pm(2)",  # 1st argument,  a parameter, dim 2.
            "X = Vi("
            + str(self.g64.shape[1])
            + ") ",  # 2nd argument, indexed by i, dim D.
            "Y = Vj(" + str(self.f64.shape[1]) + ") ",
        ]  # 3rd argument, indexed by j, dim D.

        formula = "(Elem(P,0) * X) + (Elem(P,1) * Y)"

        # Pytorch version
        my_routine = Genred(formula, aliases, reduction_op="LogSumExp", axis=1)
        tmp = my_routine(self.p64, self.f64, self.g64, backend="auto")
        res = torch.dot(
            torch.ones_like(tmp).view(-1), tmp.view(-1)
        )  # equivalent to tmp.sum() but avoiding contiguity pb
        gamma_keops = torch.autograd.grad(res, [self.f64, self.g64], create_graph=False)

        # Torch reference
        tmp = self.p64[0] * self.f64 + self.p64[1] * self.g64.T
        res_ref = torch.exp(tmp).sum(dim=1)
        tmp2 = torch.exp(tmp.T) / res_ref.reshape(1, -1)
        gamma_ref = [
            torch.ones(self.M, dtype=torch.float64, device=device) * self.p64[0],
            self.p64[1] * tmp2.T.sum(dim=0),
        ]

        # compare output
        assert_torch_allclose(
            gamma_keops[0].reshape(-1), gamma_ref[0].reshape(-1), atol=1e-6
        )
        assert_torch_allclose(
            gamma_keops[1].reshape(-1), gamma_ref[1].reshape(-1), atol=1e-6
        )

    ############################################################
    def test_non_contiguity(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = [
            "P = Pm(2)",  # 1st argument,  a parameter, dim 2.
            "X = Vi("
            + str(self.x64.shape[1])
            + ") ",  # 2nd argument, indexed by i, dim D.
            "Y = Vj(" + str(self.y64.shape[1]) + ") ",
        ]  # 3rd argument, indexed by j, dim D.

        formula = "Pow((X|Y),2) * ((Elem(P,0) * X) + (Elem(P,1) * Y))"

        my_routine = Genred(formula, aliases, reduction_op="Sum", axis=1)
        yc_tmp = self.y64.t().contiguous().t()  # create a non contiguous copy

        # check output
        self.assertFalse(yc_tmp.is_contiguous())
        my_routine(self.p64, self.x64, yc_tmp, backend="auto")

    ############################################################
    def test_heterogeneous_var_aliases(self):
        ############################################################
        from pykeops.torch import Genred

        aliases = ["p=Pm(0,1)", "x=Vi(1,3)", "y=Vj(2,3)"]
        formula = "Square(p-Var(3,1,1))*Exp(-SqNorm2(y-x))"

        # Call cuda kernel
        myconv = Genred(formula, aliases, reduction_op="Sum", axis=1)
        gamma_keops = myconv(self.sigma64, self.x64, self.y64, self.g64, backend="auto")

        # Torch reference
        sqdist = torch.sum((self.x64[:, None, :] - self.y64[None, :, :]) ** 2, dim=2)
        gamma_ref = torch.sum(
            (self.sigma64 - self.g64.T) ** 2 * torch.exp(-sqdist), dim=1
        )

        # compare output
        assert_torch_allclose(gamma_keops.reshape(-1), gamma_ref.reshape(-1), atol=1e-6)

    ############################################################
    def test_invkernel(self):
        ############################################################
        import torch
        from pykeops.torch.operations import KernelSolve

        formula = "Exp(-oos2*SqDist(x,y))*b"
        aliases = [
            "x = Vi(" + str(self.D) + ")",  # First arg   : i-variable, of size D
            "y = Vj(" + str(self.D) + ")",  # Second arg  : j-variable, of size D
            "b = Vj(" + str(self.E) + ")",  # Third arg  : j-variable, of size Dv
            "oos2 = Pm(1)",
        ]  # Fourth arg  : scalar parameter

        Kinv = KernelSolve(formula, aliases, "b", axis=1)

        c = Kinv(self.x64, self.x64, self.a64, self.sigma64, alpha=self.alpha64)
        if torch.__version__ >= "1.8":
            torchsolve = lambda A, B: torch.linalg.solve(A, B)
        else:
            torchsolve = lambda A, B: torch.solve(B, A)[0]
        c_ = torchsolve(
            self.alpha64 * torch.eye(self.M, device=device, dtype=torch.float64)
            + torch.exp(
                -torch.sum((self.x64[:, None, :] - self.x64[None, :, :]) ** 2, dim=2)
                * self.sigma64
            ),
            self.a64,
        )

        assert_torch_allclose(c.reshape(-1), c_.reshape(-1), atol=1e-4)

        (u,) = torch.autograd.grad(c, self.x64, self.e64)
        (u_,) = torch.autograd.grad(c_, self.x64, self.e64)
        assert_torch_allclose(u.reshape(-1), u_.reshape(-1), atol=1e-4)

    ############################################################
    def test_cg_solver_stops_immediately_when_x0_is_good(self):
        ############################################################

        import torch
        from pykeops.torch import LazyTensor

        alpha = 2.0

        x_i = LazyTensor(self.x64[:, None, :])
        x_j = LazyTensor(self.x64[None, :, :])
        K_xx = (((x_i - x_j).abs()).sum(-1)).exp()

        b = K_xx @ self.f64 + alpha * self.f64

        x = K_xx.solve(b, alpha=alpha, x0=self.f64, eps=1e-12)
        assert_torch_allclose(self.f64, x)

        stream = io.StringIO()
        with redirect_stdout(stream):
            x = K_xx.solve(b, alpha=alpha, x0=self.f64, eps=1e-12, verbose=True)

        assert_torch_allclose(self.f64, x)
        output = stream.getvalue()
        self.assertIn("'status': 'Converged'", output)
        self.assertIn("'niter': 0", output)
        self.assertIn("'x0_provided': True", output)

    ############################################################
    def test_softmax(self):
        ############################################################

        import torch
        from pykeops.torch import Genred

        formula = "SqDist(x,y)"
        formula_weights = "b"
        aliases = [
            "x = Vi(" + str(self.D) + ")",  # First arg   : i-variable, of size D
            "y = Vj(" + str(self.D) + ")",  # Second arg  : j-variable, of size D
            "b = Vj(" + str(self.E) + ")",
        ]  # third arg : j-variable, of size Dv

        softmax_op = Genred(
            formula,
            aliases,
            reduction_op="SumSoftMaxWeight",
            axis=1,
            formula2=formula_weights,
        )

        c = softmax_op(self.x64, self.y64, self.b64)

        # compare with direct implementation
        cc = 0
        for k in range(self.D):
            xk = self.x64[:, k][:, None]
            yk = self.y64[:, k][:, None]
            cc += (xk - yk.t()) ** 2
        cc -= torch.max(cc, dim=1)[0][:, None]  # subtract the max for robustness
        cc = torch.exp(cc) @ self.b64 / torch.sum(torch.exp(cc), dim=1)[:, None]

        assert_torch_allclose(c.reshape(-1), cc.reshape(-1), atol=1e-6)

    ############################################################
    def test_pickle(self):
        ############################################################
        from pykeops.torch import Genred
        import pickle

        formula = "SqDist(x,y)"
        aliases = [
            "x = Vi(" + str(self.D) + ")",  # First arg   : i-variable, of size D
            "y = Vj(" + str(self.D) + ")",  # Second arg  : j-variable, of size D
        ]

        kernel_instance = Genred(formula, aliases, reduction_op="Sum", axis=1)

        # serialize/pickle
        serialized_kernel = pickle.dumps(kernel_instance)
        # deserialize/unpickle
        deserialized_kernel = pickle.loads(serialized_kernel)

        self.assertTrue(type(kernel_instance), type(deserialized_kernel))

    ############################################################
    def test_LazyTensor_sum(self):
        ############################################################
        import torch
        from pykeops.torch import LazyTensor

        full_results = []
        for use_keops in [True, False]:
            results = []

            # N.B.: We could loop over float32 and float64, but this would take longer...
            for x, l, y, s in [(self.X32, self.L32, self.Y32, self.S32)]:  # Float32
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)

                if use_keops:
                    x_i = LazyTensor(x_i)
                    l_i = LazyTensor(l_i)
                    y_j = LazyTensor(y_j)
                    s_p = LazyTensor(s_p)

                D_ij = (0.5 * (l_i * x_i - y_j) ** 2 / s_p).sum(-1)
                K_ij = (-D_ij).exp()
                a_i = K_ij.sum(self.nbatchdims + 1)
                if use_keops:
                    a_i = a_i.squeeze(-1)
                [g_x, g_y, g_s] = torch.autograd.grad(
                    (a_i**2).sum(), [x, y, s], create_graph=True
                )
                [g_xx] = torch.autograd.grad((g_x**2).sum(), [x], create_graph=True)

                results += [a_i, g_x, g_y, g_s, g_xx]

            full_results.append(results)

        for res_keops, res_torch in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            assert_torch_allclose(
                res_keops.reshape(-1), res_torch.reshape(-1), atol=1e-3
            )

    ############################################################
    def test_LazyTensor_logsumexp(self):
        ############################################################
        import torch
        from pykeops.torch import LazyTensor

        full_results = []
        for use_keops in [True, False]:
            results = []

            # N.B.: We could loop over float32 and float64, but this would take longer...
            for x, l, y, s in [(self.X64, self.L64, self.Y64, self.S64)]:  # Float64
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)

                if use_keops:
                    x_i = LazyTensor(x_i)
                    l_i = LazyTensor(l_i)
                    y_j = LazyTensor(y_j)
                    s_p = LazyTensor(s_p)

                D_ij = ((l_i * x_i + y_j).relu() * s_p / 9).sum(-1)
                K_ij = -1 / (1 + D_ij)
                a_i = K_ij.logsumexp(self.nbatchdims + 1)
                if use_keops:
                    a_i = a_i.squeeze(-1)
                [g_x, g_y, g_s] = torch.autograd.grad(
                    (1.0 * a_i).sum(), [x, y, s], create_graph=True
                )

                # N.B. (Joan, sept 2020) commenting out the 2nd order gradient computation here,
                # since it slows down too much the compilation currently, when using Cuda 11.
                #
                [g_xs] = torch.autograd.grad((g_x.abs()).sum(), [s], create_graph=True)
                results += [a_i, g_x, g_y, g_s, g_xs]

            full_results.append(results)

        for res_keops, res_torch in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            assert_torch_allclose(
                res_keops.reshape(-1), res_torch.reshape(-1), atol=1e-5
            )

    ############################################################
    # Test min reduction with chunk without batches
    def test_LazyTensor_min_chunked(self):
        ############################################################
        from pykeops.torch import LazyTensor
        import torch

        X64 = torch.rand((self.M, 990), dtype=torch.float64, device=device)
        Y64 = torch.rand((self.N, 990), dtype=torch.float64, device=device)

        X32 = X64.to(torch.float32).clone().requires_grad_(True)
        Y32 = Y64.to(torch.float32).clone().requires_grad_(True)

        full_results = []
        for use_keops in [True, False]:
            results = []

            for x, y in [(X32, Y32), (X64, Y64)]:
                x_i = x.unsqueeze(-2)
                y_j = y.unsqueeze(-3)

                if use_keops:
                    x_i = LazyTensor(x_i)
                    y_j = LazyTensor(y_j)

                K_ij = ((-(((x_i + y_j)) ** 2)).exp()).sum(-1, keepdim=True)

                if use_keops:
                    m, am = K_ij.min_argmin(dim=0)
                else:
                    m, am = K_ij.min(dim=0)

                results += [m, am]

            full_results.append(results)

        for res_keops, res_torch in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            assert_torch_allclose(
                res_keops.reshape(-1), res_torch.reshape(-1), atol=1e-5
            )

    ############################################################
    def test_LazyTensor_min(self):
        ############################################################
        from pykeops.torch import LazyTensor

        full_results = []
        for use_keops in [True, False]:
            results = []

            # N.B.: We could loop over float32 and float64, but this would take longer...
            for x, l, y, s in [(self.X32, self.L32, self.Y32, self.S32)]:  # Float32
                x_i = x.unsqueeze(-2)
                l_i = l.unsqueeze(-2)
                y_j = y.unsqueeze(-3)
                s_p = s.unsqueeze(-2).unsqueeze(-2)

                if use_keops:
                    x_i = LazyTensor(x_i)
                    l_i = LazyTensor(l_i)
                    y_j = LazyTensor(y_j)
                    s_p = LazyTensor(s_p)

                D_ij = ((1 + ((l_i * x_i + y_j).relu() * s_p) ** 2).log()).sum(
                    -1, keepdim=True
                )
                K_ij = (D_ij**1.5 + 1).cos() * (D_ij * (3.2 + s_p)).sin()

                if use_keops:
                    m, am = K_ij.min_argmin(dim=self.nbatchdims)
                else:
                    m, am = K_ij.min(dim=self.nbatchdims)

                results += [m, am]

            full_results.append(results)

        for res_keops, res_torch in zip(full_results[0], full_results[1]):
            self.assertTrue(res_keops.shape == res_torch.shape)
            assert_torch_allclose(
                res_keops.reshape(-1), res_torch.reshape(-1), atol=1e-5
            )

    ############################################################
    def test_TensorDot_with_permute(self):
        ############################################################
        import torch
        from pykeops.torch import LazyTensor

        def my_tensordort_perm(a, b, dims=None, perm=None):
            return torch.tensordot(a, b, dims=dims).sum(3).permute(perm)

        def invert_permutation_numpy(permutation):
            return [
                idx for idx, _ in sorted(enumerate(permutation), key=lambda x: x[1])
            ]

        x = torch.randn(self.M, 2, 3, 2, 2, 4, requires_grad=True, dtype=torch.float64)
        y = torch.randn(
            self.N, 2, 4, 2, 3, 2, 3, requires_grad=True, dtype=torch.float64
        )

        dimfa, dimfb = x.shape[1:], y.shape[1:]
        contfa, contfb = [5, 1, 3], [2, 5, 3]
        perm = [4, 3, 2, 0, 1]
        perm_torch = (0,) + tuple([(i + 1) for i in invert_permutation_numpy(perm)])
        sum_f_torch2 = my_tensordort_perm(x, y, dims=(contfa, contfb), perm=perm_torch)

        f_keops = LazyTensor(x.reshape(self.M, 1, prod(dimfa))).keops_tensordot(
            LazyTensor(y.reshape(1, self.N, prod(dimfb))),
            dimfa,
            dimfb,
            tuple(i - 1 for i in contfa),
            tuple(i - 1 for i in contfb),
            tuple(perm),
        )
        sum_f_keops = f_keops.sum_reduction(dim=1)
        assert_torch_allclose(sum_f_keops.flatten(), sum_f_torch2.flatten())

        e = torch.randn_like(sum_f_torch2)
        # checking gradients
        grad_keops = torch.autograd.grad(
            sum_f_keops, x, e.reshape(self.M, -1), retain_graph=True
        )[0]
        grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0]
        assert_torch_allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4)

        grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(self.M, -1))[0]
        grad_torch = torch.autograd.grad(sum_f_torch2, y, e)[0]
        assert_torch_allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4)


if __name__ == "__main__":
    """
    run tests
    """
    unittest.main()
