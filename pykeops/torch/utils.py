import torch

from pykeops.torch import Genred, KernelSolve
from pykeops.torch.cluster import swap_axes as torch_swap_axes


# from pykeops.torch.generic.generic_red import GenredLowlevel


def is_on_device(x):
    return x.is_cuda


class torchtools:
    copy = torch.clone
    exp = torch.exp
    log = torch.log
    norm = torch.norm

    swap_axes = torch_swap_axes

    Genred = Genred
    KernelSolve = KernelSolve

    arraytype = torch.Tensor
    float_types = [float]

    # GenredLowlevel = GenredLowlevel

    @staticmethod
    def eq(x, y):
        return torch.eq(x, y)

    @staticmethod
    def transpose(x):
        return x.t()

    @staticmethod
    def permute(x, *args):
        return x.permute(*args)

    @staticmethod
    def contiguous(x):
        return x.contiguous()

    @staticmethod
    def solve(A, b):
        return torch.solve(b, A)[0].contiguous()

    @staticmethod
    def arraysum(x, axis=None):
        return x.sum() if axis is None else x.sum(dim=axis)

    @staticmethod
    def long(x):
        return x.long()

    @staticmethod
    def size(x):
        return x.numel()

    @staticmethod
    def tile(*args):
        return torch.Tensor.repeat(*args)

    @staticmethod
    def numpy(x):
        return x.detach().cpu().numpy()

    @staticmethod
    def view(x, s):
        return x.view(s)

    @staticmethod
    def is_tensor(x):
        return isinstance(x, torch.Tensor)

    @staticmethod
    def dtype(x):
        if hasattr(x, "dtype"):
            return x.dtype
        else:
            return type(x)

    @staticmethod
    def dtype(x):
        if hasattr(x, "dtype"):
            return x.dtype
        else:
            return type(x)

    @staticmethod
    def detect_complex(x):
        if type(x) == list:
            return any(type(v) == complex for v in x)
        elif type(x) == torch.Tensor:
            return torch.is_complex(x)
        else:
            return type(x) == complex

    @staticmethod
    def view_as_complex(x):
        sh = list(x.shape)
        sh[-1] //= 2
        sh += [2]
        x = x.view(sh)
        return torch.view_as_complex(x)

    @staticmethod
    def view_as_real(x):
        sh = list(x.shape)
        sh[-1] *= 2
        return torch.view_as_real(x).view(sh)

    @staticmethod
    def dtypename(dtype):
        if dtype == torch.float32:
            return "float32"
        elif dtype == torch.float64:
            return "float64"
        elif dtype == torch.float16:
            return "float16"
        elif dtype == int:
            return int
        elif dtype == list:
            return "float32"
        else:
            raise ValueError(
                "[KeOps] {} data type incompatible with KeOps.".format(dtype)
            )

    @staticmethod
    def rand(m, n, dtype, device):
        return torch.rand(m, n, dtype=dtype, device=device)

    @staticmethod
    def randn(m, n, dtype, device):
        return torch.randn(m, n, dtype=dtype, device=device)

    @staticmethod
    def zeros(
        shape,
        dtype,
        device,
        requires_grad=False,
    ):
        return torch.zeros(
            *shape, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @staticmethod
    def empty(
        shape,
        dtype,
        device,
        requires_grad=False,
    ):
        return torch.empty(
            *shape, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @staticmethod
    def eye(n, dtype, device):
        return torch.eye(n, dtype=dtype, device=device)

    @staticmethod
    def array(x, dtype, device):
        if dtype == "float32":
            dtype = torch.float32
        elif dtype == "float64":
            dtype = torch.float64
        elif dtype == "float16":
            dtype = torch.float16
        elif dtype == "int32":
            dtype = torch.int32
        else:
            raise ValueError(
                "[KeOps] data type " + dtype + " is incompatible with KeOps."
            )
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def device(x):
        if isinstance(x, torch.Tensor):
            return x.device
        else:
            return None

    @staticmethod
    def get_pointer(x):
        return x.data_ptr()

    @staticmethod
    def device_type_index(x):
        if isinstance(x, torch.Tensor):
            dev = x.device
            return dev.type, dev.index
        else:
            return None, None

    @staticmethod
    def pointer(x):
        return x.data.data_ptr()


def squared_distances(x, y):
    x_norm = (x**2).sum(1).reshape(-1, 1)
    y_norm = (y**2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
    return dist


def torch_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    _kernel = {
        "gaussian": lambda _sq, _s: torch.exp(-_sq / (_s * _s)),
        "laplacian": lambda _sq, _s: torch.exp(-torch.sqrt(_sq) / _s),
        "cauchy": lambda _sq, _s: 1.0 / (1 + _sq / (_s * _s)),
        "inverse_multiquadric": lambda _sq, _s: torch.rsqrt(1 + _sq / (_s * _s)),
    }
    return _kernel[kernel](sq, s)
