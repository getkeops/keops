import torch

from pykeops.torch import Genred, KernelSolve, default_dtype
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
    def dtype(x):
        if hasattr(x, "dtype"):
            return x.dtype
        else:
            return type(x)

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
    def rand(m, n, dtype=default_dtype, device="cpu"):
        return torch.rand(m, n, dtype=dtype, device=device)

    @staticmethod
    def randn(m, n, dtype=default_dtype, device="cpu"):
        return torch.randn(m, n, dtype=dtype, device=device)

    @staticmethod
    def zeros(shape, dtype=default_dtype, device="cpu"):
        return torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def eye(n, dtype=default_dtype, device="cpu"):
        return torch.eye(n, dtype=dtype, device=device)

    @staticmethod
    def array(x, dtype=default_dtype, device="cpu"):
        if dtype == "float32":
            dtype = torch.float32
        elif dtype == "float64":
            dtype = torch.float64
        elif dtype == "float16":
            dtype = torch.float16
        else:
            raise ValueError("[KeOps] data type incompatible with KeOps.")
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def device(x):
        if isinstance(x, torch.Tensor):
            return x.device
        else:
            return None

    @staticmethod
    def distance_function(metric):
        def euclidean(x, y):
            return ((x - y) ** 2).sum(-1)

        def manhattan(x, y):
            return ((x - y).abs()).sum(-1)

        def angular(x, y):
            return x | y

        def hyperbolic(x, y):
            return ((x - y) ** 2).sum(-1) / (x[0] * y[0])

        if metric == "euclidean":
            return euclidean
        elif metric == "manhattan":
            return manhattan
        elif metric == "angular":
            return angular
        elif metric == "hyperbolic":
            return hyperbolic
        else:
            raise ValueError("Unknown metric")

    @staticmethod
    def sort(x):
        return torch.sort(x)

    @staticmethod
    def unsqueeze(x, n):
        return torch.unsqueeze(x, n)

    @staticmethod
    def arange(n, device="cpu"):
        return torch.arange(n, device=device)

    @staticmethod
    def repeat(x, n):
        return torch.repeat_interleave(x, n)

    @staticmethod
    def to(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    @staticmethod
    def index_select(input, dim, index):
        return torch.index_select(input, dim, index)

    @staticmethod
    def norm(x, p=2, dim=-1):
        return torch.norm(x, p=p, dim=dim)

    @staticmethod
    def kmeans(x, K=10, Niter=15, metric="euclidean", device="cuda"):
        from pykeops.torch import LazyTensor

        distance = torchtools.distance_function(metric)
        N, D = x.shape
        c = x[:K, :].clone()
        x_i = LazyTensor(x.view(N, 1, D).to(device))
        for i in range(Niter):
            c_j = LazyTensor(c.view(1, K, D).to(device))
            D_ij = distance(x_i, c_j)
            cl = D_ij.argmin(dim=1).long().view(-1)
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)
            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
            c /= Ncl
            if torch.any(torch.isnan(c)) and metric == "angular":
                raise ValueError("Please normalise inputs")
        return cl, c


def squared_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
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
