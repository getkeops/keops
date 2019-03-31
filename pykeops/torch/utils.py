import torch

from pykeops.torch import Genred, default_dtype


#  from pykeops.torch.generic.generic_red import GenredLowlevel


def is_on_device(x):
    return x.is_cuda


class torchtools:
    copy = torch.clone
    exp = torch.exp
    log = torch.log
    norm = torch.norm

    # Genred = Genred
    # GenredLowlevel = GenredLowlevel

    @staticmethod
    def transpose(x): return (x.t())

    @staticmethod
    def solve(A, b): return torch.gesv(b, A)[0].contiguous()

    @staticmethod
    def arraysum(x, axis=None): return x.sum() if axis is None else x.sum(dim=axis)

    @staticmethod
    def long(x): return x.long()

    @staticmethod
    def size(x): return x.numel()

    @staticmethod
    def tile(*args): return torch.Tensor.repeat(*args)

    @staticmethod
    def numpy(x): return x.detach().cpu().numpy()

    @staticmethod
    def view(x, s): return x.view(s)

    @staticmethod
    def dtype(x): return x.dtype

    @staticmethod
    def rand(m, n, dtype=default_dtype, device='0'): return torch.rand(m, n, dtype=dtype, device=device)

    @staticmethod
    def randn(m, n, dtype=default_dtype, device='0'): return torch.randn(m, n, dtype=dtype, device=device)

    @staticmethod
    def zeros(shape, dtype=default_dtype, device='0'): return torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def eye(n, dtype=default_dtype, device='0'): return torch.eye(n, dtype=dtype, device=device)

    @staticmethod
    def array(x, dtype=default_dtype, device='0'): return torch.tensor(x, dtype=dtype, device=device)
    

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vi(1)',  # First arg   : i-variable, of size 1
               'y = Vj(1)',  # Second arg  : j-variable, of size 1
               'b = Vj(1)',  # Third arg  : j-variable, of size 1
               'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float32')
    dum = torch.rand(10, 1)
    dum2 = torch.rand(10, 1)
    my_routine(dum, dum, dum2, torch.tensor([1.0]))
    my_routine(dum, dum, dum2, torch.tensor([1.0]))


def squared_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
    return dist
