import torch

from pykeops.torch import Genred, KernelSolve, default_dtype
from pykeops.torch.cluster import swap_axes as torch_swap_axes


#  from pykeops.torch.generic.generic_red import GenredLowlevel


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
    def transpose(x): return (x.t())

    @staticmethod
    def permute(x,*args): return x.permute(*args)

    @staticmethod
    def contiguous(x): return x.contiguous()

    @staticmethod
    def solve(A, b): return torch.solve(b, A)[0].contiguous()

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
    def dtypename(dtype):
        if dtype == torch.float32:
            return 'float32'
        elif dtype == torch.float64:
            return 'float64'
        elif dtype == torch.float16:
            return 'float16'
        else:
            raise ValueError("[KeOps] data type incompatible with KeOps.")

    @staticmethod
    def rand(m, n, dtype=default_dtype, device='cpu'):
        return torch.rand(m, n, dtype=dtype, device=device)

    @staticmethod
    def randn(m, n, dtype=default_dtype, device='cpu'):
        return torch.randn(m, n, dtype=dtype, device=device)

    @staticmethod
    def zeros(shape, dtype=default_dtype, device='cpu'):
        return torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def eye(n, dtype=default_dtype, device='cpu'):
        return torch.eye(n, dtype=dtype, device=device)

    @staticmethod
    def array(x, dtype=default_dtype, device='cpu'):
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


def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    print("Warming up the Gpu (torch bindings) !!!")
    if torch.cuda.is_available():
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


def _scalar_products(u, v):
    u_i = u.unsqueeze(1)  # Shape (N,D) -> Shape (N,1,D)
    v_j = v.unsqueeze(0)  # Shape (M,D) -> Shape (1,M,D)
    return (u_i * v_j).sum(dim=2)  # N-by-M matrix, usv[i,j] = <u_i,v_j>


def torch_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    if kernel == "gaussian":
        return torch.exp(-sq / (s * s))
    elif kernel == "laplacian":
        return torch.exp(-torch.sqrt(sq) / s)
    elif kernel == "cauchy":
        return 1. / (1 + sq / (s * s))
    elif kernel == "inverse_multiquadric":
        return torch.rsqrt(1 + sq / (s * s))


def _log_sum_exp(mat, axis=0):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme,
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.

    For instance, if dim = 1 and mat is a 2d array, we output
                log(sum_j exp(mat[i,j] ))
    by factoring out the row-wise maximas.
    """
    max_rc = mat.max(dim=axis)[0]
    return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim=axis)), dim=axis))


def extract_metric_parameters(G):
    """
    From the shape of the Variable G, infers if it is supposed
    to be used as a fixed parameter or as a "j" variable,
    and whether it represents a scalar, a diagonal matrix
    or a full symmetric matrix.

    If G is a tuple, it can also encode an "i" variable.
    """
    if type(G) == tuple and G[0] == "i":  # i-variable
        G_var = G[1]
        G_cat = 0
        G_dim = G_var.shape[1]

    elif (type(G) == tuple and G[0] == "j") or len(G.shape) == 2:  # j-variable
        G_var = G
        G_cat = 1
        G_dim = G.shape[1]

    elif len(G.shape) == 1:  # Parameter
        G_var = G
        G_cat = 2
        G_dim = G.shape[0]
    else:
        raise ValueError("[KeOps] A 'metric' parameter is expected to be of dimension 1 or 2.")

    G_str = ["Vi", "Vj", "Pm"][G_cat]
    return G_var, G_dim, G_cat, G_str


def _weighted_squared_distances(g, x, y):
    x_i = x.unsqueeze(1)  # Shape (N,D) -> Shape (N,1,D)
    y_j = y.unsqueeze(0)  # Shape (M,D) -> Shape (1,M,D)

    D = x.shape[1]

    g, g_dim, g_cat, g_str = extract_metric_parameters(g)

    if g_cat == 2:  # g is a parameter
        if g_dim == 1:  # g is a scalar
            return g * ((x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] = g * |x_i-y_j|^2

        elif g_dim == D:  # g is a diagonal matrix
            g_d = g.unsqueeze(0).unsqueeze(1)  # Shape (D) -> Shape (1,1,D)
            return (g_d * (x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] =  \sum_d g_d * (x_i,d-y_j,d)^2

        elif g_dim == D ** 2:  # G is a symmetric matrix
            G = g.view(1, 1, D, D)  # Shape (D**2) -> Shape (1,1,D,D)
            xmy = x_i - y_j  # Shape (N,M,D)
            xmy_ = xmy.unsqueeze(2)  #  Shape (N,M,1,D)
            Gxmy = (G * xmy_).sum(3)  #  Shape (N,M,D,D) -> (N,M,D)
            return (xmy * Gxmy).sum(2)  # N-by-M matrix, xmy[i,j] =  < (x_i-y_j), G (x_i-y_j) >
        else:
            raise ValueError("[KeOps] We support scalar (dim=1), diagonal (dim=D) and symmetric (dim=D**2) metrics.")

    elif g_cat == 0:  # g is a 'i' variable
        if g_dim == 1:  # g_i is scalar
            return g.view(-1, 1) * ((x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] = g_i * |x_i-y_j|^2

        elif g_dim == D:  # g_i is a diagonal matrix
            g_d = g.unsqueeze(1)  # Shape (N,D) -> Shape (N,1,D)
            return (g_d * (x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] =  \sum_d g_i,d * (x_i,d-y_j,d)^2

        elif g_dim == D ** 2:  # G_i is a symmetric matrix
            G_i = g.view(-1, 1, D, D)  # Shape (N,D**2) -> Shape (N,1,D,D)
            xmy = x_i - y_j  # Shape (N,M,D)
            xmy_ = xmy.unsqueeze(2)  #  Shape (N,M,1,D)
            Gxmy = (G_i * xmy_).sum(3)  #  Shape (N,M,D,D) -> (N,M,D)
            return (xmy * Gxmy).sum(2)  # N-by-M matrix, xmy[i,j] =  < (x_i-y_j), G_i (x_i-y_j) >

        else:
            raise ValueError("[KeOps] We support scalar (dim=1), diagonal (dim=D) and symmetric (dim=D**2) metrics.")

    elif g_cat == 1:  # g is a 'j' variable
        if g_dim == 1:  # g_j is scalar
            return g.view(1, -1) * ((x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] = g_j * |x_i-y_j|^2

        elif g_dim == D:  # g_j is a diagonal matrix
            g_d = g.unsqueeze(0)  # Shape (M,D) -> Shape (1,M,D)
            return (g_d * (x_i - y_j) ** 2).sum(2)  # N-by-M matrix, xmy[i,j] =  \sum_d g_j,d * (x_i,d-y_j,d)^2

        elif g_dim == D ** 2:  # G_j is a symmetric matrix
            G_j = g.view(1, -1, D, D)  # Shape (M,D**2) -> Shape (1,M,D,D)
            xmy = x_i - y_j  # Shape (N,M,D)
            xmy_ = xmy.unsqueeze(2)  #  Shape (N,M,1,D)
            Gxmy = (G_j * xmy_).sum(3)  #  Shape (N,M,D,D) -> (N,M,D)
            return (xmy * Gxmy).sum(2)  # N-by-M matrix, xmy[i,j] =  < (x_i-y_j), G_j (x_i-y_j) >

        else:
            raise ValueError("[KeOps] We support scalar (dim=1), diagonal (dim=D) and symmetric (dim=D**2) metrics.")

    else:
        raise ValueError("[KeOps] A metric parameter should either be a vector or a 2d-tensor.")
    


class Formula:

    def __init__(self, formula_sum=None, routine_sum=None,
                 formula_log=None, routine_log=None, intvalue=None):
        if intvalue is None:
            self.formula_sum = formula_sum
            self.routine_sum = routine_sum
            self.formula_log = formula_log
            self.routine_log = routine_log
        else:
            import math

            self.formula_sum = "IntCst(" + str(intvalue) + ")"
            self.routine_sum = lambda **x: intvalue
            self.formula_log = "Log(IntCst(" + str(intvalue) + ")"
            self.routine_log = lambda **x: math.log(intvalue)
        self.intvalue = intvalue
        self.n_params = 1
        self.n_vars = 2

    def __add__(self, other):
        return Formula(formula_sum="(" + self.formula_sum + " + " + other.formula_sum + ")",
                       routine_sum=lambda **x: self.routine_sum(**x) + other.routine_sum(**x),
                       formula_log="Log((" + self.formula_sum + " + " + other.formula_sum + ") )",
                       routine_log=lambda **x: (self.routine_sum(**x) + other.routine_sum(**x)).log(),
                       )

    def __mul__(self, other):
        return Formula(formula_sum="(" + self.formula_sum + " * " + other.formula_sum + ")",
                       routine_sum=lambda **x: self.routine_sum(**x) * other.routine_sum(**x),
                       formula_log="(" + self.formula_log + " + " + other.formula_log + ")",
                       routine_log=lambda **x: (self.routine_log(**x) + other.routine_log(**x)),
                       )

    def __neg__(self):
        return Formula(formula_sum="(-" + self.formula_sum + ")",
                       routine_sum=lambda **x: - self.routine_sum(**x),
                       formula_log=None,
                       routine_log=lambda **x: None,
                       )

    def __pow__(self, other):
        """
        N.B.: other should be a Formula(intvalue=N).
        """
        return Formula(formula_sum="Pow(" + self.formula_sum + "," + str(other.intvalue) + ")",
                       routine_sum=lambda **x: self.routine_sum(**x) ** (other.intvalue),
                       formula_log="(" + other.formula_sum + " * " + self.formula_log + ")",
                       routine_log=lambda **x: other.routine_sum(**x) * self.routine_log(**x),
                       )
