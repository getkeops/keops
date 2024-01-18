import time
import torch
from pykeops.torch import KernelSolve, LazyTensor

if torch.__version__ >= "1.8":
    torchsolve = lambda A, B: torch.linalg.solve(A, B)
else:
    torchsolve = lambda A, B: torch.solve(B, A)[0]

###############################################################################
# Define our dataset:
#

B = 3  # batch dim
N = 5000 if torch.cuda.is_available() else 500  # Number of points
D = 2  # Dimension of the ambient space
Dv = 2  # Dimension of the vectors (= number of linear problems to solve)
sigma = 0.1  # Radius of our RBF kernel

x = torch.rand(B, N, D, requires_grad=True)
b = torch.rand(B, N, Dv)
g = torch.Tensor([0.5 / sigma**2])  # Parameter of the Gaussian RBF kernel

sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None


def my_kernel(x, g):
    x_i = LazyTensor(x[:, :, None, :])
    x_j = LazyTensor(x[:, None, :, :])
    D_ij = ((x_i - x_j) ** 2).sum(axis=3)
    return (-g * D_ij).exp()


alpha = 0.01

print("Solving a Gaussian linear system, with {} points in dimension {}.".format(N, D))
sync()
start = time.time()
b_j = LazyTensor(b[:, None, :, None])
c = my_kernel(x, g).solve(b_j, alpha=alpha)
sync()
end = time.time()
print("Timing (KeOps implementation):", round(end - start, 5), "s")

###############################################################################
# Compare with a straightforward PyTorch implementation:
#

sync()
start = time.time()
K_xx = alpha * torch.eye(N) + torch.exp(
    -torch.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, dim=3) / (2 * sigma**2)
)
c_py = torchsolve(K_xx, b)
sync()
end = time.time()
print("Timing (PyTorch implementation):", round(end - start, 5), "s")
print("Relative error = ", (torch.norm(c - c_py) / torch.norm(c_py)).item())
