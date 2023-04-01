import torch
import typing
import pykeops.torch as ktorch

import pykeops

pykeops.clean_pykeops()


def cross_knn_search(
    A: torch.Tensor, B: torch.Tensor, k: int
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    D_ij = ((X_i - X_j) ** 2).sum(-1)
    D, I = D_ij.Kmin_argKmin(K=k, dim=1)
    D = torch.sqrt(D)
    return D, I


input_a = torch.randn(204, 405 * 61)
input_b = torch.randn(205, 405 * 61)

D, I = cross_knn_search(input_b, input_a, 1)
