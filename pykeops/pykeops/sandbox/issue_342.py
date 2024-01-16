import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import from_matrix
import nvsmi

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type(
        "torch.cuda.FloatTensor"
    )  # All tensors should be on the GPU
    device = "cuda"

M, N = (10, 10)  # Dimensions
T = 5  # Number of iterations


def bsr_test(update_ranges=True):
    t = torch.linspace(0, 2 * np.pi, M + 1)[:-1]
    x = torch.stack((0.4 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
    x = x + 0.01 * torch.randn(x.shape)

    y = torch.randn((N, 2))
    y = y / 10 + torch.tensor([0.6, 0.6])

    eps = 0.05  # Size of our square bins
    x_labels = grid_cluster(x, eps)  # class labels
    y_labels = grid_cluster(y, eps)  # class labels
    torch.cuda.synchronize()

    x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
    y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)
    torch.cuda.synchronize()

    x, x_labels = sort_clusters(x, x_labels)
    y, y_labels = sort_clusters(y, y_labels)
    torch.cuda.synchronize()

    sigma = 0.05  # Characteristic length of interaction
    # Compute a coarse Boolean mask:
    D = ((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2).sum(2)
    keep = D < (4 * sigma) ** 2

    ranges_ij = from_matrix(x_ranges, y_ranges, keep)
    torch.cuda.synchronize()

    x_, y_ = x / sigma, y / sigma
    x_i, y_j = LazyTensor(x_[:, None, :]), LazyTensor(y_[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(dim=2)  # Symbolic (M,N,1) matrix of squared distances
    K = (-D_ij / 2).exp()  # Symbolic (M,N,1) Gaussian kernel matrix

    b = torch.randn((N, 1))

    if update_ranges:
        K.ranges = ranges_ij

    a_sparse = K @ b

    return a_sparse


for it in range(T):
    a = bsr_test(update_ranges=True)
    if (it + 1) % 1000 == 0:
        print("Iteration: " + str(it), flush=True)
        print("Memory allocated: " + str(torch.cuda.memory_allocated()), flush=True)
        print("Memory cached: " + str(torch.cuda.memory_reserved()), flush=True)
        print(nvsmi.get_gpu_processes()[0], flush=True)
        print("------", flush=True)
