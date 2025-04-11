import time
import numpy as np
import torch
from pykeops.torch import LazyTensor


# Import clustering functions from KeOps
from pykeops.torch.cluster import (
    grid_cluster,
    cluster_ranges_centroids,
    sort_clusters,
    from_matrix,
)


def test_block_sparse_reduction():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32

    M, N = (5000, 5000) if use_cuda else (500, 500)

    # Create two point clouds
    t = torch.linspace(0, 2 * np.pi, M + 1)[:-1]
    # Generate points with a circular pattern and add a bit of noise
    x = torch.stack((0.4 + 0.4 * torch.cos(t), 0.5 + 0.3 * torch.sin(t)), dim=1)
    x = x + 0.01 * torch.randn(x.shape)
    x = x.to(device=device, dtype=dtype)

    # Create the second point cloud with random perturbations
    y = torch.randn(N, 2, device=device, dtype=dtype)
    y = y / 10 + torch.tensor([0.6, 0.6], device=device, dtype=dtype)

    # Clustering: group points into bins
    eps = 0.05

    x_labels = grid_cluster(x.cpu(), eps).to(device)
    y_labels = grid_cluster(y.cpu(), eps).to(device)

    # Compute ranges and centroids for each cluster
    x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
    y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)

    # Sort the points so clusters are contiguous in memory
    x, x_labels = sort_clusters(x, x_labels)
    y, y_labels = sort_clusters(y, y_labels)

    # Compute a binary mask indicating which clusters interact
    D = ((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2).sum(2)
    # Use a threshold
    sigma = 0.05
    keep = D < (4 * sigma) ** 2

    # Convert the binary mask into ranges (for sparse reduction)
    ranges_ij = from_matrix(x_ranges, y_ranges, keep)

    # Prepare LazyTensors for convolution
    x_ = x / sigma
    y_ = y / sigma
    x_i = LazyTensor(x_[:, None, :])  # shape: (M,1,2)
    y_j = LazyTensor(y_[None, :, :])  # shape: (1,N,2)

    # Define the Gaussian kernel (dense mode)
    D_ij = ((x_i - y_j) ** 2).sum(dim=2)
    K = (-D_ij / 2).exp()

    # Create a random signal supported by y
    b = torch.randn(N, 1, device=device, dtype=dtype)

    # Compute the dense convolution result
    t0 = time.time()
    a_full = K @ b
    t_full = time.time() - t0

    # Compute the sparse convolution result using the ranges
    K.ranges = ranges_ij
    t0 = time.time()
    a_sparse = K @ b
    t_sparse = time.time() - t0

    # Compute the relative error between dense and sparse results
    rel_error = 100 * (a_sparse - a_full).abs().sum() / a_full.abs().sum()
    rel_error = rel_error.item()
    print(
        "Dense time: {:.4f}s, Sparse time: {:.4f}s, Relative error: {:.4f}%".format(
            t_full, t_sparse, rel_error
        )
    )

    # Assert that the relative error is below a tolerance
    assert rel_error < 0.2, "Relative error {}% is above tolerance".format(rel_error)


if __name__ == "__main__":
    test_block_sparse_reduction()
