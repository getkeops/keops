"""
Datasets for the benchmarks
==========================================

"""

import os
import numpy as np
import urllib.request

synthetic = {
    # Key   :  metric,    Ntrain, Ntest, D,
    "R^D a": ("euclidean", 10**4, 10**4, 3),
    "R^D b": ("euclidean", 10**6, 10**4, 3),
    "R^D c": ("euclidean", 10**6, 10**4, 10),
    "R^D d": ("euclidean", 10**6, 10**4, 100),
    "R^D e": ("euclidean", 10**7, 10**4, 100),
    "R^D f": ("manhattan", 10**6, 10**4, 10),
    "R^D g": ("manhattan", 10**6, 10**4, 100),
    "S^{D-1}": ("angular", 10**6, 10**4, 10),
    "H^D": ("hyperbolic", 10**6, 10**4, 10),
}

downloaded = {
    # Key     :    metric, filename, url
    "MNIST a": (
        "euclidean",
        "mnist-784-euclidean.hdf5",
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
    ),
    "MNIST b": (
        "manhattan",
        "mnist-784-euclidean.hdf5",
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
    ),
    "GloVe25": (
        "angular",
        "glove-25-angular.hdf5",
        "http://ann-benchmarks.com/glove-25-angular.hdf5",
    ),
    "GloVe100": (
        "angular",
        "glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
    ),
}


def get_dataset(key):
    data_folder = "benchmark_datasets/"

    filename = None

    true_indices = None
    true_values = None

    if key in synthetic.keys():
        metric, Ntrain, Ntest, D = synthetic[key]

        if metric == "hyperbolic":
            x_train = 0.5 + np.random.rand(Ntrain, D)
            x_test = 0.5 + np.random.rand(Ntest, D)
        else:
            x_train = np.random.randn(Ntrain, D)
            x_test = np.random.randn(Ntest, D)

    else:
        import h5py

        metric, filename, url = downloaded[key]
        filename = data_folder + filename

        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            urllib.request.urlretrieve(url, filename)

        f = h5py.File(filename, "r")

        x_train = f["train"][()]
        x_test = f["test"][()]
        true_indices = f["neighbors"][()]
        true_values = f["distances"][()]

    #  With the angular metric, all the points are normalized:
    if metric == "angular":
        x_train /= np.linalg.norm(x_train, axis=1, keepdims=True)
        x_test /= np.linalg.norm(x_test, axis=1, keepdims=True)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return {
        "train": x_train,
        "test": x_test,
        "metric": metric,
        "output": true_indices,
        # "true_distances": true_values,
    }


from benchmark_utils import tensor
from pykeops.torch import LazyTensor


def ground_truth(x_train, x_test, K, metric):
    # Setup the K-NN estimator:
    x_train = tensor(x_train)
    x_test = tensor(x_test)

    # Encoding as KeOps LazyTensors:
    X_i = LazyTensor(x_test[:, None, :])
    X_j = LazyTensor(x_train[None, :, :])

    # Symbolic distance matrix:
    if metric == "euclidean":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif metric == "manhattan":
        D_ij = (X_i - X_j).abs().sum(-1)
    elif metric == "angular":
        D_ij = -(X_i | X_j)
    elif metric == "hyperbolic":
        D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])

    # K-NN query:
    indices = D_ij.argKmin(K, dim=1)
    return indices.cpu().numpy()


from copy import deepcopy


def generate_samples(key):
    dataset = get_dataset(key)

    def samples(K):
        KNN_dataset = deepcopy(dataset)
        KNN_dataset["output"] = ground_truth(
            KNN_dataset["train"], KNN_dataset["test"], K, KNN_dataset["metric"]
        )
        return KNN_dataset

    return samples
