"""
Scaling up multi-head attention layers
===========================================================

Let's compare the performances of PyTorch and KeOps 
for simple attention computations, with an increasing
number of tokens, attention heads and embedding features.

.. note::
    In this demo, we use exact **bruteforce** computations 
    (tensorized for PyTorch and online for KeOps), without leveraging any multiscale
    or low-rank (Nystroem/multipole) decomposition of the attention matrix.
    First support for these approximation schemes is scheduled for
    May-June 2021.

"""


##############################################
# Setup
# ---------------------

import numpy as np
import torch
from matplotlib import pyplot as plt

from functools import partial
from benchmark_utils import flatten, random_normal, full_benchmark

from torch.nn import MultiheadAttention as MultiheadAttention_torch
from pykeops.torch import MultiheadAttention as MultiheadAttention_keops

use_cuda = torch.cuda.is_available()

##############################################
# Benchmark specifications:
#

# Sequence lengths that we'll loop upon:
problem_sizes = [2 ** k for k in range(7, 18)]


##############################################
# Synthetic data:


def generate_sequences(
    target_source_len, embed_dim=1, device="cuda", lang="torch", batchsize=1, **kwargs
):
    """Generates query, key and value arrays.

    Args:
        target_source_len (int): length of the target and source sequences.
        embed_dim (int): dimension of the feature vectors. Default to 1.
        device (str, optional): "cuda", "cpu", etc. Defaults to "cuda".
        lang (str, optional): "torch", "numpy", etc. Defaults to "torch".
        batchsize (int, optional): number of experiments to run in parallel. Defaults to 1.

    Returns:
        3-uple of arrays: query, key, value
    """
    randn = random_normal(device=device, lang=lang)
    target_len = target_source_len
    source_len = target_source_len

    if callable(batchsize):
        batchsize = batchsize(target_source_len)

    query = randn((target_len, batchsize, embed_dim))
    key = randn((source_len, batchsize, embed_dim))
    value = randn((source_len, batchsize, embed_dim))

    return query, key, value


#################################################
# Our main experiment: benchmark the forward and
# forward+backward passes through the multi-head attention
# layer using both PyTorch and KeOps backends.
#


def run_experiment(embed_dim=1, num_heads=1):

    generate_samples = partial(generate_sequences, embed_dim=embed_dim)

    batchmems_torch = {
        2 ** 7: 2 ** 15,
        2 ** 8: 2 ** 14,
        2 ** 9: 2 ** 12,
        2 ** 10: 2 ** 8,
    }
    batchmems_keops = {
        2 ** 7: 2 ** 20,
        2 ** 8: 2 ** 19,
        2 ** 9: 2 ** 18,
        2 ** 10: 2 ** 17,
        2 ** 11: 2 ** 12,
        2 ** 12: 2 ** 11,
        2 ** 13: 2 ** 10,
        2 ** 14: 2 ** 9,
        2 ** 15: 2 ** 8,
        2 ** 16: 2 ** 7,
    }

    def batchsize_fun(n, use_keops=False, **kwargs):
        batchmems = batchmems_keops if use_keops else batchmems_torch
        batchmem = batchmems.get(n, 1)
        if batchmem <= embed_dim:
            batchsize = 1
        else:
            batchsize = batchmem // embed_dim
        return batchsize

    batchsize_torch = partial(batchsize_fun, use_keops=False)
    batchsize_keops = partial(batchsize_fun, use_keops=True)

    def attention(query, key, value, use_keops=False, backward=False, **kwargs):

        MHA = MultiheadAttention_keops if use_keops else MultiheadAttention_torch
        layer = MHA(embed_dim, num_heads)
        if use_cuda:
            layer = layer.cuda()

        def to_call(query, key, value, **kwargs):
            if backward:
                query.requires_grad = True
                key.requires_grad = True
                value.requires_grad = True

                out, _ = layer(query, key, value)
                out.sum().backward()
                return out

            else:
                return layer(query, key, value)

        return to_call

    routines = [
        (
            attention,
            "PyTorch forward+backward",
            {"batchsize": batchsize_torch, "use_keops": False, "backward": True},
        ),
        (
            attention,
            "PyTorch forward",
            {"batchsize": batchsize_torch, "use_keops": False},
        ),
        (
            attention,
            "KeOps forward+backward",
            {"batchsize": batchsize_keops, "use_keops": True, "backward": True},
        ),
        (
            attention,
            "KeOps forward",
            {"batchsize": batchsize_keops, "use_keops": True},
        ),
    ]

    full_benchmark(
        f"Multi-head attention (embed_dim={embed_dim},n_heads={num_heads},heads_dim={embed_dim//num_heads})",
        routines,
        generate_samples,
        problem_sizes=problem_sizes,
        loops=[10, 1],
        max_time=1,
        red_time=0.1,
        linestyles=[
            "o-b",
            "s:b",
            "^-r",
            "<:r",
        ],
        xlabel="Sequence length",
    )


##############################################
# Embeddings of dimension 64
# ----------------------------------
#
# Embedding of dimension 64 = 64 heads of dimension 1.

run_experiment(embed_dim=64, num_heads=64)

##############################################
# Embedding of dimension 64 = 16 heads of dimension 4.

run_experiment(embed_dim=64, num_heads=16)

##############################################
# Embedding of dimension 64 = 1 head of dimension 64.

run_experiment(embed_dim=64, num_heads=1)


##############################################
# Embeddings of dimension 256
# ------------------------------
#
# Embedding of dimension 256 = 64 heads of dimension 4.

run_experiment(embed_dim=256, num_heads=64)

##############################################
# Embedding of dimension 256 = 16 heads of dimension 16.

run_experiment(embed_dim=256, num_heads=16)

##############################################
# Embedding of dimension 256 = 4 heads of dimension 64.

run_experiment(embed_dim=256, num_heads=4)


##############################################
# Embeddings of dimension 1,024
# ----------------------------------
#
# Embedding of dimension 1,024 = 256 heads of dimension 4.

run_experiment(embed_dim=1024, num_heads=256)

##############################################
# Embedding of dimension 1,024 = 32 heads of dimension 32.

run_experiment(embed_dim=1024, num_heads=32)

##############################################
# Embedding of dimension 1,024 = 8 heads of dimension 128.

run_experiment(embed_dim=1024, num_heads=8)


plt.show()
