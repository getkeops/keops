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

# Numbers of samples that we'll loop upon:
problem_sizes = flatten(
    [[1 * 10 ** k, 2 * 10 ** k, 5 * 10 ** k] for k in [2, 3, 4, 5]] + [[10 ** 6]]
)


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

    query = randn((target_len, batchsize, embed_dim))
    key = randn((source_len, batchsize, embed_dim))
    value = randn((source_len, batchsize, embed_dim))

    return query, key, value


#################################################
# Experiment:


def run_experiment(embed_dim, num_heads, batchsize=10):

    generate_samples = partial(generate_sequences, embed_dim=embed_dim)

    attention_torch = MultiheadAttention_torch(embed_dim, num_heads)
    attention_keops = MultiheadAttention_keops(embed_dim, num_heads)

    if use_cuda:
        routines = [
            (attention_torch, "PyTorch", {"batchsize": batchsize}),
            (attention_keops, "KeOps", {"batchsize": batchsize}),
        ]

        full_benchmark(
            "Multi-head attention (batch)",
            routines,
            generate_samples,
            problem_sizes=problem_sizes,
        )


##############################################
#

run_experiment(256, 16, batchsize=10)

plt.show()
