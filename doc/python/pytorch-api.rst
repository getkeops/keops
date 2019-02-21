PyTorch API
============

:mod:`pykeops.torch` - Generic reductions, with full support of PyTorch's ``autograd`` engine:

.. currentmodule:: pykeops.torch
.. autosummary:: 

    Genred
    Genred.__call__

:mod:`pykeops.torch` - Math-friendly aliases:

.. currentmodule:: pykeops.torch
.. autosummary:: 
    generic_sum
    generic_logsumexp
    generic_argmin
    generic_argkmin
    
:mod:`pykeops.torch` - The ``kernel_product`` helper:

.. currentmodule:: pykeops.torch
.. autosummary:: 

    Kernel
    kernel_product
    kernel_formulas
    Formula

:mod:`pykeops.torch.cluster` - Block-sparse reductions, allowing you to go beyond the baseline quadratic complexity of kernel operations with the optional ``ranges`` argument:

.. currentmodule:: pykeops.torch.cluster
.. autosummary:: 

    cluster_centroids
    cluster_ranges
    cluster_ranges_centroids
    from_matrix
    grid_cluster
    sort_clusters
    swap_axes


.. automodule:: pykeops.torch
    :members:
    :special-members:

.. automodule:: pykeops.torch.cluster
    :members:
