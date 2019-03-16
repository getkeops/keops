PyTorch API
============

:mod:`pykeops.torch` - :doc:`Generic reductions <generic-syntax>`, with full support of PyTorch's ``autograd`` engine:

.. currentmodule:: pykeops.torch
.. autosummary:: 
    Genred

:mod:`pykeops.torch` - :doc:`Math-friendly <generic-reduction>` aliases:

.. currentmodule:: pykeops.torch
.. autosummary:: 
    generic_sum
    generic_logsumexp
    generic_argmin
    generic_argkmin
    
:mod:`pykeops.torch` - The :doc:`kernel_product <kernel-product>` helper:

.. currentmodule:: pykeops.torch
.. autosummary:: 

    Kernel
    kernel_product
    kernel_formulas
    Formula

:mod:`pykeops.torch.cluster` - :doc:`Block-sparse reductions <sparsity>`, allowing you to go beyond the baseline quadratic complexity of kernel operations with the optional ``ranges`` argument:

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
