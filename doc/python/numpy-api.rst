NumPy API
============

:mod:`pykeops.numpy` - :doc:`Generic reductions <generic-syntax>`:

.. currentmodule:: pykeops.numpy
.. autosummary:: 
    Genred


:mod:`pykeops.numpy` - Conjugate gradient solver for :doc:`Generic linear systems <generic-syntax>`:

.. currentmodule:: pykeops.numpy
.. autosummary:: 
    KernelSolve

:mod:`pykeops.numpy` - :doc:`Math-friendly <generic-reduction>` aliases:

.. currentmodule:: pykeops.numpy
.. autosummary:: 
    generic_sum
    generic_logsumexp
    generic_argmin
    generic_argkmin


:mod:`pykeops.numpy.cluster` - :doc:`Block-sparse reductions <sparsity>`, allowing you to go beyond the baseline quadratic complexity of kernel operations with the optional **ranges** argument:

.. currentmodule:: pykeops.numpy.cluster
.. autosummary:: 

    cluster_centroids
    cluster_ranges
    cluster_ranges_centroids
    from_matrix
    grid_cluster
    sort_clusters
    swap_axes


.. automodule:: pykeops.numpy
    :members:


.. automodule:: pykeops.numpy.cluster
    :members:

