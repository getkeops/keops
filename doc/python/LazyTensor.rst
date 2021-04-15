.. _`part.lazytensor`:

LazyTensors
##################

Overview
========

**High-level** interface for the KeOps routines is provided by the :mod:`pykeops.numpy.LazyTensor <pykeops.numpy.LazyTensor>` and :mod:`pykeops.torch.LazyTensor <pykeops.torch.LazyTensor>` modules, which allow users to perform **efficient, semi-symbolic computations** on large NumPy arrays and PyTorch tensors. As displayed on this website's :doc:`front page, <../index>` these new tensor types can be used with **very little overhead** in existing Python programs:

.. code-block:: python

    # Create two arrays with 3 columns and a (huge) number of lines, on the GPU
    import torch
    x = torch.randn(1000000, 3, requires_grad=True).cuda()
    y = torch.randn(2000000, 3).cuda()

    # Turn our Tensors into KeOps symbolic variables:
    from pykeops.torch import LazyTensor
    x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (1e6, 1, 3)
    y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, 2e6,3)

    # We can now perform large-scale computations, without memory overflows:
    D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (1e6,2e6,1) matrix of squared distances
    K_ij = (- D_ij).exp()               # Symbolic (1e6,2e6,1) Gaussian kernel matrix
    # Note that in fact nothing has been computed yet, everything will be done in the final reduction step

    # Now we come back to vanilla PyTorch Tensors or NumPy arrays using
    # reduction operations such as .sum(), .logsumexp() or .argmin().
    # Here, the kernel density estimation   a_i = sum_j exp(-|x_i-y_j|^2)
    # is computed using a CUDA online map-reduce routine that has a linear
    # memory footprint and outperforms standard PyTorch implementations
    # by two orders of magnitude. All actual computations are performed at this step.
    a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1), 
    g_x = torch.autograd.grad((a_i ** 2).sum(), [x])  # KeOps supports autograd!


Documentation
=============

Starting with the :doc:`KeOps 101 tutorial <../_auto_tutorials/a_LazyTensors/plot_lazytensors_a>`,
most examples in our :doc:`gallery <../_auto_tutorials/index>`
rely on :mod:`LazyTensors <pykeops.common.lazy_tensor.GenericLazyTensor>`:
going through this collection of **real-life demos** is probably
the best way of getting familiar with the KeOps user interface.

Going further, please refer to the :mod:`GenericLazyTensor <pykeops.common.lazy_tensor.GenericLazyTensor>` (common), :mod:`numpy.LazyTensor <pykeops.numpy.LazyTensor>` (NumPy) and :mod:`torch.LazyTensor <pykeops.torch.LazyTensor>` (PyTorch) API documentations for an exhaustive list of all supported operations.

