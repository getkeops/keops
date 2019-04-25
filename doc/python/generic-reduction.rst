Math-friendly syntax
####################

The :class:`Genred` operator provides a pythonic interface for the KeOps library.
To let users code with maximum efficiency, we also propose
some math-friendly **syntactic sugar** for 
`NumPy <https://github.com/getkeops/keops/blob/master/pykeops/numpy/generic/generic_ops.py>`_ and
`PyTorch <https://github.com/getkeops/keops/blob/master/pykeops/torch/generic/generic_ops.py>`_:


.. code-block:: python

    from pykeops.numpy import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin
    from pykeops.torch import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin

These functions are simple wrappers around the :class:`numpy.Genred <pykeops.numpy.Genred>`
and :class:`torch.Genred <pykeops.torch.Genred>` classes: they let users
specify the reduction operation and axis with **strings**
instead of **keyword arguments**.

Documentation
=============

See the :doc:`numpy <api/numpy/GenericOps>` or the :doc:`pytorch <api/pytorch/GenericOps>` API documentation to get a complete syntax.

An example
==========

For instance, coming back to the :ref:`previous example <part.example>`,
computing a Gaussian-RBF kernel product

.. math::

 \text{for } i = 1, \cdots, 1000, \quad\quad a_i =  \sum_{j=1}^{2000} \exp(-\gamma\|x_i-y_j\|^2) \,\cdot\, b_j.

can be done with:

.. code-block:: python
    
    import torch
    from pykeops.torch import generic_sum
    
    # Notice that the parameter gamma is a dim-1 vector, *not* a scalar:
    gamma  = torch.tensor([.5])
    # Generate the data as pytorch tensors. If you intend to compute gradients, don't forget the `requires_grad` flag!
    x = torch.randn(1000,3)
    y = torch.randn(2000,3)
    b = torch.randn(2000,2)
    
    gaussian_conv = generic_sum(
        'Exp(-G * SqDist(X,Y)) * B',  # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
        'A = Vi(2)',          # Output indexed by "i",        of dim 2
        'G = Pm(1)',          # First arg  is a parameter,    of dim 1
        'X = Vi(3)',          # Second arg is indexed by "i", of dim 3
        'Y = Vj(3)',          # Third arg  is indexed by "j", of dim 3
        'B = Vj(2)' )         # Fourth arg is indexed by "j", of dim 2

    # N.B.: a.shape == [1000, 2]
    a = gaussian_conv(gamma, x, y, b)

    # By explicitly specifying the backend, you can try to optimize your pipeline:
    a = gaussian_conv(gamma, x, y, b, backend='GPU')
    a = gaussian_conv(gamma, x, y, b, backend='CPU')
