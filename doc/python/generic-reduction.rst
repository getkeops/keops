Math-friendly syntax
====================

The ``Genred`` operator provides a pythonic interface for the KeOps library.
To let researchers use our code with maximum efficiency, we also propose
some math-friendly syntaxic sugar for 
`NumPy <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/numpy/generic/generic_ops.py>`_ and
`PyTorch <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/torch/generic/generic_ops.py>`_:


.. code-block:: python

    from pykeops.numpy import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin
    from pykeops.torch import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin

These functions are simple wrappers around the ``pykeops.numpy.Genred``
and ``pykeops.torch.Genred`` modules: they let users
specify the reduction operation and axis with **strings**
instead of **keyword arguments**.
They can be used just like the ``Genred(...)`` constructor,
and accept the following arguments:

1. ``formula`` : a *string*, just as in the :ref:`standard wrapper <part.instantiation>`.
2. ``output`` : a *string* of the form ``"Out = [Vx|Vy](Dim)"``, where

    - ``Out`` is a dummy alphanumerical alias.
    - ``Vx`` or ``Vy`` specifies whether the output is indexed by :math:`i` or :math:`j`, with a reduction on the other index.
    - ``Dim`` is the dimension of the output variable. It should be coherent with ``formula``.

3. ``*aliases`` : the (unwrapped) list of aliases.
4. ``cuda_type`` and ``opt_arg``, optional arguments just like in the :ref:`standard wrapper <part.instantiation>`.

Example
-------

Coming back to the :ref:`previous example <part.example>`, 
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
        'Exp(-G * SqDist(X,Y)) * B', # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
        'A = Vx(2)',          # Output indexed by "i",        of dim 2
        'G = Pm(1)',          # First arg  is a parameter,    of dim 1
        'X = Vx(3)',          # Second arg is indexed by "i", of dim 3
        'Y = Vy(3)',          # Third arg  is indexed by "j", of dim 3
        'B = Vy(2)' )         # Fourth arg is indexed by "j", of dim 2

    # N.B.: a.shape == [1000, 2]
    a = gaussian_conv(gamma, x, y, b)

    # By explicitly specifying the backend, you can try to optimize your pipeline:
    a = gaussian_conv(gamma, x, y, b, backend='GPU')
    a = gaussian_conv(gamma, x, y, b, backend='CPU')


















