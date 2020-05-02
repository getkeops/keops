.. _`part.genred`:

Generic reductions
##################

Overview
========

The low-level interface of KeOps is the :mod:`Genred` module, which allows us to **define and reduce** generic operations. Depending on the framework, we can import :mod:`Genred` using either:

.. code-block:: python

    from pykeops.numpy import Genred  # for NumPy users, or...
    from pykeops.torch import Genred  # for PyTorch users.
    
In both cases, :mod:`Genred` is a class with no methods: its instantiation simply returns a **numerical function** that can be called at will.

1. **Instantiation**: :mod:`Genred(...)` takes as input a bunch of *strings* that specify the desired computation. It returns a **python function** or **PyTorch layer**, callable on numpy arrays or torch tensors. The syntax is:

  .. code-block:: python

    my_red = Genred(formula, aliases, reduction_op='Sum', axis=0, dtype='float32')

2. **Call**: The variable ``my_red`` now refers to a callable object wrapped around a set of custom Cuda routines. It may be used on any set of arrays (either NumPy arrays or Torch tensors) with the correct shapes, as described in the ``aliases`` argument:

  .. code-block:: python

    result = my_red(arg_1, arg_2, ..., arg_p, backend='auto', device_id=-1, ranges=None)



Documentation
=============

See the :class:`numpy.Genred<pykeops.numpy.Genred>` or :class:`torch.Genred<pykeops.torch.Genred>` API documentations for a complete description of the syntax at **instantiation** and **call** times.



.. _`part.example`:

An example
==========

Using the generic syntax, computing a Gaussian-RBF kernel product

.. math::

 \text{for } i = 1, \cdots, 1000, \quad\quad a_i =  \sum_{j=1}^{2000} \exp(-\gamma\|x_i-y_j\|^2) \,\cdot\, b_j.

can be done with:

.. code-block:: python
    
    import torch
    from pykeops.torch import Genred
    
    # Notice that the parameter gamma is a dim-1 vector, *not* a scalar:
    gamma  = torch.tensor([.5])
    # Generate the data as pytorch tensors. If you intend to compute gradients, don't forget the `requires_grad` flag!
    x = torch.randn(1000,3)
    y = torch.randn(2000,3)
    b = torch.randn(2000,2)
    
    gaussian_conv = Genred('Exp(-G * SqDist(X,Y)) * B', # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                           ['G = Pm(1)',          # First arg  is a parameter,    of dim 1
                            'X = Vi(3)',          # Second arg is indexed by "i", of dim 3
                            'Y = Vj(3)',          # Third arg  is indexed by "j", of dim 3
                            'B = Vj(2)'],         # Fourth arg is indexed by "j", of dim 2
                           reduction_op='Sum',
                           axis=1)                # Summation over "j"

    
    # N.B.: a.shape == [1000, 2]
    a = gaussian_conv(gamma, x, y, b)

    # By explicitly specifying the backend, you can try to optimize your pipeline:
    a = gaussian_conv(gamma, x, y, b, backend='GPU')
    a = gaussian_conv(gamma, x, y, b, backend='CPU')


**More examples** can be found in the :doc:`gallery <../_auto_examples/index>`.
