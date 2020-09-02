Generic solver
##############

Overview
========



On top of the :mod:`Genred` interface, KeOps provides a simple
`conjugate gradient solver <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ 
which can be used to solve
large-scale `Kriging/regression <https://en.wikipedia.org/wiki/Kriging>`_ 
problems on the GPU: the :mod:`KernelSolve` module.
Depending on the framework, we can import it using either:

.. code-block:: python

    from pykeops.numpy import KernelSolve  # for NumPy users, or...
    from pykeops.torch import KernelSolve  # for PyTorch users.
    
In both cases, :mod:`KernelSolve` is a class with no methods: its instantiation simply returns a **numerical function** that can be called on arbitrary input tensors.


1. **Instantiation**: :mod:`KernelSolve(...)` takes as input a bunch of *strings* that specify the desired computation. It returns a **python function** or **PyTorch layer**, callable on numpy arrays or torch tensors. The syntax is:

  .. code-block:: python

    K_inv = KernelSolve(formula, aliases, varinvalias, alpha=1e-10, axis=0, dtype='float32')

2. **Call**: The variable ``K_inv`` now refers to a callable object wrapped around a set of custom Cuda routines. It may be used on any set of arrays (either NumPy arrays or Torch tensors) with the correct shapes, as described in the ``aliases`` argument:

  .. code-block:: python

    a = K_inv( arg_1, arg_2, ..., arg_p, backend='auto', device_id=-1, ranges=None)


Documentation
=============

See the :class:`numpy.KernelSolve <pykeops.numpy.KernelSolve>` or :class:`torch.KernelSolve <pykeops.torch.KernelSolve>`  API documentation for the syntax at **instantiation** and **call** times.


An example
==========

Using the generic syntax, solving a ridge regression problem with a Cauchy kernel

.. math::

 \text{for } i = 1, \cdots, 10 000, \quad b_i =  0.1 \cdot a_i +\sum_{j=1}^{10 000} \frac{1}{1+\|x_i-x_j\|^2}\cdot a_j

with respect to the :math:`a_i`'s can be done with:

.. code-block:: python
    
    import torch
    from pykeops.torch import Genred, KernelSolve
    
    formula = "Inv(IntCst(1) + SqDist(X,Y)) * A"  # Positive definite Cauchy kernel
    aliases = ["X = Vi(3)",  # 1st arg: one point per  line,  in dimension 3
               "Y = Vj(3)",  # 2nd arg: one point per column, in dimension 3
               "A = Vj(1)"]  # 3rd arg: one scalar per column
    
    K = Genred(formula, aliases, axis=1)        # Sum with respect to "j"
    K_inv = KernelSolve(formula, aliases, "A",  # Solve with respect to "A"
                        axis=1, alpha=.1)       # Add 0.1 to the diagonal of the kernel matrix

    # Generate the data as PyTorch tensors:
    x = torch.randn(10000, 3)
    b = torch.randn(10000, 1)
    
    a = K_inv(x, x, b)  # N.B.: a.shape == [10000, 1]
    mean_squared_error = ((K(x, x, a) + .1*a - b)**2).sum().sqrt() / len(x)


**More examples** can be found in the :doc:`examples <../_auto_examples/index>` , :doc:`tutorials <../_auto_tutorials/index>` and :doc:`benchmark <../_auto_benchmarks/plot_benchmark_invkernel>`.

