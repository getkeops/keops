Generic reductions
==================

The low-level interface of KeOps is the :mod:`Genred` module, which allows users to **define and reduce** generic operations. Depending on your framework, you may import :mod:`Genred` using either:

.. code-block:: python

    from pykeops.numpy import Genred  # for NumPy users, or...
    from pykeops.torch import Genred  # for PyTorch users.
    
In both cases, :mod:`Genred` is a class with no methods: its instantiation simply returns a **numerical function** that can be called at will.


1. **Instantiation**: :mod:`Genred(...)` takes as input a bunch of *strings* that specify the desired computation. It returns a python *function*, callable on numpy arrays or torch tensors. :ref:`The syntax <part.instantiation>` is:

  .. code-block:: python

    my_red = Genred(formula, aliases, reduction_op='Sum', axis=0, cuda_type='float32')

2. **Call**: The variable **my_red** now refers to a Python function wrapped around a set of custom Cuda routines. It may be applied to any set of arrays (either NumPy arrays or Torch tensors) with the correct shapes, as encoded in the **aliases** argument:

  .. code-block:: python

    result = my_red( arg_1, arg_2, ..., arg_p, backend='auto', device_id=-1, ranges=None)


Examples are given :ref:`below <part.example>` and in the :doc:`gallery <../_auto_examples/index>`.
We now fully document the syntax at :ref:`instantiation <part.instantiation>` and :ref:`call <part.call>` times.

.. _`part.instantiation`:

Instantiation: create a new KeOps routine
-----------------------------------------

:func:`pykeops.numpy.Genred` and :func:`pykeops.torch.Genred` take as input *five arguments* that describe a new computation:

1. **formula**: a *string* that specifies a symbolic formula with our :ref:`custom syntax <part.mathOperation>`. Please use aliases instead of the cumbersome ``Vx(cat,dim)`` placeholders.

2. **aliases**: a *list of strings* of the form ``"AL = TYPE(DIM)"`` that specify the categories and dimensions of the input variables. Here:

  - ``AL`` is an alphanumerical alias, used in the ``formula``.
  - ``TYPE`` is a *category*. One of

            =========   ===================================================================
            ``Vx``       indexation by :math:`i`
            ``Vy``       indexation by :math:`j`
            ``Pm``       no indexation, the input tensor is a *vector* and not a 2D array
            =========   ===================================================================

  - ``DIM`` is an integer, the dimension of the current variable.

  Crucially, unlike in our :ref:`C++ interface <part.varCategory>`, you don't have to specify by hand the *indices* of each variable: they are implicitly encoded in the ordering of the list of aliases.

3. **reduction_op** (optional, default value = ``"Sum"``): a *string* that specifies the reduction used on the formula. The first column of the :ref:`table of reductions <part.reduction>` lists the possible values. **N.B.:** As of today, vector-valued output is only supported for the ``"Sum"`` reduction. All the other reductions expect the ``formula`` to be scalar-valued.


4. **axis**  (optional, default value = 0): an *integer* that specifies the dimension of the "kernel matrix" that is reduced. Possible values are:

  - **axis** = 0: reduction with respect to :math:`i`, ouputs a :math:`j` variable.
  - **axis** = 1: reduction with respect to :math:`j`, ouputs a :math:`i` variable.

5. **cuda_type** (optional, default value = ``"float32"``): a *string* that specifies the numerical **dtype** of the input and output arrays. Possible values are:

  - **cuda_type** = ``"float32"`` or ``"float"``.
  - **cuda_type** = ``"float64"`` or ``"double"``.



.. _`part.call`:

Call: compute a value on the GPU
--------------------------------

The output of a :func:`Genred(...)` instantiation is 
a Python function that can be called directly on numerical tensors.
Its arguments are:

1. ***args** (NumPy arrays or PyTorch tensors): the input numerical arrays, which should all have the same **dtype**, be *contiguous* and live on the *same device*. KeOps expects one array per alias, with the following compatibility rules:
  
  - All ``Vx(Dim_k)`` variables are **2d-tensors** with the same number :math:`M` of lines and ``Dim_k`` columns.
  - All ``Vy(Dim_k)`` variables are **2d-tensors** with the same number :math:`N` of lines and ``Dim_k`` columns.
  - All ``Pm(Dim_k)`` variables are **1d-tensors** (vectors) of size ``Dim_k``.

2. **backend** (optional, default value = ``"auto"``): a *string* that specifies the algorithm used to compute and reduce the numerical values on the device. Possible values are:

  - **backend** = ``"auto"`` : let KeOps decide which backend is best suited to your data, using a simple heuristic based on the tensors' shapes.
  - **backend** = ``"CPU"`` : run a `for loop <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/CpuConv.cpp>`_ on a single CPU core.
  - **backend** = ``"GPU_1D"`` : use a `simple multithreading scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
  - **backend** = ``"GPU_2D"`` : use a more sophisticated `2D parallelization scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
  - **backend** = ``"GPU"`` : let KeOps decide which one of the ``"GPU_1D"`` or the ``"GPU_2D"`` scheme will run faster on the given input.


3. **device_id** (optional, default value = -1): an *integer* that specifies the GPU that should be used to perform the computation; a negative value lets your system choose the default GPU. This argument is only useful if your system has access to several GPUs.

4. **ranges** (optional, default value = None = full reduction): a 6-uple of integer tensors that specifies a **block-sparse** reduction mask, thus allowing you to **skip useless computations** as often as possible. Its use is described in a :doc:`dedicated page of the documentation <sparsity>`.

The output of a KeOps call is always a **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1 or **axis** = 0, respectively) and a number of columns that is inferred from the **formula**.

.. _`part.example`:

Example
-------

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
                            'X = Vx(3)',          # Second arg is indexed by "i", of dim 3
                            'Y = Vy(3)',          # Third arg  is indexed by "j", of dim 3
                            'B = Vy(2)'],         # Fourth arg is indexed by "j", of dim 2
                           reduction_op='Sum',
                           axis=1)                # Summation over "j"

    
    # N.B.: a.shape == [1000, 2]
    a = gaussian_conv(gamma, x, y, b)

    # By explicitly specifying the backend, you can try to optimize your pipeline:
    a = gaussian_conv(gamma, x, y, b, backend='GPU')
    a = gaussian_conv(gamma, x, y, b, backend='CPU')
