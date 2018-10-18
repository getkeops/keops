Generic reductions
==================

We provide two generic operations, that allow users to **define and reduce** custom operations using either a reduction. Pure numpy user should import:

.. code-block:: python

    from pykeops.numpy import Genred
    
and pyTorch user should import:

.. code-block:: python

    from pykeops.torch import Genred

In both cases, ``Genred`` is a class with no method: it can just be instantiated (one time) and then called (as many time as you need).

1. **Instantiation**: Basically, instantiation takes as input *strings* specifying the computation and return a python *function*, callable on numpy arrays or torch tensors . To instantiate an object of type ``Genred``, :ref:`the signature <part.instantiation>` is as follow:

.. code-block:: python

    myconv = Genred(formula, aliases, reduction_op='Sum', axis=0, cuda_type='float32')


2. **Call**: The variable ``myconv`` now contains a function using the compiled C++/Cuda code to compute the formula. It may then be applied on any arrays (numpy array or torch tensor ``x``, ``y`` says) with the right shape as follow:

.. code-block:: python

    result = myconv(x, y, backend='auto', device_id=-1)


The :ref:`signature at call time <part.call>` is defined below.


Complete example are given Section :ref:`part.example` below and :doc:`../_auto_examples/index`.

.. _`part.instantiation`:

Signature at instantiation
--------------------------

Formula
^^^^^^^
The first string ``formula`` specifies the desired symbolic *formula*, see :ref:`part.mathOperation`.


Aliases
^^^^^^^

The second argument ``alias`` is a list of strings of the form ``XX = TYPE(INT)``, where:

  - ``XX`` is an alphanumerical name, used in the *formula*;
  - ``TYPE`` is a *category*. One of

            =========   ===================================================================
            ``Vx``       indexation by ``i``
            ``Vy``       indexation by ``j``
            ``Pm``       no indexation, the input tensor is a *vector* and not a 2D array
            =========   ===================================================================

  - ``INT`` is an integer corresponding to the dimension (ie number of columns).

Details may also be found here: :ref:`part.varCategory`.


``reduction_op`` (keyword arg)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a string giving the reduction type. **Default** is ``reduction_op='sum'``. See :doc:`../api/math-operations`.


``axis``  (keyword arg)
^^^^^^^^^^^^^^^^^^^^^^^

An integer with binary value:

- ``axis=0`` **(default)**: generate a reduction over the first dimension (with respect to ``i``) and ouput a ``j`` variable.
- ``axis=1``: generate a reduction over the second dimension (with respect to ``j``) and ouput a ``i`` variable.


``cuda_type`` (keyword arg)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a string. It may be one of :

- ``cuda_type='float32'`` or ``'float'`` **(default)**
- ``cuda_type='float64'`` or ``'double'``

All the input arrays should then be of the declared type.


.. _`part.call`:

Signature at call
-----------------

\*arg
^^^^^

The first (non keyword) args should be numpy arrays or pytorch tensors. The order and shapes of these input variables should correspond to the orders declares in ``formula`` and ``aliases``.

``backends`` (keyword arg)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The callable routines given by ``Genred`` accepts an optional keyword argument ``backend``. Setting its value by hand may be useful while debugging and optimizing your code. Supported values are:

- ``backends="auto"`` **(default)**, let KeOps decide which backend is best suited to your data, using a simple heuristic based on the tensors' shapes.
- ``backends="CPU"``, run a `for loop <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/CpuConv.cpp>`_ on a single CPU core.
- ``backends="GPU_1D"``, use a `simple multithreading scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
- ``backends="GPU_2D"``, use a more sophisticated `2D parallelization scheme <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
- ``backends="GPU"``, let KeOps decide which one of ``"GPU_1D"`` or ``"GPU_2D"`` method will run faster on the given input. Note that if your data is already located on the GPU, KeOps won't have to load it "by hand".


``device_id`` (keyword arg)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an integer. It is useful only if your system have multi GPU. If it is 0 or positive it corresponds to the index of the GPU device to use to perform the operation. A negative value ``device_id=-1`` **(default)** let the system choose the default GPU for you.


.. _`part.example`:

Example
-------

Using the generic syntax,computing a Gaussian-RBF kernel product

.. math::

 \text{for } i = 1, \cdots, 1000, \quad\quad a_i =  \sum_{j=1}^{2000} \exp(-\sigma\|x_i-y_j\|^2) \,\cdot\, b_j.

can be done with:

.. code-block:: python
    
    import torch
    from pykeops.torch import Genred
    
    # Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
    sigma  = torch.tensor([.5])
    # Generate the data as pytorch tensors. If you intend to compute gradients, don't forget the `requires_grad` flag!
    x = torch.randn(1000,3)
    y = torch.randn(2000,3)
    b = torch.randn(2000,2)
    
    gaussian_conv = Genred('Exp(-S*SqDist(X,Y))', # f(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                           ['S = Pm(1)',          # First arg  is a parameter,    of dim 1
                            'X = Vx(3)',          # Second arg is indexed by "i", of dim 3
                            'Y = Vy(3)',          # Third arg  is indexed by "j", of dim 3
                            'B = Vy(2)'],         # Fourth arg is indexed by "j", of dim 2
                           reduction_op='sum',
                           axis=0)

    # By explicitely specifying the backend, you can try to optimize your pipeline:
    a = gaussian_conv(sigma, x, y, b)
    a = gaussian_conv(sigma, x, y, b, backend='GPU')
    a = gaussian_conv(sigma, x, y, b, backend='CPU')