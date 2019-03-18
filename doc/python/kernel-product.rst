The kernel_product helper
=============================

On top of the low-level operators, we also provide a **kernel name parser** that lets you quickly define and work with most of the kernel products used in shape analysis.  This high-level interface is only compatible with the PyTorch backend and relies on two operators:

.. code-block:: python

    from pykeops.torch import Kernel, kernel_product

- :mod:`pykeops.torch.Kernel` is the name parser: it turns a string identifier (say, ``"gaussian(x,y) * linear(u,v)**2"``) into a set of KeOps formulas.
- :func:`pykeops.torch.kernel_product` is the "numerical" torch routine. It takes as input a dict of parameters and a set of input tensors, to return a fully differentiable torch variable.

**A quick example:** here is how you can compute a *fully differentiable* Gaussian-RBF kernel product:

.. code-block:: python

    import torch
    from pykeops.torch import Kernel, kernel_product

    # Generate the data as pytorch tensors
    x = torch.randn(1000,3, requires_grad=True)
    y = torch.randn(2000,3, requires_grad=True)
    b = torch.randn(2000,2, requires_grad=True)

    # Pre-defined kernel: using custom expressions is also possible!
    # Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
    sigma  = torch.tensor([.5], requires_grad=True)
    params = {
        "id"      : Kernel("gaussian(x,y)"),
        "gamma"   : .5 / sigma**2,
    }

    # Depending on the inputs' types, 'a' is a CPU or a GPU variable.
    # It can be differentiated wrt. x, y, b and sigma.
    a = kernel_product(params, x, y, b)


The Kernel parser
---------------------

**Kernel names.** The cornerstone of our high-level syntax is the :mod:`pykeops.torch.Kernel` constructor, that takes as input a **string** name and returns a pre-processed kernel identifier. A valid kernel name is built from a small set of atomic formulas, acting on arbitrary pairs of variables and combined using:

- integer constants, 
- the addition ``+``, 
- the product ``*``,
- the integer exponentiation ``**k``.

**Parameters and variables.** Every kernel name is associated to a list of *atomic formulas* (that will require **parameters**) and a list of **pairs of variables**, ordered as they are in the name string. Both *parameters* and *variables* will be required as inputs by :func:`pykeops.torch.kernel_product`. A few examples:

- ``"gaussian(x,y)"`` : one formula and one pair of variables.
- ``"gaussian(x,y) * linear(u,v)**2"`` : two formulas and two pairs of variables.
- ``"cauchy(x,y) + gaussian(x,y) * (1 + cauchy(u,v)**2)``: **three** formulas (``cauchy``, ``gaussian`` and ``cauchy`` once again) with **two** pairs of variables (``(x,y)`` first, ``(u,v)`` second)

Note that by convention, pairs of variables should be denoted by single-letter, non-overlapping duets: ``"gaussian(x',yy)"`` or ``"gaussian(x,y) + cauchy(y,z)"`` are not supported.

Atomic formulas
^^^^^^^^^^^^^^^

As of today, the `pre-defined kernel names <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/torch/kernel_product/kernels.py>`_ are:


==============================  =====================================================     ======================================
``linear(x,y)``                 :math:`\langle x,y\rangle`                                the :math:`L^2` scalar product
``gaussian(x,y)``               :math:`\exp(-\langle x-y, G\, (x-y)\rangle)`              the standard RBF kernel
``laplacian(x,y)``              :math:`\exp(-\sqrt{\langle x-y, G\, (x-y)\rangle})`       the exponential pointy kernel
``cauchy(x,y)``                 :math:`1/(1+\langle x-y, G\, (x-y)\rangle)`               a heavy-tail kernel
``inverse_multiquadric(x,y)``   :math:`1/\sqrt{1+\langle x-y, G\, (x-y)\rangle}`          a very heavy-tail kernel
``distance(x,y)``               :math:`\sqrt{\langle x-y, G\, (x-y)\rangle}`              a Euclidean norm
==============================  =====================================================     ======================================

**Defining your own formulas** is also possible, and documented in the second part of this :doc:`example <../_auto_examples/pytorch/kernel_product_syntax>`.


**Parameters.** With the exception of the linear kernel (which accepts ``None`` as its parameter), all these kernels act on arbitrary vectors of dimension `D` and are parametrized by a variable ``G`` that can represent :

=======================================  ===============================
Parameter :math:`G`                      Dimension of the tensor ``G``
=======================================  ===============================
scalar                                   dim-1 vector
diagonal matrix                          dim-`D` vector
symmetric `D`-by-`D` matrix              dim-`D*D` vector
j-varying scalar                         `N`-by-1 array
j-varying diagonal matrix                `N`-by-`D` array
j-varying symmetric `D`-by-`D` matrix    `N`-by-`D*D` array
=======================================  ===============================

If required by the user, a kernel-id can thus be used to represent non-uniform, non-radial kernels as documented in the :doc:`anisotropic_kernels example <../_auto_examples/pytorch/plot_anisotropic_kernels>`.

The kernel_product routine
------------------------------

Having created our kernel-id, and with a few torch tensors at hand, we can feed the :func:`pykeops.torch.kernel_product` numerical routine with the appropriate input. More precisely, if :mod:`Kernel("my_kernel_name...")` defines a kernel with **F formulas** and **V variable pairs**, :func:`pykeops.torch.kernel_product` will accept the following arguments:

1. A **parameters** dict with the following entries:

  - ``"id" = Kernel("my_kernel_name...")`` - **mandatory**: the kernel id, as documented above.
  - ``"gamma" = (G_0, G_1, ..., G_(F-1))`` - **mandatory**: a list or tuple of formula parameters - one per formula. As documented above, each of them can be either ``None``, a torch vector or a torch 2D tensor. Note that if **F=1**, we also accept the use of ``"gamma" = G_0`` instead of ``(G_0,)``.
  - ``"backend" = ["auto"] | "pytorch" | "CPU" | "GPU" | "GPU_1D" | "GPU_2D"`` - optional: the same set of options as in ``Genred``, with an additionnal **pure-vanilla-pytorch** backend that does *not* rely on the KeOps engine.
  - ``"mode"`` - optional, default value = ``"sum"`` : the **operation** performed on the data. The possible values are documented :ref:`below <part.kernel_modes>`.
  
2. A tuple **(X_0, ..., X_(V-1))** of torch tensors, with the same size `M` along the dimension 0. Note that if V=1, we also accept ``X_0`` in place of ``(X_0,)``.
3. A tuple ``(Y_0, ..., Y_(V-1))`` of torch tensors, with the same size `N` along the dimension 0. We should have ``X_k.size(1) == Y_k.size(1)`` for ``0 <= k <= V-1``. Note that if ``V=1``, we also accept ``Y_0`` in place of ``(Y_0,)``.
4. A torch tensor ``B`` of shape `N`-by-`E`, with `N` lines and an arbitrary number `E` of columns.
5. (optional:) A keyword argument ``mode``, a *string* whose value supersedes that of ``parameters["mode"]``.
6. (optional:) A keyword argument ``backend``, a *string* whose value supersedes that of ``parameters["backend"]``.

Then, provided that these conditions are satisfied,

.. code-block:: python

   a = kernel_product( { "id"    : Kernel("my_kernel..."),
                         "gamma" : (G_0, G_1, ..., G_(F-1)),
                         "backend" : "auto",
                         "mode"    : "sum",    },
                         (X_0,...,X_(V-1)), (Y_0,...,Y_(V-1)), B,   mode = "sum" )

defines a fully-differentiable `M`-by-`E` torch tensor:

.. math::

    a_i =  \sum_j \text{my_kernel}_{G_0, G_1, ...}(\,x^0_i,x^1_i,...\,;\,y^0_j,y^1_j,...\,) \,\cdot\, b_j,

where the kernel parameters :math:`G_k` may possibly be indexed by :math:`j`.

.. _`part.kernel_modes`:

Kernel modes
^^^^^^^^^^^^

Kernel computations are not limited to simple kernel products. We thus provide a high-level interface for the `following operations <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/pykeops/torch/kernel_product/features_kernels.py>`_:

- **Sum.** If ``mode == 'sum'``,

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, mode='sum')


.. math::

  a_i ~=~  \sum_j K_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,) \,\cdot\, b_j.

- **Log-Sum-Exp.** If ``mode == 'lse'``,

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, mode='lse')

.. math::

  a_i =  \log \sum_j \exp \big( \log(K)_{G_0, ...}(\,x^0_i,...\,;\,y^0_j,...\,) \,+\, b_j \big).

- **Scaled Log-Sum-Exp.** If ``mode == 'log_scaled'``, ``kernel_product`` accepts two additional tensor parameters ``U`` (`M`-by-1) and ``V`` (`N`-by-1) :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, U, V, mode='log_scaled')

.. math::

  a_i =  \sum_j \exp \big( \log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,u_i\,+\,v_j\big)\,\cdot\, b_j.

- **Log scaled Log-Sum-Exp.** If ``mode == 'log_scaled_lse'``, ``kernel_product`` accepts two additional tensor parameters ``U`` (`M`-by-1) and ``V`` (`N`-by-1) :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, U, V, mode='log_scaled_lse')

.. math::

  a_i =  \log \sum_j \exp \big( \log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,u_i\,+\,v_j\,+\, b_j\big).

- **Log scaled barycenter.** If ``mode == 'log_scaled_barycenter'``, ``kernel_product`` accepts three additional tensor parameters ``U`` (`M`-by-1), ``V`` (`N`-by-1) and ``C`` (`M`-by-`E`) :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, U, V, C, mode='log_scaled_barycenter')

.. math::

  a_i =  \sum_j \exp \big( \log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,u_i\,+\,v_j\big)\,\cdot\, (b_j-c_i).

- **Log-Sum-Exp mult_i.** If ``mode == 'lse_mult_i'``, ``kernel_product`` accepts an additional tensor parameter ``H`` (`M`-by-1) :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), B, H, mode='lse_mult_i')

.. math::

  a_i =  \log \sum_j \exp \big( \,h_i\cdot\log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,b_j\big).

- **Sinkhorn cost.** If ``mode == 'sinkhorn_cost'``, ``kernel_product`` accepts two tensor parameters ``S`` (`M`-by-1) and ``T`` (`N`-by-1) **instead** of ``B`` :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), S, T, mode='sinkhorn_cost')

.. math::

  a_i =  \sum_j -\log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,) \,\cdot\, \exp \big( \log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,s_i\,+\,t_j\big).


- **Sinkhorn primal cost.** If ``mode == 'sinkhorn_primal'``, ``kernel_product`` accepts four tensor parameters ``S`` (`M`-by-1), ``T`` (`N`-by-1), ``U`` (`M`-by-1) and ``V`` (`N`-by-1) **instead** of ``B`` :

.. code-block:: python

  a = kernel_product(params, (X_0,...), (Y_0,...), S, T, U, V, mode='sinkhorn_primal')

.. math::

  a_i =  \sum_j (u_i+v_j-1)\,\cdot\, \exp \big( \log(K)_{G_0,...}(\,x^0_i,...\,;\,y^0_j,...\,)\,+\,s_i\,+\,t_j\big).

**If you think that other kernel-operations should be supported, feel free to ask!**


Example: Varifold kernel on a product space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now showcase the computation of a **Cauchy-Binet varifold kernel** on a product space of (point,orientation) pairs.  Given:

- a set :math:`(x_i)` of target points in :math:`\mathbb{R}^3`;
- a set :math:`(u_i)` of target orientations in :math:`\mathbb{S}^2`, encoded as unit-norm vectors in :math:`\mathbb{R}^3`;
- a set :math:`(y_j)` of source points in :math:`\mathbb{R}^3`;
- a set :math:`(v_j)` of source orientations in :math:`\mathbb{S}^2`, encoded as unit-norm vectors in :math:`\mathbb{R}^3`;
- a set :math:`(b_j)` of source signal values in :math:`\mathbb{R}^4`;

we will compute the "target" signal values

.. math::

 a_i ~=~  \sum_j K(\,x_i,u_i\,;\,y_j,v_j\,)\,\cdot\, b_j ~=~ \sum_j k(x_i,y_j)\cdot \langle u_i, v_j\rangle^2 \cdot b_j,

where :math:`k(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / \sigma^2)`.

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from pykeops.torch import Kernel, kernel_product

    M, N = 1000, 2000 # number of "i" and "j" indices
    # Generate the data as pytorch tensors.

    # First, the "i" variables:
    x = torch.randn(M,3) # Positions,    in R^3
    u = torch.randn(M,3) # Orientations, in R^3 (for example)

    # Then, the "j" ones:
    y = torch.randn(N,3) # Positions,    in R^3
    v = torch.randn(N,3) # Orientations, in R^3

    # The signal b_j, supported by the (y_j,v_j)'s
    b = torch.randn(N,4)

    # Pre-defined kernel: using custom expressions is also possible!
    # Notice that the parameter sigma is a dim-1 vector, *not* a scalar:
    sigma  = torch.tensor([.5])
    params = {
        # The "id" is defined using a set of special function names
        "id"      : Kernel("gaussian(x,y) * (linear(u,v)**2) "),
        # gaussian(x,y) requires a standard deviation; linear(u,v) requires no parameter
        "gamma"   : ( 1./sigma**2 , None ) ,
    }

    # Don't forget to normalize the orientations:
    u = F.normalize(u, p=2, dim=1)
    v = F.normalize(v, p=2, dim=1)

    # We're good to go! Notice how we grouped together the "i" and "j" features:
    a = kernel_product(params, (x,u), (y,v), b)
    # a.shape == [1000, 4]

