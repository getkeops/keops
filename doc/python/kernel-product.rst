The kernel_product helper
##########################

On top of the low-level operators, we also provide a **kernel name parser** that lets you quickly define and work with most of the kernel products used in shape analysis.  This high-level interface is only compatible with the PyTorch backend and relies on two operators:

.. code-block:: python

    from pykeops.torch import Kernel, kernel_product

- :class:`pykeops.torch.Kernel  <pykeops.torch.Kernel>` is the name parser: it turns a string identifier (say, ``"gaussian(x,y) * linear(u,v)**2"``) into a set of KeOps formulas.
- :func:`pykeops.torch.kernel_product()  <pykeops.torch.kernel_product>` is the "numerical" torch routine. It takes as input a dict of parameters and a set of input tensors, to return a fully differentiable torch variable.

**A quick example:** here is how we can compute a *fully differentiable* Gaussian-RBF kernel product:

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



Documentation
==============

See the :doc:`API documentation<api/pytorch/KernelProduct>` for the syntax of the :class:`pykeops.torch.Kernel<pykeops.torch.Kernel>` parser and the :func:`pykeops.torch.kernel_product()<pykeops.torch.kernel_product>` routine.


An example
==========

We now showcase the computation of a **Cauchy-Binet varifold kernel** on a product space of (point,orientation) pairs.  Given:

- a set :math:`(x_i)` of **target points** in :math:`\mathbb{R}^3`;
- a set :math:`(u_i)` of **target orientations** in :math:`\mathbb{S}^2`, encoded as unit-norm vectors in :math:`\mathbb{R}^3`;
- a set :math:`(y_j)` of **source points** in :math:`\mathbb{R}^3`;
- a set :math:`(v_j)` of **source orientations** in :math:`\mathbb{S}^2`, encoded as unit-norm vectors in :math:`\mathbb{R}^3`;
- a set :math:`(b_j)` of **source signal values** in :math:`\mathbb{R}^4`;

we will compute the **target signal values**

.. math::

 a_i ~=~  \sum_j K(\,x_i,u_i\,;\,y_j,v_j\,)\,\cdot\, b_j ~=~ \sum_j k(x_i,y_j)\cdot \langle u_i, v_j\rangle^2 \cdot b_j,

where :math:`k(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`.

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
        "gamma"   : ( .5 / sigma**2 , None ) ,
    }

    # Don't forget to normalize the orientations:
    u = F.normalize(u, p=2, dim=1)
    v = F.normalize(v, p=2, dim=1)

    # We're good to go! Notice how we grouped together the "i" and "j" features:
    a = kernel_product(params, (x,u), (y,v), b)
    # a.shape == [1000, 4]

