.. figure:: _static/logo/keops_logo.png
   :width: 100% 
   :alt: Keops logo

Kernel Operations on the GPU, with autodiff, without memory overflows
---------------------------------------------------------------------

The KeOps library lets you compute generic reductions of **large 2d arrays** whose entries are given by a mathematical formula. It combines a tiled reduction scheme with an automatic differentiation engine, and can be used through Matlab, NumPy or PyTorch backends.
It is perfectly suited to the computation of **Kernel dot products**
and the associated gradients,
even when the full kernel matrix does *not* fit into the GPU memory.

Using the PyTorch backend, a typical sample of code looks like:

.. code-block:: python

    import torch
    from pykeops.torch import generic_sum

    # Kernel density estimator between point clouds in R^3
    my_conv = generic_sum( 'Exp( -SqNorm2(x-y) )',    # formula
                            'a = Vx(1)',  # output:    1 scalar per line
                            'x = Vx(3)',  # 1st input: dim-3 vector per line
                            'y = Vy(3)')  # 2nd input: dim-3 vector per column

    # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
    x = torch.randn( 1000000, 3, requires_grad=True).cuda()
    y = torch.randn( 2000000, 3).cuda()
    a = my_conv(x,y) # shape (1000000, 1), a_i = sum_j exp(-|x_i-y_j|^2)
    g_x = torch.autograd.grad((a**2).sum(), [x])  # KeOps supports autodiff!

KeOps allows you to leverage your GPU without compromising on usability.
It provides:

* Linear (instead of quadratic) memory footprint for Kernel operations.
* Support for a wide range of mathematical formulas.
* Seamless computation of derivatives, up to arbitrary orders.
* Sum, LogSumExp, Min, Max but also ArgMin, ArgMax or K-min reductions.
* Support for multi GPU.

KeOps can thus be used in a wide variety of settings, 
from shape analysis (LDDMM, optimal transport...)
to machine learning (kernel methods, k-means...)
or kriging (aka. Gaussian process regression).
More details are provided below:

* :doc:`Documentation <api/why_using_keops>`.
* `Source code <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops>`_
* :doc:`Learning KeOps syntax with examples <_auto_examples/index>`
* :doc:`Tutorials gallery <_auto_tutorials/index>`

Projects using KeOps
--------------------

* `Deformetrica <http://www.deformetrica.org>`_ 
* `FshapesTk <https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk>`_
* `Shapes toolbox <https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox>`_

Authors
-------

Feel free to contact us for any bug report or feature request:

- `Benjamin Charlier <http://imag.umontpellier.fr/~charlier/>`_
- `Jean Feydy <http://www.math.ens.fr/~feydy/>`_
- `Joan Alexis Glaunès <http://www.mi.parisdescartes.fr/~glaunes/>`_

Related project
---------------

You may also be interested in `Tensor Comprehensions <https://facebookresearch.github.io/TensorComprehensions/introduction.html>`_.

Table of content
----------------

.. toctree::
   :maxdepth: 2

   api/installation
   api/why_using_keops

.. toctree::
   :maxdepth: 2
   :caption: KeOps

   api/math-operations
   api/autodiff
   api/road-map

.. toctree::
   :maxdepth: 2
   :caption: PyKeops

   python/index
   _auto_examples/index
   _auto_tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: KeopsLab

   matlab/index

.. toctree::
   :maxdepth: 2
   :caption: Keops++

   cpp/index

