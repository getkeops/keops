.. figure:: _static/logo/keops_logo.png
   :width: 100% 
   :alt: Keops logo

Kernel Operations on the GPU, with autodiff, without memory overflows
---------------------------------------------------------------------

The KeOps library lets you compute generic reductions of **large 2d arrays** whose entries are given by a mathematical formula. It combines a **tiled reduction scheme** with an **automatic differentiation** engine, and can be used through Matlab, NumPy or PyTorch backends.
It is perfectly suited to the computation of **Kernel dot products**
and the associated gradients,
even when the full kernel matrix does *not* fit into the GPU memory.

Using the PyTorch backend, a typical sample of code looks like:

.. code-block:: python

    import torch
    from pykeops.torch import Genred

    # Kernel density estimator between point clouds in R^3
    my_conv = Genred('Exp(-SqNorm2(x - y))',  # formula
                     ['x = Vi(3)',            # 1st input: dim-3 vector per line
                      'y = Vj(3)'],           # 2nd input: dim-3 vector per column
                     reduction_op='Sum',      # we also support LogSumExp, Min, etc.
                     axis=1)                  # reduce along the lines of the kernel matrix

    # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
    x = torch.randn(1000000, 3, requires_grad=True).cuda()
    y = torch.randn(2000000, 3).cuda()
    a = my_conv(x, y)                               # shape (1000000, 1), a_i = sum_j exp(-|x_i-y_j|^2)
    g_x = torch.autograd.grad((a ** 2).sum(), [x])  # KeOps supports autodiff!

KeOps allows you to leverage your GPU without compromising on usability.
It provides:

* **Linear** (instead of quadratic) **memory footprint** for Kernel operations.
* Support for a wide range of mathematical **formulas**.
* Seamless computation of **derivatives**, up to arbitrary orders.
* Sum, LogSumExp, Min, Max but also ArgMin, ArgMax or K-min **reductions**.
* An interface for **block-sparse** and coarse-to-fine strategies.
* Support for **multi GPU** configurations.

KeOps can thus be used in a wide variety of settings, 
from shape analysis (LDDMM, optimal transport...)
to machine learning (kernel methods, k-means...)
or kriging (aka. Gaussian process regression).
More details are provided below:

* :doc:`Documentation <api/why_using_keops>`
* `Source code <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops>`_
* :doc:`Learning KeOps with examples <_auto_examples/index>`
* :doc:`Gallery of tutorials <_auto_tutorials/index>`
* :doc:`Benchmarks <_auto_benchmarks/index>`

**KeOps is licensed** under the `MIT license <https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/licence.txt>`_.

Projects using KeOps
--------------------

As of today, KeOps provides core routines for:

* `Deformetrica <http://www.deformetrica.org>`_, a shape analysis software
  developed by the `Aramis <https://www.inria.fr/en/teams/aramis>`_ Inria team.
* `GeomLoss <http://www.kernel-operations.io/geomloss>`_, a multiscale
  implementation of Kernel and **Wasserstein distances** that scales up to
  **millions of samples** on modern hardware.
* `FshapesTk <https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk>`_ and the
  `Shapes toolbox <https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox>`_,
  two research-oriented `LDDMM <https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping>`_ toolkits.

Authors
-------

Feel free to contact us for any bug report or feature request:

- `Benjamin Charlier <http://imag.umontpellier.fr/~charlier/>`_
- `Jean Feydy <http://www.math.ens.fr/~feydy/>`_
- `Joan Alexis Glaunès <http://www.mi.parisdescartes.fr/~glaunes/>`_


Table of content
----------------

.. toctree::
   :maxdepth: 2

   api/why_using_keops
   api/installation

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
   _auto_benchmarks/index
   _auto_tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: KeopsLab

   matlab/index

.. toctree::
   :maxdepth: 2
   :caption: Keops++

   cpp/index

