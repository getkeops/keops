Conclusion
=====================


The previous sections uncovered the inner workings of the KeOps
library. After pages and pages of technical derivations, 
**users can now reap the reward of our year-long investment in low-level software
engineering** through the user-friendly :mod:`pykeops.numpy.LazyTensor` or :mod:`pykeops.torch.LazyTensor` wrappers.


History of the project
-------------------------

**Our starting point: Computational Anatomy.**
Back in 2017, we started working on the KeOps library to give to our
esteemed colleagues of the 
`(medical) shape analysis community <https://en.wikipedia.org/wiki/Computational_anatomy>`_ an easy
access to the CUDA routines of the 
`fShapes toolkit <https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk>`_ – a 
Matlab toolbox that relies
extensively on Gaussian kernel products. This initial target was reached
pretty quickly: today, the reference `Deformetrica <http://www.deformetrica.org/>`_ 
software – maintained by the `Aramis Inria team <http://www.aramislab.fr/>`_
at the 
`ICM Institute for Brain and Spinal Cord <https://icm-institute.org/en/>`_ – 
is fully reliant on
the **PyTorch+KeOps framework**. Most of our collaborators use one of
the KeOps bindings to implement their shape processing pipelines.

As discussed in :doc:`one of our tutorials <../_auto_tutorials/surface_registration/plot_LDDMM_Surface>`, 
new “`LDDMM <https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping>`_” codebases for
statistical shape modelling are ten times slimmer (and easier to
maintain!) than they were just three years ago: **graduate students can
now get started in days instead of months**, and we expect to witness many
progresses in the field as research teams get relieved from the burden
of low-level C++ development. As far as our specialized community
of mathematicians is concerned, with more than 1,000 downloads per
month on the `PyPi repository <https://pypi.org/project/pykeops/>`_, 
KeOps is already a success.

**Reaching a wider audience.**
In 2018-2019, after several interactions with colleagues in 
**machine learning** and **optimal transport conferences**, we quickly realized that our
generic Map-Reduce engine could be used to **solve problems that go way
beyond neuro-anatomy**. Provided that some effort was made to improve the
general user experience, KeOps :mod:`LazyTensors` could be a game changer
for engineers and researchers in many applied fields.

**Today**, after months of patient re-packaging and documentation, KeOps
is a **fully-fledged open source library**
(`MIT License <https://en.wikipedia.org/wiki/MIT_License>`_) whose development
can be tracked on `GitHub <https://github.com/getkeops/keops>`_.
The Python bindings are easy to install through the PyPi
repository (:mod:`pip install pykeops`), with numerous examples available
in our :doc:`galleries <../_auto_tutorials/index>`.



**Place of KeOps in the scientific ecosystem.**
The KeOps package has **no claim to set the state-of-the-art** in High
Performance Calculus: when implemented properly, hand-written CUDA
schemes will always outperform naive GPU loops, be it for (approximate)
nearest neighbor search or B-spline interpolation.

However, as it combines a **reasonable level of performance** with the
**flexibility of a deep learning interface**, KeOps can unlock
research programs by significantly increasing the productivity of
developers. Allowing our colleagues in computational anatomy to **benefit**
from the “deep learning revolution” **without having to focus exclusively
on convolutional neural networks** was the main ambition of this work; we
now hope that this localized success can be replicated in **other fields**.



Supported reductions and formulas
---------------------------------

As discussed in our 
:doc:`introductory tutorials <../_auto_tutorials/a_LazyTensors/plot_lazytensors_a>`, 
:mod:`LazyTensors` can be built
from **any valid NumPy array or PyTorch tensor** and support a wide
range of mathematical operations. Generic, broadcasted computations
define valid programs:


.. code-block:: python

    import torch
    from pykeops.torch import LazyTensor

    A, B, M, N, D = 7, 3, 100000, 200000, 10
    x_i = LazyTensor( torch.randn(A, B, M, 1, D) )  # "i"-variable
    l_i = LazyTensor( torch.randn(1, 1, M, 1, D) )  # "i"-variable
    y_j = LazyTensor( torch.randn(1, B, 1, N, D) )  # "j"-variable
    s   = LazyTensor( torch.rand( A, 1, 1, 1, 1) )  # parameter

    F_ij = (x_i ** 1.5 + y_j / l_i).cos()            # Algebraic expression
    F_ij = F_ij - (x_i | y_j)                        # Scalar product
    F_ij = F_ij + (x_i[:,:,:,:,2] * s.relu() * y_j)  # Indexing, ReLU activation

    a_j = F_ij.sum(dim=2)  # a_j.shape = [7, 3, 200000, 10]


:mod:`LazyTensors` fully support automatic differentiation – up to
arbitrary orders – as well as a decent collection of reduction
operations. On top of the :mod:`.sum()`, :mod:`@` (matrix multiplication)
and :mod:`.logsumexp()` operators which have already been discussed in
depth, users may rely on :mod:`.min()`, :mod:`.argmin()`, 
:mod:`.min_argmin()`,
:mod:`.max()`, :mod:`.argmax()`, :mod:`.max_argmax()`, :mod:`.Kmin(K=...)`,
:mod:`.argKmin(K=...)` or :mod:`.min_argKmin(K=...)` methods to implement
their algorithms.

**Linear solver.**
Interestingly, KeOps also provides support for the resolution of
**large “mathematical” linear systems** – a critical operation in geology
(Kriging), imaging (splines), statistics (Gaussian Process regression)
and data sciences (kernel regression). Assuming that the :mod:`LazyTensor`
“**K_xx**” encodes a symmetric, positive definite matrix :math:`K_{xx}`,
the :mod:`.solve()` method:

.. code-block:: python

    a_i = K_xx.solve(b_i, alpha=alpha)

returns the solution:

.. math::

   \begin{aligned}
   a^{\star}
   =
   \operatorname*{argmin}_a  \|\, (\alpha\operatorname{Id}+K_{xx})\,a \,-\,b\,\|^2_2
   =
   (\alpha \operatorname{Id} + K_{xx})^{-1} b, \label{eq:pykeops_solver}\end{aligned}

of the linear system ":math:`(\alpha \operatorname{Id}~+~ K_{xx})\,a = b`", computed with a
conjugate gradient scheme.

**Using KeOps as a backend for high-level libraries.**
Going further, as discussed in 
our :ref:`gallery <part.backend>`, 
:mod:`LazyTensors` can be neatly interfaced
with the high-quality solvers of the 
`Scipy <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_
and 
`GPytorch <https://gpytorch.readthedocs.io/en/latest/examples/14_KeOps_Integration/KeOps_GP_Regression.html>`_ libraries. Preliminary
results with the maintainers of the latter already show remarkable
improvements to the state-of-the-art: re-running the benchmarks of
`(Wang et al., 2019) <https://arxiv.org/abs/1903.08114>`_ with a new KeOps backend, 
exact
Gaussian Process regressions that took **7 hours** to train on a cluster
of 8 top-drawer V100 GPUs (:math:`\texttt{3DRoad}` dataset, :math:`\mathrm{N} = \texttt{278,319}`,
:math:`\mathrm{D} = \texttt{3}`) 
can now be performed in **15 minutes** on a single gaming
chip, the Nvidia GeForce RTX 2080 Ti.


Future works
------------

Our :doc:`gallery of tutorials <../_auto_tutorials/index>` 
showcases an eclectic collection of
applications in machine learning, statistics, optimal transport theory
and computational anatomy.
We carry on working towards a **closer integration** with the **Python**
scientific stack and
will improve/implement **R** and **Julia** bindings in months to come.
We also plan to implement boilerplate features such as
row- and column-wise indexing, block-wise definition of LazyTensors
and a full support of tensor variables. **Additional low-level profiling**
should also help us to converge towards **optimal runtimes**.

By making our routines freely available to the general public, we hope to
**help the applied maths community to catch up with the state-of-the-art**
in computer science: **in 2019**, bruteforce quadratic algorithms should
have no problem scaling up to **millions of samples** in minutes; clever
approximation schemes are only needed if users intend to perform
real-time analysis or scale to Gigabytes of data.



**Our long-term goal: fast approximation schemes.**
Long-term, our main challenge will be to reconcile KeOps with the
**rich literature** in numerical mathematics that focuses on fast
**approximation schemes for kernel dot products** – which are often
referred to as discrete **convolutions** in computational geometry or
discrete **integral operators** in physics. 
Adapting ideas from the 
`Nyström <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations>`_,
`Fast Multipole <https://math.nyu.edu/faculty/greengar/shortcourse_fmm.pdf>`_ and 
`Fast & Free Memory Methods <https://arxiv.org/abs/1909.05600>`_ to GPU chips,
we hope
to **let users trade time for accuracy** with
a simple ``K.tol = 1e-3`` interface by 2020.







