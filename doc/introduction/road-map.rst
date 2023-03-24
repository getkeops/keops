Changelog and Road map
=======================


Design philosophy
--------------------

KeOps is developed by researchers, for researchers.
It is meant to be used by students and mathematicians
who have original ideas but lack the programming expertise
to implement them efficiently.
By providing cutting edge support for a wide range
of important but "non-Euclidean" computations, we hope to 
**stimulate research in computational mathematics and geometric data analysis**.

To reach this goal, we focus on a simple
but powerful abstraction: :doc:`symbolic matrices <why_using_keops>`.
We prioritize the support of widely available hardware
(CPUs, GPUs) over domain-specific accelerators
and strive to provide an 
**optimal level of performance without ever compromising on flexibility**.
Going forward, our main target is to let KeOps:

#. Remain easy to use, with a transparent syntax and a **smooth user experience**.

#. Open new paths for researchers, with comprehensive support for **generic formulas** and **approximation strategies**. 

#. Bring advanced numerical methods to a global audience, through a seamless **integration with standard libraries**.

.. |br| raw:: html

  <br/>


Road map
----------

In this context, we can summarize our plans
for 2022-2024 in three main axes.

**A) User experience:**

#. **Compatibility with the NumPy API.** 
   The :mod:`LazyTensor <pykeops.torch.LazyTensor>` module
   follows common PyTorch and NumPy conventions
   but cannot always be used as a plug-in replacement.
   To mitigate this compatibility issue, we intend to implement a new 
   :mod:`SymbolicTensor` wrapper that will be 100% compatible with NumPy
   while maintaining support of the :mod:`LazyTensor` API for
   the sake of backward compatibility.
   Among other features, we will notably
   add support for indexing methods
   and a :mod:`.dense()` 
   `conversion routine <https://github.com/getkeops/keops/issues/126>`_.
   This will allow users to turn small- and medium-sized
   symbolic matrices into
   dense arrays for e.g. debugging and display purposes.

#. **High-dimensional vectors.**
   As detailed in our benchmarks for 
   :doc:`kernel matrix-vector products <../_auto_benchmarks/plot_benchmark_high_dimension>`
   and K-Nearest Neighbors queries
   as well as our
   `NeurIPS 2020 paper <https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf>`_,
   KeOps is currently best suited to computations
   in spaces of dimension D < 100.
   We are now working on implementing
   efficient C++/CUDA schemes for high-dimensional variables,
   with partial support already available for
   e.g. squared Euclidean norms and dot products.
   Going further, we intend to provide support for 
   `tensor cores <https://github.com/getkeops/keops/issues/100>`_
   and quantized numerical types
   after the release of our new compilation engine.
   |br|


**B) Flexibility of the engine, approximation strategies:**

#. **Expand the KeOps syntax.** 
   Our new compilation engine will streamline
   the development of new mathematical formulas.
   Among other features, we intend to focus on
   **integer numerical types** (both for computations
   and indexing) and **tensor variables**.
   Adding support for symbolic matrices
   that have **more than two "symbolic" axes**
   is also a significant target.
   |br| |br|

#. **Block-wise matrices and sparsity masks.**
   At a higher level, we intend to support the 
   **block-wise construction** of symbolic matrices
   using a :mod:`BlockTensor([[A, B], [C, D]])` syntax.
   This module is inspired by SciPy's 
   `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_ wrapper
   and will be especially useful for applications
   to mathematical modelling and physics.
   
   Providing a `user-friendly interface <https://github.com/getkeops/keops/issues/121>`_
   for **block-wise sparsity masks**,
   band-diagonal and triangular matrices will also be
   of interest for applications to e.g. imaging sciences.
   |br|

#. **Approximation strategies.**
   Finally, we intend to progressively add support for
   approximate reduction schemes that allow
   users to **trade time for precision**.
   We are currently implementing
   IVF-like methods for K-NN search and
   `Nyström-like <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations>`_ **approximations** for sum reductions.
   Going further, we have started preliminary work on the
   `Fast and Free Memory method <https://arxiv.org/pdf/1909.05600.pdf>`_
   and other advanced strategies that best
   leverage the **geometric structure** of the computation.
   Implementing these methods on the GPU
   without loss of generality is a significant challenge,
   but KeOps provides us with the perfect platform
   to tackle it effectively.
   Long-term, we hope to provide a simple
   ``K.tol = 1e-3`` syntax for a wide range of
   symbolic matrices and help these advanced
   numerical methods to reach a global audience.



**C) Integration with the wider scientific software ecosystem:**

#. **Standard frameworks.**
   Improving the compatibility of KeOps
   with scientific computing frameworks
   is a major priority.
   Beyond PyTorch, NumPy, Matlab and R
   that are already supported,
   we are very much open to :doc:`contributions <contributing>`
   that would be related to e.g. `Julia <https://github.com/getkeops/keops/issues/144>`_ 
   or `TensorFlow <https://github.com/getkeops/keops/issues/135>`_.
   We follow closely 
   `standardization efforts <https://data-apis.org/blog/array_api_standard_release/>`_ 
   for tensor computing APIs.
   |br| |br|

#. **Domain-specific libraries.** 
   Going further, we work to let KeOps
   interact seamlessly with higher-level libraries
   such as 
   :doc:`SciPy <../_auto_tutorials/backends/plot_scipy>` 
   and 
   :doc:`GPyTorch <../_auto_tutorials/backends/plot_gpytorch>`.
   We are actively working on integration with
   `PyTorch_geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ 
   and the
   `Python Optimal Transport (POT) <https://pythonot.github.io>`_ libraries,
   which are close to our own research interests.
   In the long run, interactions with 
   `scikit-learn <https://scikit-learn.org/stable/>`_ 
   and
   `UMAP <https://umap-learn.readthedocs.io>`_
   would also be most relevant,
   but are significantly more challenging
   to setup due to the structure of their codebases.
   The 
   `cuML <https://docs.rapids.ai/api/cuml/stable/>`_
   repository could provide us with a convenient
   interface to these libraries: 
   preliminary plans are detailed on our
   GitHub `project page <https://github.com/getkeops/keops/projects>`_.
   |br| |br|


As detailed in our :doc:`contribution guide <contributing>`,
we warmly welcome help on our `GitHub repository <https://github.com/getkeops/keops/>`_
and keep the door open for internships and collaborations
that are related to this library.
So far, KeOps has been primarily developed by 
French mathematicians working in Paris and Montpellier...
but we'd be happy to diversify the team!


Changelog
---------

Our `Changelog <https://github.com/getkeops/keops/blob/main/CHANGELOG.md>`_
can be found on the `KeOps Github repository <https://github.com/getkeops/keops/>`_.
