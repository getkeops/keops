Why using KeOps?
################

Scalable kernel operations
==========================

KeOps can be used on a broad class of problems (:ref:`part.formula`).
But the first motivation behind this library is simple:
we intend to accelerate the computation of Gaussian convolutions on point clouds, 
also known as **RBF kernel products** on sampled data. 

Working in a vector space :math:`\mathbb{R}^{\mathrm{D}}`, let
us consider for **large values** of :math:`\mathrm{M}` and :math:`\mathrm{N}`:

- a **target** point cloud :math:`x_1, \dots, x_{\mathrm{M}} \in  \mathbb{R}^{\mathrm{D}}`, encoded as a :math:`\mathrm{M}\times\mathrm{D}` array;
- a **source** point cloud :math:`y_1, \dots, y_{\mathrm{N}} \in  \mathbb{R}^{\mathrm{D}}`, encoded as a :math:`\mathrm{N}\times\mathrm{D}` array;
- a **signal** :math:`b_1, \dots, b_{\mathrm{N}} \in  \mathbb{R}`, attached to the :math:`y_j`'s and encoded as a :math:`\mathrm{N}\times 1` vector.

Then, KeOps allows us to compute efficiently
the vector of :math:`\mathrm{M}` values :math:`a_1, \dots, a_{\mathrm{M}} \in \mathbb{R}` given by:

.. math::
    a_i \gets  \sum_{j=1}^{\mathrm{N}} k(x_i,y_j)\, b_j~,  \qquad i=1,\dots,\mathrm{M}~,

where :math:`k(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`
is a Gaussian kernel of deviation :math:`\sigma > 0`.
Thanks to the KeOps **automatic differentiation** engine,
we can also get access to the gradient of the :math:`a_i`'s with respect to the :math:`x_i`'s:

.. math::
   a_i' \gets  \sum_{j=1}^{\mathrm{N}} \partial_x k(x_i,y_j)\, b_j~,  \qquad i=1,\dots,\mathrm{M}~,

without having to write
the formula :math:`\partial_x k(x_i,y_j) = -\tfrac{1}{\sigma^2}(x_i - y_j) \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`
in our programs.


.. _part.formula:

A generic framework
=========================================

Going further, KeOps supports a wide range of **mathematical** and 
**deep learning computations**. Let's say that we have at hand:

- a collection :math:`p^1, p^2, ..., p^{\mathrm{P}}` of vectors;
- a collection :math:`x^1_i, x^2_i, ..., x^{\mathrm{X}}_i` of vector sequences, indexed by an integer :math:`i` that ranges from 1 to :math:`\mathrm{M}`;
- a collection :math:`y^1_j, y^2_j, ..., y^{\mathrm{Y}}_j` of vector sequences, indexed by an integer :math:`j` that ranges from 1 to :math:`\mathrm{N}`;
- a vector-valued function :math:`F(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)` on these input vectors, such as a small neural network;
- a :math:`\operatorname{Reduction}` or "*pooling*" operator such as a sum, a max or an arg-min.

Then, referring to the :math:`p`'s as **parameters**, the :math:`x_i`'s as **i-variables** and the :math:`y_j`'s as **j-variables**, the KeOps library allows us to compute efficiently **any expression** :math:`a_i` of the form:

.. math::
    a_i \gets \operatorname{Reduction}_{j=1,...,\mathrm{N}}\limits \big[ F(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)  \big]~, \qquad i=1,\dots,\mathrm{M}~,

alongside its **derivatives** with respect to all variables and parameters.


.. |br| raw:: html

  <br/>

Examples of applications
=========================

This type of computation is common in machine learning and applied mathematics:

- A **kernel matrix-vector product** is implemented using
  a sum reduction and a formula :math:`F(x_i,y_j,b_j)=k(x_i,y_j)\cdot b_j` that is weighted by a suitable `kernel function <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_ :math:`k(x,y)`. As detailed in the section above, the computation reads:

  .. math::
    a_i \gets \sum_{j=1}^{\mathrm{N}} k(x_i,y_j)\, b_j~,  \qquad i=1,\dots,\mathrm{M}~.

  This operation is key to `spline regression <https://en.wikipedia.org/wiki/Smoothing_spline>`_, `kernel methods <https://en.wikipedia.org/wiki/Kernel_method>`_ and the study of `Gausian processes <https://en.wikipedia.org/wiki/Gaussian_process>`_.
  In physics, we often use Newton or Coulomb kernels such as :math:`k(x,y)=1/\|x-y\|^2`: accelerating kernel products is the first step towards fast `N-body simulations <https://en.wikipedia.org/wiki/N-body_simulation>`_. |br|  |br|

- **K-Nearest Neighbors queries** are implemented using an "arg-K-min" reduction
  that returns, for all index :math:`i`, the indices :math:`(j_1,\dots,j_{\mathrm{K}})` that correspond to the K smallest values of a distance or similarity metric :math:`F(x_i,y_j)`. For instance, in a Euclidean space, we compute:

  .. math::
    a_i \gets \arg_{\mathrm{K}} \min_{j=1,\,\dots\,,\,\mathrm{N}} \|x_i - y_j\|^2 ~,  \qquad i=1,\dots,\mathrm{M}~,

  where :math:`\| x - y \|^2 = \sum_k (x[k] - y[k])^2` is a sum of squared distances.
  K-NN search is a key building block for numerous methods in data sciences, from `simple classifiers <https://scikit-learn.org/stable/modules/neighbors.html>`_ to advanced methods in `topological data analysis <https://en.wikipedia.org/wiki/Topological_data_analysis>`_ and `dimensionality reduction <https://umap-learn.readthedocs.io/en/latest/>`_. KeOps intends to provide fast runtimes for **all types of metrics**, beyond the standard Euclidean distance and cosine similarity: we refer to our :doc:`benchmarks <../_auto_benchmarks/plot_benchmark_KNN>` for an extensive discussion. |br|  |br|

- In **computer graphics** and **geometric deep learning**, we implement
  **point cloud convolutions** and 
  **message passing layers** using a function:
  
  .. math::
    F(p,x_i,y_j,f_j)=\text{Window}(x_i,y_j)\cdot \text{Filter}(p,x_i,y_j,f_j)
    
  that is the product of a neighborhood :math:`\text{Window}(x_i,y_j)` between point positions :math:`x_i`, :math:`y_j` and of a parametric filter that is applied to a collection of feature vectors :math:`f_j`. The reduction or "pooling" operator is usually a (weighted) sum or a maximum.

  Most architectures in computer vision rely on K-Nearest Neighbors graphs (":math:`x_i \leftrightarrow y_j`") to define sparse neighborhood windows. These are equal to 1 if :math:`y_j` is one of the closest neighbors of :math:`x_i` and 0 otherwise. The point convolution then reads:

  .. math::
    a_i \gets \sum_{\substack{j \text{ such that }\\ x_i \leftrightarrow y_j}} \text{Filter}(p,x_i,y_j,f_j) ~.
  
  Crucially, KeOps now also lets users work with **global point convolutions** without compromising on performances: we refer to the Section 5.3 of our `NeurIPS 2020 paper <http://jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf>`_ and to `this presentation <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full.pdf>`_ of quasi-geodesic convolutions on protein surfaces for a detailed discussion. |br|  |br|

- In **natural language processing**,
  we implement **attention layers** for `transformer networks <https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>`_ using an exponentiated dot product :math:`F(q_i,k_j)=\exp(\langle q_i,k_j\rangle/ \sqrt{\mathrm{D}})` between *query* (:math:`q_i`) and *key* (:math:`k_j`) vectors of dimension :math:`\mathrm{D}`. The reduction is a normalized matrix-vector product with an array of *value* vectors :math:`v_j` (a **soft maximum**) and the overall computation reads:

  .. math::
    a_i \gets \frac{
     \sum_{j=1}^{\mathrm{N}}
     \exp\big[ \langle q_i,k_j\rangle / \sqrt{\mathrm{D}} \big]~\cdot~ v_j
    }{
    \sum_{j=1}^{\mathrm{N}}
    \exp\big[ \langle q_i,k_j\rangle / \sqrt{\mathrm{D}}\big]
    }~.

  It can be implemented efficiently using the KeOps "Sum-SoftMax-Weight" reduction.  |br| |br|


- We implement the **Fourier transform** with
  a sum reduction and a complex exponential:
  
  .. math::
    \widehat{f_i} = \widehat{f}(\omega_i)
    ~\gets~ 
    \sum_{j=1}^{\mathrm{N}}
    \exp(i\langle \omega_i,x_j\rangle)~\cdot~ f_j ~.

  This expression evaluates the spectral content at frequency :math:`\omega_i` of a signal :math:`f` that is represented by sampled values :math:`f_j=f(x_j)` at locations :math:`x_j`.
  KeOps thus allows users to implement efficient `Fourier-Stieltjes transforms <https://en.wikipedia.org/wiki/Fourier_transform#Fourier%E2%80%93Stieltjes_transform>`_ on **non-uniform data** using both real- and complex-valued trigonometric functions. |br|  |br|

- In **optimization theory**,
  we implement the `Legendre-Fenchel transform <https://en.wikipedia.org/wiki/Legendre_transformation>`_
  or `convex conjugate <https://en.wikipedia.org/wiki/Convex_conjugate>`_
  of an arbitrary function :math:`f(x)`
  that is sampled on a point cloud :math:`x_1, \dots, x_\mathrm{N}`
  with a vector of values :math:`f_j = f(x_j)`
  using a dot product and a maximum reduction:

  .. math::
    \forall u_i \in \mathbb{R}^\mathrm{D},~~
    f^*_i = f^*(u_i) ~\gets~
    \max_{j=1,\, \dots\,,\,\mathrm{N}} \langle u_i, x_j\rangle - f_j.


- In **imaging sciences**, 
  we implement the `distance transform <https://en.wikipedia.org/wiki/Distance_transform>`_ 
  of a binary mask :math:`m_j = m(y_j) \in \{0, 1\}`
  that is defined on the rectangle domain :math:`[\![1, \text{W} ]\!] \times [\![1, \text{H} ]\!]`
  using a minimum reduction and a squared distance function:
  
  .. math::
    \forall x_i \in [\![1, \text{W} ]\!] \times [\![1, \text{H} ]\!],~~
    d_i = d(x_i) ~\gets~ 
    \min_{y_j \in [\![1, \text{W} ]\!] \times [\![1, \text{H} ]\!]}
    \|x_i-y_j\|^2 - \log(m_j) .

  We note that just like the Legendre-Fenchel transform,
  the distance transform is **separable** and can be implemented
  efficiently on 2D and 3D grids.
  Just as with `separable Gaussian convolution <https://en.wikipedia.org/wiki/Gaussian_blur#Implementation>`_,
  the trick is to apply the transform **successively** on the lines 
  and columns of the image.
  Thanks to its native support for batch processing,
  KeOps is ideally suited to these manipulations:
  it can be used to implement all types of fast separable transforms 
  on the GPU. |br|  |br|


- In `optimal transport theory <https://optimaltransport.github.io/book/>`_, 
  we implement the **C-transform** using a "min" reduction and a formula :math:`F(x_i,y_j,g_j)=\text{C}(x_i,y_j) -g_j` that penalizes the value of the ground cost function :math:`\text{C}` by that of the dual potential :math:`g` :

  .. math::
    a_i \gets \min_{j=1,\, \dots\,,\,\mathrm{N}} \big[ \text{C}(x_i,y_j) - g_j \big],  \qquad i=1,\dots,\mathrm{M}~.
  
  Going further, numerically stable **Sinkhorn iterations** correspond to the case where the minimum in the C-transform is replaced by a (rescaled) log-sum-exp reduction, known as a **soft minimum** at temperature :math:`\varepsilon > 0`:

  .. math::
    a_i \gets - \varepsilon \cdot \log \sum_{j=1}^{\mathrm{N}} \exp \tfrac{1}{\varepsilon} \big[ g_j - \text{C}(x_i,y_j) \big],  \qquad i=1,\dots,\mathrm{M}~.

  As detailed in our `NeurIPS 2020 paper <https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf>`_, KeOps speeds up modern optimal transport solvers by **one to three orders of magnitude**, from standard auction iterations to multiscale Sinkhorn loops. A collection of reference solvers is provided by the `GeomLoss library <https://www.kernel-operations.io/geomloss>`_, that now scales up to millions of samples in seconds. |br|  |br|

- Numerous **particle** and **swarming** models
  rely on **interaction steps** that fit this template to update the positions and inner states of their agents. For instance, on modest gaming hardware, KeOps can scale up simulations of `Vicsek-like systems <https://en.wikipedia.org/wiki/Vicsek_model>`_ to 
  `millions of active swimmers and flyers <https://arxiv.org/pdf/2101.10864.pdf>`_: this allows researchers to make original conjectures on their models with a minimal amount of programming effort.



Crucially, we can understand all these computations as **reductions of "symbolic" matrices** whose coefficients are given, for all indices :math:`i` and :math:`j`, by a mathematical formula :math:`F(p, x_i, y_j)`.
As detailed on the :doc:`front page <../index>` of this website,
**the KeOps library is built around this remark**. We introduce a new type of "symbolic" tensor that lets users implement all these operations efficiently, with a small memory footprint. 
Under the hood, operations on KeOps :mod:`LazyTensors <pykeops.common.lazy_tensor.GenericLazyTensor>` avoid storing in memory the matrix of values :math:`F(p,x_i,y_j)` and rely instead on fast C++/CUDA routines that are compiled on demand.
We refer to our :doc:`guided tour of the KeOps++ engine <../engine/index>` for more details.


High performances
=================

KeOps fits within a thriving ecosystem of Python/C++ libraries for scientific computing. So how does it compare with other acceleration frameworks such as 
`Numba <https://numba.pydata.org>`_, 
`Halide <https://halide-lang.org>`_, 
`TVM <https://tvm.apache.org>`_,
`Julia <https://julialang.org>`_ or 
`JAX/XLA <https://github.com/google/jax>`_?
To answer this question, let us now briefly explain the relationship between our library and the wider software environment for tensor computing.


Tensor computing on the GPU
----------------------------

In recent years, deep learning frameworks such as `PyTorch  <http://pytorch.org>`_, 
`JAX <https://github.com/google/jax>`_ and `TensorFlow <http://www.tensorflow.org>`_ have evolved into fully-fledged applied math libraries. With negligible overhead, they bring **automatic differentiation** and **seamless GPU support** to research communities that were used to Matlab, NumPy and other tensor-centric frameworks.

Unfortunately though, **no magic** is involved: optimized C++/CUDA schemes still have to be written for every operation that is provided to end-users, from matrix-vector products to fast Fourier transforms. Supporting all the standard mathematical routines thus comes at a **huge engineering cost** for the developers of the main frameworks. 

As of today, efforts in the machine learning community have been mostly focused on the operations that are needed to implement **(Convolutional) Neural Networks**: 
linear algebra routines on **dense matrices** and convolutions on **grids**, such as images and volumes.
This 
Even if other operations are also supported, they seldom
benefit from the same level of integration.

The memory bottleneck
-----------------------

As a consequence of this focus on matrix manipulations, 
the standard way of computing a Gaussian kernel convolution with PyTorch or Numpy is to create and store in memory the full :math:`\mathrm{M}\times\mathrm{N}` kernel matrix :math:`K_{i,j}=K(x_i,y_j)`, before computing :math:`(a_i) = (K_{i,j}) (b_j)` as a matrix-vector product. 
This method leverages the 
But for large datasets (say, :math:`M,N \geqslant 10,000`), it is not a realistic option: **large matrices just don't fit in GPU memories**.

KeOps is all about **letting researchers break through this memory bottleneck**. 
We rely on **online map-reduce schemes** to provide CUDA routines that "sum" the coefficients :math:`K_{i,j}\cdot b_j` as they are computed, without ever storing the full matrix :math:`K` in memory.


KeOps: a specialized tool
---------------------------

As evidenced by our :doc:`benchmarks <../_auto_benchmarks/index>`,
the KeOps routines **outperform** their standard counterparts
**by two orders of magnitude** in many settings.
On top of a reduced memory usage, they can also bring
a considerable speed-up to methods 
in machine learning, computational physics and other applied fields.



Is KeOps going to speed-up your program?
-----------------------------------------


Main features
==============

Feel free to browse through our :doc:`gallery of tutorials <../_auto_tutorials/index>`
for examples of applications.
Among other features, KeOps supports:

- :doc:`Non-radial kernels <../_auto_examples/pytorch/plot_anisotropic_kernels>`, `neural networks <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>`_ and other arbitrary formulas.
- Most common reduction operations: Summations, stabilized :doc:`LogSumExp reductions <../_auto_examples/pytorch/plot_generic_syntax_pytorch_LSE>`, :doc:`Min <../_auto_tutorials/kmeans/plot_kmeans_numpy>`, Max, :doc:`ArgKMin <../_auto_tutorials/knn/plot_knn_numpy>`, :doc:`SoftMin <../_auto_examples/numpy/plot_test_softmax_numpy>`, Softmax...
- Batch processing and :doc:`block-wise sparsity masks <../python/sparsity>`.
- :doc:`High-order derivatives <../_auto_tutorials/surface_registration/plot_LDDMM_Surface>` with respect to all parameters and variables.
- The resolution of positive definite linear systems using a :doc:`conjugate gradient solver <../_auto_benchmarks/plot_benchmark_invkernel>`.

