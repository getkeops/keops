Why using KeOps?
################

Scalable kernel operations
==========================

KeOps can be used on a broad class of problems (:ref:`part.formula`).
But the first motivation behind this library is very simple:
we want to accelerate the computation of Gaussian convolutions on point clouds, 
also known as **RBF kernel products** on sampled data. 

Working in a vector space :math:`\mathbb{R}^{\mathrm{D}}`, let
us consider for **large values** of :math:`\mathrm{M}` and :math:`\mathrm{N}`:

- a **target** point cloud :math:`x_1, \dots, x_{\mathrm{M}} \in  \mathbb{R}^{\mathrm{D}}`, encoded as a :math:`\mathrm{M}\times\mathrm{D}` array;
- a **source** point cloud :math:`y_1, \dots, y_{\mathrm{N}} \in  \mathbb{R}^{\mathrm{D}}`, encoded as a :math:`\mathrm{N}\times\mathrm{D}` array;
- a **signal** :math:`b_1, \dots, b_{\mathrm{N}} \in  \mathbb{R}`, attached to the :math:`y_j`'s and encoded as a :math:`\mathrm{N}\times 1` vector.

Then, KeOps allows us to compute efficiently
the vector of :math:`\mathrm{M}` values :math:`a_1, \dots, a_{\mathrm{M}} \in \mathbb{R}` given by:

.. math::
    a_i =  \sum_{j=1}^{\mathrm{N}} K(x_i,y_j)\, b_j~,  \qquad i=1,\dots,\mathrm{M}~,

where :math:`K(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`
is Gaussian kernel of deviation :math:`\sigma > 0`.
Thanks to the KeOps **automatic differentiation** engine,
we can also get access to the gradient of the :math:`a_i`'s with respect to the :math:`x_i`'s:

.. math::
   a_i' =  \sum_{j=1}^{\mathrm{N}} \partial_x K(x_i,y_j)\, b_j~,  \qquad i=1,\dots,\mathrm{M}~,

without having to write
the formula :math:`\partial_x K(x_i,y_j) = -\tfrac{1}{\sigma^2}(x_i - y_j) \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`
in our programs.


.. _part.formula:

A generic framework that suits your needs
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
    a_i = \operatorname{Reduction}_{j=1,\cdots,N}\limits \big[ F(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)  \big]~, \qquad i=1,\cdots,M~,

alongside its **derivatives** with respect to all the variables and parameters.

This type of computation is common in machine learning and applied mathematics:

- When the .
  The kernel, Gaussian process regression.
  In computational physics, accelerating these 
  N-body

- K-Nearest Neighbors queries.

- Message passing

- Attention layers.

- Fourier transform

- C-transform (min), LogSumExp: Sinkhorn iterations.

- Vicsek, swarming model.

Interaction steps.

In terms of tensor operations, they correspond to reductions on
"symbolic" matrices whose coefficients are given by the mathematical formula
:math:`F(p, x_i, y_j)` for all indices :math:`i` and :math:`j`.
As detailed on the :doc:`front page <../index>` of this website,
this Our main user interface 

From a computational perspective, . Deep down, this is : see our :doc:`guided tour of the KeOps inner engine <../engine/index>` for more details.


High performances
=================

In recent years, deep learning frameworks such as `PyTorch  <http://pytorch.org>`_, 
`JAX <https://github.com/google/jax>`_ and `TensorFlow <http://www.tensorflow.org>`_ have evolved into fully-fledged applied math libraries. With negligible overhead, they bring **automatic differentiation** and **seamless GPU support** to research communities that were used to Matlab, NumPy and other tensor-centric frameworks.

Unfortunately though, **no magic** is involved: optimized C++/CUDA codes still have to be written for every operation that is provided to end-users, from matrix-vector products to fast Fourier transforms. Supporting all the standard mathematical routines thus comes at a **huge engineering cost** for the developers of the main frameworks. 
This

As of today, their efforts have been mostly focused on the operations that are needed to implement **(Convolutional) Neural Networks**: 
dense **linear algebra** routines and convolutions on **grids**, such as images and volumes.
This 
Even if other operations are also supported, they seldom
benefit from the same level of integration.

As a consequence of this focus on matrix manipulations, 
the standard way of computing a Gaussian kernel convolution with PyTorch or Numpy is to create and store in memory the full :math:`\mathrm{M}\times\mathrm{N}` kernel matrix :math:`K_{i,j}=K(x_i,y_j)`, before computing :math:`(a_i) = (K_{i,j}) (b_j)` as a matrix-vector product. 
This method leverages the 
But for large datasets (say, :math:`M,N \geqslant 10,000`), it is not a realistic option: **large matrices just don't fit in GPU memories**.

KeOps is all about **letting researchers break through this memory bottleneck**. 
We rely on **online map-reduce schemes** to provide CUDA routines that "sum" the coefficients :math:`K_{i,j}\cdot b_j` as they are computed, without ever storing the full matrix :math:`K` in memory.


As evidenced by our :doc:`benchmarks <../_auto_benchmarks/index>`,
the KeOps routines **outperform** their standard counterparts
**by two orders of magnitude** in many settings.
On top of a reduced memory usage, they can also bring
a considerable speed-up to methods 
in machine learning, computational physics and other applied fields.



Is KeOps going to speed-up your program?
========================================


Main features
==============

Feel free to browse through our :doc:`gallery of tutorials <../_auto_tutorials/index>`
for examples of applications.


- Most common reduction operations: Summation, stabilized :doc:`LogSumExp reduction <../_auto_examples/pytorch/plot_generic_syntax_pytorch_LSE>`, :doc:`Min <../_auto_tutorials/kmeans/plot_kmeans_numpy>`, Max, :doc:`ArgKMin <../_auto_tutorials/knn/plot_knn_numpy>`, :doc:`SoftMin <../_auto_examples/numpy/plot_test_softmax_numpy>`, Softmax...
- :doc:`Block-sparse reductions <../python/sparsity>` and kernel matrices.
- Custom high-level (``'gaussian(x,y) * (1+linear(u,v)**2)'``) and low-level (``'Exp(-G*SqDist(X,Y)) * ( IntCst(1) + Pow((U|V), 2) )'``) syntaxes to compute general formulas.
- :doc:`High-order derivatives with respect to all parameters and variables <../_auto_tutorials/surface_registration/plot_LDDMM_Surface>`.
- :doc:`Non-radial kernels <../_auto_examples/pytorch/plot_anisotropic_kernels>`.
- Inverse of positive definite linear systems through the classes :class:`torch.KernelSolve <pykeops.torch.KernelSolve>` (see also :doc:`here <../_auto_examples/pytorch/plot_test_invkernel_torch>`) and  :class:`numpy.KernelSolve <pykeops.numpy.KernelSolve>` (see also :doc:`here <../_auto_examples/numpy/plot_test_invkernel_numpy>`)

