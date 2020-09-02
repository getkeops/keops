Why using KeOps?
################

Scalable kernel operations
==========================

KeOps can now be used on a broad class of problems (:ref:`part.formula`).
But at heart, the main motivation behind this library is the need to compute fast and scalable Gaussian convolutions (**RBF kernel products**). For
**very large values** of :math:`M` and :math:`N`, given :

- a **target** point cloud :math:`(x_i)_{i=1}^M \in  \mathbb R^{M \times D}`,
- a **source** point cloud :math:`(y_j)_{j=1}^N \in  \mathbb R^{N \times D}`,
- a **signal** :math:`(b_j)_{j=1}^N \in  \mathbb R^{N}` attached to the :math:`y_j`'s,

KeOps allows you to compute efficiently
the array :math:`(a_i)_{i=1}^M \in  \mathbb R^{M}` given by

.. math::
    a_i =  \sum_j K(x_i,y_j) b_j,  \qquad i=1,\cdots,M,

where :math:`K(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)`.
On top of this, thanks to KeOps' **automatic differentiation** module,
you can also get access to the gradient of the :math:`a_i`'s with respect to the :math:`x_i`'s:

.. math::
   a_i' =  \sum_j \partial_x K(x_i,y_j) b_j,  \qquad i=1,\cdots,M,

without having to code
the formula :math:`\partial_x K(x_i,y_j) = -\tfrac{1}{\sigma^2}(x_i - y_j) \exp(-\|x_i - y_j\|^2 / 2 \sigma^2)` !

High performances
=================

In recent years, Deep Learning frameworks such as `PyTorch  <http://pytorch.org>`_ or `TensorFlow <http://www.tensorflow.org>`_ have evolved into fully-fledged applied math libraries: with negligible overhead, they bring **automatic differentiation** and **seamless GPU support** to research communities that were used to Matlab, NumPy and other tensor-centric frameworks.

Unfortunately, though, *no magic* is involved: optimised CUDA codes still have to be written for every single operation provided to end-users. Supporting all the standard mathematical routines thus comes at a **huge engineering cost** for the developers of the main frameworks.  As of 2019, this effort has been mostly restricted to the operations needed to implement **Convolutional Neural Networks**: linear algebra routines and convolutions on *grids*, images and volumes. 

Consequently, in array-centric frameworks, the standard way of computing Gaussian convolutions is to create and store in memory the full :math:`M`-by-:math:`N` kernel matrix :math:`K_{i,j}=K(x_i,y_j)`, before computing :math:`(a_i) = (K_{i,j}) (b_j)` as a matrix product.  
But for large datasets (say, :math:`M,N \geqslant 10,000`), this is not realistic: **large matrices just don't fit in GPU memories**.

KeOps is all about **letting researchers break through this memory bottleneck**. 
We rely on **online map-reduce schemes** to provide CUDA routines that "sum" the coefficients :math:`K_{i,j}\cdot b_j` as they are computed, without ever storing the full matrix :math:`K` in memory.


As evidenced by our :doc:`benchmarks <../_auto_benchmarks/index>`,
the KeOps routines **outperform** their standard counterparts
**by two orders of magnitude** on modern hardware:
on top of a reduced memory usage, they can also bring
a considerable speed-up to all applications 
that rely on massive - but simple - map-reduce operations.

.. _part.formula:

A generic framework that suits your needs
=========================================

KeOps supports **generic operations**, way beyond the simple case of kernel convolutions.
Let's say that we have at hand:

- a collection :math:`p^1, p^2, ..., p^P` of vectors.
- a collection :math:`x^1_i, x^2_i, ..., x^X_i` of vector sequences, indexed by an integer :math:`i` ranging from 1 to :math:`M`.
- a collection :math:`y^1_j, y^2_j, ..., y^Y_j` of vector sequences, indexed by an integer :math:`j` ranging from 1 to :math:`N`.
- a vector-valued function :math:`f(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)` on these input vectors.

Then, referring to the :math:`p`'s as **parameters**, the :math:`x_i`'s as **x-variables** and the :math:`y_j`'s as **y-variables**, the KeOps library allows us to compute efficiently *any* expression :math:`a_i` of the form

.. math::
    a_i = \operatorname{Reduction}_{j=1,\cdots,N}\limits \big[ f(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)  \big], \qquad i=1,\cdots,M

alongside its **derivatives** with respect to all the variables and parameters.


Feel free to browse through our :doc:`gallery of tutorials <../_auto_tutorials/index>`
for examples of applications.


Features
========

- Most common reduction operations: Summation, stabilized :doc:`LogSumExp reduction <../_auto_examples/pytorch/plot_generic_syntax_pytorch_LSE>`, :doc:`Min <../_auto_tutorials/kmeans/plot_kmeans_numpy>`, Max, :doc:`ArgKMin <../_auto_tutorials/knn/plot_knn_numpy>`, :doc:`SoftMin <../_auto_examples/numpy/plot_test_softmax_numpy>`, Softmax...
- :doc:`Block-sparse reductions <../python/sparsity>` and kernel matrices.
- Custom high-level (``'gaussian(x,y) * (1+linear(u,v)**2)'``) and low-level (``'Exp(-G*SqDist(X,Y)) * ( IntCst(1) + Pow((U|V), 2) )'``) syntaxes to compute general formulas.
- :doc:`High-order derivatives with respect to all parameters and variables <../_auto_tutorials/surface_registration/plot_LDDMM_Surface>`.
- :doc:`Non-radial kernels <../_auto_examples/pytorch/plot_anisotropic_kernels>`.
- Inverse of positive definite linear systems through the classes :class:`torch.KernelSolve <pykeops.torch.KernelSolve>` (see also :doc:`here <../_auto_examples/pytorch/plot_test_invkernel_torch>`) and  :class:`numpy.KernelSolve <pykeops.numpy.KernelSolve>` (see also :doc:`here <../_auto_examples/numpy/plot_test_invkernel_numpy>`)