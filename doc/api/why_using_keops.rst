Why using KeOps?
================

Scalable kernel operations
--------------------------

Even though KeOps can now be used on a broad class of formulas (:ref:`part.formula`),
the first motivation for writing this library was the need to compute fast and scalable Gaussian convolutions (aka. **RBF kernel products**). For
**very large values** of :math:`M` and :math:`N`, given :

- a target point cloud :math:`(x_i)_{i=1}^M \in  \mathbb R^{M \times D}`,
- a source point cloud :math:`(y_j)_{j=1}^N \in  \mathbb R^{N \times D}`,
- a signal :math:`(b_j)_{j=1}^N \in  \mathbb R^{N}` attached to the :math:`y_j`'s,

KeOps allows you to compute
the array :math:`(a_i)_{i=1}^M \in  \mathbb R^{M}` given by

.. math::
    a_i =  \sum_j K(x_i,y_j) b_j,  \qquad i=1,\cdots,M,

where :math:`K(x_i,y_j) = \exp(-\|x_i - y_j\|^2 / \sigma^2)`.
On top of this, thanks to KeOps' automatic differentiation module,
you can also compute the gradient of the :math:`a_i`'s with respect to the :math:`x_i`'s:

.. math::
   a_i' =  \sum_j \partial_x K(x_i,y_j) b_j,  \qquad i=1,\cdots,M,

without even having to code
the formula :math:`\partial_x K(x_i,y_j) = -\tfrac{2}{\sigma^2}(x_i - y_j) \exp(-\|x_i - y_j\|^2 / \sigma^2)` !

High performances
-----------------

In recent years, Deep Learning frameworks such as `PyTorch  <http://pytorch.org>`_, `TensorFlow <http://www.tensorflow.org>`_ or `Theano <http://deeplearning.net/software/theano/>`_ have evolved into fully-fledged applied math libraries: with negligible overhead, they bring **automatic differentiation** and **seamless GPU support** to research communities that were used to Matlab and NumPy.

Unfortunately, though, *no magic* is involved: in the backyard, optimised CUDA codes still have to be written for every operation provided to end-users. Supporting all the standard mathematical routines thus comes at a **huge engineering cost** for the developers of the main frameworks.  As of 2018, this effort has been mostly restricted to the operations needed to implement Convolutional Neural Networks: linear algebra routines and convolutions on *grids*. 


In array-centric frameworks, a standard way of computing Gaussian convolutions is thus to create and store in memory the full :math:`M`-by-:math:`N` kernel matrix :math:`K_{i,j}=K(x_i,y_j)`, before computing :math:`(a_i) = (K_{i,j}) (b_j)` as a matrix product.  
But for large datasets (say, :math:`M,N \geqslant 10,000`), this is not realistic: **large matrices just don't fit in GPU memories**.

This is where we fit in: KeOps is all about **letting researchers break through the memory bottleneck**. Relying on an **online map-reduce scheme**, we provide CUDA 
routines that "sum" the coefficients :math:`K_{i,j}\cdot b_j` as they are computed,
without ever storing the full matrix :math:`K` in memory.




.. figure:: ../_static/benchmark.png
   :width: 100% 
   :alt: benchmark keops

.. _part.formula:

A generic framework that fits your needs
----------------------------------------

KeOps supports **generic operations**, way beyond the simple case of kernel convolutions.
Let's say that you have at hand:

- a collection :math:`p^1, p^2, ..., p^P` of vectors.
- a collection :math:`x^1_i, x^2_i, ..., x^X_i` of vector sequences, indexed by an integer :math:`i` ranging from 1 to :math:`M`.
- a collection :math:`y^1_j, y^2_j, ..., y^Y_j` of vector sequences, indexed by an integer :math:`j` ranging from 1 to :math:`N`.
- a vector-valued function :math:`f(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)` on these input vectors.

Then, referring to the :math:`p`'s as **parameters**, the :math:`x`'s as **x-variables** and the :math:`y`'s as **y-variables**, the KeOps library allows you to compute efficiently *any* expression :math:`a_i` of the form

.. math::
    a_i = \operatorname{Reduction}_{j=1,\cdots,N}\limits \big[ f(p^1, p^2,..., x^1_i, x^2_i,..., y^1_j, y^2_j, ...)  \big], \qquad i=1,\cdots,M

alongside its **derivatives** with respect to all the variables and parameters.

As of today, we support:

- Various reduction operations: Summation, (online, numerically stable) :doc:`LogSumExp reduction <../_auto_examples/plot_generic_syntax_pytorch_LSE>`, :doc:`min <../_auto_tutorials/kmeans/plot_kmeans_numpy>`, max, ...
- Custom high-level (``'gaussian(x,y) * (1+linear(u,v)**2)'``) and low-level (``'Exp(-G*SqDist(X,Y)) * ( IntCst(1) + Pow((U|V), 2) )'``) syntaxes to compute general formulas.
- :doc:`High-order derivatives with respect to all parameters and variables <../_auto_tutorials/surface_registration/plot_LDDMM_Surface>`.
- :doc:`Non-radial kernels <../_auto_examples/plot_anisotropic_kernels>`.
