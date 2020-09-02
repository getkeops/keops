KeOps Lazy tensors
================================


**Current state of the art.**
The level of performance provided by KeOps 
may surprise readers who grew accustomed to
the limitations of tensor-centric frameworks. 
As discussed in previous sections, 
common knowledge in the machine
learning community asserts that “kernel” computations can *not* scale to
large point clouds with the CUDA backends of modern libraries:
:math:`\mathrm{N}`-by-:math:`\mathrm{N}` kernel matrices **stop fitting contiguously on
the Device memory** as soon as :math:`\mathrm{N}` exceeds some threshold in the
10,000–50,000 range that depends on the GPU chip.

Focusing on the key operation involved, the *kernel dot product* or
*discrete convolution*:

.. math::

   \begin{aligned}
    \begin{array}{ccccl}
       \text{KP} & : & \mathbb{R}^{\mathrm{M}\times \mathrm{D}}\times \mathbb{R}^{\mathrm{N}\times \mathrm{D}} \times \mathbb{R}^{\mathrm{N}\times\mathrm{E}} 
         & \to & \mathbb{R}^{\mathrm{M}\times\mathrm{E}} \\
        & & \big( (x_i), (y_j), (b_j)\big) & \mapsto & 
        (a_i) ~~\text{with}~~
        a_i = \sum_{j=1}^\mathrm{N} k(x_i, y_j)\,b_j~,
   \end{array} \end{aligned}

most authors are tempted to introduce an :math:`\mathrm{M}`-by-:math:`\mathrm{N}`
kernel matrix :math:`K_{ij} = k(x_i,y_j)` and implement the operation
above as a matrix dot product

.. math::

   \begin{aligned}
   \text{KP}\big((x_i), (y_j), (b_j)\big) ~~=~~
   (K_{ij}) \cdot (b_j)~. \end{aligned}

To accelerate computations, a flourishing literature has then focused
on the construction of **low-rank approximations** of the linear
operator :math:`(K_{ij})`. Common
methods rely on random sampling schemes, multiscale decompositions of
the input data or take advantage of specific properties of the kernel
function :math:`k` – in our case, a convenient Gaussian blob.

Our focus: exact Map-Reduce computations
-------------------------------------------

As discussed in detail in 
our conclusion, such approximation strategies have a long history and a clear
intrinsic value. Nevertheless, acknowledging the fact that 
**progresses can also be made through low-level software engineering**, 
we decide to
tackle this problem in a completely different way. Brushing aside the
elegant but *inefficient* matrix decomposition written above,
the KeOps package **directly optimizes** the kernel sum 
by understanding it as a **Map-Reduce composition** of the
operators:

.. math::

   \begin{aligned}
   &\begin{array}{ccccc}
       \quad\quad\text{Map} & : & \mathbb{R}^{\mathrm{D}}\times \mathbb{R}^{\mathrm{D}} \times \mathbb{R}^{\mathrm{E}} 
         & \to & \mathbb{R}^{\mathrm{E}} \\
        & & ( x, y, b ) & \mapsto & 
        k(x, y)\,b
   \end{array} \\[.5cm]
   &
   \begin{array}{ccccc}
     \text{and}\quad\text{Reduce} & : & \mathbb{R}^{\mathrm{E}} \times \mathbb{R}^{\mathrm{E}} 
       & \to & \mathbb{R}^{\mathrm{E}} \\
      & & ( a, a' ) & \mapsto & 
      a + a'
   \end{array}
    \label{eq:KP_map_reduce}\end{aligned}

over the *indexing* indices :math:`i \in \left[\!\left[ 1,\mathrm{M}\right]\!\right]` and *reduction*
indices :math:`j\in \left[\!\left[ 1,\mathrm{N}\right]\!\right]`.

**Parenthesis: are we re-packaging the wheel?**
This approach is **common in the computer graphics literature** but tends to
be strictly limited to C++/CUDA programming guides: with its
emphasis on real-time rendering and explicit models, the graphics
community never felt the need to develop high-level libraries that would
be suited to machine learning research.

In this context, **our scientific contribution does not lie in any new
theorem or algorithm**. Described in the next few sections, the tools on
which KeOps relies (the backpropagation algorithm, online
Map-Reduce CUDA schemes and symbolic variadic templating) are all very
well-known in their respective communities. But as we **combine** them in
a versatile framework, endowed with a transparent interface and a
comprehensive documentation, we allow them to **reach a much wider
audience** and have, hopefully, a positive and fertilizing impact.

A generic framework 
-------------------------

The seminal Theano library combined the flexibility of high-level
Python frameworks with a first-class support of convolutional
architectures on the GPU. In the same vein, the KeOps package 
**puts the spotlight on Map-Reduce schemes for (off-grid) sampled data**, an
algorithmic structure that we deem to be relevant in many fields that
are related to data sciences and shape analysis.

Removing all the Python sugar coating, the workhorse of our
library is a **Generic Reduction** (:mod:`Genred`) operator that supports
a wide family of formulas. Let us assume that we have:  

    #. A collection :math:`p^1, p^2, \dots, p^P` of vectors.

    #. A collection :math:`x^1_i, x^2_i, \dots, x^X_i` of vector sequences,
       indexed by an integer :math:`i` in :math:`\left[\!\left[ 1,\mathrm{M}\right]\!\right]`.

    #. A collection :math:`y^1_j, y^2_j, \dots, y^Y_j` of vector sequences,
       indexed by an integer :math:`j` in :math:`\left[\!\left[ 1,\mathrm{N}\right]\!\right]`.

    #. A vector-valued formula
       :math:`F(p^1, p^2, \dots, x^1_i, x^2_i, \dots, y^1_j, y^2_j, \dots)` on
       these input vectors.

    #. A :math:`\operatorname{Reduction}` operation that may be a sum, an
       arg-min, a log-sum-exp, etc.


Then, referring to the :math:`p`\ ’s as **parameters**, the
:math:`x_i`\ ’s as :math:`i`-**variables** and the :math:`y_j`\ ’s as
:math:`j`-**variables**, a single KeOps :mod:`Genred` call allows
users to compute efficiently the expression

  .. math::

     \begin{aligned}
         a_i ~&=~ 
         \operatorname{Reduction}_{j=1,\dots,\mathrm{N}}\limits 
         \big[ F(p^1, p^2, \dots, x^1_i, x^2_i, \dots, y^1_j, y^2_j, \dots)  \big] \qquad
         \text{for}~~ i=1,\dots,\mathrm{M},  \label{eq:keops_genred}\end{aligned}

alongside its **derivatives** with respect to all variables and
parameters. As showcased in 
our gallery of tutorials, this level of generality allows KeOps to handle
K-Nearest-Neighbors classification, K-Means clustering, Gaussian
Mixture Model-fitting and many other tasks.

The :mod:`LazyTensor` abstraction
-------------------------------------

Implementation details are covered in the next few sections but probably
won’t interest most mathematicians. Wary of making users step outside of
the convenient tensor-centric paradigm, we give a **matrix-like interface**
to the generic reduction above. Through a new
:mod:`LazyTensor` wrapper for NumPy arrays and PyTorch tensors,
users may specify formulas :math:`F` without ever leaving the comfort of
a NumPy-like interface.

KeOps :mod:`LazyTensors` embody the concept of
“symbolic” tensors that are not sparse in the traditional sense, but can
nevertheless be handled more efficiently than large
:math:`\mathrm{M}`-by-:math:`\mathrm{N}` arrays by using:

    #. **A symbolic mathematical formula** :math:`F`, the “**.formula**”
       attribute that is encoded as a well-formed string, manipulated with
       Python operations and parsed at reduction time.

    #. **A collection of “small” data arrays** :math:`p`, :math:`x` 
       and :math:`y`, the “**.variables**” list of *parameters*, :math:`i`- 
       and :math:`j`-variables that are needed to evaluate the formula :math:`F`.


Coming back to the example of the previous section, 
we may display the :mod:`LazyTensor` **K\_ij** using:

.. code-block:: python

    >>> print(K_ij) 
    ... KeOps LazyTensor 
    ...     formula: Exp((Minus(Sum(Square((Var(0,3,0) - Var(1,3,1))))) / Var(2,1,2))) 
    ...     shape: (1000, 1000)

Here, the **Var(index, dimension, [ i | j | parameter ] )** placeholders
refer to the data arrays **q\_i**, **q\_j** and **1/(2\*s*s)** that are
stored in the list of **K_ij.variables**. As we call a supported
reduction operator such as the matrix dot-product “**@**” on **K\_ij**,
this information is fed to the :mod:`Genred` engine and a result is
returned as a genuine, differentiable PyTorch tensor: things just
work smoothly, with full support of **operator broadcasting** and 
**batch dimensions**.