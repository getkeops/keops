Formulas
========

.. _`part.varCategory`:

Variables category
------------------

KeOps uses a low-level syntax written in C++/Cuda to define virtually any reduction operation of the form

.. math::

   \alpha_i = \operatorname{Reduction}_j \big[ f(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]

or

.. math::

   \beta_j = \operatorname{Reduction}_i \big[ f(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]

where "Reduction" can be the Sum, LogSumExp, min,... operations (see :ref:`part.reduction` for the full list)


Each of the variables :math:`x^k_{\iota_k}` is specified by its positional index ``k``, its category :math:`\iota_k\in\{i,j,\emptyset\}` (meaning that the variable is indexed by ``i``, by ``j``, or is a fixed parameter across indices) and its dimension :math:`d_k`. These three characteristics are encoded as follows :

* category is given via the keywords

=========  ============================
 keyword    meanning
=========  ============================
 ``Vx``     variable indexed by ``i``
 ``Vy``     variable indexed by ``j``
 ``Pm``     parameter
=========  ============================

* positional index ``k`` and dimension :math:`d_k` are given as two integer parameters put into parenthesis after the category-specifier keyword.

For instance, ``Vx(2,4)`` specifies a variable indexed by ``i``, given as the third (``k=2``) input in the function call, and representing a vector of dimension 4.

Of course, using the same index ``k`` for two different variables is not allowed and will be rejected by the compiler.

An example
----------

Asume we want to compute the following sum

.. math::

  f(p,x,y,\beta)_i = \left(\sum_{j=1}^M (p -\beta_j )^2 \exp(x_i^u + y_j^u) \right)_{u=1,2,3} \in \mathbb R^3


where :math:`p \in \mathbb R` is a constant, :math:`x \in \mathbb R^{N\times 3}`, :math:`y \in \mathbb R^{M\times 3}`, :math:`\beta \in \mathbb R^M`. From the "variables" symbolic placeholders, one can build the function ``f`` using the syntax 

.. code-block:: cpp

    Square(Pm(0,1)-Vy(3,1))*Exp(Vx(1,3)+Vy(2,3))

in which ``+`` and ``-`` denote the usual addition of vectors, ``Exp`` is the (element-wise) exponential function and ``*`` denotes scalar-vector multiplication.

The operations available are listed :ref:`part.mathOperation`.

Variables can be given aliases, allowing us to write human-readable expressions for our formula. For example, one may define ``p=Pm(0,1)``, ``x=Vx(1,3)``, ``y=Vy(2,3)``, ``beta=Vy(3,1)``, and write the previous computation as

.. code-block:: cpp

    Square(p-beta)*Exp(x+y)

.. _`part.mathOperation`:

Math operators
--------------

Here is a list of the implemented operations that can be used in formulas:

======================   =========================================================================================================
``f * g``                 scalar-vector multiplication (if ``f`` is scalar) or vector-vector element-wise multiplication
``f + g``                 addition of two vectors
``f - g``                 difference between two vectors or minus sign
``f / g``                 element-wise division
``(f | g)``               scalar product between vectors
``Exp(f)``                element-wise exponential function
``Log(f)``                element-wise natural logarithm
``Pow(f, N)``             N-th power of ``f`` (element-wise), where N is a fixed-size integer
``Powf(f, g)``            power operation ``-`` alias for ``Exp(g*Log(f))``
``Square(f)``             element-wise square
``Sqrt(f)``               element-wise square root
``Rsqrt(f)``              element-wise inverse square root
``SqNorm2(f)``            squared L2 norm, same as ``(f|f)``
``Norm2(f)``              L2 norm, same as ``Sqrt((f|f))``
``Normalize(f)``          normalize vector, same as ``Rsqrt(SqNorm2(f)) * f``
``Grad(f,x,e)``           gradient of ``f`` with respect to the variable ``x``, with ``e`` as the "grad_output" to backpropagate
``IntCst(N)``             integer constant N
``Zero(N)``               vector of zeros of size N
``Inv(f)``                element-wise inverse (1 ./ f)
``IntInv(N)``             alias for ``Inv(IntCst(N)) : 1/N``
``Elem(f, M)``            extract M-th element of vector ``f``
``ElemT(f, N, M)``        insert scalar value ``f`` at position M in a vector of zeros of length N
``Extract(f, M, D)``      extract sub-vector from vector ``f`` (M is starting index, D is dimension of sub-vector)
``ExtractT(f, M, D)``     insert vector ``f`` in a larger vector of zeros (M is starting index, D is dimension of output)
``Concat(f, g)``          concatenation of vectors ``f`` and ``g``
``MatVecMult(f, g)``      matrix-vector product ``f x g``: ``f`` is vector interpreted as matrix (column-major), ``g`` is vector
``VecMatMult(f, g)``      vector-matrix product ``f x g``: ``f`` is vector, ``g`` is vector interpreted as matrix (column-major)
``TensorProd(f, g)``      tensor product ``f x g^T`` : ``f`` and ``g`` are vectors of sizes m and n, output is of size mn.
``GradMatrix(f, v)``      matrix of gradient (i.e. transpose of the jacobian matrix)
======================   =========================================================================================================


.. _`part.reduction`:

Reductions
----------

Here is a list of the implemented operations that can be used to reduce an array:

===========      ===========================      ============================================
Sum              summation                        :math:`\sum(\cdots)`
LogSumExp        log-sum-exp                      :math:`\log\left(\sum\exp(\cdots)\right)`
Min              min                              :math:`\min(\cdots)`
ArgMin           argmin                           :math:`\text{argmin}(\cdots)`
MinArgMin        minargmin                        :math:`(\min(...),\text{argmin}(\cdots))`
Max              max                              :math:`\max(\cdots)`
ArgMax           argmax                           :math:`\text{argmax}(\cdots)`
MaxArgMax        maxargmax                        :math:`(\max(...),\text{argmax}(\cdots))`
KMin             K first order statistics         :math:`(\cdots)_{(1)},\ldots,(\cdots)_{(K)}`
ArgKMin          indices of order statistics      :math:`(1),\ldots,(K)`
KMinArgKMin      (KMin,ArgKMin)                   :math:`\left((\cdots)_{(1)},\ldots,(\cdots)_{(K)},(1),\ldots,(K)\right)`
===========      ===========================      ============================================


