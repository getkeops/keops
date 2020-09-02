.. _`part.generic_formulas`:

Formulas and syntax
###################


KeOps lets us define any reduction operation of the form

.. math::

   \alpha_i = \operatorname{Reduction}_j\limits \big[ F(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]

or

.. math::

   \beta_j = \operatorname{Reduction}_i\limits \big[ F(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]

where :math:`F` is a symbolic formula, the :math:`x^k_{\iota_k}`'s are vector variables
and 
:math:`\text{Reduction}` is a Sum, LogSumExp or any other standard operation (see :ref:`part.reduction` for the full list of supported reductions).

We now describe the symbolic syntax that 
can be used through all KeOps bindings.

.. _`part.varCategory`:

Variables: category, index and dimension
========================================


At a low level, every variable :math:`x^k_{\iota_k}` is specified by its **category** :math:`\iota_k\in\{i,j,\emptyset\}` (meaning that the variable is indexed by :math:`i`, by :math:`j`, or is a fixed parameter across indices), its **positional index** :math:`k` and its **dimension** :math:`d_k`. 

In practice, the category :math:`\iota_k` is given through a keyword

=========  ============================================================
 keyword    meaning
=========  ============================================================
 ``Vi``     variable indexed by :math:`i`
 ``Vj``     variable indexed by :math:`j`
 ``Pm``     parameter
=========  ============================================================

followed by a :math:`(k,d_k)` or (index,dimension) pair of integers.
For instance, ``Vi(2,4)`` specifies a variable indexed by :math:`i`, given as the third (:math:`k=2`) input in the function call, and representing a vector of dimension :math:`d_k=4`.

**N.B.:** Using the same index ``k`` for two variables with different dimensions or categories is not allowed and will be rejected by the compiler.


.. _`part.reservedWord`:

Reserved words
===============

=========  ============================================================
 keyword    meaning
=========  ============================================================
 ``Ind``    indexes sequences
=========  ============================================================

followed by a sequence  :math:`(i_0, i_1, \cdots)` of integers. For instance, ``Ind(2,4,2,5,12)`` can be used as parameters for some operations.

.. _`part.mathOperation`:

Math operators
==============

To define formulas with KeOps, you can use simple arithmetics:

======================   =========================================================================================================
``f * g``                 scalar-vector multiplication (if ``f`` is scalar) or vector-vector element-wise multiplication
``f + g``                 addition of two vectors
``f - g``                 difference between two vectors or minus sign
``f / g``                 element-wise division (N.B. ``f`` can be scalar, in fact ``f / g`` is the same as ``f * Inv(g)``)
``(f | g)``               scalar product between vectors
======================   =========================================================================================================

Elementary functions:

======================   =========================================================================================================
``Inv(f)``                element-wise inverse ``1 ./ f``
``Exp(f)``                element-wise exponential function
``Log(f)``                element-wise natural logarithm
``Sin(f)``                element-wise sine function
``Cos(f)``                element-wise cosine function
``Pow(f, P)``             ``P``-th power of ``f`` (element-wise), where ``P`` is a fixed integer
``Powf(f, g)``            power operation, alias for ``Exp(g*Log(f))``
``Square(f)``             element-wise square, faster than ``Pow(f,2)``
``Sqrt(f)``               element-wise square root, faster than ``Powf(f,.5)``
``Rsqrt(f)``              element-wise inverse square root, faster than ``Powf(f,-.5)``
``Abs(f)``                element-wise absolute value
``Sign(f)``               element-wise sign function (``-1`` if ``f<0``, ``0`` if ``f=0``, ``1`` if ``f>0``)
``Step(f)``               element-wise step function (``0`` if ``f<0``, ``1`` if ``f>=0``)
``ReLU(f)``               element-wise ReLU function (``0`` if ``f<0``, ``f`` if ``f>=0``)
======================   =========================================================================================================


Simple vector operations:

=========================   =============================================================================================================
``SqNorm2(f)``               squared L2 norm, same as ``(f|f)``
``Norm2(f)``                 L2 norm, same as ``Sqrt((f|f))``
``Normalize(f)``             normalize vector, same as ``Rsqrt(SqNorm2(f)) * f``
``SqDist(f, g)``              squared L2 distance, same as ``SqNorm2(f - g)``
=========================   =============================================================================================================

Generic squared Euclidean norms, with support for scalar, diagonal and full (symmetric)
matrices. If ``f`` is a vector of size `N`, depending on the size of
``s``, ``WeightedSqNorm(s,f)`` may refer to:

- a weighted L2 norm :math:`s[0]\cdot\sum_{0\leqslant i < N} f[i]^2`  if ``s`` is a vector of size 1.
- a separable norm :math:`\sum_{0\leqslant i < N} s[i]\cdot f[i]^2`  if ``s`` is a vector of size `N`.
- a full anisotropic norm :math:`\sum_{0\leqslant i,j < N} s[iN+j] f[i] f[j]`  if ``s`` is a vector of size `N * N` such that ``s[i*N+j]=s[j*N+i]`` (i.e. stores a symmetric matrix).

============================   =============================================================================================================
``WeightedSqNorm(s, f)``         generic squared euclidean norm
``WeightedSqDist(s, f, g)``      generic squared distance, same as ``WeightedSqNorm(s, f-g)``
============================   =============================================================================================================

Constants and padding/concatenation operations:

======================   =========================================================================================================
``IntCst(N)``             integer constant N
``IntInv(N)``             alias for ``Inv(IntCst(N))`` : 1/N
``Zero(N)``               vector of zeros of size N
``Sum(f)``                sum of elements of vector ``f``
``Max(f)``                max of elements of vector ``f``
``Min(f)``                min of elements of vector ``f``
``ArgMax(f)``             argmax of elements of vector ``f``
``ArgMin(f)``             argmin of elements of vector ``f``
``Elem(f, M)``            extract M-th element of vector ``f``
``ElemT(f, N, M)``        insert scalar value ``f`` at position M in a vector of zeros of length N
``Extract(f, M, D)``      extract sub-vector from vector ``f`` (M is starting index, D is dimension of sub-vector)
``ExtractT(f, M, D)``     insert vector ``f`` in a larger vector of zeros (M is starting index, D is dimension of output)
``Concat(f, g)``          concatenation of vectors ``f`` and ``g``
``OneHot(f, D)``          encodes a (rounded) scalar value as a one-hot vector of dimension D
======================   =========================================================================================================

Elementary dot products:

==============================================     ====================================================================================================================================================================================================================================================================================
``MatVecMult(f, g)``                                matrix-vector product ``f x g``: ``f`` is vector interpreted as matrix (column-major), ``g`` is vector
``VecMatMult(f, g)``                                vector-matrix product ``f x g``: ``f`` is vector, ``g`` is vector interpreted as matrix (column-major)
``TensorProd(f, g)``                                tensor cross product ``f x g^T``: ``f`` and ``g`` are vectors of sizes M and N, output is of size MN.
``TensorDot(f, g, dimf, dimg, contf, contg)``       tensordot product ``f : g``(similar to `numpy's tensordot <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html>`_ in the spirit): ``f`` and ``g`` are tensors of sizes listed in ``dimf`` and ``dimg`` :ref:`index sequences <part.reservedWord>` and contracted along the dimensions listed in ``contf`` and ``contg`` :ref:`index sequences <part.reservedWord>`. The ``MatVecMult``, ``VecMatMult`` and ``TensorProd`` operations are special cases of ``TensorDot``.
==============================================     ====================================================================================================================================================================================================================================================================================

Symbolic gradients:

======================   =========================================================================================================
``Grad(f,x,e)``           gradient of ``f`` with respect to the variable ``x``, with ``e`` as the "grad_input" to backpropagate
``GradMatrix(f, v)``      matrix of gradient (i.e. transpose of the jacobian matrix)
======================   =========================================================================================================


.. _`part.reduction`:

Reductions
==========

The operations that can be used to reduce an array are described in the following table.

=========================    =====================  ============================================================================================================================  =========================================================================
code name                    arguments              mathematical expression                                                                                                       remarks
                                                    (reduction over j)
=========================    =====================  ============================================================================================================================  =========================================================================
``Sum``                      ``f``                  :math:`\sum_j f_{ij}`                                                                                        
``Max_SumShiftExp``          ``f`` (scalar)         :math:`(m_i,s_i)` with :math:`\left\{\begin{array}{l}m_i=\max_j f_{ij}\\s_i=\sum_j\exp(f_{ij}-m_i)\end{array}\right.`         - core KeOps reduction for ``LogSumExp``.
                                                                                                                                                                                  - gradient is a pseudo-gradient, should not be used by itself
``LogSumExp``                ``f`` (scalar)         :math:`\log\left(\sum_j\exp(f_{ij})\right)`                                                                                   only in Python bindings
``Max_SumShiftExpWeight``    ``f`` (scalar), ``g``  :math:`(m_i,s_i)` with :math:`\left\{\begin{array}{l}m_i=\max_j f_{ij}\\s_i=\sum_j\exp(f_{ij}-m_i)g_{ij}\end{array}\right.`   - core KeOps reduction for ``LogSumExpWeight`` and ``SumSoftMaxWeight``.
                                                                                                                                                                                  - gradient is a pseudo-gradient, should not be used by itself
``LogSumExpWeight``          ``f`` (scalar), ``g``  :math:`\log\left(\sum_j\exp(f_{ij})g_{ij}\right)`                                                                             only in Python bindings
``SumSoftMaxWeight``         ``f`` (scalar), ``g``  :math:`\left(\sum_j\exp(f_{ij})g_{ij}\right)/\left(\sum_j\exp(f_{ij})\right)`                                                 only in Python bindings
``Min``                      ``f``                  :math:`\min_j f_{ij}`                                                                                                         no gradient
``ArgMin``                   ``f``                  :math:`\text{argmin}_jf_{ij}`                                                                                                 gradient xreturns zeros
``Min_ArgMin``               ``f``                  :math:`\left(\min_j f_{ij} ,\text{argmin}_j f_{ij}\right)`                                                                    no gradient
``Max``                      ``f``                  :math:`\max_j f_{ij}`                                                                                                         no gradient
``ArgMax``                   ``f``                  :math:`\text{argmax}_j f_{ij}`                                                                                                gradient returns zeros
``Max_ArgMax``               ``f``                  :math:`\left(\max_j f_{ij},\text{argmax}_j f_{ij}\right)`                                                                     no gradient
``KMin``                     ``f``, ``K`` (int)     :math:`\begin{array}{l}\left[\min_j f_{ij},\ldots,\min^{(K)}_jf_{ij}\right]                                                   no gradient
                                                    \\(\min^{(k)}\text{means k-th smallest value})\end{array}`                                                                     
``ArgKMin``                  ``f``, ``K`` (int)     :math:`\left[\text{argmin}_jf_{ij},\ldots,\text{argmin}^{(K)}_j f_{ij}\right]`                                                gradient returns zeros
``KMin_ArgKMin``             ``f``, ``K`` (int)     :math:`\left([\min^{(1...K)}_j f_{ij} ],[\text{argmin}^{(1...K)}_j f_{ij}]\right)`                                            no gradient
=========================    =====================  ============================================================================================================================  =========================================================================

**N.B.:** All these reductions, except ``Max_SumShiftExp`` and ``LogSumExp``, are vectorized : whenever the input ``f`` or ``g`` is vector-valued, the output will be vector-valued, with the corresponding reduction applied element-wise to each component.

**N.B.:** All reductions accept an additional optional argument that specifies wether the reduction is performed over the j or the i index.
(see :ref:`part.cppapi` and :ref:`part.genred`)



.. _`formula.example`:

An example
==========

Assume we want to compute the sum

.. math::

  F(p,x,y,a)_i = \left(\sum_{j=1}^N (p -a_j )^2 \exp(x_i^u + y_j^u) \right)_{i=1,\ldots,M, u=1,2,3} \in \mathbb R^{M\times 3}


where:

- :math:`p \in \mathbb R` is a **parameter**,
- :math:`x \in \mathbb R^{M\times 3}` is an **x-variable** indexed by :math:`i`,
- :math:`y \in \mathbb R^{N\times 3}` is an **y-variable** indexed by :math:`j`,
- :math:`a \in \mathbb R^N` is an **y-variable** indexed by :math:`j`.

Using the **variable placeholders** presented above and the
mathematical operations listed in :ref:`part.mathOperation`,
we can define ``F`` as a **symbolic string**

.. code-block:: cpp

    F = "Sum_Reduction( Square( Pm(0,1) - Vj(3,1) )  *  Exp( Vi(1,3) + Vj(2,3) ), 1 )"

in which ``+`` and ``-`` denote the usual addition of vectors, ``Exp`` is the (element-wise) exponential function and ``*`` denotes scalar-vector multiplication.
The second argument ``1`` of the ``Sum_Reduction`` operator
indicates that the summation is performed with respect to the :math:`j`
index: a ``0`` would have been associated with an :math:`i`-reduction.

Note that in all bindings, variables can be defined through **aliases**.
In this example, we may write ``p=Pm(0,1)``, ``x=Vi(1,3)``, ``y=Vj(2,3)``, ``a=Vj(3,1)`` and thus give ``F`` through a much friendlier expression:

.. code-block:: cpp

    F = "Sum_Reduction( Square(p - a) * Exp(x + y), 1 )"
