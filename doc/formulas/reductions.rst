Reductions
================================


**Following the same design principles**, :math:`\operatorname{Reduction}`
operators are implemented in the 
`keops/core/reductions/*.h <https://github.com/getkeops/keops/tree/master/keops/core/reductions>`_ headers.
Taking as input an arbitrary symbolic formula :mod:`F`, 
:mod:`Reduction<F>`
templates encode generic Map-Reduce schemes 
and should implement a few standard routines.

Summation
--------------

In the case of the simple **Sum** reduction 
(`Sum_Reduction.h <https://github.com/getkeops/keops/blob/master/keops/core/reductions/Sum_Reduction.h>`_ header), 
these can be described as:

#. An :mod:`InitializeReduction` method, which **fills up the running buffer**
   “:math:`a`” of 
   :doc:`our Map-Reduce algorithm <../engine/map_reduce_schemes>`
   – a vector of size :mod:`F::DIM` – with zeros
   **before the start of the loop** on the reduction index :math:`j`.
   |br|

#. A :mod:`ReducePair` method, which takes as input a pointer to the
   running buffer :math:`a`, a pointer to the result
   :math:`F_{i,j} = F(p^1,\dots,x^1_i,\dots,y^1_j,\dots)` and 
   **implements the in-place reduction**:

   .. math::

      \begin{aligned}
            a~\gets~a~+~F_{i,j}.
            \end{aligned}

#. A :mod:`FinalizeOutput` method, which **post-processes the buffer**
   :math:`a` before saving its value in the output array. This is a
   useful step for **argmin**-like reductions; but in the case of the
   **sum**, no post-processing is needed.


The online Log-Sum-Exp trick
--------------------------------

More interestingly, the 
`Max_SumShiftExp_Reduction.h <https://github.com/getkeops/keops/blob/master/keops/core/reductions/Max_SumShiftExp_Reduction.h>`_ 
header implements an
**online version** of the well-known 
`Log-Sum-Exp trick <https://en.wikipedia.org/wiki/LogSumExp>`_: 
a **factorization of the maximum** in the computation of

.. math::

   \begin{aligned}
   \log \sum_{j=1}^\mathrm{N} \exp (F_{i,j})
   ~=~
   m_i~+~
   \log \sum_{j=1}^\mathrm{N} \exp (F_{i,j}\,-\,m_i),
   ~~\text{with}~~
   m_i~=~ \max_{j=1}^\mathrm{N} F_{i,j} \label{eq:logsumexp_trick}\end{aligned}

that ensures the computation of this important quantity – the linchpin
of maximum likelihood estimators and entropic Optimal Transport solvers
– without numeric overflows.

Merging the content of our **C++ header** and of the 
**Python post-processing step** implemented in 
`pykeops/common/operations.py <https://github.com/getkeops/keops/blob/master/pykeops/common/operations.py>`_,
assuming that :math:`F_{i,j} = F(p^1,\dots,x^1_i,\dots,y^1_j,\dots)` is
a scalar quantity, we may describe its behaviour as follows:

#. The :mod:`InitializeReduction` method ensures that our running buffer
   :math:`a` is a **vector of size 2** that encodes the current value of the
   inner summation as an **explicit (exponent, mantissa)** or “(maximum,
   residual)” pair of float numbers: at any stage of the
   computation, the pair :math:`(m,r)` encodes the positive number
   :math:`e^{m}\cdot r` with the required precision. We initially set
   the value of :math:`a` to
   :math:`(-\infty, 0)\simeq e^{-\infty}\cdot 0`.
   |br|

#. The :mod:`ReducePair` method takes as input a pointer to the result
   :math:`F_{i,j}` of the computation, a pointer to the running buffer
   :math:`a = (m, r) \simeq e^m\cdot r` and 
   **implements the in-place update**:

   .. math::

      \begin{aligned}
                (m,r)
                ~\gets~
                \begin{cases}
                      \big( ~m~, ~\,r + \phantom{r\cdot{}} e^{F_{i,j} - m} \big) & \text{if}~ m \geqslant F_{i,j}\\
                      \big( F_{i,j},~ 1 + r \cdot e^{m - F_{i,j}} \big) & \text{otherwise.}
                \end{cases}
                \end{aligned}

   This is a **numerically stable way of writing the sum reduction**:

   .. math::

      \begin{aligned}
                e^m \cdot r
                ~\gets~
                e^m\cdot r \, +\, e^{F_{i,j}}
                ~=~
                \begin{cases}
                      ~e^m~\cdot(\,r+ \phantom{r\cdot{}} e^{F_{i,j}-m}) & \text{if}~ m \geqslant F_{i,j}\\
                      e^{F_{i,j}}\cdot (1 + r\cdot e^{m-F_{i,j}}) & \text{otherwise.}
                \end{cases}
                \end{aligned}

#. :mod:`FinalizeOutput` **post-processes the buffer**
   :math:`a = (m,r) \simeq e^{m}\cdot r` by applying the final
   “:math:`\log`” operation, returning a value of :math:`m+\log(r)` for
   the full reduction.


.. |br| raw:: html

  <br/><br/>
