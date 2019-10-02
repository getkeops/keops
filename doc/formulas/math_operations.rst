Math operations
================================

**Key files.**
As detailed 
`earlier <../engine/repository>`_, 
our parsing grammar for symbolic formulas is
described in terms of **abstract C++ types** 
implemented in the 
`keops/core/formulas/*/*.h <https://github.com/getkeops/keops/tree/master/keops/core/formulas>`_
headers. These files provide a **comprehensive list of mathematical
operators** and rely on the primitives implemented in the
`keops/core/pack <https://github.com/getkeops/keops/tree/master/keops/core/pack>`_ 
and 
`keops/core/autodiff <https://github.com/getkeops/keops/tree/master/keops/core/autodiff>`_ 
subfolders: 
abstract
`unary <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/UnaryOp.h>`_
and 
`binary <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/BinaryOp.h>`_ 
operators, 
`tuples <https://github.com/getkeops/keops/blob/master/keops/core/pack/Pack.h>`_  of variables and parameters, 
`variables <https://github.com/getkeops/keops/tree/master/keops/core/autodiff>`_, 
`gradients <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/Grad.h>`_.

In practice
-----------------

To give a glimpse of **how KeOps works under the hood**, let us present
a small excerpt from the 
`formulas/maths/ <https://github.com/getkeops/keops/tree/master/keops/core/formulas/maths>`_ 
subfolder – the declaration
of the :mod:`Log(...)` operator
in the `Log.h <https://github.com/getkeops/keops/blob/master/keops/core/formulas/maths/Log.h>`_ 
header:

.. code-block:: cpp

    // 1. Declare a new Unary Operation - the pointwise logarithm:
    template < class F >
    struct Log : UnaryOp<Log,F> {
    
        // 2. Declare a new attribute: dimension of Log(F) = dimension of F:
        static const int DIM = F::DIM;
    
        // 3. Utility method: pointwise Logarithm should be displayed as "Log":
        static void PrintIdString(std::stringstream& str) { str << "Log"; }

        // 4. Actual C++ implementation of our operator:
        static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
            for(int k = 0; k < DIM; k++)
                out[k] = log( outF[k] );
	}

        // 5. Define a new alias for the "backward" operator of F...
        template < class V, class GRADIN >
        using DiffTF = typename F::template DiffT<V,GRADIN>;
    
        // 6. And use it to implement the "backward" of Log(F):
        template < class V, class GRADIN >
        using DiffT = DiffTF<V, Mult< Inv<F>, GRADIN> >;
    };

    // 7. Define a user-friendly alias "Log(...)" for "Log<...>":
    #define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()


As evidenced here, the implementation of a new operator goes through 
**seven compulsory steps**:

#. The **declaration** of a new operation as an instance of the abstract
   :mod:`UnaryOp` or :mod:`BinaryOp` templates. These are defined in the
   `keops/core/autodiff <https://github.com/getkeops/keops/tree/master/keops/core/autodiff>`_ 
   folder with a set of standard methods and
   attributes. The operand :mod:`F` of :mod:`Log<F>` is an arbitrary formula,
   **recursively encoded as a templated structure**.
   |br|

#. The specification of a few **standard attributes**. Here, the
   dimension of the vector :mod:`Log(F)` – accessed as :mod:`Log<F>::DIM` in
   typical C++ fashion – is equal to that of :mod:`F`. Our logarithm
   is applied pointwise and does not affect the shape of the underlying
   vector.
   |br|

#. The specification of some **utility methods**. Here, the string
   identifier :mod:`PrintIdString` may be used to access the formula that
   is encoded within any KeOps C++ template.
   |br|

#. The actual **implementation** of our operator, that is to be executed
   **within the Thread memory of each CUDA core**. As specified in the
   abstract definition of :mod:`UnaryOp`, the inline method 
   :mod:`Operation`
   takes as input a C++ array :mod:`outF`, the vector-valued output
   of our operand :mod:`F`. It computes the relevant pointwise logarithms
   using a standard CUDA-maths routine and stores them in a new :mod:`out`
   buffer of size :mod:`Log<F>::DIM`. In practice, modern C++
   compilers may simplify this operation as an in-place modification of
   the values stored in :mod:`outF`.
   |br|

#. **Prepare the chain rule** by defining an alias for the adjoint
   “backward” operator of the operand :mod:`F` with respect to an arbitrary
   differentiation variable :mod:`V`. As explained in
   our :doc:`introduction to backpropagation <../autodiff_gpus/backpropagation>`, 
   the new operator
   :math:`\partial_{\texttt{V}} F` is a formal expression that takes as
   input the variables “:math:`x = (p^1,\dots,x^1_i,\dots,y^1_j,\dots)`”
   of :mod:`F` and a new vector “:math:`a`” of size :mod:`F::DIM`, the
   gradient vector :mod:`GRADIN` or “:math:`x_i^*`” that is backpropagated
   through the whole computational graph. Understood as the **adjoint or
   “transpose” of the differential** of :mod:`F`, the application of this
   operator is encoded within KeOps as a new templated expression
   :mod:`F::DiffT<V,GRADIN>` that should implement the computation of
   :math:`\partial_\texttt{V} F \cdot \texttt{GRADIN}`.
   |br|

#. **Implement the chain rule** recursively, using the templated
   expression above: :mod:`DiffTF = F::DiffT<V,GRADIN>`. Here, the
   C++ declaration:

   .. math::

      \begin{aligned}
          \texttt{Log<F>::DiffT<V,GRADIN> = F::DiffT<V, Mult< Inv<F>, GRADIN> >}
      \end{aligned}

   simply encodes the well-known fact that with pointwise computations,

   .. math::

      \begin{aligned}
              \partial_{\texttt{V}} \big[ \log \circ F \big] (p, x_i, y_j) \, \cdot \,\texttt{GRADIN}
              ~=~ 
              \partial_{\texttt{V}} F (p, x_i, y_j) \, \cdot \,
              \frac{\texttt{GRADIN}}{F(p, x_i, y_j)}~.
            \end{aligned}

#. **Declare a convenient alias for the operation.**
   This arcane formulation relies on classes
   defined in the
   `keops/core/pre_headers.h <https://github.com/getkeops/keops/blob/master/keops/core/pre_headers.h>`_
   header.

Contributing with a new operation
-------------------------------------

Advanced users may wish to **extend the existing engine with home-made
operators**, injecting their C++ code within the KeOps
Map-Reduce kernels. Doing so is now relatively easy: 
having implemented a
custom instance of the :mod:`UnaryOp` or :mod:`BinaryOp` templates in 
a new `keops/core/formulas/*/*.h <https://github.com/getkeops/keops/tree/master/keops/core/formulas>`_ header, 
contributors should simply
remember to add their file to the
`list of KeOps includes <https://github.com/getkeops/keops/blob/master/keops/keops_includes.h>`_
and write a LazyTensor method in the
`pykeops/common/lazy_tensor.py <https://github.com/getkeops/keops/blob/master/pykeops/common/lazy_tensor.py>`_ 
module. 

To **get merged** in the 
`main KeOps repository <https://github.com/getkeops/keops>`_, 
which is hosted on GitHub, writing a simple 
**unit test** in the 
`pykeops/test/ <https://github.com/getkeops/keops/tree/master/pykeops/test>`_ 
folder and an **adequate description** in the
**pull request** should then be enough.


.. |br| raw:: html

  <br/><br/>
