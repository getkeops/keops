Structure of the repository
================================

But how does KeOps handle symbolic formulas on the GPU? How can its
routines outperform the CUDA backends of Deep Learning frameworks by
such a wide margin?
To answer these questions, we need to dive into the mixed
**C++/Python/Matlab codebase** of the `KeOps package <https://github.com/getkeops/keops>`_, whose
structure may be summarized as follows:

-  The `pykeops/ <(https://github.com/getkeops/keops/tree/master/pykeops>`_ folder, 
   with `common/ <https://github.com/getkeops/keops/tree/master/pykeops/common>`_, 
   `numpy/ <https://github.com/getkeops/keops/tree/master/pykeops/numpy>`_ and 
   `torch/ <https://github.com/getkeops/keops/tree/master/pykeops/torch>`_
   subfolders contains our Python wrappers and relies on the
   fantastic `PyBind11 <https://pybind11.readthedocs.io/en/stable/>`_ library.
   |br|

-  The `keopslab/ <https://github.com/getkeops/keops/tree/master/keopslab>`_ 
   folder provides a collection of entry points for
   Matlab scripts.
   |br|

-  The `keops/ <https://github.com/getkeops/keops/tree/master/keops>`_
   folder contains our C++ files and the associated
   compilation scripts. The generic KeOps engine that we are now
   about to discuss is implemented in the 
   `core/ <https://github.com/getkeops/keops/tree/master/keops/core>`_ subfolder which
   contains:
   |br|

   -  The 
      `link_autodiff.cpp <https://github.com/getkeops/keops/blob/master/keops/core/link_autodiff.cpp>`_ 
      and 
      `link_autodiff.cu <https://github.com/getkeops/keops/blob/master/keops/core/link_autodiff.cu>`_ 
      **“main” C++ files**, which define the methods that binding libraries
      may use to create high-level modules.
      |br|

   -  The `pack/ <https://github.com/getkeops/keops/blob/master/keops/core/pack>`_ 
      subfolder, which defines **abstract types for lists
      and tuples** within the C++ templating system. Using advanced
      concepts that were introduced with the C++11 revision,
      this file allows us to drive the **nvcc** compiler with
      declarative “variadic templating” and generate routines that
      manipulate an arbitrary number of parameters, :math:`i`- and
      :math:`j`-variables.
      |br|

   -  The `autodiff/ <https://github.com/getkeops/keops/blob/master/keops/core/autodiff>`_  
      subfolder, which defines the primitives of
      the KeOps symbolic syntax: 
      `variables <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/Var.h>`_, 
      abstract `unary <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/UnaryOp.h>`_ 
      and
      `binary <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/BinaryOp.h>`_
      operations, 
      `gradients <https://github.com/getkeops/keops/blob/master/keops/core/autodiff/Grad.h>`_.
      |br|

   -  The `mapreduce/GpuConv*_.cu <https://github.com/getkeops/keops/tree/master/keops/core/mapreduce>`_ 
      CUDA files, which implement our massively
      parallel Map-Reduce schemes. **These files contain the core logic
      of the KeOps library.**
      |br|

   -  The `mapreduce/CpuConv*_.cpp <https://github.com/getkeops/keops/tree/master/keops/core/mapreduce>`_ 
      C++ files, which implement simple
      Map-Reduce schemes using **standard “for” loops**. They may be
      used to test the correctness of our parallel implementations and
      provide a fall-back mode to users who do not have access to GPU
      chips on their machines.
      |br|

   -  The `reductions/ <https://github.com/getkeops/keops/tree/master/keops/core/reductions>`_ 
      subfolder, which implements the supported
      :math:`\operatorname{Reduction}` operations: sum, arg-min,
      log-sum-exp, etc.
      |br|

   -  The `formulas/ <https://github.com/getkeops/keops/tree/master/keops/core/formulas>`_ 
      subfolder, which implement the atomic
      operations that users may combine to define vector-valued formulas
      :math:`F`.

As evidenced here, the KeOps engine is heavily reliant on **modern
features of the C++ language**: every time :mod:`Genred` encounters a
new kind of generic operation (up to the values of
:math:`\mathrm{M}`, :math:`\mathrm{N}` and the data arrays which are free to change
between every call), the string that specifies a generic formula is
parsed by the compiler and a new “**.dll**” or “**.so**” shared object
is generated before being executed on the relevant Python or
Matlab tensors.


.. |br| raw:: html

  <br/><br/>