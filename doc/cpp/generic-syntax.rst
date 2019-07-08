Using the syntax in C++/Cuda code
=================================

The expressions and variables presented in the common documentation all
correspond to specific C++ types of variables defined by the KeOps
library. The C++ keyword ``auto`` allows us to define them without
having to worry about explicit type naming:

.. code:: cpp

   auto p = Pm(0,1);
   auto a = Vj(1,1);
   auto x = Vi(2,3);
   auto y = Vj(3,3);
   auto f = Square(p-y)*Exp(x+y);


Here, the ``f`` variable represents a symbolic computation; as a C++ object, it is completely useless.

The list of available operations to build a formula is given in the following table: :ref:`part.mathOperation`. 

Next we define the type of reduction that we want to perform, as follows:

.. code:: cpp

   auto Sum_f = Sum_Reduction(f,0);


This means that we want to perform a sum reduction over the "j" indices, resulting in a "i"-indexed output. 
We would use ``Sum_Reduction(f,1)`` for a reduction over the "i" indices.

The list of available reductions is given in the following table: :ref:`part.reduction`. The code name of the reduction must be followed by ``_Reduction`` in C++ code.

The convolution operation is then performed using one of these three calls:

.. code:: cpp

   EvalRed<CpuConv>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
   EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
   EvalRed<GpuConv2D_FromHost>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);

where ``pc``, ``pp``, ``pa``, ``px``, and ``py`` are pointers to their respective arrays in (Cpu) memory, ``pc`` denoting the output. These three functions correspond to computations performed repectively on the Cpu, on the Gpu with a "1D" tiling algorithm, and with a "2D" tiling algorithm.

For a minimal working example code, see the files
`./keops/examples/test_simple.cpp <https://github.com/getkeops/keops/tree/master/keops/examples/test_simple.cpp>`_ and
`./keops/examples/test_simple.cu <https://github.com/getkeops/keops/tree/master/keops/examples/test_simple.cu>`_

If data arrays are already located in the GPU memory, these functions should be favored:

.. code:: cpp

   EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
   EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);

