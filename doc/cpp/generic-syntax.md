# Using the syntax in C++/Cuda code

The expressions and variables presented in the common documentation all correspond to specific C++ types of variables defined by the KeOps library.
The C++ keyword `auto` allows us to define them without having to worry about explicit type naming:

```cpp
auto p = Pm(0,1);
auto a = Vj(1,1);
auto x = Vi(2,3);
auto y = Vj(3,3);
auto f = Square(p-y)*Exp(x+y);
```

Here, the `f` variable represents a symbolic computation; as a C++ object, it is completely useless.
However, we can retrieve its *type* -- which contains all the relevant information -- using the `decltype` keyword :

```cpp
using F = decltype(f);
```

The convolution operation is then performed using one of these three calls:

```cpp
CpuConv(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv1D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```

where `pc`, `pp`, `pa`, `px`, and `py` are pointers to their respective arrays in (Cpu) memory, `pc` denoting the output. These three functions correspond to computations performed repectively on the Cpu, on the Gpu with a "1D" tiling algorithm, and with a "2D" tiling algorithm.

If data arrays are already located in the GPU memory, these functions should be favored:

```cpp
GpuConv1D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```